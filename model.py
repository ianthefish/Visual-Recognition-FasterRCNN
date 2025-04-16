import os
import json
import torch
import torch.utils.data as data
from plot import plot_batch_losses, plot_loss_and_map
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
from dataset import DigitDataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class Config:
    DATA_ROOT = "nycu-hw2-data"
    TRAIN_JSON = os.path.join(DATA_ROOT, "train.json")
    VALID_JSON = os.path.join(DATA_ROOT, "valid.json")
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VALID_DIR = os.path.join(DATA_ROOT, "valid")
    TEST_DIR = os.path.join(DATA_ROOT, "test")

    BATCH_SIZE = 8
    NUM_WORKERS = 0

    NUM_CLASSES = 10 + 1

    LEARNING_RATE = 0.006

    NUM_EPOCHS = 15

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONFIDENCE_THRESHOLD = 0.7

    CHECKPOINT_PATH = "checkpoint_best.pth"
    CHECKPOINT_DIR = "checkpoints"
    PLOT_DIR = "plots"

    PRED_JSON_PATH = "pred.json"
    PRED_CSV_PATH = "pred.csv"


config = Config()

os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(config.PLOT_DIR, exist_ok=True)


def get_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets


def collate_fn_test(batch):
    images = []
    image_ids = []
    for image, image_id in batch:
        images.append(image)
        image_ids.append(image_id)
    return images, image_ids


class ExtraConvFPN(torch.nn.Module):
    def __init__(self, fpn_module, channels=256):
        super().__init__()
        self.fpn_module = fpn_module
        # create one extra conv layer to be applied to each feature map.
        self.extra_conv = torch.nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        features = self.fpn_module(x)
        refined_features = {}
        for k, feat in features.items():
            refined_features[k] = self.extra_conv(feat)
        return refined_features


def get_model(num_classes):
    """
    Build a FasterRCNN-ResNet50-FPN model
    """
    weight = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weight)

    # the extra convolution layer ofr FPN

    # model.backbone.fpn = ExtraConvFPN(model.backbone.fpn)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if os.path.exists(config.CHECKPOINT_PATH):
        print(f"Loading weights from {config.CHECKPOINT_PATH}")
        model.load_state_dict(
            torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
        )

    return model


# training
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    batch_losses = []

    for images, targets in tqdm(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_value = losses.item()
        total_loss += loss_value
        batch_losses.append(loss_value)

    # return both total epoch loss and individual batch losses
    return total_loss / len(data_loader), batch_losses


def evaluate_coco(model, data_loader, device, annotation_file):
    """
    Evaluate the model using full COCO-style mAP.

    Parameters:
      - model: the detection model
      - data_loader: validation DataLoader
      - device: device on which the model is running
      - annotation_file: path to the COCO-format ground truth annotation file

    Returns:
      - mAP: the primary COCO mAP metric
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, target in enumerate(targets):
                image_id = (
                    target["image_id"].item()
                    if isinstance(target["image_id"], torch.Tensor)
                    else target["image_id"]
                )

                pred = outputs[i]
                for box, label, score in zip(
                    pred["boxes"], pred["labels"], pred["scores"]
                ):
                    if score >= config.CONFIDENCE_THRESHOLD:
                        x_min, y_min, x_max, y_max = box.tolist()
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                        predictions.append(
                            {
                                "image_id": image_id,
                                "category_id": label.item(),
                                "bbox": bbox,
                                "score": score.item(),
                            }
                        )

    temp_pred_file = os.path.join(
        config.PLOT_DIR,
        "temp_predictions_coco.json"
    )
    with open(temp_pred_file, "w") as f:
        json.dump(predictions, f)
    print(f"Saved temporary predictions file: {temp_pred_file}")

    coco_gt = COCO(annotation_file)
    coco_dt = coco_gt.loadRes(temp_pred_file)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = coco_eval.stats[0]
    return mAP


# evaluation
def evaluate(model, data_loader, device):
    model.eval()

    with torch.no_grad():
        predictions = []

        for images, targets in tqdm(data_loader):
            images = [image.to(device) for image in images]
            outputs = model(images)

            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                pred = outputs[i]

                for box, label, score in zip(
                    pred["boxes"], pred["labels"], pred["scores"]
                ):
                    if score >= config.CONFIDENCE_THRESHOLD:
                        predictions.append(
                            {
                                "image_id": image_id,
                                "bbox": box.tolist(),
                                "score": score.item(),
                                "category_id": label.item(),
                            }
                        )

    return predictions


# task1: generate predictions for digit detection
def generate_task1_predictions(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, image_ids in tqdm(data_loader):
            images = [image.to(device) for image in images]
            outputs = model(images)

            for i, image_id in enumerate(image_ids):
                pred = outputs[i]

                for box, label, score in zip(
                    pred["boxes"], pred["labels"], pred["scores"]
                ):
                    if score >= config.CONFIDENCE_THRESHOLD:
                        x_min, y_min, x_max, y_max = box.tolist()
                        w = x_max - x_min
                        h = y_max - y_min

                        predictions.append(
                            {
                                "image_id": image_id,
                                "bbox": [x_min, y_min, w, h],
                                "score": score.item(),
                                "category_id": label.item(),
                            }
                        )

    return predictions


# task2: generate predictions for whole number recognition
def generate_task2_predictions(task1_predictions):
    # group predictions by image_id
    predictions_by_image = {}
    for pred in task1_predictions:
        image_id = pred["image_id"]
        if image_id not in predictions_by_image:
            predictions_by_image[image_id] = []
        predictions_by_image[image_id].append(pred)

    task2_predictions = []
    for image_id, preds in predictions_by_image.items():
        if not preds:
            # no digits detected
            task2_predictions.append({"image_id": image_id, "pred_label": -1})
            continue

        preds.sort(key=lambda x: x["bbox"][0])

        # convert category_id to digit
        whole_number = "".join([str(p["category_id"] - 1) for p in preds])

        task2_predictions.append(
            {"image_id": image_id, "pred_label": int(whole_number)}
        )

    return task2_predictions


def train_model():
    # prepare datasets and loaders
    train_dataset = DigitDataset(
        json_file=config.TRAIN_JSON,
        img_dir=config.TRAIN_DIR,
        transforms=get_transform(),
    )
    valid_dataset = DigitDataset(
        json_file=config.VALID_JSON,
        img_dir=config.VALID_DIR,
        transforms=get_transform(),
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # initialize model, optimizer, and scheduler
    model = get_model(config.NUM_CLASSES)
    model.to(config.DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config.LEARNING_RATE,
        momentum=0.9,
        weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.35
    )

    train_losses = []
    val_metrics = []
    lr_history = []
    best_map = float(0)

    print(f"Using device: {config.DEVICE}")

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)

        epoch_loss, batch_losses = train_one_epoch(
            model, optimizer, train_loader, config.DEVICE
        )
        train_losses.append(epoch_loss)
        print(f"Training Loss: {epoch_loss:.4f}, Learning Rate: {current_lr}")

        # plot batch loss for the current epoch if desired
        plot_batch_losses(epoch, batch_losses, config.PLOT_DIR)

        # evaluate using full COCO mAP on validation set
        print("Evaluating on validation set using full COCO mAP...")
        mAP = evaluate_coco(
            model,
            valid_loader,
            config.DEVICE,
            config.VALID_JSON
        )
        val_metrics.append(mAP)
        print(f"Validation COCO mAP: {mAP:.4f}")

        # save model checkpoint for the epoch
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
                "lr": current_lr,
            },
            checkpoint_path,
        )
        print(f"checkpoint saved to {checkpoint_path}")

        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)
            print(f"best model saved to {config.CHECKPOINT_PATH}")

        lr_scheduler.step()

        plot_loss_and_map(
            train_losses,
            val_metrics,
            lr_history,
            config.PLOT_DIR
        )

    plot_loss_and_map(train_losses, val_metrics, lr_history, config.PLOT_DIR)
    metrics_df = pd.DataFrame(
        {
            "epoch": list(range(1, config.NUM_EPOCHS + 1)),
            "train_loss": train_losses,
            "learning_rate": lr_history,
            "validation_mAP": val_metrics,
        }
    )
    metrics_csv_path = os.path.join(config.PLOT_DIR, "training_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"training metrics saved to {metrics_csv_path}")

    return model


# generate predictions
def generate_predictions():
    # load trained model
    model = get_model(config.NUM_CLASSES)
    model.load_state_dict(
        torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    )
    model.to(config.DEVICE)
    model.eval()

    # prepare test dataset
    test_dataset = DigitDataset(
        json_file=None,
        img_dir=config.TEST_DIR,
        transforms=get_transform(),
        is_test=True,
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn_test,
    )

    task1_predictions = generate_task1_predictions(
        model,
        test_loader,
        config.DEVICE
    )

    # save task1 predictions
    with open(config.PRED_JSON_PATH, "w") as f:
        json.dump(task1_predictions, f)
    print(f"task1 predictions saved to {config.PRED_JSON_PATH}")

    task2_predictions = generate_task2_predictions(task1_predictions)

    # save task2 predictions
    df = pd.DataFrame(task2_predictions)
    df.to_csv(config.PRED_CSV_PATH, index=False)
    print(f"task2 predictions saved to {config.PRED_CSV_PATH}")


def main():
    # either load a checkpoint or train from scratch
    if not os.path.exists(config.CHECKPOINT_PATH):
        print("No checkpoint found. Training model...")
        train_model()
    else:
        print(f"Checkpoint found at {config.CHECKPOINT_PATH}.")

    # train_model()

    generate_predictions()


if __name__ == "__main__":
    main()
