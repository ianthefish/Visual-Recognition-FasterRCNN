import os
import json
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class DigitDataset(data.Dataset):
    def __init__(self, json_file, img_dir, transforms=None, is_test=False):
        self.img_dir = img_dir
        self.transforms = transforms
        self.is_test = is_test

        if not is_test:
            with open(json_file, "r") as f:
                self.coco_data = json.load(f)

            self.id_to_filename = {
                img["id"]: img["file_name"] for img in self.coco_data["images"]
            }
            self.annotations = {}
            for ann in self.coco_data["annotations"]:
                image_id = ann["image_id"]
                if image_id not in self.annotations:
                    self.annotations[image_id] = []
                self.annotations[image_id].append(ann)

            self.image_ids = list(self.id_to_filename.keys())
        else:
            self.image_files = sorted(
                os.listdir(img_dir), key=lambda x: int(os.path.splitext(x)[0])
            )
            self.image_ids = [
                int(os.path.splitext(file)[0]) for file in self.image_files
            ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        if not self.is_test:
            filename = self.id_to_filename[image_id]
            img_path = os.path.join(self.img_dir, filename)
        else:
            img_path = os.path.join(self.img_dir, f"{image_id}.png")

        img = Image.open(img_path).convert("RGB")

        if self.transforms:
            image_tensor = self.transforms(img)
        else:
            image_tensor = transforms.ToTensor()(img)

        if self.is_test:
            return image_tensor, image_id

        annotations = self.annotations.get(image_id, [])
        boxes, labels = [], []

        for ann in annotations:
            x_min, y_min, w, h = ann["bbox"]
            x_max = x_min + w
            y_max = y_min + h
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann["category_id"])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(image_id)
        }

        return image_tensor, target
