import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_loss_and_map(train_losses, val_metrics, lr_history, save_dir):
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    color = "tab:blue"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color=color)
    ax1.plot(epochs, train_losses, color=color, marker="o", label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Validation mAP", color=color)
    ax2.plot(epochs, val_metrics, color=color, marker="s", label="Validation mAP")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Training Loss and Validation mAP Over Epochs")
    fig.tight_layout()

    plot_path = os.path.join(save_dir, "loss_and_map.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Loss and mAP plot saved to {plot_path}")


def plot_metrics(train_losses, val_metrics, lr_history, plot_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "training_loss.png"))
    plt.close()

    if val_metrics:
        plt.figure(figsize=(10, 5))
        epochs = [i * 5 for i in range(len(val_metrics))]
        plt.plot(epochs, val_metrics, marker="o")
        plt.title("Validation Detection Rate Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Detection Rate")
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, "validation_metrics.png"))
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(lr_history) + 1), lr_history, marker="o")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.yscale("log")
    plt.savefig(os.path.join(plot_dir, "learning_rate.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.scatter(lr_history, train_losses)
    plt.title("Learning Rate vs. Training Loss")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.xscale("log")
    plt.savefig(os.path.join(plot_dir, "lr_vs_loss.png"))
    plt.close()


def plot_batch_losses(epoch, batch_losses, plot_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(batch_losses) + 1), batch_losses, marker=".")
    plt.title(f"Batch Losses for Epoch {epoch+1}")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"batch_losses_epoch_{epoch+1}.png"))
    plt.close()


def visualize_predictions(image_path, predictions, output_path=None):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for pred in predictions:
        x_min, y_min, w, h = pred["bbox"]
        category_id = pred["category_id"]
        score = pred["score"]

        cv2.rectangle(
            img,
            (int(x_min), int(y_min)),
            (int(x_min + w), int(y_min + h)),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            img,
            f"{category_id}: {score:.2f}",
            (int(x_min), int(y_min) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()
