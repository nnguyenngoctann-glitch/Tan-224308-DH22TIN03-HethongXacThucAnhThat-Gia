import argparse
import copy
import os
import time
import warnings

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

# Cho phep doc anh kich thuoc lon trong qua trinh train dataset thuc te.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.simplefilter("ignore", Image.DecompressionBombWarning)


def pil_rgb_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        img.load()
        # Chan anh qua lon gay MemoryError trong DataLoader worker.
        max_pixels = 40_000_000
        if img.width * img.height > max_pixels:
            img.thumbnail((4096, 4096), Image.Resampling.BILINEAR)
        return img.convert("RGB")


def build_dataloaders(data_root: str, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform, loader=pil_rgb_loader)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform, loader=pil_rgb_loader)

    if len(train_dataset.classes) != 2:
        raise ValueError(
            f"Can dung 2 lop du lieu, nhung tim thay {len(train_dataset.classes)} lop trong {train_dir}"
        )

    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Class train/val khong khop nhau.")

    print(f"Classes: {train_dataset.classes}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def build_model(device: torch.device) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model = model.to(device)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = running_correct / max(total, 1)
    return epoch_loss, epoch_acc


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(args.data_root, args.batch_size, args.num_workers)
    model = build_model(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                },
                args.output,
            )
            print(f"Saved best model to {args.output} (val_acc={best_val_acc:.4f})")

    elapsed = time.time() - start_time
    print(f"Training done in {elapsed / 60:.2f} minutes. Best Val Acc: {best_val_acc:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 for AI image detection")
    parser.add_argument("--data-root", type=str, default="du_lieu", help="Root folder containing train/val/test")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for Adam")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--output", type=str, default="best_efficientnet_b0.pth", help="Best model output file")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
