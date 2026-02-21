import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import warnings
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

# Dong bo voi train: cho phep anh lon va bo qua canh bao decompression bomb.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.simplefilter("ignore", Image.DecompressionBombWarning)


def pil_rgb_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def build_test_loader(data_root: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, List[str]]:
    test_dir = os.path.join(data_root, "test")
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform, loader=pil_rgb_loader)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Classes: {test_dataset.classes}")
    print(f"Test samples: {len(test_dataset)}")
    return test_loader, test_dataset.classes


def build_model(device: torch.device) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model.to(device)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint: {checkpoint_path}")


def confusion_matrix_2class(y_true: List[int], y_pred: List[int]) -> torch.Tensor:
    cm = torch.zeros((2, 2), dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_metrics(cm: torch.Tensor) -> Tuple[float, float, float, float]:
    total = cm.sum().item()
    accuracy = ((cm[0, 0] + cm[1, 1]).item() / total) if total > 0 else 0.0

    precisions = []
    recalls = []
    f1s = []
    for cls_idx in range(2):
        tp = cm[cls_idx, cls_idx].item()
        fp = cm[:, cls_idx].sum().item() - tp
        fn = cm[cls_idx, :].sum().item() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    precision_macro = sum(precisions) / 2
    recall_macro = sum(recalls) / 2
    f1_macro = sum(f1s) / 2
    return accuracy, precision_macro, recall_macro, f1_macro


def roc_points_binary(y_true: List[int], y_score_pos: List[float]) -> Tuple[List[float], List[float], float]:
    thresholds = sorted(set(y_score_pos), reverse=True)
    thresholds = [1.1] + thresholds + [-0.1]

    fprs: List[float] = []
    tprs: List[float] = []
    p_count = sum(y_true)
    n_count = len(y_true) - p_count

    for thr in thresholds:
        y_hat = [1 if s >= thr else 0 for s in y_score_pos]
        tp = sum(1 for t, p in zip(y_true, y_hat) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_hat) if t == 0 and p == 1)

        tpr = tp / p_count if p_count > 0 else 0.0
        fpr = fp / n_count if n_count > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    points = sorted(zip(fprs, tprs), key=lambda x: x[0])
    sorted_fprs = [p[0] for p in points]
    sorted_tprs = [p[1] for p in points]

    auc = 0.0
    for i in range(1, len(sorted_fprs)):
        x0, x1 = sorted_fprs[i - 1], sorted_fprs[i]
        y0, y1 = sorted_tprs[i - 1], sorted_tprs[i]
        auc += (x1 - x0) * (y0 + y1) / 2

    return sorted_fprs, sorted_tprs, auc


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_loader, class_names = build_test_loader(args.data_root, args.batch_size, args.num_workers)
    if len(class_names) != 2:
        raise ValueError(f"Can dung 2 lop de danh gia, nhung tim thay {len(class_names)} lop.")

    model = build_model(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    y_score_pos: List[float] = []
    sample_lines: List[str] = []

    image_paths = test_loader.dataset.samples

    with torch.no_grad():
        sample_idx = 0
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_score_pos.extend(probs[:, 1].cpu().tolist())

            probs_cpu = probs.cpu().tolist()
            labels_cpu = labels.cpu().tolist()
            preds_cpu = preds.cpu().tolist()

            for b in range(len(labels_cpu)):
                img_path = image_paths[sample_idx][0]
                line = (
                    f"{img_path} | "
                    f"p({class_names[0]})={probs_cpu[b][0]:.4f}, "
                    f"p({class_names[1]})={probs_cpu[b][1]:.4f} | "
                    f"pred={class_names[preds_cpu[b]]} | true={class_names[labels_cpu[b]]}"
                )
                sample_lines.append(line)
                sample_idx += 1

    cm = confusion_matrix_2class(y_true, y_pred)
    accuracy, precision, recall, f1 = compute_metrics(cm)

    print("\n=== Metrics ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\n=== Confusion Matrix ===")
    print(f"Rows=true, Cols=pred ({class_names[0]}, {class_names[1]})")
    print(cm)

    fprs, tprs, auc = roc_points_binary(y_true, y_score_pos)
    plt.figure(figsize=(6, 6))
    plt.plot(fprs, tprs, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.roc_output, dpi=150)
    print(f"\nSaved ROC curve: {args.roc_output}")

    if args.show_plot:
        plt.show()

    print("\n=== Xac suat du doan tung lop ===")
    for line in sample_lines:
        print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-B0 on test dataset")
    parser.add_argument("--data-root", type=str, default="du_lieu", help="Root folder containing train/val/test")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_efficientnet_b0.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--roc-output", type=str, default="roc_curve.png", help="Output path for ROC figure")
    parser.add_argument("--show-plot", action="store_true", help="Show ROC plot window")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
