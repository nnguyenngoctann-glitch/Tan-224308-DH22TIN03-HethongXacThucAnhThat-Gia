import argparse
import csv
import json
import os
import platform
import time
from datetime import datetime, timezone
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
try:
    from backend.model_factory import SUPPORTED_BACKBONES, build_model
except ImportError:
    from model_factory import SUPPORTED_BACKBONES, build_model

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


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
    return test_loader, test_dataset.classes


def load_checkpoint(model, checkpoint_path: str, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    model.load_state_dict(checkpoint)
    return {}


def confusion_matrix_2class(y_true: List[int], y_pred: List[int]) -> torch.Tensor:
    cm = torch.zeros((2, 2), dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_metrics(cm: torch.Tensor) -> Tuple[float, float, float, float]:
    total = cm.sum().item()
    accuracy = ((cm[0, 0] + cm[1, 1]).item() / total) if total > 0 else 0.0

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

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


def evaluate_model(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score_pos: List[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_score_pos.extend(probs[:, 1].cpu().tolist())

    cm = confusion_matrix_2class(y_true, y_pred)
    accuracy, precision, recall, f1 = compute_metrics(cm)
    fprs, tprs, auc = roc_points_binary(y_true, y_score_pos)

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "roc_fpr": fprs,
        "roc_tpr": tprs,
    }


def get_process_rss_mb() -> float | None:
    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
            return proc.memory_info().rss / (1024 * 1024)
        except Exception:
            return None

    if platform.system().lower() != "windows":
        return None

    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
        GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        ok = GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        if not ok:
            return None
        return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        return None


def benchmark_latency(model, loader: DataLoader, device: torch.device, max_samples: int) -> dict:
    model.eval()
    timings_ms: List[float] = []
    processed = 0
    ram_peak_mb = None

    with torch.no_grad():
        for images, _ in loader:
            for i in range(images.size(0)):
                if processed >= max_samples:
                    break
                x = images[i : i + 1].to(device, non_blocking=True)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                timings_ms.append((t1 - t0) * 1000.0)
                processed += 1

                rss_mb = get_process_rss_mb()
                if rss_mb is not None:
                    ram_peak_mb = rss_mb if ram_peak_mb is None else max(ram_peak_mb, rss_mb)

            if processed >= max_samples:
                break

    if not timings_ms:
        return {
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "throughput_img_per_s": None,
            "latency_samples": 0,
            "ram_peak_mb": None,
        }

    timings_ms_sorted = sorted(timings_ms)
    p50_idx = int(0.50 * (len(timings_ms_sorted) - 1))
    p95_idx = int(0.95 * (len(timings_ms_sorted) - 1))
    total_s = sum(timings_ms) / 1000.0

    return {
        "latency_p50_ms": timings_ms_sorted[p50_idx],
        "latency_p95_ms": timings_ms_sorted[p95_idx],
        "throughput_img_per_s": (len(timings_ms) / total_s) if total_s > 0 else None,
        "latency_samples": len(timings_ms),
        "ram_peak_mb": ram_peak_mb,
    }


def maybe_save_roc(roc_fpr: List[float], roc_tpr: List[float], auc: float, output_path: str) -> None:
    if not output_path:
        return
    plt.figure(figsize=(6, 6))
    plt.plot(roc_fpr, roc_tpr, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def write_csv(summary: dict, output_csv: str) -> None:
    fieldnames = [
        "model",
        "checkpoint",
        "device",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc",
        "latency_p50_ms",
        "latency_p95_ms",
        "throughput_img_per_s",
        "ram_peak_mb",
        "vram_peak_mb",
        "val_acc_from_checkpoint",
        "timestamp_utc",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: summary.get(k) for k in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export metrics for model checkpoint")
    parser.add_argument("--data-root", type=str, default="du_lieu")
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
        choices=sorted(SUPPORTED_BACKBONES),
        help="Model backbone",
    )
    parser.add_argument("--checkpoint", type=str, default="best_efficientnet_b0.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latency-samples", type=int, default=300)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--roc-output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = args.backbone.lower()

    output_json = args.output_json or f"metrics_{backbone}.json"
    output_csv = args.output_csv or f"metrics_{backbone}.csv"
    roc_output = args.roc_output or f"roc_curve_{backbone}.png"

    loader, class_names = build_test_loader(args.data_root, args.batch_size, args.num_workers)
    if len(class_names) != 2:
        raise ValueError(f"Expected 2 classes, got {len(class_names)}")

    model = build_model(backbone, num_classes=2).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    eval_out = evaluate_model(model, loader, device)
    latency_out = benchmark_latency(model, loader, device, max_samples=args.latency_samples)

    vram_peak_mb = None
    if device.type == "cuda":
        vram_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    maybe_save_roc(eval_out["roc_fpr"], eval_out["roc_tpr"], eval_out["roc_auc"], roc_output)

    summary = {
        "model": backbone,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "classes": class_names,
        "accuracy": eval_out["accuracy"],
        "precision_macro": eval_out["precision_macro"],
        "recall_macro": eval_out["recall_macro"],
        "f1_macro": eval_out["f1_macro"],
        "roc_auc": eval_out["roc_auc"],
        "confusion_matrix": eval_out["confusion_matrix"],
        "latency_p50_ms": latency_out["latency_p50_ms"],
        "latency_p95_ms": latency_out["latency_p95_ms"],
        "throughput_img_per_s": latency_out["throughput_img_per_s"],
        "latency_samples": latency_out["latency_samples"],
        "ram_peak_mb": latency_out["ram_peak_mb"],
        "vram_peak_mb": vram_peak_mb,
        "val_acc_from_checkpoint": checkpoint.get("val_acc") if isinstance(checkpoint, dict) else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "notes": {
            "roc_curve_file": roc_output,
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_csv(summary, output_csv)

    print("Exported metrics:")
    print(f"- JSON: {output_json}")
    print(f"- CSV : {output_csv}")
    print(f"- ROC : {roc_output}")
    print(f"Accuracy={summary['accuracy']:.4f} | F1={summary['f1_macro']:.4f} | AUC={summary['roc_auc']:.4f}")
    print(
        f"Latency p50={summary['latency_p50_ms']:.2f} ms | "
        f"p95={summary['latency_p95_ms']:.2f} ms | "
        f"Throughput={summary['throughput_img_per_s']:.2f} img/s | "
        f"RAM peak={summary['ram_peak_mb']:.2f} MB"
    )


if __name__ == "__main__":
    main()
