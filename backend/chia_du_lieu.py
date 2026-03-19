import argparse
import os
import random
import shutil

# Dinh dang file anh hop le
DINH_DANG_ANH = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chia du lieu anh thanh train/val/test")
    parser.add_argument("--source-root", type=str, default="du_lieu_nguon", help="Thu muc chua du lieu nguon")
    parser.add_argument("--target-root", type=str, default="du_lieu", help="Thu muc dich train/val/test")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ti le train")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ti le val")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ti le test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--classes", type=str, default="that,gia", help="Danh sach lop, cach nhau boi dau phay")
    return parser.parse_args()


def ensure_output_dirs(target_root: str, classes: list[str], splits: list[tuple[str, float]]) -> None:
    for split_name, _ in splits:
        for class_name in classes:
            os.makedirs(os.path.join(target_root, split_name, class_name), exist_ok=True)


def list_image_files(folder: str) -> list[str]:
    files: list[str] = []
    if not os.path.isdir(folder):
        return files

    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in DINH_DANG_ANH:
            files.append(path)
    return files


def gather_class_images(source_root: str, class_name: str) -> tuple[list[str], list[str]]:
    """
    Ho tro 2 kieu cau truc nguon:
    1) du_lieu_nguon/<class_name>
    2) du_lieu_nguon/<dataset con>/<class_name>
    """
    candidate_dirs: list[str] = []

    direct_dir = os.path.join(source_root, class_name)
    if os.path.isdir(direct_dir):
        candidate_dirs.append(direct_dir)

    for item in os.listdir(source_root):
        subdir = os.path.join(source_root, item)
        if not os.path.isdir(subdir):
            continue
        class_dir = os.path.join(subdir, class_name)
        if os.path.isdir(class_dir):
            candidate_dirs.append(class_dir)

    all_files: list[str] = []
    for folder in candidate_dirs:
        all_files.extend(list_image_files(folder))

    # Loai trung lap theo duong dan file
    all_files = list(dict.fromkeys(all_files))
    return all_files, candidate_dirs


def split_items(items: list[str], splits: list[tuple[str, float]]) -> dict[str, list[str]]:
    random.shuffle(items)
    total = len(items)

    train_ratio = splits[0][1]
    val_ratio = splits[1][1]

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def copy_without_overwrite(paths: list[str], target_dir: str) -> tuple[int, int]:
    copied = 0
    skipped = 0

    for src_path in paths:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(target_dir, filename)

        if os.path.exists(dst_path):
            skipped += 1
            continue

        shutil.copy2(src_path, dst_path)
        copied += 1

    return copied, skipped


def main() -> None:
    args = parse_args()

    classes = [x.strip() for x in args.classes.split(",") if x.strip()]
    if len(classes) < 2:
        raise ValueError("Can it nhat 2 lop trong --classes, vi du: that,gia")

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("Tong train-ratio + val-ratio + test-ratio phai = 1.0")

    splits = [
        ("train", args.train_ratio),
        ("val", args.val_ratio),
        ("test", args.test_ratio),
    ]

    if not os.path.isdir(args.source_root):
        raise FileNotFoundError(f"Khong tim thay thu muc nguon: {args.source_root}")

    random.seed(args.seed)
    ensure_output_dirs(args.target_root, classes, splits)

    print("Bat dau chia du lieu...")
    print(f"Nguon: {os.path.abspath(args.source_root)}")
    print(f"Dich : {os.path.abspath(args.target_root)}")
    print(f"Classes: {classes}")

    for class_name in classes:
        images, source_dirs = gather_class_images(args.source_root, class_name)
        split_map = split_items(images, splits)

        train_dir = os.path.join(args.target_root, "train", class_name)
        val_dir = os.path.join(args.target_root, "val", class_name)
        test_dir = os.path.join(args.target_root, "test", class_name)

        train_copied, train_skipped = copy_without_overwrite(split_map["train"], train_dir)
        val_copied, val_skipped = copy_without_overwrite(split_map["val"], val_dir)
        test_copied, test_skipped = copy_without_overwrite(split_map["test"], test_dir)

        print(f"\n[{class_name}]")
        print(f"- Source folders: {len(source_dirs)}")
        for src in source_dirs:
            print(f"  + {src}")
        print(f"- Total images: {len(images)}")
        print(
            f"- Split => train={len(split_map['train'])}, val={len(split_map['val'])}, test={len(split_map['test'])}"
        )
        print(
            f"- Copy  => copied={train_copied + val_copied + test_copied}, "
            f"skipped={train_skipped + val_skipped + test_skipped}"
        )

    print("\nHoan tat.")


if __name__ == "__main__":
    main()
