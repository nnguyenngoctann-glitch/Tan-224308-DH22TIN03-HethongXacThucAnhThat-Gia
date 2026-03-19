import os
from pathlib import Path
from urllib.request import urlopen


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as resp, dst.open("wb") as f:
        f.write(resp.read())


def main() -> None:
    checkpoint_path = os.getenv(
        "CHECKPOINT_PATH", "artifacts/bs16/best_efficientnet_b0.pth"
    )
    model_url = os.getenv("MODEL_URL", "").strip()
    if not model_url:
        print("MODEL_URL not set. Skip download.")
        return

    dst = Path(checkpoint_path)
    if dst.exists():
        print(f"Checkpoint already exists: {dst}")
        return

    print(f"Downloading model to {dst} ...")
    download(model_url, dst)
    print("Download complete.")


if __name__ == "__main__":
    main()
