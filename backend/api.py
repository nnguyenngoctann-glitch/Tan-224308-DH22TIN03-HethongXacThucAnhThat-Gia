import io
import os
import base64

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFile
from pydantic import BaseModel
from torch import nn
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

try:
    from backend.grad_cam import create_gradcam_overlay, EfficientNetB0GradCAM
except ImportError:
    from grad_cam import create_gradcam_overlay, EfficientNetB0GradCAM


class PredictResponse(BaseModel):
    label: str
    confidence: float


class PredictWithCamResponse(PredictResponse):
    cam_overlay_base64: str
    cam_heatmap_base64: str


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_model(device: torch.device) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model.to(device)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Khong tim thay checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def parse_labels() -> list[str]:
    # Mac dinh phu hop thu tu class khi train voi ImageFolder: ['gia', 'that'].
    labels_raw = os.getenv("CLASS_LABELS", "gia,that")
    labels = [x.strip() for x in labels_raw.split(",") if x.strip()]
    if len(labels) != 2:
        raise ValueError("CLASS_LABELS phai co dung 2 nhan, vi du: gia,that")
    return labels


def is_image_file(filename: str) -> bool:
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    _, ext = os.path.splitext(filename.lower())
    return ext in valid_ext


CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "best_efficientnet_b0.pth")
MODEL_TYPE = "efficientnet_b0"
UNCERTAIN_THRESHOLD = float(os.getenv("UNCERTAIN_THRESHOLD", "0.7"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_LABELS = parse_labels()
IMAGE_TRANSFORM = build_transform()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
FRONTEND_INDEX = os.path.join(FRONTEND_DIR, "index.html")

# Tranh loi do anh qua lon trong luong API.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = FastAPI(title="AI Image Detector API")
MODEL = build_model(DEVICE)
load_checkpoint(MODEL, CHECKPOINT_PATH, DEVICE)
MODEL.eval()
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    if not os.path.exists(FRONTEND_INDEX):
        raise HTTPException(status_code=404, detail="Khong tim thay frontend/index.html")
    return FileResponse(FRONTEND_INDEX)


@app.get("/style.css")
def serve_style() -> FileResponse:
    css_path = os.path.join(FRONTEND_DIR, "style.css")
    if not os.path.exists(css_path):
        raise HTTPException(status_code=404, detail="Khong tim thay frontend/style.css")
    return FileResponse(css_path, media_type="text/css")


@app.get("/app.js")
def serve_script() -> FileResponse:
    js_path = os.path.join(FRONTEND_DIR, "app.js")
    if not os.path.exists(js_path):
        raise HTTPException(status_code=404, detail="Khong tim thay frontend/app.js")
    return FileResponse(js_path, media_type="application/javascript")


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_type": MODEL_TYPE,
        "checkpoint": CHECKPOINT_PATH,
        "labels": CLASS_LABELS,
        "uncertain_threshold": UNCERTAIN_THRESHOLD,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Thieu ten file upload.")
    if not is_image_file(file.filename):
        raise HTTPException(status_code=400, detail="File khong phai dinh dang anh hop le.")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Khong doc duoc anh: {exc}") from exc

    x = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1)[0]
        confidence, pred_idx = torch.max(probs, dim=0)

    conf = float(confidence.item())
    label = CLASS_LABELS[pred_idx.item()] if conf >= UNCERTAIN_THRESHOLD else "uncertain"
    return PredictResponse(label=label, confidence=round(conf, 4))


@app.post("/predict-with-cam", response_model=PredictWithCamResponse)
async def predict_with_cam(file: UploadFile = File(...)) -> PredictWithCamResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Thieu ten file upload.")
    if not is_image_file(file.filename):
        raise HTTPException(status_code=400, detail="File khong phai dinh dang anh hop le.")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Khong doc duoc anh: {exc}") from exc

    try:
        overlay, cam_np, pred_idx, probs = create_gradcam_overlay(
            model=MODEL,
            device=DEVICE,
            image=image,
            transform=IMAGE_TRANSFORM,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Khong tao duoc Grad-CAM: {exc}") from exc

    confidence = float(probs[pred_idx].item())
    label = CLASS_LABELS[pred_idx] if confidence >= UNCERTAIN_THRESHOLD else "uncertain"

    cam_color = EfficientNetB0GradCAM._cam_to_color(cam_np)  # noqa: SLF001
    cam_pil = Image.fromarray(cam_color).convert("RGB")
    cam_img = EfficientNetB0GradCAM.pil_to_png_bytes(cam_pil)
    overlay_img = EfficientNetB0GradCAM.pil_to_png_bytes(overlay)

    return PredictWithCamResponse(
        label=label,
        confidence=round(confidence, 4),
        cam_overlay_base64=base64.b64encode(overlay_img).decode("utf-8"),
        cam_heatmap_base64=base64.b64encode(cam_img).decode("utf-8"),
    )
