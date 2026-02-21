import io
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import cm
from torch import nn


class EfficientNetB0GradCAM:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

        self.target_layer = self._find_last_conv_layer()
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    def _find_last_conv_layer(self) -> nn.Module:
        if not hasattr(self.model, "features"):
            raise ValueError("Model khong co thuoc tinh 'features' de tim conv layer cuoi.")

        last_conv = None
        for module in self.model.features.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module

        if last_conv is None:
            raise ValueError("Khong tim thay Conv2d layer trong model.features.")
        return last_conv

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> tuple[np.ndarray, int, torch.Tensor]:
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("input_tensor phai co shape [1, C, H, W].")

        input_tensor = input_tensor.to(self.device)
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        target_idx = pred_idx if class_idx is None else class_idx

        score = logits[:, target_idx]
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Khong lay duoc activation/gradient tu hook.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.detach().cpu().numpy()
        return cam_np, pred_idx, probs[0].detach().cpu()

    @staticmethod
    def _cam_to_color(cam_np: np.ndarray) -> np.ndarray:
        color_map = cm.get_cmap("jet")
        heatmap = color_map(cam_np)[..., :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        return heatmap

    def overlay_on_image(
        self,
        original_image: Image.Image,
        cam_np: np.ndarray,
        alpha: float = 0.45,
    ) -> Image.Image:
        heatmap = self._cam_to_color(cam_np)
        heatmap_img = Image.fromarray(heatmap).convert("RGB")

        base_img = original_image.convert("RGB").resize(
            (heatmap_img.width, heatmap_img.height),
            Image.Resampling.BILINEAR,
        )
        overlay = Image.blend(base_img, heatmap_img, alpha=alpha)
        return overlay

    def heatmap_image(self, cam_np: np.ndarray) -> Image.Image:
        return Image.fromarray(self._cam_to_color(cam_np)).convert("RGB")

    @staticmethod
    def pil_to_png_bytes(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def create_gradcam_overlay(
    model: nn.Module,
    device: torch.device,
    image: Image.Image,
    transform,
    class_idx: Optional[int] = None,
    alpha: float = 0.45,
) -> tuple[Image.Image, np.ndarray, int, torch.Tensor]:
    gradcam = EfficientNetB0GradCAM(model=model, device=device)
    input_tensor = transform(image).unsqueeze(0)
    cam_np, pred_idx, probs = gradcam.generate_cam(input_tensor=input_tensor, class_idx=class_idx)
    overlay = gradcam.overlay_on_image(original_image=image, cam_np=cam_np, alpha=alpha)
    return overlay, cam_np, pred_idx, probs


# Backward compatibility for old imports.
EfficientNetB3GradCAM = EfficientNetB0GradCAM
