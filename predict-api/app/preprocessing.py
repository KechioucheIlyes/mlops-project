from __future__ import annotations

from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_image_as_tensor(image_bytes: bytes, image_size: int) -> torch.Tensor:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    transform = build_transform(image_size)
    tensor = transform(image).unsqueeze(0)
    return tensor