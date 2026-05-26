"""Bundled demo assets."""

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
SAMPLE_IMAGE = ASSETS_DIR / "demo_workshop.png"


def ensure_demo_image() -> Path:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if SAMPLE_IMAGE.exists():
        return SAMPLE_IMAGE
    img = Image.new("RGB", (400, 300), color=(40, 44, 52))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 80, 350, 220], outline=(100, 180, 255), width=4)
    draw.text((120, 140), "DEMO ASSET", fill=(220, 230, 255))
    img.save(SAMPLE_IMAGE)
    return SAMPLE_IMAGE


def demo_image_bytes() -> bytes:
    path = ensure_demo_image()
    return path.read_bytes()
