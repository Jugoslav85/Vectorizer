"""
Vectorizer engine — vtracer only.

All images (gradients, illustrations, logos, art) go through vtracer.
Input is capped at MAX_INPUT_PX on the longest side to control memory/time.
"""
import io
import vtracer
from PIL import Image

MAX_INPUT_PX = 1200  # cap longest side before processing


def _cap_size(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) <= MAX_INPUT_PX:
        return img
    scale = MAX_INPUT_PX / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    print(f'[engine] resized {w}x{h} → {new_w}x{new_h}', flush=True)
    return img.resize((new_w, new_h), Image.LANCZOS)


def vectorize(image_data: bytes, **kwargs) -> str:
    img = Image.open(io.BytesIO(image_data)).convert('RGBA')
    img = _cap_size(img)
    w, h = img.size
    print(f'[engine] input {w}x{h}', flush=True)

    pixels = list(img.getdata())
    print('[engine] using vtracer…', flush=True)
    svg = vtracer.convert_pixels_to_svg(pixels, (w, h), **kwargs)
    print(f'[engine] {svg.count("<path")} paths', flush=True)
    return svg
