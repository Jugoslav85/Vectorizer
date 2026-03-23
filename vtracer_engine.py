"""
Vectorizer engine — vtracer only, file-based API.

Sizing (area-based):
- Below 1.5MP  → upscale to 2MP
- Above 2MP    → downscale to 2MP
- 1.5–2MP      → untouched

Pre-processing: GaussianBlur → MaxFilter → UnsharpMask → Posterize
"""
import io
import tempfile
import os
import math
import vtracer
from PIL import Image, ImageFilter, ImageOps

MAX_PIXELS   = 2_000_000
MIN_PIXELS   = 1_500_000
TARGET_SMALL = 2_000_000


def _resize(img: Image.Image) -> Image.Image:
    w, h = img.size
    pixels = w * h
    if pixels < MIN_PIXELS:
        scale = math.sqrt(TARGET_SMALL / pixels)
        new_w, new_h = int(w * scale), int(h * scale)
        print(f'[engine] upscaled {w}x{h} → {new_w}x{new_h}', flush=True)
        return img.resize((new_w, new_h), Image.LANCZOS)
    if pixels > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / pixels)
        new_w, new_h = int(w * scale), int(h * scale)
        print(f'[engine] downscaled {w}x{h} → {new_w}x{new_h}', flush=True)
        return img.resize((new_w, new_h), Image.LANCZOS)
    print(f'[engine] size ok {w}x{h} ({pixels/1_000_000:.2f}MP)', flush=True)
    return img


def _preprocess(img: Image.Image, bits: int, radius: float, percent: int,
                threshold: int, blur_radius: float) -> Image.Image:
    rgb = img.convert('RGB')
    # Smooth curved edges before tracing
    if blur_radius > 0:
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    # Dilate slightly to preserve small elements
    rgb = rgb.filter(ImageFilter.MaxFilter(1))
    # Recover sharp hard edges
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    posterized = ImageOps.posterize(rgb, bits)
    if img.mode == 'RGBA':
        posterized = posterized.convert('RGBA')
        posterized.putalpha(img.getchannel('A'))
    return posterized


def vectorize(image_data: bytes,
              posterize_bits: int   = 7,
              unsharp_radius: float = 0.5,
              unsharp_percent: int  = 90,
              unsharp_threshold: int = 4,
              blur_radius: float    = 0.8,
              **kwargs) -> str:

    img = Image.open(io.BytesIO(image_data)).convert('RGBA')
    img = _resize(img)
    w, h = img.size
    print(f'[engine] input {w}x{h}', flush=True)

    img = _preprocess(img, posterize_bits, unsharp_radius, unsharp_percent,
                      unsharp_threshold, blur_radius)
    print(f'[engine] preprocessed (blur={blur_radius}, unsharp r={unsharp_radius} '
          f'p={unsharp_percent} t={unsharp_threshold}, posterize={posterize_bits}bits)', flush=True)

    inp = out = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name, format='PNG')
            inp = f.name
        out = inp.replace('.png', '.svg')
        print('[engine] using vtracer (file-based)…', flush=True)
        vtracer.convert_image_to_svg_py(inp, out, **kwargs)
        svg = open(out, encoding='utf-8').read()
        print(f'[engine] {svg.count("<path")} paths', flush=True)
        return svg
    finally:
        if inp and os.path.exists(inp): os.unlink(inp)
        if out and os.path.exists(out): os.unlink(out)
