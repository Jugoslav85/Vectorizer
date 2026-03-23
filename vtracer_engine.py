"""
Vectorizer engine — vtracer only, file-based API.

Sizing (area-based):
- Below 1.5MP  → upscale to 2MP
- Above 2MP    → downscale to 2MP
- 1.5–2MP      → untouched

Pre-processing pipeline:
  (optional) rembg background removal
  → GaussianBlur → MaxFilter → UnsharpMask → Posterize → vtracer
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

# Lazy-load rembg session to avoid import cost on startup
# _rembg_session = None

# def _get_rembg_session():
#     global _rembg_session
#     if _rembg_session is None:
#         from rembg import new_session
#        _rembg_session = new_session('birefnet-general')
#         print('[engine] rembg session loaded', flush=True)
#     return _rembg_session


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


def _remove_background(img: Image.Image) -> Image.Image:
    """Remove image background using rembg U2Net model."""
    from rembg import remove
    print('[engine] removing background…', flush=True)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    result_bytes = remove(buf.getvalue(), session=_get_rembg_session())
    result = Image.open(io.BytesIO(result_bytes)).convert('RGBA')
    print('[engine] background removed', flush=True)
    return result


def _preprocess(img: Image.Image, bits: int, radius: float,
                percent: int, threshold: int, blur_radius: float) -> Image.Image:
    rgb = img.convert('RGB')
    if blur_radius > 0:
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    rgb = rgb.filter(ImageFilter.MaxFilter(1))
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
              remove_bg: bool       = False,
              **kwargs) -> str:

    img = Image.open(io.BytesIO(image_data)).convert('RGBA')

    # Background removal before resize (works better at full res)
    if remove_bg:
        img = _remove_background(img)

    img = _resize(img)
    w, h = img.size
    print(f'[engine] input {w}x{h}', flush=True)

    img = _preprocess(img, posterize_bits, unsharp_radius, unsharp_percent,
                      unsharp_threshold, blur_radius)
    print(f'[engine] preprocessed (blur={blur_radius}, posterize={posterize_bits}bits)', flush=True)

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
