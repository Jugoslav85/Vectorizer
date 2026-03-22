"""
Vectorizer engine.

Two-path pipeline:
1. Gradient detection — smooth gradients get native SVG gradient elements.
   Uses corner sampling + radial overlays for complex 2D gradients.
2. vtracer — all other images (illustrations, logos, art).

Also caps input to MAX_INPUT_PX before any processing.
"""
import io
import numpy as np
import vtracer
from PIL import Image

MAX_INPUT_PX = 1500  # cap longest side before processing


# ── Resize cap ────────────────────────────────────────────────────────────────

def _cap_size(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) <= MAX_INPUT_PX:
        return img
    scale = MAX_INPUT_PX / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    print(f'[engine] resized {w}x{h} → {new_w}x{new_h}', flush=True)
    return img.resize((new_w, new_h), Image.LANCZOS)


# ── Gradient detection ────────────────────────────────────────────────────────

def _is_gradient(a: np.ndarray) -> bool:
    """True if image has no hard edges (pure smooth gradient)."""
    dh = np.abs(a[1:,:].astype(np.int16) - a[:-1,:].astype(np.int16)).max(axis=2)
    dv = np.abs(a[:,1:].astype(np.int16) - a[:,:-1].astype(np.int16)).max(axis=2)
    return float((dh > 50).mean()) < 0.01 and float((dv > 50).mean()) < 0.01


# ── Gradient SVG generation ───────────────────────────────────────────────────

def _hex(color) -> str:
    r, g, b = int(color[0]), int(color[1]), int(color[2])
    return f'#{r:02x}{g:02x}{b:02x}'

def _sample(a, y_frac, x_frac, radius=0.05):
    """Sample average color around a fractional position."""
    h, w = a.shape[:2]
    y = int(y_frac * (h - 1))
    x = int(x_frac * (w - 1))
    r = max(1, int(min(h, w) * radius))
    y1, y2 = max(0, y-r), min(h, y+r)
    x1, x2 = max(0, x-r), min(w, x+r)
    return a[y1:y2, x1:x2].mean(axis=(0,1))

def _gradient_to_svg(img: Image.Image) -> str:
    a    = np.array(img.convert('RGB'))
    h, w = a.shape[:2]

    # Sample key points
    tl  = _sample(a, 0.0, 0.0)
    tr  = _sample(a, 0.0, 1.0)
    bl  = _sample(a, 1.0, 0.0)
    br  = _sample(a, 1.0, 1.0)
    c   = _sample(a, 0.5, 0.5)
    tm  = _sample(a, 0.0, 0.5)   # top middle
    bm  = _sample(a, 1.0, 0.5)   # bottom middle
    ml  = _sample(a, 0.5, 0.0)   # middle left
    mr  = _sample(a, 0.5, 1.0)   # middle right

    # Determine primary gradient direction
    horiz_var = float(np.std(a[h//2,:].astype(float), axis=0).mean())
    vert_var  = float(np.std(a[:,w//2].astype(float), axis=0).mean())
    both      = horiz_var > 8 and vert_var > 8

    defs = []
    rects = []
    gid   = 0

    if not both:
        # Simple 1D gradient
        if vert_var >= horiz_var:
            stops = _make_stops(a, axis=0, n=24)
            defs.append(f'<linearGradient id="g0" x1="0" y1="0" x2="0" y2="1" gradientUnits="objectBoundingBox">{stops}</linearGradient>')
        else:
            stops = _make_stops(a, axis=1, n=24)
            defs.append(f'<linearGradient id="g0" x1="0" y1="0" x2="1" y2="0" gradientUnits="objectBoundingBox">{stops}</linearGradient>')
        rects.append(f'<rect width="{w}" height="{h}" fill="url(#g0)"/>')
    else:
        # Complex 2D gradient — build from:
        # 1. Diagonal base gradient (TL→BR)
        # 2. Opposite diagonal (TR→BL) blended
        # 3. Radial overlays at corner colors and center hotspot

        # Base: TL to BR
        defs.append(f'''<linearGradient id="g_diag1" x1="0" y1="0" x2="1" y2="1" gradientUnits="objectBoundingBox">
      <stop offset="0%" stop-color="{_hex(tl)}"/>
      <stop offset="50%" stop-color="{_hex(c)}"/>
      <stop offset="100%" stop-color="{_hex(br)}"/>
    </linearGradient>''')
        rects.append(f'<rect width="{w}" height="{h}" fill="url(#g_diag1)"/>')

        # TR corner radial overlay
        defs.append(f'''<radialGradient id="r_tr" cx="100%" cy="0%" r="70%" gradientUnits="objectBoundingBox">
      <stop offset="0%" stop-color="{_hex(tr)}" stop-opacity="0.85"/>
      <stop offset="100%" stop-color="{_hex(tr)}" stop-opacity="0"/>
    </radialGradient>''')
        rects.append(f'<rect width="{w}" height="{h}" fill="url(#r_tr)"/>')

        # BL corner radial overlay
        defs.append(f'''<radialGradient id="r_bl" cx="0%" cy="100%" r="70%" gradientUnits="objectBoundingBox">
      <stop offset="0%" stop-color="{_hex(bl)}" stop-opacity="0.85"/>
      <stop offset="100%" stop-color="{_hex(bl)}" stop-opacity="0"/>
    </radialGradient>''')
        rects.append(f'<rect width="{w}" height="{h}" fill="url(#r_bl)"/>')

        # Center hotspot if significantly different from diagonal blend
        center_blend = (tl + br) / 2
        center_diff  = float(np.abs(c.astype(float) - center_blend).mean())
        if center_diff > 15:
            defs.append(f'''<radialGradient id="r_c" cx="50%" cy="50%" r="55%" gradientUnits="objectBoundingBox">
      <stop offset="0%" stop-color="{_hex(c)}" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="{_hex(c)}" stop-opacity="0"/>
    </radialGradient>''')
            rects.append(f'<rect width="{w}" height="{h}" fill="url(#r_c)"/>')

        # Extra edge radial overlays for TM and BM if they add info
        for name, cx, cy, color in [
            ('tm', '50%', '0%',   tm),
            ('bm', '50%', '100%', bm),
            ('ml', '0%',  '50%',  ml),
            ('mr', '100%','50%',  mr),
        ]:
            # Only add if edge color is meaningfully different from nearby corners
            diff = float(np.abs(color.astype(float) - c.astype(float)).mean())
            if diff > 20:
                defs.append(f'''<radialGradient id="r_{name}" cx="{cx}" cy="{cy}" r="60%" gradientUnits="objectBoundingBox">
      <stop offset="0%" stop-color="{_hex(color)}" stop-opacity="0.5"/>
      <stop offset="100%" stop-color="{_hex(color)}" stop-opacity="0"/>
    </radialGradient>''')
                rects.append(f'<rect width="{w}" height="{h}" fill="url(#r_{name})"/>')

    defs_str  = '\n    '.join(defs)
    rects_str = '\n  '.join(rects)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <defs>
    {defs_str}
  </defs>
  {rects_str}
</svg>'''


def _make_stops(a, axis, n=24):
    h, w = a.shape[:2]
    stops = []
    for i in range(n):
        t = i / (n - 1)
        if axis == 0:
            y  = int(t * (h-1))
            color = a[y, w//3:2*w//3].mean(axis=0)
        else:
            x  = int(t * (w-1))
            color = a[h//3:2*h//3, x].mean(axis=0)
        stops.append(f'<stop offset="{t:.1%}" stop-color="{_hex(color)}"/>')
    return '\n      '.join(stops)


# ── Main vectorize function ───────────────────────────────────────────────────

def vectorize(image_data: bytes, **kwargs) -> str:
    img = Image.open(io.BytesIO(image_data)).convert('RGBA')

    # Cap size
    img = _cap_size(img)
    w, h = img.size

    a = np.array(img.convert('RGB'))
    print(f'[engine] input {w}x{h}', flush=True)

    if _is_gradient(a):
        print('[engine] gradient detected → SVG gradient', flush=True)
        svg = _gradient_to_svg(img)
        print(f'[engine] done ({len(svg)} chars)', flush=True)
        return svg

    # vtracer path
    pixels = list(img.getdata())
    print('[engine] using vtracer…', flush=True)
    svg = vtracer.convert_pixels_to_svg(pixels, (w, h), **kwargs)
    print(f'[engine] {svg.count("<path")} paths', flush=True)
    return svg
