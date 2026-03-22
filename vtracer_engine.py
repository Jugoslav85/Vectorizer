"""
Vectorizer engine.

Two-path pipeline:
1. Gradient detection — if the image has no hard edges (pure gradient),
   generate a native SVG linearGradient or mesh of gradients. 
   This produces a perfect smooth SVG gradient, not stepped bands.
   
2. vtracer — for all other images (illustrations, logos, photos),
   run vtracer at full resolution.
"""
import io
import re
import numpy as np
import vtracer
from PIL import Image


# ── Gradient detection ────────────────────────────────────────────────────────

def _is_gradient(a: np.ndarray) -> bool:
    """
    Returns True if the image is a smooth gradient with no hard edges.
    Hard edge = neighboring pixels differ by >50 in any channel.
    If less than 1% of transitions are hard edges, it's a gradient.
    """
    dh = np.abs(a[1:,:].astype(np.int16) - a[:-1,:].astype(np.int16)).max(axis=2)
    dv = np.abs(a[:,1:].astype(np.int16) - a[:,:-1].astype(np.int16)).max(axis=2)
    return float((dh > 50).mean()) < 0.01 and float((dv > 50).mean()) < 0.01


def _sample_gradient_stops(a: np.ndarray, axis: int, n_stops: int = 20) -> list:
    """
    Sample n_stops colors along the given axis (0=vertical, 1=horizontal).
    Returns list of (offset_float, hex_color) tuples.
    Averages across the middle third of the perpendicular axis for stability.
    """
    h, w = a.shape[:2]
    stops = []
    for i in range(n_stops):
        t = i / (n_stops - 1)
        if axis == 0:  # vertical
            y  = int(t * (h - 1))
            x1, x2 = w // 3, 2 * w // 3
            color = a[y, x1:x2].mean(axis=0).astype(int)
        else:  # horizontal
            x  = int(t * (w - 1))
            y1, y2 = h // 3, 2 * h // 3
            color = a[y1:y2, x].mean(axis=0).astype(int)
        stops.append((t, f'#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}'))
    return stops


def _gradient_to_svg(img: Image.Image) -> str:
    """
    Convert a gradient image to an SVG using native linearGradient elements.
    Handles: vertical, horizontal, and 2D (diagonal/mesh) gradients.
    """
    a   = np.array(img.convert('RGB'))
    h, w = a.shape[:2]

    # Determine gradient direction
    horiz_var = float(np.std(a[0,:].astype(float), axis=0).mean())   # top row variance
    vert_var  = float(np.std(a[:,0].astype(float), axis=0).mean())   # left col variance

    # Check if it varies significantly in both directions (2D gradient)
    both = horiz_var > 10 and vert_var > 10

    if both:
        # 2D gradient: use two overlapping gradients (horizontal + vertical)
        # Sample both axes
        h_stops = _sample_gradient_stops(a, axis=1, n_stops=16)  # horizontal
        v_stops = _sample_gradient_stops(a, axis=0, n_stops=16)  # vertical

        h_stop_els = '\n      '.join(
            f'<stop offset="{p:.1%}" stop-color="{c}"/>' for p, c in h_stops
        )
        v_stop_els = '\n      '.join(
            f'<stop offset="{p:.1%}" stop-color="{c}" stop-opacity="0.6"/>' for p, c in v_stops
        )

        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <defs>
    <linearGradient id="hg" x1="0" y1="0" x2="1" y2="0" gradientUnits="objectBoundingBox">
      {h_stop_els}
    </linearGradient>
    <linearGradient id="vg" x1="0" y1="0" x2="0" y2="1" gradientUnits="objectBoundingBox">
      {v_stop_els}
    </linearGradient>
  </defs>
  <rect width="{w}" height="{h}" fill="url(#hg)"/>
  <rect width="{w}" height="{h}" fill="url(#vg)"/>
</svg>'''

    elif vert_var >= horiz_var:
        # Vertical gradient (top to bottom)
        stops    = _sample_gradient_stops(a, axis=0, n_stops=20)
        stop_els = '\n      '.join(
            f'<stop offset="{p:.1%}" stop-color="{c}"/>' for p, c in stops
        )
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="0" y2="1" gradientUnits="objectBoundingBox">
      {stop_els}
    </linearGradient>
  </defs>
  <rect width="{w}" height="{h}" fill="url(#g)"/>
</svg>'''

    else:
        # Horizontal gradient (left to right)
        stops    = _sample_gradient_stops(a, axis=1, n_stops=20)
        stop_els = '\n      '.join(
            f'<stop offset="{p:.1%}" stop-color="{c}"/>' for p, c in stops
        )
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="0" gradientUnits="objectBoundingBox">
      {stop_els}
    </linearGradient>
  </defs>
  <rect width="{w}" height="{h}" fill="url(#g)"/>
</svg>'''

    return svg


# ── Main vectorize function ───────────────────────────────────────────────────

def vectorize(image_data: bytes, **kwargs) -> str:
    img = Image.open(io.BytesIO(image_data)).convert('RGBA')
    w, h = img.size
    a    = np.array(img.convert('RGB'))

    print(f'[engine] input {w}x{h}', flush=True)

    if _is_gradient(a):
        print('[engine] detected gradient → generating SVG gradient', flush=True)
        svg = _gradient_to_svg(img)
        print(f'[engine] gradient SVG done ({len(svg)} chars)', flush=True)
        return svg

    # Not a gradient — use vtracer
    pixels = list(img.getdata())
    print(f'[engine] using vtracer…', flush=True)
    svg = vtracer.convert_pixels_to_svg(pixels, (w, h), **kwargs)
    print(f'[engine] vtracer done — {svg.count("<path")} paths', flush=True)
    return svg
