"""
Vectorizer engine — vtracer, file-based API.

Sizing (area-based):
- Below 1.5MP  → upscale to 2MP
- Above 2MP    → downscale to 2MP
- 1.5–2MP      → untouched

Pipeline options (selected by mode):
  COLOR:   BilateralFilter → MaxFilter → UnsharpMask → Posterize/Quantise → vtracer (color)
  LINEART: Contrast boost → BilateralFilter → Otsu threshold → vtracer (binary)
  TEXT:    Contrast boost → Text region binarisation → vtracer (binary)

Post-processing:
  RDP path simplification → short path removal
"""
import io
import re
import tempfile
import os
import math
import struct
import zlib
import vtracer
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw

MAX_PIXELS   = 2_000_000
MIN_PIXELS   = 1_500_000
TARGET_SMALL = 2_000_000

# ── Resize ────────────────────────────────────────────────────────────────────
def _resize(img: Image.Image) -> Image.Image:
    w, h = img.size
    pixels = w * h
    if pixels < MIN_PIXELS:
        scale = math.sqrt(TARGET_SMALL / pixels)
        nw, nh = int(w * scale), int(h * scale)
        print(f'[engine] upscaled {w}x{h} → {nw}x{nh}', flush=True)
        return img.resize((nw, nh), Image.LANCZOS)
    if pixels > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / pixels)
        nw, nh = int(w * scale), int(h * scale)
        print(f'[engine] downscaled {w}x{h} → {nw}x{nh}', flush=True)
        return img.resize((nw, nh), Image.LANCZOS)
    print(f'[engine] size ok {w}x{h} ({pixels/1_000_000:.2f}MP)', flush=True)
    return img


# ── Bilateral filter (edge-preserving blur) ──────────────────────────────────
def _bilateral_filter(img: Image.Image, radius: float = 1.5) -> Image.Image:
    """
    Approximated bilateral filter using Pillow only.
    Blurs noise while preserving hard edges better than Gaussian.
    Strategy: blur a copy, then restore edges from the original via UnsharpMask.
    """
    if radius <= 0:
        return img
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    # Restore edges: blend original back using UnsharpMask
    restored = blurred.filter(ImageFilter.UnsharpMask(radius=radius*1.5, percent=60, threshold=2))
    return restored


# ── Otsu threshold (auto black/white) ────────────────────────────────────────
def _otsu_threshold(gray: Image.Image) -> Image.Image:
    """Find optimal threshold automatically using Otsu's method."""
    hist = gray.histogram()
    total = sum(hist)
    sum_all = sum(i * hist[i] for i in range(256))
    sum_b = 0
    w_b = 0
    max_var = 0
    best_t = 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / w_b
        mean_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (mean_b - mean_f) ** 2
        if var > max_var:
            max_var = var
            best_t = t
    return gray.point(lambda p: 255 if p >= best_t else 0, '1').convert('L')


# ── Text region detection + binarisation ─────────────────────────────────────
def _binarise_text_regions(img: Image.Image) -> Image.Image:
    """
    Detect high-contrast text-like regions and binarise them.
    Uses local variance to find areas with text-like patterns.
    Non-text areas stay as colour.
    """
    rgb = img.convert('RGB')
    gray = img.convert('L')
    w, h = img.size

    # Use a sliding window to find high-variance regions (text-like)
    block = 24  # block size for variance detection
    text_mask = Image.new('L', (w, h), 0)

    for y in range(0, h, block):
        for x in range(0, w, block):
            box = (x, y, min(x+block, w), min(y+block, h))
            region = gray.crop(box)
            pixels = list(region.getdata())
            if len(pixels) < 4:
                continue
            mean = sum(pixels) / len(pixels)
            variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)
            # High variance = likely text or detailed line art
            if variance > 800:
                draw = ImageDraw.Draw(text_mask)
                draw.rectangle(box, fill=255)

    # Dilate mask slightly to cover full letters
    text_mask = text_mask.filter(ImageFilter.MaxFilter(3))

    # Binarise the text regions using Otsu
    binary = _otsu_threshold(gray)

    # Composite: use binary where text mask is active, colour elsewhere
    result = rgb.copy()
    binary_rgb = Image.merge('RGB', [binary, binary, binary])
    result.paste(binary_rgb, mask=text_mask)
    return result


# ── Colour quantisation ───────────────────────────────────────────────────────
def _quantise(img: Image.Image, n_colors: int) -> Image.Image:
    """K-means style colour quantisation for cleaner colour boundaries."""
    # PIL's quantize uses median cut which is close enough and fast
    rgb = img.convert('RGB')
    quantised = rgb.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT, dither=0)
    return quantised.convert('RGB')


# ── Standard colour preprocessing ────────────────────────────────────────────
def _preprocess_color(img: Image.Image, bits: int, unsharp_radius: float,
                      unsharp_percent: int, unsharp_threshold: int,
                      blur_radius: float) -> Image.Image:
    """Standard pipeline for colour images: bilateral → unsharp → posterize."""
    rgb = img.convert('RGB')
    rgb = _bilateral_filter(rgb, blur_radius)
    rgb = rgb.filter(ImageFilter.MaxFilter(1))
    rgb = rgb.filter(ImageFilter.UnsharpMask(
        radius=unsharp_radius, percent=unsharp_percent, threshold=unsharp_threshold))
    posterized = ImageOps.posterize(rgb, bits)
    if img.mode == 'RGBA':
        posterized = posterized.convert('RGBA')
        posterized.putalpha(img.getchannel('A'))
    return posterized


# ── Line art / binary preprocessing ──────────────────────────────────────────
def _preprocess_lineart(img: Image.Image) -> Image.Image:
    """
    Pipeline for line art and text: boost contrast → bilateral → Otsu threshold.
    Returns a high-contrast B&W image for vtracer binary mode.
    """
    rgb = img.convert('RGB')
    # Boost contrast strongly
    rgb = ImageEnhance.Contrast(rgb).enhance(2.5)
    rgb = ImageEnhance.Sharpness(rgb).enhance(2.0)
    # Bilateral-style smooth to reduce noise while keeping edges
    rgb = _bilateral_filter(rgb, radius=0.8)
    # Convert to greyscale and apply Otsu
    gray = rgb.convert('L')
    binary = _otsu_threshold(gray)
    return binary.convert('RGB')


# ── SVG path simplification (RDP algorithm) ───────────────────────────────────
def _rdp_distance(point, start, end):
    """Perpendicular distance from point to line segment."""
    if start == end:
        return math.hypot(point[0]-start[0], point[1]-start[1])
    dx, dy = end[0]-start[0], end[1]-start[1]
    denom = math.hypot(dx, dy)
    return abs(dy*point[0] - dx*point[1] + end[0]*start[1] - end[1]*start[0]) / denom

def _rdp(points, epsilon):
    """Ramer-Douglas-Peucker path simplification."""
    if len(points) < 3:
        return points
    max_dist = 0
    max_idx = 0
    for i in range(1, len(points)-1):
        d = _rdp_distance(points[i], points[0], points[-1])
        if d > max_dist:
            max_dist = d
            max_idx = i
    if max_dist > epsilon:
        left = _rdp(points[:max_idx+1], epsilon)
        right = _rdp(points[max_idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]

def _simplify_svg_paths(svg: str, epsilon: float = 0.3) -> str:
    """
    Apply RDP simplification to SVG path data.
    Reduces node count on straight-ish segments while preserving curves.
    Only simplifies L (line) segments — curves (C,S,Q) are left intact.
    """
    def simplify_path(d: str) -> str:
        # Split into commands
        tokens = re.findall(r'[MmLlCcSsQqZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', d)
        result = []
        i = 0
        line_points = []
        line_start_cmd = None

        def flush_lines():
            nonlocal line_points, line_start_cmd
            if len(line_points) >= 2:
                simplified = _rdp(line_points, epsilon)
                if simplified:
                    result.append('L')
                    for pt in simplified[1:]:  # skip first (already placed by M or prev L)
                        result.append(f'{pt[0]:.2f},{pt[1]:.2f}')
            line_points = []
            line_start_cmd = None

        while i < len(tokens):
            t = tokens[i]
            if t in ('M', 'm'):
                flush_lines()
                result.append(t)
                i += 1
                if i+1 < len(tokens):
                    x, y = float(tokens[i]), float(tokens[i+1])
                    result.append(f'{x:.2f},{y:.2f}')
                    line_points = [(x, y)]
                    i += 2
            elif t == 'L':
                i += 1
                while i+1 < len(tokens) and tokens[i] not in 'MmLlCcSsQqZz':
                    x, y = float(tokens[i]), float(tokens[i+1])
                    line_points.append((x, y))
                    i += 2
            elif t in ('C', 'c', 'S', 's', 'Q', 'q'):
                flush_lines()
                result.append(t)
                i += 1
                # Consume all coordinate pairs for this curve command
                while i < len(tokens) and tokens[i] not in 'MmLlCcSsQqZz':
                    result.append(tokens[i])
                    i += 1
            elif t in ('Z', 'z'):
                flush_lines()
                result.append(t)
                i += 1
            else:
                result.append(t)
                i += 1

        flush_lines()
        return ' '.join(result)

    def replace_path(m):
        d = m.group(1)
        simplified = simplify_path(d)
        return f'd="{simplified}"'

    return re.sub(r'd="([^"]+)"', replace_path, svg)


# ── Short path removal ────────────────────────────────────────────────────────
def _remove_short_paths(svg: str, min_size: float = 2.0) -> str:
    """Remove paths whose bounding box is smaller than min_size px."""
    def is_tiny(d: str) -> bool:
        nums = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)
        if len(nums) < 4:
            return True
        coords = [float(n) for n in nums]
        xs = coords[0::2]
        ys = coords[1::2]
        if not xs or not ys:
            return True
        return (max(xs)-min(xs)) < min_size and (max(ys)-min(ys)) < min_size

    return re.sub(
        r'<path[^>]*d="([^"]+)"[^/]*/?>',
        lambda m: '' if is_tiny(m.group(1)) else m.group(0),
        svg
    )


# ── Auto-detect image type ────────────────────────────────────────────────────
def _detect_mode(img: Image.Image) -> str:
    """
    Analyse image to suggest best processing mode.
    Returns: 'color', 'lineart', or 'text'
    """
    rgb = img.convert('RGB')
    # Sample colours
    small = rgb.resize((64, 64), Image.LANCZOS)
    pixels = list(small.getdata())
    # Count unique colours (approximate)
    quantised = small.quantize(colors=16)
    unique_colors = len(set(quantised.getdata()))
    # Measure edge density via UnsharpMask response
    gray = small.convert('L')
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_pixels = list(edges.getdata())
    edge_density = sum(1 for p in edge_pixels if p > 30) / len(edge_pixels)
    print(f'[engine] detect: {unique_colors} colours, edge_density={edge_density:.2f}', flush=True)
    if unique_colors <= 4 and edge_density > 0.15:
        return 'lineart'
    if unique_colors <= 8 and edge_density > 0.25:
        return 'lineart'
    return 'color'


# ── Main entry point ──────────────────────────────────────────────────────────
def vectorize(image_data: bytes,
              posterize_bits: int    = 7,
              unsharp_radius: float  = 0.5,
              unsharp_percent: int   = 90,
              unsharp_threshold: int = 4,
              blur_radius: float     = 0.8,
              engine_mode: str       = 'auto',   # 'auto' | 'color' | 'lineart' | 'text'
              simplify: bool         = True,
              simplify_epsilon: float = 0.3,
              **kwargs) -> str:

    img = Image.open(io.BytesIO(image_data)).convert('RGBA')
    img = _resize(img)
    w, h = img.size
    print(f'[engine] input {w}x{h}', flush=True)

    # Auto-detect mode
    mode = engine_mode
    if mode == 'auto':
        mode = _detect_mode(img)
    print(f'[engine] mode={mode}', flush=True)

    # Select pipeline and vtracer params
    if mode == 'lineart':
        # Pure B&W — for sketches, outlines, black & white line drawings
        processed = _preprocess_lineart(img)
        kwargs['colormode'] = 'bw'
        kwargs.setdefault('filter_speckle', 4)
        kwargs.setdefault('corner_threshold', 60)
        kwargs.setdefault('length_threshold', 4.0)
        kwargs.setdefault('splice_threshold', 45)
        kwargs.setdefault('mode', 'spline')
        kwargs.setdefault('path_precision', 3)
        print('[engine] lineart pipeline → binary vtracer', flush=True)

    elif mode == 'text':
        # Mixed content (colour + text/logos): preserve colours faithfully,
        # just clean up edges and reduce noise for cleaner tracing.
        # Key principle: minimum colour distortion, maximum edge clarity.
        rgb = img.convert('RGB')

        # Step 1: Gentle colour correction — boost saturation slightly to
        # compensate for any fading, DO NOT boost contrast (crushes light tones)
        rgb = ImageEnhance.Color(rgb).enhance(1.15)

        # Step 2: Bilateral filter — smooth noise without blurring edges
        rgb = _bilateral_filter(rgb, radius=0.6)

        # Step 3: Gentle sharpness to crisp up text boundaries only
        rgb = ImageEnhance.Sharpness(rgb).enhance(1.4)

        # Step 4: Light unsharp mask — recovers edge definition after smoothing
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=0.8, percent=80, threshold=3))

        # Step 5: Quantise with more colours to preserve gradient richness
        # 48 colours is enough to approximate gradients without banding
        rgb = _quantise(rgb, n_colors=48)

        # Preserve alpha
        if img.mode == 'RGBA':
            rgb = rgb.convert('RGBA')
            rgb.putalpha(img.getchannel('A'))
        processed = rgb

        # Moderate corner threshold — sharper than colour but not as aggressive as lineart
        # Higher color_precision to preserve more colour layers
        kwargs.setdefault('colormode', 'color')
        kwargs.setdefault('mode', 'spline')
        kwargs.setdefault('corner_threshold', 30)
        kwargs.setdefault('length_threshold', 3.5)
        kwargs.setdefault('splice_threshold', 30)
        kwargs.setdefault('path_precision', 2)
        print('[engine] text/mixed pipeline → gentle colour-preserving + edge clarity', flush=True)

    else:  # color
        processed = _preprocess_color(img, posterize_bits, unsharp_radius,
                                      unsharp_percent, unsharp_threshold, blur_radius)
        kwargs.setdefault('colormode', 'color')
        kwargs.setdefault('mode', 'spline')
        kwargs.setdefault('corner_threshold', 1)
        kwargs.setdefault('length_threshold', 3.5)
        kwargs.setdefault('splice_threshold', 1)
        print(f'[engine] color pipeline (blur={blur_radius}, posterize={posterize_bits}bits)', flush=True)

    inp = out = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            processed.save(f.name, format='PNG')
            inp = f.name
        out = inp.replace('.png', '.svg')
        print('[engine] vtracer running…', flush=True)
        vtracer.convert_image_to_svg_py(inp, out, **kwargs)
        svg = open(out, encoding='utf-8').read()
        paths_before = svg.count('<path')
        print(f'[engine] {paths_before} paths before post-processing', flush=True)

        # Post-processing
        svg = _remove_short_paths(svg, min_size=2.0)
        if simplify:
            svg = _simplify_svg_paths(svg, epsilon=simplify_epsilon)
        paths_after = svg.count('<path')
        print(f'[engine] {paths_after} paths after post-processing '
              f'(removed {paths_before - paths_after})', flush=True)

        return svg
    finally:
        if inp and os.path.exists(inp): os.unlink(inp)
        if out and os.path.exists(out): os.unlink(out)
