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


# ── Layer separation vectorisation ───────────────────────────────────────────

def _detect_edges_pillow(img: Image.Image, threshold: int = 20) -> Image.Image:
    """
    Detect hard edges using Pillow FIND_EDGES filter.
    Returns a greyscale mask — white where edges are, black elsewhere.
    """
    gray = img.convert('L')
    edges = gray.filter(ImageFilter.FIND_EDGES)
    # Threshold — only keep strong edges
    edges = edges.point(lambda p: 255 if p > threshold else 0)
    # Dilate slightly to capture full stroke width
    edges = edges.filter(ImageFilter.MaxFilter(1))
    return edges


def _detect_gradients(img: Image.Image, block: int = 16,
                       var_threshold: float = 400) -> Image.Image:
    """
    Detect gradient/complex areas using local variance.
    Returns a mask — white where gradients are detected.
    """
    gray = img.convert('L')
    w, h = img.size
    grad_mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(grad_mask)

    for y in range(0, h, block):
        for x in range(0, w, block):
            box = (x, y, min(x+block, w), min(y+block, h))
            region = gray.crop(box)
            pixels = list(region.getdata())
            if len(pixels) < 4:
                continue
            mean = sum(pixels) / len(pixels)
            variance = sum((p - mean)**2 for p in pixels) / len(pixels)
            if variance > var_threshold:
                draw.rectangle(box, fill=200)

    # Smooth the mask
    grad_mask = grad_mask.filter(ImageFilter.GaussianBlur(radius=block//2))
    return grad_mask


def _apply_mask(img: Image.Image, mask: Image.Image,
                invert: bool = False) -> Image.Image:
    """Apply a mask to an image, returning RGBA with transparency where mask is black."""
    rgba = img.convert('RGBA')
    if invert:
        mask = mask.point(lambda p: 255 - p)
    # Ensure mask is L mode
    m = mask.convert('L')
    rgba.putalpha(m)
    return rgba


def _merge_svgs(svgs: list) -> str:
    """
    Merge multiple SVG strings into one, stacking layers in order.
    Background rectangles stripped from upper layers so they don't cover lower ones.
    """
    if not svgs:
        return ''
    if len(svgs) == 1:
        return svgs[0]

    import re as _re

    first = svgs[0]
    svg_open_match = _re.search(r'<svg[^>]+>', first)
    if not svg_open_match:
        return first
    svg_open = svg_open_match.group(0)

    def get_inner(svg_str, strip_bg=False):
        s = _re.sub(r'<\?xml[^>]+\?>', '', svg_str)
        s = _re.sub(r'<!DOCTYPE[^>]+>', '', s)
        s = _re.sub(r'<svg[^>]+>', '', s)
        s = _re.sub(r'</svg>', '', s)
        if strip_bg:
            # Remove vtracer background rectangle (first rect element)
            s = _re.sub(r'<rect[^/]*/>', '', s, count=1)
            s = _re.sub(r'<rect[^>]+></rect>', '', s, count=1)
        return s.strip()

    merged_content = []
    for i, svg in enumerate(svgs):
        inner = get_inner(svg, strip_bg=(i > 0))
        if inner:
            merged_content.append('  <g id="layer{}">'.format(i))
            merged_content.append(inner)
            merged_content.append('  </g>')

    return svg_open + '\n' + '\n'.join(merged_content) + '\n</svg>'


def _vectorize_layer(img_pil: Image.Image, mode: str, **kwargs) -> str:
    """
    Vectorize a single PIL image layer, returns SVG string.
    mode: 'bw' or 'color'
    """
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    raw = buf.getvalue()

    inp = out = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img_pil.save(f.name, format='PNG')
            inp = f.name
        out = inp.replace('.png', '_layer.svg')
        vtracer.convert_image_to_svg_py(inp, out, colormode=mode, **kwargs)
        return open(out, encoding='utf-8').read()
    finally:
        if inp and os.path.exists(inp): os.unlink(inp)
        if out and os.path.exists(out): os.unlink(out)


def vectorize_layered(image_data: bytes,
                      simplify: bool = True,
                      simplify_epsilon: float = 0.3) -> str:
    """
    Single-pass composite approach:
    1. Preprocess image for good colour (bilateral + quantise)
    2. Detect edges and burn them into the colour image as hard boundaries
    3. Run vtracer once on the composite → clean edges + preserved colours

    This avoids the SVG merge alignment problems of the three-pass approach.
    """
    img = Image.open(io.BytesIO(image_data)).convert('RGBA')
    img = _resize(img)
    w, h = img.size
    print(f'[layered] input {w}x{h}', flush=True)

    rgb = img.convert('RGB')

    # ── Step 1: Build colour base layer ──────────────────────────────────────
    print('[layered] building colour base…', flush=True)
    colour = rgb.copy()
    # Gentle saturation boost to separate colours
    colour = ImageEnhance.Color(colour).enhance(1.2)
    # Bilateral to smooth noise within colour regions
    colour = _bilateral_filter(colour, radius=0.8)
    # Quantise to clean colour set
    colour = _quantise(colour, n_colors=64)

    # ── Step 2: Build edge mask ───────────────────────────────────────────────
    print('[layered] detecting edges…', flush=True)
    # Use original (not colour-processed) for edge detection — more accurate
    edge_mask = _detect_edges_pillow(rgb, threshold=20)

    # ── Step 3: Burn edges into colour image ──────────────────────────────────
    # Where edges are detected, push those pixels toward black in the colour image
    # This creates hard, dark boundaries that vtracer traces as clean paths
    print('[layered] burning edges into colour image…', flush=True)
    colour_arr = list(colour.getdata())
    edge_arr = list(edge_mask.getdata())
    composite_arr = []
    for i, (r, g, b) in enumerate(colour_arr):
        edge_strength = edge_arr[i] / 255.0
        if edge_strength > 0.3:
            # Darken toward black proportionally to edge strength
            factor = 1.0 - (edge_strength * 0.85)
            composite_arr.append((
                int(r * factor),
                int(g * factor),
                int(b * factor)
            ))
        else:
            composite_arr.append((r, g, b))

    composite = Image.new('RGB', (w, h))
    composite.putdata(composite_arr)

    # ── Step 4: Final sharpening on composite ────────────────────────────────
    # Recover any edge softening from the bilateral step
    composite = composite.filter(
        ImageFilter.UnsharpMask(radius=0.8, percent=100, threshold=2))

    # Restore alpha if present
    if img.mode == 'RGBA':
        composite = composite.convert('RGBA')
        composite.putalpha(img.getchannel('A'))

    # ── Step 5: Single vtracer pass ───────────────────────────────────────────
    print('[layered] running vtracer on composite…', flush=True)
    inp = out = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            composite.save(f.name, format='PNG')
            inp = f.name
        out = inp.replace('.png', '_layered.svg')
        vtracer.convert_image_to_svg_py(
            inp, out,
            colormode='color',
            mode='spline',
            corner_threshold=28,
            length_threshold=3.2,
            filter_speckle=3,
            color_precision=8,
            layer_difference=2,
            splice_threshold=30,
            path_precision=3,
        )
        svg = open(out, encoding='utf-8').read()
        paths = svg.count('<path')
        print(f'[layered] {paths} paths before post-processing', flush=True)

        svg = _remove_short_paths(svg, min_size=2.0)
        if simplify:
            svg = _simplify_svg_paths(svg, epsilon=simplify_epsilon)

        print(f'[layered] {svg.count("<path")} paths after post-processing',
              flush=True)
        return svg
    finally:
        if inp and os.path.exists(inp): os.unlink(inp)
        if out and os.path.exists(out): os.unlink(out)



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

    # Layered mode runs its own pipeline
    if engine_mode == 'layered':
        print('[engine] layered mode → three-layer separation', flush=True)
        return vectorize_layered(
            image_data,
            simplify=simplify,
            simplify_epsilon=simplify_epsilon,
        )

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
        # smooth letter/logo edges using the same bilateral approach as lineart.
        rgb = img.convert('RGB')

        # Step 1: Strong saturation boost to push minority colours (red, gold)
        # far enough from dominant colours that quantisation won't merge them
        rgb = ImageEnhance.Color(rgb).enhance(1.5)

        # Step 2: Contrast boost — just enough to separate colour boundaries
        # without crushing light tones (keep below 1.4)
        rgb = ImageEnhance.Contrast(rgb).enhance(1.3)

        # Step 3: Bilateral filter with higher radius — same as lineart magic.
        # Smooths noise WITHIN colour regions, preserves hard edges between them.
        # This is what makes letter curves smooth instead of jagged.
        rgb = _bilateral_filter(rgb, radius=1.2)

        # Step 4: Strong sharpening AFTER bilateral, BEFORE quantisation.
        # Bilateral softened edges slightly — unsharp mask recovers them hard.
        # Hard edges going into vtracer = smooth spline output.
        rgb = ImageEnhance.Sharpness(rgb).enhance(1.8)
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=1))

        # Step 5: Quantise to 80 colours — more colours = minority colours
        # (red, gold) less likely to be merged into dominant green/blue
        rgb = _quantise(rgb, n_colors=80)

        # Preserve alpha
        if img.mode == 'RGBA':
            rgb = rgb.convert('RGBA')
            rgb.putalpha(img.getchannel('A'))
        processed = rgb

        # High corner_threshold — same principle as lineart.
        # Only corners at genuine direction changes, everything else smooth curves.
        kwargs.setdefault('colormode', 'color')
        kwargs.setdefault('mode', 'spline')
        kwargs.setdefault('corner_threshold', 55)
        kwargs.setdefault('length_threshold', 3.0)
        kwargs.setdefault('splice_threshold', 45)
        kwargs.setdefault('path_precision', 3)
        print('[engine] text/mixed → sat1.5+contrast1.3+bilateral1.2+unsharp150+quant80',
              flush=True)

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
