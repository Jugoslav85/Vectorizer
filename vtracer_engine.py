"""
Vectorizer engine — vtracer backend.

No blurring anywhere in the pipeline. Sharp edges preserved throughout.

Pipeline (colour):
  iterative_upscale     (small images — LANCZOS only, no post-step blur)
  -> guided_filter      (edge-preserving noise reduction via NumPy)
  -> median_filter      (speckle removal, preserves hard edges)
  -> max_colour_quantise (hard palette cap)
  -> colour_dedup       (merge near-identical palette entries)
  -> vtracer color
  -> svg_colour_dedup   (merge near-identical SVG fill colours)
  -> remove_short_paths
  -> rdp_simplify

Pipeline (lineart):
  iterative_upscale
  -> guided_filter
  -> median_filter
  -> contrast + sharpness boost
  -> otsu_threshold
  -> morph_close
  -> vtracer bw
  -> remove_short_paths
  -> rdp_simplify
"""

import io
import re
import tempfile
import os
import math
import struct
import vtracer

from PIL import Image, ImageFilter, ImageOps, ImageEnhance

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False
    print('[engine] numpy not available — guided filter disabled', flush=True)


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

TARGET_SIMPLE = 600_000    # 0.6MP — logos, flat art
TARGET_NORMAL = 1_000_000  # 1.0MP — illustrations, lineart
MIN_UPSCALE   = 300_000    # upscale anything below 0.3MP


def _count_unique_colours(img):
    small = img.convert('RGB').resize((64, 64), Image.LANCZOS)
    return len(set(small.quantize(colors=64, dither=0).getdata()))


def _iterative_upscale(img, target_px):
    """
    For small images: upscale in 1.5x steps rather than one big LANCZOS jump.
    LANCZOS is already the sharpest resampling filter available — no post-step
    sharpening needed and no blur introduced.
    """
    w, h   = img.size
    pixels = w * h
    if pixels >= MIN_UPSCALE:
        return img

    print(f'[engine] iterative upscale {w}x{h} -> target {target_px//1000}k px', flush=True)
    current = img
    while True:
        cw, ch = current.size
        cpx    = cw * ch
        if cpx >= target_px:
            break
        scale   = min(1.5, math.sqrt(target_px / cpx))
        nw, nh  = int(cw * scale), int(ch * scale)
        current = current.resize((nw, nh), Image.LANCZOS)
        print(f'[engine]   step -> {nw}x{nh}', flush=True)
    return current


def _resize_down(img, target_px):
    """Downscale only — used when image is already large enough."""
    w, h   = img.size
    pixels = w * h
    if pixels <= target_px:
        return img
    scale  = math.sqrt(target_px / pixels)
    nw, nh = int(w * scale), int(h * scale)
    print(f'[engine] downscaled {w}x{h} -> {nw}x{nh}', flush=True)
    return img.resize((nw, nh), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Guided filter (edge-preserving smooth, pure NumPy)
# ---------------------------------------------------------------------------

def _guided_filter(img, radius=4, eps=0.02):
    """
    Fast guided filter (He et al. 2013) — edge-preserving noise reduction.
    Smooths noise INSIDE colour regions while keeping colour boundaries crisp.
    eps controls edge sensitivity: lower = sharper edges preserved.
    Returns image unchanged if NumPy not available (no blur fallback).
    """
    if not _NUMPY:
        return img  # skip rather than blur

    rgb  = img.convert('RGB')
    arr  = np.asarray(rgb, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]

    def box(a, r):
        """2D box filter via cumulative sum — O(n) regardless of radius."""
        s = a.cumsum(axis=0).cumsum(axis=1)
        # Pad with zeros
        s = np.pad(s, ((r+1, 0), (r+1, 0)) + ((0,),)*(s.ndim-2), mode='constant')
        area = (2*r+1)**2
        return (s[2*r+2:, 2*r+2:] - s[2*r+2:, :w] - s[:h, 2*r+2:] + s[:h, :w]) / area

    out_channels = []
    # Use luminance as guide
    guide = arr.mean(axis=2)

    for c in range(3):
        p     = arr[:, :, c]
        mean_I = box(guide, radius)
        mean_p = box(p,     radius)
        mean_Ip= box(guide * p, radius)
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I  = box(guide * guide, radius) - mean_I * mean_I
        a_k    = cov_Ip / (var_I + eps)
        b_k    = mean_p - a_k * mean_I
        mean_a = box(a_k, radius)
        mean_b = box(b_k, radius)
        q      = np.clip(mean_a * guide + mean_b, 0, 1)
        out_channels.append((q * 255).astype(np.uint8))

    result = np.stack(out_channels, axis=2)
    out    = Image.fromarray(result, 'RGB')

    # Restore alpha if original had it
    if img.mode == 'RGBA':
        out = out.convert('RGBA')
        out.putalpha(img.getchannel('A'))
    return out


# ---------------------------------------------------------------------------
# Colour quantisation with hard palette cap
# ---------------------------------------------------------------------------

def _quantise_palette(img, max_colors):
    """
    Hard palette cap using Pillow median-cut quantisation.
    Returns RGB image with at most max_colors distinct colours.
    """
    rgb       = img.convert('RGB')
    quantised = rgb.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT, dither=0)
    return quantised.convert('RGB')


# ---------------------------------------------------------------------------
# Colour deduplication (pre-vtracer)
# ---------------------------------------------------------------------------

def _colour_dedup(img, threshold=12):
    """
    Merge near-identical colours in the image palette before tracing.
    Two colours within `threshold` Euclidean distance in RGB space are merged
    into their average. Reduces the number of colour layers vtracer creates,
    speeding up tracing and producing cleaner, fewer paths.
    Works on already-quantised images for best effect.
    """
    if not _NUMPY:
        return img

    rgb  = img.convert('RGB')
    arr  = np.asarray(rgb)
    flat = arr.reshape(-1, 3).astype(np.float32)

    # Get unique colours
    unique = np.unique(flat, axis=0)
    if len(unique) <= 1:
        return img

    # Build merge mapping: for each colour, find closest within threshold
    mapping = {}
    merged  = []
    used    = set()

    for i, col in enumerate(unique):
        if i in used:
            continue
        # Find all colours within threshold
        dists   = np.sqrt(((unique - col) ** 2).sum(axis=1))
        close   = np.where(dists <= threshold)[0]
        cluster = unique[close]
        centroid = cluster.mean(axis=0).round().astype(np.uint8)
        for idx in close:
            mapping[tuple(unique[idx].astype(int))] = tuple(centroid)
            used.add(idx)

    if len(mapping) == len(unique):
        return img  # nothing to merge

    # Apply mapping
    result = arr.copy()
    for orig_col, new_col in mapping.items():
        if orig_col == new_col:
            continue
        mask = np.all(arr == np.array(orig_col, dtype=np.uint8), axis=2)
        result[mask] = np.array(new_col, dtype=np.uint8)

    merged_count = sum(1 for o, n in mapping.items() if o != n)
    print(f'[engine] colour_dedup: merged {merged_count} near-identical colours (threshold={threshold})', flush=True)
    return Image.fromarray(result, 'RGB')


# ---------------------------------------------------------------------------
# SVG colour deduplication (post-vtracer)
# ---------------------------------------------------------------------------

def _hex_to_rgb(h):
    h = h.lstrip('#')
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    return struct.unpack('BBB', bytes.fromhex(h))


def _rgb_to_hex(r, g, b):
    return f'#{r:02x}{g:02x}{b:02x}'


def _svg_colour_dedup(svg, threshold=10):
    """
    Merge near-identical fill colours directly in SVG output.
    After vtracer runs, similar colours that escaped pre-processing get merged
    here. Reduces layer count and file size without changing visual appearance.
    """
    # Extract all fill colours
    fills = re.findall(r'fill="#([0-9a-fA-F]{3,6})"', svg)
    if not fills:
        return svg

    unique_fills = list(set(fills))
    rgb_fills    = []
    for f in unique_fills:
        try:
            rgb_fills.append((f, _hex_to_rgb(f)))
        except Exception:
            pass

    # Build merge map
    colour_map = {}
    used       = set()
    for i, (hex_a, rgb_a) in enumerate(rgb_fills):
        if hex_a in used:
            continue
        cluster    = [rgb_a]
        cluster_hex = [hex_a]
        for j, (hex_b, rgb_b) in enumerate(rgb_fills):
            if i == j or hex_b in used:
                continue
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(rgb_a, rgb_b)))
            if dist <= threshold:
                cluster.append(rgb_b)
                cluster_hex.append(hex_b)
        centroid = tuple(round(sum(c[k] for c in cluster) / len(cluster)) for k in range(3))
        centroid_hex = _rgb_to_hex(*centroid)
        for h in cluster_hex:
            colour_map[h.lower()] = centroid_hex
            used.add(h)

    # Apply replacements
    merged_count = sum(1 for o, n in colour_map.items() if o != n.lstrip('#'))
    if merged_count:
        print(f'[engine] svg_colour_dedup: merged {merged_count} fill colours', flush=True)

    def replace_fill(m):
        orig = m.group(1).lower()
        replacement = colour_map.get(orig, orig)
        return f'fill="#{replacement.lstrip("#")}"'

    return re.sub(r'fill="#([0-9a-fA-F]{3,6})"', replace_fill, svg)


# ---------------------------------------------------------------------------
# Otsu threshold
# ---------------------------------------------------------------------------

def _otsu_threshold(gray):
    hist    = gray.histogram()
    total   = sum(hist)
    sum_all = sum(i * hist[i] for i in range(256))
    sum_b, w_b, max_var, best_t = 0, 0, 0, 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b  += t * hist[t]
        mean_b  = sum_b / w_b
        mean_f  = (sum_all - sum_b) / w_f
        var     = w_b * w_f * (mean_b - mean_f) ** 2
        if var > max_var:
            max_var = var
            best_t  = t
    return gray.point(lambda p: 255 if p >= best_t else 0, '1').convert('L')


# ---------------------------------------------------------------------------
# Morphological close
# ---------------------------------------------------------------------------

def _morph_close(img, size=3):
    """Dilate then erode — joins small gaps in lineart strokes."""
    size    = size if size % 2 == 1 else size + 1
    dilated = img.filter(ImageFilter.MaxFilter(size))
    return dilated.filter(ImageFilter.MinFilter(size))


# ---------------------------------------------------------------------------
# SVG path simplification (RDP)
# ---------------------------------------------------------------------------

def _rdp_distance(point, start, end):
    if start == end:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    dx, dy = end[0] - start[0], end[1] - start[1]
    denom  = math.hypot(dx, dy)
    return abs(dy*point[0] - dx*point[1] + end[0]*start[1] - end[1]*start[0]) / denom


def _rdp(points, epsilon):
    if len(points) < 3:
        return points
    max_dist, max_idx = 0, 0
    for i in range(1, len(points) - 1):
        d = _rdp_distance(points[i], points[0], points[-1])
        if d > max_dist:
            max_dist, max_idx = d, i
    if max_dist > epsilon:
        left  = _rdp(points[:max_idx + 1], epsilon)
        right = _rdp(points[max_idx:],      epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def _simplify_svg_paths(svg, epsilon=0.3):
    def simplify_path(d):
        tokens = re.findall(
            r'[MmLlCcSsQqZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', d)
        result, line_points = [], []

        def flush_lines():
            nonlocal line_points
            if len(line_points) >= 2:
                simplified = _rdp(line_points, epsilon)
                if simplified:
                    result.append('L')
                    for pt in simplified[1:]:
                        result.append(f'{pt[0]:.2f},{pt[1]:.2f}')
            line_points = []

        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t in ('M', 'm'):
                flush_lines()
                result.append(t)
                i += 1
                if i + 1 < len(tokens):
                    x, y = float(tokens[i]), float(tokens[i + 1])
                    result.append(f'{x:.2f},{y:.2f}')
                    line_points = [(x, y)]
                    i += 2
            elif t == 'L':
                i += 1
                while i + 1 < len(tokens) and tokens[i] not in 'MmLlCcSsQqZz':
                    x, y = float(tokens[i]), float(tokens[i + 1])
                    line_points.append((x, y))
                    i += 2
            elif t in ('C', 'c', 'S', 's', 'Q', 'q'):
                flush_lines()
                result.append(t)
                i += 1
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

    return re.sub(r'd="([^"]+)"',
                  lambda m: f'd="{simplify_path(m.group(1))}"', svg)


# ---------------------------------------------------------------------------
# Short path removal
# ---------------------------------------------------------------------------

def _remove_short_paths(svg, min_size=2.0):
    def is_tiny(d):
        nums = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)
        if len(nums) < 4:
            return True
        coords = [float(n) for n in nums]
        xs, ys = coords[0::2], coords[1::2]
        if not xs or not ys:
            return True
        return (max(xs) - min(xs)) < min_size and (max(ys) - min(ys)) < min_size

    return re.sub(
        r'<path[^>]*d="([^"]+)"[^/]*/?>',
        lambda m: '' if is_tiny(m.group(1)) else m.group(0),
        svg,
    )


# ---------------------------------------------------------------------------
# Auto-detect mode
# ---------------------------------------------------------------------------

def _detect_mode(img):
    small        = img.convert('RGB').resize((64, 64), Image.LANCZOS)
    unique_cols  = len(set(small.quantize(colors=16, dither=0).getdata()))
    gray         = small.convert('L')
    edge_pixels  = list(gray.filter(ImageFilter.FIND_EDGES).getdata())
    edge_density = sum(1 for p in edge_pixels if p > 30) / len(edge_pixels)
    print(f'[engine] detect: {unique_cols} colours, edge_density={edge_density:.2f}', flush=True)
    if unique_cols <= 4 and edge_density > 0.15:
        return 'lineart'
    if unique_cols <= 8 and edge_density > 0.25:
        return 'lineart'
    return 'color'


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def vectorize(
    image_data,
    # Pre-processing
    median_size          = 3,      # MedianFilter kernel (1 = off, odd)
    morph_close_size     = 3,      # Morphological close for lineart (1 = off)
    posterize_bits       = 6,      # fallback when max_colors = 0
    # Quality controls
    max_colors           = 32,     # hard palette cap before vtracer (0 = use posterize)
    guided_filter_radius = 4,      # guided filter radius (0 = off)
    guided_filter_eps    = 0.02,   # guided filter edge sensitivity
    color_dedup_thresh   = 12,     # pre-vtracer colour merge distance (0 = off)
    svg_dedup_thresh     = 10,     # post-vtracer SVG fill merge distance (0 = off)
    # Engine
    engine_mode          = 'auto',
    # vtracer
    color_precision      = 6,
    layer_difference     = 4,
    filter_speckle       = 6,
    corner_threshold     = 48,
    splice_threshold     = 70,
    # Post-processing
    simplify             = True,
    simplify_epsilon     = 0.3,
    **kwargs,
):
    img = Image.open(io.BytesIO(image_data)).convert('RGBA')

    # ── Adaptive resize ───────────────────────────────────────────────────────
    unique_cols = _count_unique_colours(img)
    target_px   = TARGET_SIMPLE if unique_cols <= 8 else TARGET_NORMAL
    print(f'[engine] {unique_cols} unique colours -> target {target_px//1000}k px', flush=True)

    w, h   = img.size
    pixels = w * h

    if pixels < MIN_UPSCALE:
        img = _iterative_upscale(img, target_px)
    else:
        img = _resize_down(img, target_px)

    # ── Mode selection ────────────────────────────────────────────────────────
    mode = engine_mode
    if mode == 'auto':
        mode = _detect_mode(img)
    print(f'[engine] mode={mode}', flush=True)

    # ── Pipeline ──────────────────────────────────────────────────────────────

    if mode == 'lineart':
        rgb = img.convert('RGB')

        # 1. Guided filter (edge-preserving noise reduction, no blurring)
        if guided_filter_radius > 0:
            rgb = _guided_filter(rgb, radius=guided_filter_radius, eps=guided_filter_eps)

        # 2. Median filter
        if median_size > 1:
            ms  = median_size if median_size % 2 == 1 else median_size + 1
            rgb = rgb.filter(ImageFilter.MedianFilter(size=ms))

        # 3. Contrast + sharpness boost
        rgb    = ImageEnhance.Contrast(rgb).enhance(2.5)
        rgb    = ImageEnhance.Sharpness(rgb).enhance(2.0)

        # 4. Otsu threshold
        gray   = rgb.convert('L')
        binary = _otsu_threshold(gray)

        # 5. Morphological close
        if morph_close_size > 1:
            binary = _morph_close(binary, size=morph_close_size)

        processed = binary.convert('RGB')

        vtracer_kwargs = dict(
            colormode        = 'bw',
            mode             = 'spline',
            filter_speckle   = filter_speckle,
            corner_threshold = corner_threshold,
            length_threshold = 6.0,
            splice_threshold = splice_threshold,
            path_precision   = 2,
            max_iterations   = 1,
        )
        print(
            f'[engine] lineart: guided={guided_filter_radius} median={median_size} '
            f'morph={morph_close_size} corner={corner_threshold} splice={splice_threshold}',
            flush=True,
        )

    else:  # color
        rgb = img.convert('RGB')

        # 1. Guided filter (edge-preserving noise reduction, no blurring)
        if guided_filter_radius > 0:
            rgb = _guided_filter(rgb, radius=guided_filter_radius, eps=guided_filter_eps)

        # 2. Median filter — removes speckle without blurring edges
        if median_size > 1:
            ms  = median_size if median_size % 2 == 1 else median_size + 1
            rgb = rgb.filter(ImageFilter.MedianFilter(size=ms))

        # 3. Quantise to hard colour cap
        if max_colors > 0:
            rgb = _quantise_palette(rgb, max_colors)
        else:
            rgb = ImageOps.posterize(rgb, posterize_bits)

        # 4. Colour deduplication — merge near-identical palette entries
        if color_dedup_thresh > 0:
            rgb = _colour_dedup(rgb, threshold=color_dedup_thresh)

        # Restore alpha
        if img.mode == 'RGBA':
            rgba = rgb.convert('RGBA')
            rgba.putalpha(img.getchannel('A'))
            processed = rgba
        else:
            processed = rgb

        vtracer_kwargs = dict(
            colormode        = 'color',
            mode             = 'spline',
            filter_speckle   = filter_speckle,
            color_precision  = color_precision,
            layer_difference = layer_difference,
            corner_threshold = corner_threshold,
            length_threshold = 6.0,
            splice_threshold = splice_threshold,
            path_precision   = 2,
            max_iterations   = 1,
        )
        print(
            f'[engine] color: guided={guided_filter_radius} median={median_size} '
            f'max_colors={max_colors} dedup={color_dedup_thresh} '
            f'corner={corner_threshold} splice={splice_threshold}',
            flush=True,
        )

    # ── Run vtracer ───────────────────────────────────────────────────────────
    inp = out = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            processed.save(f.name, format='PNG')
            inp = f.name
        out = inp.replace('.png', '.svg')
        print('[engine] vtracer running...', flush=True)
        vtracer.convert_image_to_svg_py(inp, out, **vtracer_kwargs)
        svg = open(out, encoding='utf-8').read()

        paths_before = svg.count('<path')
        print(f'[engine] {paths_before} paths before post-processing', flush=True)

        # ── Post-processing ───────────────────────────────────────────────────
        # SVG colour deduplication
        if mode == 'color' and svg_dedup_thresh > 0:
            svg = _svg_colour_dedup(svg, threshold=svg_dedup_thresh)

        svg = _remove_short_paths(svg, min_size=2.0)

        if simplify and svg.count('<path') > 150:
            svg = _simplify_svg_paths(svg, epsilon=simplify_epsilon)

        paths_after = svg.count('<path')
        print(
            f'[engine] {paths_after} paths after post-processing '
            f'(removed {paths_before - paths_after})',
            flush=True,
        )
        return svg

    finally:
        if inp and os.path.exists(inp): os.unlink(inp)
        if out and os.path.exists(out): os.unlink(out)
