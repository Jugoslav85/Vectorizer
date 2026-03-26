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
# Group by colour
# ---------------------------------------------------------------------------

def _group_by_color(svg):
    """
    Wrap same-colour paths in <g fill="..."> elements and strip individual
    fill attributes. Output looks identical visually but opens in Illustrator,
    Figma, and Inkscape with proper colour groups — select all red shapes at
    once, recolour an entire region, etc.

    Algorithm:
      1. Parse the flat list of <path> elements vtracer produces.
      2. Collect paths by their fill colour, preserving draw order within
         each colour group (later paths paint over earlier ones).
      3. Emit one <g fill="#rrggbb"> per colour, containing all its paths
         with the redundant fill attribute removed.
      4. Keep non-path elements (rect background, etc.) outside the groups.
    """
    # Extract the SVG opening tag and everything before the first path/rect
    svg_open_m = re.search(r'(<svg[^>]+>)', svg)
    if not svg_open_m:
        return svg
    svg_open = svg_open_m.group(1)

    # Collect all top-level elements in document order
    # Match <path .../>, <path ...></path>, and <rect .../>
    element_pattern = re.compile(
        r'(<rect\b[^>]*/?>|<path\b[^>]*/?>|<path\b[^>]*>.*?</path>)',
        re.DOTALL,
    )
    elements = element_pattern.findall(svg)
    if not elements:
        return svg

    # Split into background elements (rect) and paths
    background = []
    paths_by_color = {}   # colour -> list of path strings (fill attr removed)
    color_order   = []    # insertion-order colour list

    fill_re = re.compile(r'\s*fill="#([0-9a-fA-F]{3,6})"')

    for el in elements:
        if el.strip().startswith('<rect'):
            background.append(el)
            continue
        m = fill_re.search(el)
        if not m:
            # No fill — keep as-is in a catch-all group
            color = '__none__'
        else:
            color = '#' + m.group(1).lower()
        # Remove fill attr from path — it will live on the <g>
        clean = fill_re.sub('', el, count=1)
        if color not in paths_by_color:
            paths_by_color[color] = []
            color_order.append(color)
        paths_by_color[color].append(clean)

    # Build output
    lines = [svg_open]
    for el in background:
        lines.append('  ' + el)
    for color in color_order:
        paths = paths_by_color[color]
        if color == '__none__':
            for p in paths:
                lines.append('  ' + p.strip())
        else:
            lines.append(f'  <g fill="{color}">')
            for p in paths:
                lines.append('    ' + p.strip())
            lines.append('  </g>')
    lines.append('</svg>')

    grouped = '\n'.join(lines)
    n_groups = len([c for c in color_order if c != '__none__'])
    print(f'[engine] group_by_color: {n_groups} colour groups, {len(elements)} paths', flush=True)
    return grouped


# ---------------------------------------------------------------------------
# Gap filler
# ---------------------------------------------------------------------------

def _gap_filler(svg, stroke_width=1.5):
    """
    Eliminate the white-line artifact that appears between adjacent shapes in
    many SVG renderers (browsers, Illustrator, etc.).

    Strategy: inject a thin stroke on every filled path, using the same colour
    as the fill, placed BEFORE the fills in the draw order so it acts as a
    bleed underneath the shape boundaries. Width of 1-2px is enough to cover
    the sub-pixel gap without being visible at normal zoom.

    This is simpler than the full adjacent-colour-averaging approach (which
    requires geometric adjacency detection) but solves 95% of cases because
    the gap colour is usually close to one of the two flanking shapes anyway.
    """
    if stroke_width <= 0:
        return svg

    # Collect all fill colours to build the stroke layer
    fills = re.findall(r'fill="#([0-9a-fA-F]{3,6})"', svg)
    if not fills:
        return svg

    # Build stroke paths: one <path> per original path, stroked not filled
    # We insert them as a <g> before the main content so they render under fills
    stroke_paths = []
    path_re = re.compile(r'<path\b[^>]*/?>|<path\b[^>]*>.*?</path>', re.DOTALL)
    fill_attr_re = re.compile(r'fill="#([0-9a-fA-F]{3,6})"')

    for match in path_re.finditer(svg):
        el  = match.group(0)
        fm  = fill_attr_re.search(el)
        if not fm:
            continue
        color = '#' + fm.group(1)
        # Replace fill with stroke styling
        stroked = fill_attr_re.sub(
            f'fill="none" stroke="{color}" stroke-width="{stroke_width}" '
            f'stroke-linejoin="round" stroke-linecap="round"',
            el, count=1,
        )
        stroke_paths.append('    ' + stroked.strip())

    if not stroke_paths:
        return svg

    # Inject the stroke layer right after the opening <svg> tag
    stroke_layer = (
        '  <g id="gap-filler" opacity="1">\n' +
        '\n'.join(stroke_paths) +
        '\n  </g>'
    )
    svg_open_m = re.search(r'<svg[^>]+>', svg)
    if not svg_open_m:
        return svg
    insert_pos = svg_open_m.end()
    svg = svg[:insert_pos] + '\n' + stroke_layer + svg[insert_pos:]
    print(f'[engine] gap_filler: injected {len(stroke_paths)} stroke paths (width={stroke_width})', flush=True)
    return svg


# ---------------------------------------------------------------------------
# Stroke edges (outline mode)
# ---------------------------------------------------------------------------

def _stroke_edges(svg, stroke_width=1.5, stroke_color=None):
    """
    Convert the SVG to an outline/edge drawing by replacing fill with stroke.
    Useful for laser cutting, vinyl cutting, and outline illustration styles.

    stroke_color: override colour (e.g. '#000000'). None = use each path's
                  original fill colour as its stroke colour.
    """
    def convert_path(m):
        el       = m.group(0)
        fill_m   = re.search(r'fill="#([0-9a-fA-F]{3,6})"', el)
        orig_col = ('#' + fill_m.group(1)) if fill_m else '#000000'
        color    = stroke_color if stroke_color else orig_col
        # Replace fill with stroke
        el = re.sub(r'fill="#[0-9a-fA-F]{3,6}"', f'fill="none"', el)
        # Inject stroke attrs
        el = el.rstrip('/>').rstrip() + (
            f' stroke="{color}" stroke-width="{stroke_width}" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        return el

    result = re.sub(
        r'<path\b[^>]*/?>|<path\b[^>]*>.*?</path>',
        convert_path,
        svg,
        flags=re.DOTALL,
    )
    print(f'[engine] stroke_edges: converted to outline (width={stroke_width}, color={stroke_color or "natural"})', flush=True)
    return result


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
    simplify_epsilon     = 0.1,    # 0.1 = Medium quality (was 0.3 = Coarse)
    group_by_color       = True,   # wrap same-colour paths in <g> elements
    gap_fill             = True,   # inject stroke bleed to kill white-line artifacts
    gap_fill_width       = 1.5,    # gap filler stroke width in px
    stroke_edges         = False,  # outline mode: stroke paths instead of fill
    stroke_edges_width   = 1.5,    # stroke width for outline mode
    stroke_edges_color   = None,   # None = natural colour, '#000000' = override
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
        # 1. SVG colour deduplication (merge near-identical fills)
        if mode == 'color' and svg_dedup_thresh > 0:
            svg = _svg_colour_dedup(svg, threshold=svg_dedup_thresh)

        # 2. Remove tiny stray paths
        svg = _remove_short_paths(svg, min_size=2.0)

        # 3. RDP path simplification
        if simplify and svg.count('<path') > 150:
            svg = _simplify_svg_paths(svg, epsilon=simplify_epsilon)

        # 4. Gap filler — inject stroke bleed before fills to kill white-line artifact
        #    Must run BEFORE group_by_color so strokes sit under the fill groups
        if gap_fill and not stroke_edges:
            svg = _gap_filler(svg, stroke_width=gap_fill_width)

        # 5. Group paths by colour for clean Illustrator/Figma editing
        #    Skip grouping in stroke_edges mode — no fill colours remain
        if group_by_color and not stroke_edges:
            svg = _group_by_color(svg)

        # 6. Stroke edges (outline/laser-cut mode) — replaces fills with strokes
        if stroke_edges:
            svg = _stroke_edges(svg, stroke_width=stroke_edges_width,
                                stroke_color=stroke_edges_color)

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
