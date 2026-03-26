"""
Vectorizer engine — vtracer backend.

Colour pipeline:
  resize -> median_filter -> posterize -> vtracer -> remove_short_paths
  -> rdp_simplify -> gap_fill -> group_by_color

Lineart pipeline:
  resize -> median_filter -> contrast_boost -> otsu_threshold
  -> morph_close -> vtracer bw -> remove_short_paths -> rdp_simplify
"""

import io
import re
import tempfile
import os
import math
import vtracer

from PIL import Image, ImageFilter, ImageOps, ImageEnhance


# ---------------------------------------------------------------------------
# Resize — 1MP cap, iterative upscale for tiny images
# ---------------------------------------------------------------------------

TARGET_PX   = 1_000_000   # 1MP
MIN_UPSCALE =   300_000   # upscale below 0.3MP


def _resize(img):
    w, h   = img.size
    pixels = w * h

    if pixels < MIN_UPSCALE:
        # Iterative 1.5x steps — sharper than one big jump
        print(f'[engine] upscaling {w}x{h}', flush=True)
        while True:
            cw, ch = img.size
            if cw * ch >= TARGET_PX:
                break
            scale  = min(1.5, math.sqrt(TARGET_PX / (cw * ch)))
            nw, nh = int(cw * scale), int(ch * scale)
            img    = img.resize((nw, nh), Image.LANCZOS)
        return img

    if pixels > TARGET_PX:
        scale  = math.sqrt(TARGET_PX / pixels)
        nw, nh = int(w * scale), int(h * scale)
        print(f'[engine] downscaled {w}x{h} -> {nw}x{nh}', flush=True)
        return img.resize((nw, nh), Image.LANCZOS)

    print(f'[engine] size ok {w}x{h}', flush=True)
    return img


# ---------------------------------------------------------------------------
# Otsu threshold (auto black/white for lineart)
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
# Morphological close (join gaps in lineart strokes)
# ---------------------------------------------------------------------------

def _morph_close(img, size=3):
    size    = size if size % 2 == 1 else size + 1
    dilated = img.filter(ImageFilter.MaxFilter(size))
    return dilated.filter(ImageFilter.MinFilter(size))


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
        return (max(xs) - min(xs)) < min_size and (max(ys) - min(ys)) < min_size

    return re.sub(
        r'<path[^>]*d="([^"]+)"[^/]*/?>',
        lambda m: '' if is_tiny(m.group(1)) else m.group(0),
        svg,
    )


# ---------------------------------------------------------------------------
# RDP path simplification
# ---------------------------------------------------------------------------

def _rdp_dist(p, a, b):
    if a == b:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return abs(dy*p[0] - dx*p[1] + b[0]*a[1] - b[1]*a[0]) / math.hypot(dx, dy)


def _rdp(pts, eps):
    if len(pts) < 3:
        return pts
    d, idx = max(((_rdp_dist(pts[i], pts[0], pts[-1]), i) for i in range(1, len(pts)-1)))
    if d > eps:
        return _rdp(pts[:idx+1], eps)[:-1] + _rdp(pts[idx:], eps)
    return [pts[0], pts[-1]]


def _simplify_svg_paths(svg, epsilon=0.1):
    def simplify_path(d):
        tokens = re.findall(
            r'[MmLlCcSsQqZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', d)
        result, line_pts = [], []

        def flush():
            nonlocal line_pts
            if len(line_pts) >= 2:
                s = _rdp(line_pts, epsilon)
                if s:
                    result.append('L')
                    for pt in s[1:]:
                        result.append(f'{pt[0]:.2f},{pt[1]:.2f}')
            line_pts = []

        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t in ('M', 'm'):
                flush(); result.append(t); i += 1
                if i + 1 < len(tokens):
                    x, y = float(tokens[i]), float(tokens[i+1])
                    result.append(f'{x:.2f},{y:.2f}')
                    line_pts = [(x, y)]; i += 2
            elif t == 'L':
                i += 1
                while i + 1 < len(tokens) and tokens[i] not in 'MmLlCcSsQqZz':
                    line_pts.append((float(tokens[i]), float(tokens[i+1]))); i += 2
            elif t in ('C','c','S','s','Q','q'):
                flush(); result.append(t); i += 1
                while i < len(tokens) and tokens[i] not in 'MmLlCcSsQqZz':
                    result.append(tokens[i]); i += 1
            elif t in ('Z', 'z'):
                flush(); result.append(t); i += 1
            else:
                result.append(t); i += 1
        flush()
        return ' '.join(result)

    return re.sub(r'd="([^"]+)"', lambda m: f'd="{simplify_path(m.group(1))}"', svg)


# ---------------------------------------------------------------------------
# Gap filler — thin stroke bleed under fills kills white-line artifacts
# ---------------------------------------------------------------------------

def _gap_filler(svg, stroke_width=1.5):
    if stroke_width <= 0:
        return svg
    stroke_paths = []
    fill_re = re.compile(r'fill="#([0-9a-fA-F]{3,6})"')
    for m in re.finditer(r'<path\b[^>]*/?>|<path\b[^>]*>.*?</path>', svg, re.DOTALL):
        el = m.group(0)
        fm = fill_re.search(el)
        if not fm:
            continue
        color   = '#' + fm.group(1)
        stroked = fill_re.sub(
            f'fill="none" stroke="{color}" stroke-width="{stroke_width}" '
            f'stroke-linejoin="round" stroke-linecap="round"',
            el, count=1,
        )
        stroke_paths.append('    ' + stroked.strip())
    if not stroke_paths:
        return svg
    layer = '  <g id="gap-filler">\n' + '\n'.join(stroke_paths) + '\n  </g>'
    m = re.search(r'<svg[^>]+>', svg)
    if not m:
        return svg
    svg = svg[:m.end()] + '\n' + layer + svg[m.end():]
    print(f'[engine] gap_filler: {len(stroke_paths)} stroke paths', flush=True)
    return svg


# ---------------------------------------------------------------------------
# Group by colour — wraps same-fill paths in <g> for Illustrator/Figma
# ---------------------------------------------------------------------------

def _group_by_color(svg):
    m = re.search(r'(<svg[^>]+>)', svg)
    if not m:
        return svg
    svg_open = m.group(1)

    elements = re.findall(
        r'(<rect\b[^>]*/?>|<path\b[^>]*/?>|<path\b[^>]*>.*?</path>)',
        svg, re.DOTALL,
    )
    if not elements:
        return svg

    fill_re        = re.compile(r'\s*fill="#([0-9a-fA-F]{3,6})"')
    background     = []
    paths_by_color = {}
    color_order    = []

    for el in elements:
        if el.strip().startswith('<rect'):
            background.append(el); continue
        fm    = fill_re.search(el)
        color = ('#' + fm.group(1).lower()) if fm else '__none__'
        clean = fill_re.sub('', el, count=1)
        if color not in paths_by_color:
            paths_by_color[color] = []
            color_order.append(color)
        paths_by_color[color].append(clean)

    lines = [svg_open]
    for el in background:
        lines.append('  ' + el)
    for color in color_order:
        if color == '__none__':
            for p in paths_by_color[color]:
                lines.append('  ' + p.strip())
        else:
            lines.append(f'  <g fill="{color}">')
            for p in paths_by_color[color]:
                lines.append('    ' + p.strip())
            lines.append('  </g>')
    lines.append('</svg>')

    n = len([c for c in color_order if c != '__none__'])
    print(f'[engine] group_by_color: {n} groups, {len(elements)} paths', flush=True)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Auto-detect mode
# ---------------------------------------------------------------------------

def _detect_mode(img):
    small       = img.convert('RGB').resize((64, 64), Image.LANCZOS)
    unique_cols = len(set(small.quantize(colors=16, dither=0).getdata()))
    gray        = small.convert('L')
    edges       = list(gray.filter(ImageFilter.FIND_EDGES).getdata())
    edge_density = sum(1 for p in edges if p > 30) / len(edges)
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
    engine_mode      = 'auto',
    # Pre-processing — user controlled
    median_size      = 3,     # MedianFilter size (1=off, must be odd)
    morph_close_size = 3,     # lineart gap-join size (1=off)
    posterize_bits   = 6,     # colour banding depth
    # vtracer — user controlled
    color_precision  = 6,     # 1–8
    filter_speckle   = 6,     # min path size to keep
    corner_threshold = 48,    # higher = smoother curves
    # Post-processing — user controlled
    simplify_epsilon = 0.1,   # RDP tolerance
    gap_fill         = True,  # white-line artifact fix
    # Post-processing — always on, not exposed
    gap_fill_width   = 1.5,
    group_by_color   = True,
    stroke_edges     = False,
    stroke_edges_width = 1.5,
    stroke_edges_color = None,
):
    img = Image.open(io.BytesIO(image_data)).convert('RGBA')
    img = _resize(img)

    mode = engine_mode
    if mode == 'auto':
        mode = _detect_mode(img)
    print(f'[engine] mode={mode}', flush=True)

    # ── Lineart pipeline ──────────────────────────────────────────────────────
    if mode == 'lineart':
        rgb = img.convert('RGB')

        if median_size > 1:
            ms  = median_size if median_size % 2 == 1 else median_size + 1
            rgb = rgb.filter(ImageFilter.MedianFilter(size=ms))

        rgb    = ImageEnhance.Contrast(rgb).enhance(2.5)
        rgb    = ImageEnhance.Sharpness(rgb).enhance(2.0)
        binary = _otsu_threshold(rgb.convert('L'))

        if morph_close_size > 1:
            mcs    = morph_close_size if morph_close_size % 2 == 1 else morph_close_size + 1
            binary = _morph_close(binary, size=mcs)

        processed = binary.convert('RGB')
        vtracer_kwargs = dict(
            colormode        = 'bw',
            mode             = 'spline',
            filter_speckle   = filter_speckle,
            corner_threshold = corner_threshold,
            length_threshold = 6.0,
            splice_threshold = 70,
            path_precision   = 2,
            max_iterations   = 1,
        )
        print(f'[engine] lineart: median={median_size} morph={morph_close_size} corner={corner_threshold}', flush=True)

    # ── Colour pipeline ───────────────────────────────────────────────────────
    else:
        rgb = img.convert('RGB')

        if median_size > 1:
            ms  = median_size if median_size % 2 == 1 else median_size + 1
            rgb = rgb.filter(ImageFilter.MedianFilter(size=ms))

        rgb = ImageOps.posterize(rgb, posterize_bits)

        if img.mode == 'RGBA':
            out = rgb.convert('RGBA')
            out.putalpha(img.getchannel('A'))
            processed = out
        else:
            processed = rgb

        vtracer_kwargs = dict(
            colormode        = 'color',
            mode             = 'spline',
            filter_speckle   = filter_speckle,
            color_precision  = color_precision,
            layer_difference = 4,
            corner_threshold = corner_threshold,
            length_threshold = 6.0,
            splice_threshold = 70,
            path_precision   = 2,
            max_iterations   = 1,
        )
        print(f'[engine] color: median={median_size} posterize={posterize_bits} cp={color_precision} corner={corner_threshold}', flush=True)

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

        n_before = svg.count('<path')
        print(f'[engine] {n_before} paths before post-processing', flush=True)

        svg = _remove_short_paths(svg, min_size=2.0)

        if svg.count('<path') > 150:
            svg = _simplify_svg_paths(svg, epsilon=simplify_epsilon)

        if gap_fill and not stroke_edges:
            svg = _gap_filler(svg, stroke_width=gap_fill_width)

        if group_by_color and not stroke_edges:
            svg = _group_by_color(svg)

        if stroke_edges:
            svg = _stroke_edges_fn(svg, stroke_edges_width, stroke_edges_color)

        print(f'[engine] {svg.count("<path")} paths after post-processing', flush=True)
        return svg

    finally:
        if inp and os.path.exists(inp): os.unlink(inp)
        if out and os.path.exists(out): os.unlink(out)


def _stroke_edges_fn(svg, width=1.5, color=None):
    """Outline mode: replace fills with strokes."""
    fill_re = re.compile(r'fill="#([0-9a-fA-F]{3,6})"')

    def convert(m):
        el     = m.group(0)
        fm     = fill_re.search(el)
        col    = ('#' + fm.group(1)) if fm else '#000000'
        stroke = color or col
        el     = fill_re.sub('fill="none"', el)
        el     = el.rstrip('/>').rstrip() + (
            f' stroke="{stroke}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        return el

    return re.sub(r'<path\b[^>]*/?>|<path\b[^>]*>.*?</path>', convert, svg, flags=re.DOTALL)
