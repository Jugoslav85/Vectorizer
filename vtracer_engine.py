"""
Vectorizer engine — vtracer backend.

Pre-processing:  resize to 1MP  ->  optional posterize
vtracer:         straight pass-through with user params
Post-processing: remove_short_paths -> rdp_simplify -> gap_fill -> group_by_color
"""

import io
import re
import tempfile
import os
import math
import vtracer

from PIL import Image, ImageFilter, ImageOps, ImageEnhance


# ---------------------------------------------------------------------------
# Resize — 1MP cap, iterative upscale for small images
# ---------------------------------------------------------------------------

TARGET_PX = 1_000_000
MIN_UPSCALE = 300_000


def _resize(img):
    w, h = img.size
    px = w * h
    if px < MIN_UPSCALE:
        print(f'[engine] upscaling {w}x{h}', flush=True)
        while True:
            cw, ch = img.size
            if cw * ch >= TARGET_PX:
                break
            scale = min(1.5, math.sqrt(TARGET_PX / (cw * ch)))
            img = img.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)
        return img
    if px > TARGET_PX:
        scale = math.sqrt(TARGET_PX / px)
        nw, nh = int(w * scale), int(h * scale)
        print(f'[engine] downscaled {w}x{h} -> {nw}x{nh}', flush=True)
        return img.resize((nw, nh), Image.LANCZOS)
    print(f'[engine] size ok {w}x{h}', flush=True)
    return img


# ---------------------------------------------------------------------------
# Post-processing helpers
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


def _rdp_dist(p, a, b):
    if a == b:
        return math.hypot(p[0]-a[0], p[1]-a[1])
    dx, dy = b[0]-a[0], b[1]-a[1]
    return abs(dy*p[0] - dx*p[1] + b[0]*a[1] - b[1]*a[0]) / math.hypot(dx, dy)


def _rdp(pts, eps):
    if len(pts) < 3:
        return pts
    d, idx = max((_rdp_dist(pts[i], pts[0], pts[-1]), i) for i in range(1, len(pts)-1))
    if d > eps:
        return _rdp(pts[:idx+1], eps)[:-1] + _rdp(pts[idx:], eps)
    return [pts[0], pts[-1]]


def _simplify_svg_paths(svg, epsilon=0.1):
    def simplify_path(d):
        tokens = re.findall(r'[MmLlCcSsQqZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', d)
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
                if i+1 < len(tokens):
                    x, y = float(tokens[i]), float(tokens[i+1])
                    result.append(f'{x:.2f},{y:.2f}')
                    line_pts = [(x, y)]; i += 2
            elif t == 'L':
                i += 1
                while i+1 < len(tokens) and tokens[i] not in 'MmLlCcSsQqZz':
                    line_pts.append((float(tokens[i]), float(tokens[i+1]))); i += 2
            elif t in ('C','c','S','s','Q','q'):
                flush(); result.append(t); i += 1
                while i < len(tokens) and tokens[i] not in 'MmLlCcSsQqZz':
                    result.append(tokens[i]); i += 1
            elif t in ('Z','z'):
                flush(); result.append(t); i += 1
            else:
                result.append(t); i += 1
        flush()
        return ' '.join(result)

    return re.sub(r'd="([^"]+)"', lambda m: f'd="{simplify_path(m.group(1))}"', svg)


def _gap_filler(svg, stroke_width=1.5):
    stroke_paths = []
    fill_re = re.compile(r'fill="#([0-9a-fA-F]{3,6})"')
    for m in re.finditer(r'<path\b[^>]*/?>|<path\b[^>]*>.*?</path>', svg, re.DOTALL):
        el = m.group(0)
        fm = fill_re.search(el)
        if not fm:
            continue
        color = '#' + fm.group(1)
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
    fill_re = re.compile(r'\s*fill="#([0-9a-fA-F]{3,6})"')
    background, paths_by_color, color_order = [], {}, []
    for el in elements:
        if el.strip().startswith('<rect'):
            background.append(el); continue
        fm = fill_re.search(el)
        color = ('#' + fm.group(1).lower()) if fm else '__none__'
        clean = fill_re.sub('', el, count=1)
        if color not in paths_by_color:
            paths_by_color[color] = []; color_order.append(color)
        paths_by_color[color].append(clean)
    lines = [svg_open]
    for el in background:
        lines.append('  ' + el)
    for color in color_order:
        if color == '__none__':
            for p in paths_by_color[color]: lines.append('  ' + p.strip())
        else:
            lines.append(f'  <g fill="{color}">')
            for p in paths_by_color[color]: lines.append('    ' + p.strip())
            lines.append('  </g>')
    lines.append('</svg>')
    n = len([c for c in color_order if c != '__none__'])
    print(f'[engine] group_by_color: {n} groups, {len(elements)} paths', flush=True)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def vectorize(
    image_data,
    # Mode
    colormode        = 'color',    # 'color' or 'bw'
    # Pre-processing (non-vtracer)
    posterize_bits   = 0,          # 0 = off, 2-8 = apply posterize before tracing
    # vtracer params — all at vtracer defaults
    filter_speckle   = 4,
    color_precision  = 6,
    layer_difference = 16,
    corner_threshold = 60,
    length_threshold = 4.0,
    splice_threshold = 45,
    # Post-processing
    simplify_epsilon = 0.1,        # RDP tolerance
    gap_fill         = True,       # white-line artifact fix
):
    img = Image.open(io.BytesIO(image_data)).convert('RGBA')
    img = _resize(img)

    # Optional posterize before tracing
    rgb = img.convert('RGB')
    if posterize_bits >= 2:
        rgb = ImageOps.posterize(rgb, posterize_bits)
        print(f'[engine] posterize: {posterize_bits} bits', flush=True)

    # Restore alpha if needed
    if img.mode == 'RGBA':
        out = rgb.convert('RGBA')
        out.putalpha(img.getchannel('A'))
        processed = out
    else:
        processed = rgb

    # Run vtracer
    inp = vtr_out = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            processed.save(f.name, format='PNG')
            inp = f.name
        vtr_out = inp.replace('.png', '.svg')
        print(f'[engine] vtracer: colormode={colormode} filter_speckle={filter_speckle} '
              f'color_precision={color_precision} layer_difference={layer_difference} '
              f'corner_threshold={corner_threshold} length_threshold={length_threshold} '
              f'splice_threshold={splice_threshold}', flush=True)
        vtracer.convert_image_to_svg_py(
            inp, vtr_out,
            colormode        = colormode,
            hierarchical     = 'stacked',
            mode             = 'spline',
            filter_speckle   = filter_speckle,
            color_precision  = color_precision,
            layer_difference = layer_difference,
            corner_threshold = corner_threshold,
            length_threshold = length_threshold,
            max_iterations   = 10,
            splice_threshold = splice_threshold,
            path_precision   = 8,
        )
        svg = open(vtr_out, encoding='utf-8').read()
        print(f'[engine] {svg.count("<path")} paths from vtracer', flush=True)

        # Post-processing
        svg = _remove_short_paths(svg, min_size=2.0)
        if svg.count('<path') > 150:
            svg = _simplify_svg_paths(svg, epsilon=simplify_epsilon)
        if gap_fill and colormode == 'color':
            svg = _gap_filler(svg)
        if colormode == 'color':
            svg = _group_by_color(svg)

        print(f'[engine] {svg.count("<path")} paths after post-processing', flush=True)
        return svg

    finally:
        if inp and os.path.exists(inp): os.unlink(inp)
        if vtr_out and os.path.exists(vtr_out): os.unlink(vtr_out)
