"""
Vectorizer engine — vtracer backend.

Pre-processing (all individually togglable):
  resize -> unsharp_mask -> clahe -> edge_contrast -> tonal_separation
  -> posterize -> vtracer

Post-processing (all individually togglable):
  remove_short_paths -> smart_path_retention -> rdp_simplify
  -> thin_path_stroke -> gap_fill -> group_by_color
"""

import io
import re
import tempfile
import os
import math
import vtracer

from PIL import Image, ImageFilter, ImageOps, ImageEnhance

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

TARGET_PX   = 1_000_000
MIN_UPSCALE =   300_000


def _resize(img):
    w, h = img.size
    px   = w * h
    if px < MIN_UPSCALE:
        print(f'[engine] upscaling {w}x{h}', flush=True)
        while True:
            cw, ch = img.size
            if cw * ch >= TARGET_PX:
                break
            scale = min(1.5, math.sqrt(TARGET_PX / (cw * ch)))
            img   = img.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)
        return img
    if px > TARGET_PX:
        scale  = math.sqrt(TARGET_PX / px)
        nw, nh = int(w * scale), int(h * scale)
        print(f'[engine] downscaled {w}x{h} -> {nw}x{nh}', flush=True)
        return img.resize((nw, nh), Image.LANCZOS)
    print(f'[engine] size ok {w}x{h}', flush=True)
    return img


# ---------------------------------------------------------------------------
# Pre-processing: Unsharp Mask
# ---------------------------------------------------------------------------

def _unsharp_mask(img, radius=1.5, percent=180, threshold=3):
    """
    Sharpen fine detail without blurring edges.
    output = original + (original - blurred) * strength
    The blur is internal — never reaches vtracer.
    radius 1-2px targets the detail frequency where eyes and thin lines live.
    """
    rgb     = img.convert('RGB')
    sharpened = rgb.filter(ImageFilter.UnsharpMask(
        radius=radius, percent=percent, threshold=threshold
    ))
    print(f'[engine] unsharp_mask: radius={radius} percent={percent}', flush=True)
    if img.mode == 'RGBA':
        out = sharpened.convert('RGBA')
        out.putalpha(img.getchannel('A'))
        return out
    return sharpened


# ---------------------------------------------------------------------------
# Pre-processing: CLAHE (local contrast enhancement)
# ---------------------------------------------------------------------------

def _clahe(img, clip_limit=2.0, tile_size=8):
    """
    Contrast Limited Adaptive Histogram Equalization.
    Independently boosts contrast in every local tile — lifts shadow detail
    (eyes, wrinkles, thin lines in dark regions) without blowing out highlights.
    Uses OpenCV if available (best quality), falls back to a Pillow tile
    approximation that's ~80% as effective.
    clip_limit: max contrast amplification per tile (2-8, higher = more aggressive)
    tile_size:  local tile grid size (4-16)
    """
    rgb = img.convert('RGB')

    if _CV2:
        import numpy as _np
        arr  = _np.asarray(rgb)
        lab  = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit,
                                     tileGridSize=(tile_size, tile_size))
        l_eq = clahe_obj.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        result = Image.fromarray(rgb_eq, 'RGB')
        print(f'[engine] clahe (opencv): clip={clip_limit} tile={tile_size}', flush=True)

    else:
        # Pillow fallback: equalize each tile independently then blend back
        w, h    = rgb.size
        ts      = max(16, min(w, h) // tile_size)
        result  = rgb.copy()
        from PIL import ImageDraw
        for y in range(0, h, ts):
            for x in range(0, w, ts):
                box    = (x, y, min(x+ts, w), min(y+ts, h))
                tile   = rgb.crop(box)
                eq     = ImageOps.equalize(tile)
                # Blend: 60% equalized, 40% original — avoids over-amplification
                blended = Image.blend(tile, eq, alpha=0.6)
                result.paste(blended, box)
        print(f'[engine] clahe (pillow fallback): tile_size={ts}', flush=True)

    if img.mode == 'RGBA':
        out = result.convert('RGBA')
        out.putalpha(img.getchannel('A'))
        return out
    return result


# ---------------------------------------------------------------------------
# Pre-processing: Edge-aware local contrast boost
# ---------------------------------------------------------------------------

def _edge_contrast(img, boost=1.8, dilation=3):
    """
    Detect edges, dilate the mask, then apply strong contrast boost only near
    edges. Flat colour regions stay accurate. Boundary regions get pushed apart
    so vtracer sees a clear divide instead of a gradual transition.
    boost: contrast multiplier applied near edges (1.2 - 3.0)
    dilation: how many pixels around each edge get boosted
    """
    rgb  = img.convert('RGB')
    gray = rgb.convert('L')

    # Detect edges
    edges = gray.filter(ImageFilter.FIND_EDGES)
    # Threshold to binary mask
    edges = edges.point(lambda p: 255 if p > 20 else 0)
    # Dilate to cover stroke width
    for _ in range(dilation):
        edges = edges.filter(ImageFilter.MaxFilter(3))

    # Apply contrast boost to the entire image
    boosted = ImageEnhance.Contrast(rgb).enhance(boost)

    # Composite: use boosted near edges, original elsewhere
    edge_rgb = edges.convert('RGB')
    mask     = edges  # L mode mask
    result   = Image.composite(boosted, rgb, mask)

    print(f'[engine] edge_contrast: boost={boost} dilation={dilation}', flush=True)
    if img.mode == 'RGBA':
        out = result.convert('RGBA')
        out.putalpha(img.getchannel('A'))
        return out
    return result


# ---------------------------------------------------------------------------
# Pre-processing: Tonal separation (shadow range stretch)
# ---------------------------------------------------------------------------

def _tonal_separation(img, shadow_end=80, shadow_out=160):
    """
    Stretch the shadow tonal range so dark details become visible.
    Pixels 0-shadow_end get remapped to 0-shadow_out (expanded).
    Everything above shadow_end is compressed proportionally.
    Specifically helps: eyes on dark faces, dark lines on dark backgrounds,
    shadow detail in illustrations.
    shadow_end:  input value below which shadows are stretched (50-120)
    shadow_out:  output value the shadow_end maps to (100-200)
    """
    rgb = img.convert('RGB')

    # Build a piecewise LUT: stretch shadows, compress highlights
    lut = []
    for i in range(256):
        if i <= shadow_end:
            # Shadow zone: linear stretch to shadow_out
            v = int(i * shadow_out / shadow_end)
        else:
            # Highlight zone: compress remaining range into shadow_out..255
            ratio = (i - shadow_end) / (255 - shadow_end)
            v     = int(shadow_out + ratio * (255 - shadow_out))
        lut.append(min(255, max(0, v)))

    # Apply LUT to each channel
    r, g, b = rgb.split()
    r = r.point(lut)
    g = g.point(lut)
    b = b.point(lut)
    result = Image.merge('RGB', (r, g, b))

    print(f'[engine] tonal_separation: shadow_end={shadow_end} shadow_out={shadow_out}', flush=True)
    if img.mode == 'RGBA':
        out = result.convert('RGBA')
        out.putalpha(img.getchannel('A'))
        return out
    return result


# ---------------------------------------------------------------------------
# Post-processing: Smart path retention
# ---------------------------------------------------------------------------

def _smart_path_retention(svg, proximity_px=8.0):
    """
    Context-aware path removal: keep small paths that are near other paths
    of the same colour (they're probably part of a detail cluster — an eye,
    a textured region), discard small paths that are isolated (probably noise).

    Replaces the blunt _remove_short_paths for better detail preservation.
    proximity_px: if a tiny path has a same-colour neighbour within this many
                  pixels, keep it. Otherwise remove it.
    """
    # Parse all paths with their fill and bounding box
    path_re  = re.compile(r'<path\b[^>]*d="([^"]+)"[^/]*/?>|<path\b[^>]*d="([^"]+)"[^>]*>[^<]*</path>')
    fill_re  = re.compile(r'fill="(#[0-9a-fA-F]{3,6})"')
    num_re   = re.compile(r'[-+]?[0-9]*\.?[0-9]+')

    def bbox(d):
        nums = [float(x) for x in num_re.findall(d)]
        if len(nums) < 4:
            return None
        xs, ys = nums[0::2], nums[1::2]
        return (min(xs), min(ys), max(xs), max(ys))

    def centre(bb):
        return ((bb[0]+bb[2])/2, (bb[1]+bb[3])/2)

    def size(bb):
        return (bb[2]-bb[0]), (bb[3]-bb[1])

    paths = []
    for m in re.finditer(r'<path\b[^>]*/?>|<path\b[^>]*>.*?</path>', svg, re.DOTALL):
        el   = m.group(0)
        d_m  = re.search(r'd="([^"]+)"', el)
        f_m  = fill_re.search(el)
        if not d_m:
            continue
        bb = bbox(d_m.group(1))
        if bb is None:
            continue
        w, h  = size(bb)
        paths.append({
            'el':    el,
            'fill':  f_m.group(1).lower() if f_m else None,
            'bb':    bb,
            'cx':    centre(bb)[0],
            'cy':    centre(bb)[1],
            'tiny':  w < proximity_px and h < proximity_px,
        })

    if not paths:
        return svg

    # For each tiny path, check if same-colour neighbour exists nearby
    keep_set = set()
    for i, p in enumerate(paths):
        if not p['tiny']:
            keep_set.add(i)
            continue
        # Check proximity to any non-tiny path of same colour
        survived = False
        for j, q in enumerate(paths):
            if i == j:
                continue
            if q['fill'] != p['fill']:
                continue
            dist = math.hypot(p['cx'] - q['cx'], p['cy'] - q['cy'])
            if dist <= proximity_px * 3:
                survived = True
                break
        if survived:
            keep_set.add(i)

    removed = len(paths) - len(keep_set)
    if removed == 0:
        return svg

    # Rebuild SVG replacing each path el
    path_iter = iter(paths)
    idx       = [0]

    def replace_path(m):
        p = paths[idx[0]]
        idx[0] += 1
        return p['el'] if idx[0]-1 in keep_set else ''

    result = re.sub(
        r'<path\b[^>]*/?>|<path\b[^>]*>.*?</path>',
        replace_path, svg, flags=re.DOTALL
    )
    print(f'[engine] smart_path_retention: kept {len(keep_set)}/{len(paths)} paths (removed {removed} isolated tiny)', flush=True)
    return result


# ---------------------------------------------------------------------------
# Post-processing: Thin path stroke
# ---------------------------------------------------------------------------

def _thin_path_stroke(svg, max_size=6.0, stroke_width=0.5):
    """
    Add a thin stroke to paths with a small bounding box that survived
    filter_speckle. These are probably fine lines (hair strands, thin outlines)
    that would disappear at certain zoom levels due to sub-pixel rendering.
    A 0.5px same-colour stroke fills the rendering gap without changing appearance.
    """
    fill_re = re.compile(r'fill="(#[0-9a-fA-F]{3,6})"')
    num_re  = re.compile(r'[-+]?[0-9]*\.?[0-9]+')

    def is_thin(el):
        d_m = re.search(r'd="([^"]+)"', el)
        if not d_m:
            return False
        nums = [float(x) for x in num_re.findall(d_m.group(1))]
        if len(nums) < 4:
            return False
        xs, ys = nums[0::2], nums[1::2]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        return w < max_size or h < max_size

    def add_stroke(m):
        el  = m.group(0)
        if not is_thin(el):
            return el
        fm  = fill_re.search(el)
        if not fm:
            return el
        color = fm.group(1)
        # Inject stroke attrs before the closing />
        el = el.rstrip('/>').rstrip()
        el += f' stroke="{color}" stroke-width="{stroke_width}"/>'
        return el

    result = re.sub(
        r'<path\b[^>]*/?>',
        add_stroke, svg
    )
    print(f'[engine] thin_path_stroke: applied to thin paths (max_size={max_size}px)', flush=True)
    return result


# ---------------------------------------------------------------------------
# Post-processing: Short path removal (fallback when smart retention off)
# ---------------------------------------------------------------------------

def _remove_short_paths(svg, min_size=2.0):
    def is_tiny(d):
        nums = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)
        if len(nums) < 4:
            return True
        coords = [float(n) for n in nums]
        xs, ys = coords[0::2], coords[1::2]
        return (max(xs)-min(xs)) < min_size and (max(ys)-min(ys)) < min_size
    return re.sub(
        r'<path[^>]*d="([^"]+)"[^/]*/?>',
        lambda m: '' if is_tiny(m.group(1)) else m.group(0),
        svg,
    )


# ---------------------------------------------------------------------------
# Post-processing: RDP simplification
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Post-processing: Gap filler
# ---------------------------------------------------------------------------

def _gap_filler(svg, stroke_width=1.5):
    stroke_paths = []
    fill_re      = re.compile(r'fill="(#[0-9a-fA-F]{3,6})"')
    for m in re.finditer(r'<path\b[^>]*/?>|<path\b[^>]*>.*?</path>', svg, re.DOTALL):
        el = m.group(0)
        fm = fill_re.search(el)
        if not fm:
            continue
        color   = fm.group(1)
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
# Post-processing: Group by colour
# ---------------------------------------------------------------------------

def _group_by_color(svg):
    m = re.search(r'(<svg[^>]+>)', svg)
    if not m:
        return svg
    svg_open  = m.group(1)
    elements  = re.findall(
        r'(<rect\b[^>]*/?>|<path\b[^>]*/?>|<path\b[^>]*>.*?</path>)',
        svg, re.DOTALL,
    )
    if not elements:
        return svg
    fill_re        = re.compile(r'\s*fill="(#[0-9a-fA-F]{3,6})"')
    background     = []
    paths_by_color = {}
    color_order    = []
    for el in elements:
        if el.strip().startswith('<rect'):
            background.append(el); continue
        fm    = fill_re.search(el)
        color = ('#' + fm.group(1).lstrip('#').lower()) if fm else '__none__'
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
    colormode              = 'color',   # 'color' or 'bw'

    # ── Pre-processing toggles ────────────────────────────────────────────
    unsharp_mask           = True,      # sharpen fine detail before tracing
    unsharp_radius         = 1.5,       # px — targets detail frequency
    unsharp_percent        = 180,       # strength %
    unsharp_threshold      = 3,         # edge detection threshold

    clahe                  = True,      # local contrast enhancement
    clahe_clip             = 2.0,       # amplification limit per tile
    clahe_tile             = 8,         # tile grid size

    edge_contrast          = True,      # boost contrast near detected edges
    edge_contrast_boost    = 1.8,       # multiplier (1.2–3.0)
    edge_contrast_dilation = 3,         # px around each edge to boost

    tonal_separation       = False,     # stretch shadow range (dark detail)
    tonal_shadow_end       = 80,        # input shadow cutoff (0–150)
    tonal_shadow_out       = 160,       # output value shadow_end maps to

    posterize_bits         = 0,         # 0 = off, 2–6 = reduce colours

    # ── vtracer params (all at vtracer defaults) ──────────────────────────
    filter_speckle         = 4,
    color_precision        = 6,
    layer_difference       = 16,
    corner_threshold       = 60,
    length_threshold       = 4.0,
    splice_threshold       = 45,

    # ── Post-processing toggles ───────────────────────────────────────────
    smart_path_retention   = True,      # keep small paths near same-colour neighbours
    smart_proximity        = 8.0,       # px proximity for retention decision

    simplify_epsilon       = 0.1,       # RDP path simplification tolerance

    thin_path_stroke       = True,      # add hairline stroke to thin surviving paths
    thin_path_max_size     = 6.0,       # px — paths narrower than this get a stroke
    thin_stroke_width      = 0.5,       # stroke width in px

    gap_fill               = True,      # white-line artifact fix
):
    img = Image.open(io.BytesIO(image_data)).convert('RGBA')
    img = _resize(img)

    # ── Pre-processing ────────────────────────────────────────────────────
    if unsharp_mask:
        img = _unsharp_mask(img, radius=unsharp_radius,
                            percent=unsharp_percent, threshold=unsharp_threshold)

    if clahe:
        img = _clahe(img, clip_limit=clahe_clip, tile_size=clahe_tile)

    if edge_contrast:
        img = _edge_contrast(img, boost=edge_contrast_boost,
                             dilation=edge_contrast_dilation)

    if tonal_separation:
        img = _tonal_separation(img, shadow_end=tonal_shadow_end,
                                shadow_out=tonal_shadow_out)

    rgb = img.convert('RGB')
    if posterize_bits >= 2:
        rgb = ImageOps.posterize(rgb, posterize_bits)
        print(f'[engine] posterize: {posterize_bits} bits', flush=True)

    if img.mode == 'RGBA':
        out = rgb.convert('RGBA')
        out.putalpha(img.getchannel('A'))
        processed = out
    else:
        processed = rgb

    # ── vtracer ───────────────────────────────────────────────────────────
    inp = vtr_out = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            processed.save(f.name, format='PNG')
            inp = f.name
        vtr_out = inp.replace('.png', '.svg')
        print(f'[engine] vtracer: colormode={colormode} fs={filter_speckle} '
              f'cp={color_precision} ld={layer_difference} ct={corner_threshold} '
              f'lt={length_threshold} st={splice_threshold}', flush=True)
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

        # ── Post-processing ───────────────────────────────────────────────
        if smart_path_retention:
            svg = _smart_path_retention(svg, proximity_px=smart_proximity)
        else:
            svg = _remove_short_paths(svg, min_size=2.0)

        if svg.count('<path') > 150:
            svg = _simplify_svg_paths(svg, epsilon=simplify_epsilon)

        if thin_path_stroke:
            svg = _thin_path_stroke(svg, max_size=thin_path_max_size,
                                    stroke_width=thin_stroke_width)

        if gap_fill and colormode == 'color':
            svg = _gap_filler(svg)

        if colormode == 'color':
            svg = _group_by_color(svg)

        print(f'[engine] {svg.count("<path")} paths after post-processing', flush=True)
        return svg

    finally:
        if inp and os.path.exists(inp):     os.unlink(inp)
        if vtr_out and os.path.exists(vtr_out): os.unlink(vtr_out)
