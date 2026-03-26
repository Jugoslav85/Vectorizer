"""
Vectorizer engine — vtracer backend.

Pre-processing (individually togglable):
  resize
  -> unsharp_mask       (sharpen fine detail, default on, 100%)
  -> clahe              (local contrast, photos only, default off)
  -> lab_shadow_lift    (lift dark detail via LAB L-channel, no colour shift)
  -> saturation_boost   (separate similar colours for vtracer)
  -> global_contrast    (uniform contrast lift)
  -> colour_quantise    (k-means palette reduction, replaces posterize)
  -> posterize          (channel-based banding, legacy)
  -> vtracer

Post-processing (individually togglable):
  -> smart_path_retention / remove_short_paths
  -> rdp_simplify
  -> thin_path_stroke
  -> gap_fill
  -> group_by_color
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

def _unsharp_mask(img, radius=1.0, percent=100, threshold=3):
    """Sharpen fine detail. The internal blur never reaches vtracer."""
    rgb = img.convert('RGB').filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    )
    print(f'[engine] unsharp_mask: r={radius} p={percent}', flush=True)
    if img.mode == 'RGBA':
        out = rgb.convert('RGBA'); out.putalpha(img.getchannel('A')); return out
    return rgb


# ---------------------------------------------------------------------------
# Pre-processing: CLAHE (photos only)
# ---------------------------------------------------------------------------

def _clahe(img, clip_limit=2.0, tile_size=8):
    """Local contrast enhancement. Photos/portraits only — destroys flat art."""
    rgb = img.convert('RGB')
    if _CV2:
        import numpy as _np
        arr  = _np.asarray(rgb)
        lab  = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl   = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        lab2 = cv2.merge([cl.apply(l), a, b])
        result = Image.fromarray(cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB), 'RGB')
        print(f'[engine] clahe (cv2): clip={clip_limit}', flush=True)
    else:
        w, h   = rgb.size
        ts     = max(16, min(w, h) // tile_size)
        result = rgb.copy()
        for y in range(0, h, ts):
            for x in range(0, w, ts):
                box = (x, y, min(x+ts, w), min(y+ts, h))
                tile = rgb.crop(box)
                result.paste(Image.blend(tile, ImageOps.equalize(tile), 0.6), box)
        print(f'[engine] clahe (pillow): tile={ts}', flush=True)
    if img.mode == 'RGBA':
        out = result.convert('RGBA'); out.putalpha(img.getchannel('A')); return out
    return result


# ---------------------------------------------------------------------------
# Pre-processing: LAB Shadow Lift
# ---------------------------------------------------------------------------

def _lab_shadow_lift(img, strength=0.4, shadow_end=100):
    """
    Lift dark detail via the LAB L-channel only.
    Colours (A and B channels) are never touched — zero colour shift.
    strength: how much to brighten shadows (0.1–0.8)
    shadow_end: L values below this are lifted (0–180 in LAB space)
    """
    if not _NUMPY:
        # Pillow fallback: curves on luminance only
        gray  = img.convert('L')
        lut   = []
        for i in range(256):
            if i < shadow_end:
                v = int(i + (shadow_end - i) * strength)
            else:
                v = i
            lut.append(min(255, v))
        rgb = img.convert('RGB')
        r, g, b = rgb.split()
        # Apply only to luminance-equivalent — approximate via each channel
        # Better than nothing but not colour-accurate
        lifted = Image.merge('RGB', (r.point(lut), g.point(lut), b.point(lut)))
        print(f'[engine] lab_shadow_lift (pillow approx): str={strength}', flush=True)
        if img.mode == 'RGBA':
            out = lifted.convert('RGBA'); out.putalpha(img.getchannel('A')); return out
        return lifted

    arr  = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
    # Convert to LAB
    # Simple sRGB -> XYZ -> LAB (D65)
    def srgb_to_linear(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def linear_to_lab(rgb_lin):
        M = np.array([[0.4124564,0.3575761,0.1804375],
                      [0.2126729,0.7151522,0.0721750],
                      [0.0193339,0.1191920,0.9503041]])
        xyz = rgb_lin @ M.T
        xyz /= np.array([0.95047, 1.0, 1.08883])
        e = 0.008856
        xyz = np.where(xyz > e, xyz ** (1/3), 7.787 * xyz + 16/116)
        L = 116 * xyz[..., 1] - 16
        A = 500 * (xyz[..., 0] - xyz[..., 1])
        B = 200 * (xyz[..., 1] - xyz[..., 2])
        return L, A, B

    def lab_to_linear(L, A, B):
        fy = (L + 16) / 116
        fx = A / 500 + fy
        fz = fy - B / 200
        e  = 0.008856
        x  = np.where(fx**3 > e, fx**3, (fx - 16/116) / 7.787)
        y  = np.where(fy**3 > e, fy**3, (fy - 16/116) / 7.787)
        z  = np.where(fz**3 > e, fz**3, (fz - 16/116) / 7.787)
        xyz = np.stack([x, y, z], axis=-1) * np.array([0.95047, 1.0, 1.08883])
        M_inv = np.array([[ 3.2404542,-1.5371385,-0.4985314],
                          [-0.9692660, 1.8760108, 0.0415560],
                          [ 0.0556434,-0.2040259, 1.0572252]])
        return np.clip(xyz @ M_inv.T, 0, None)

    def linear_to_srgb(c):
        return np.where(c <= 0.0031308, 12.92*c, 1.055*c**(1/2.4) - 0.055)

    lin  = srgb_to_linear(arr)
    L, A, B = linear_to_lab(lin)

    # Lift L values below shadow_end
    lab_shadow = shadow_end / 100 * 100  # map to LAB L scale (0-100)
    mask = L < lab_shadow
    L    = np.where(mask, L + (lab_shadow - L) * strength, L)

    lin2   = lab_to_linear(L, A, B)
    result = np.clip(linear_to_srgb(lin2), 0, 1)
    out    = Image.fromarray((result * 255).astype(np.uint8), 'RGB')
    print(f'[engine] lab_shadow_lift: str={strength} shadow_end={shadow_end}', flush=True)
    if img.mode == 'RGBA':
        out2 = out.convert('RGBA'); out2.putalpha(img.getchannel('A')); return out2
    return out


# ---------------------------------------------------------------------------
# Pre-processing: Saturation Boost
# ---------------------------------------------------------------------------

def _saturation_boost(img, factor=1.3):
    """
    Boost saturation before tracing — makes similar colours more distinct
    so vtracer creates separate layers for regions that would otherwise merge.
    factor: 1.0 = no change, 1.2-1.5 = mild, 2.0 = strong
    """
    rgb    = img.convert('RGB')
    result = ImageEnhance.Color(rgb).enhance(factor)
    print(f'[engine] saturation_boost: {factor}x', flush=True)
    if img.mode == 'RGBA':
        out = result.convert('RGBA'); out.putalpha(img.getchannel('A')); return out
    return result


# ---------------------------------------------------------------------------
# Pre-processing: Global Contrast
# ---------------------------------------------------------------------------

def _global_contrast(img, factor=1.2):
    """
    Uniform contrast boost. Preserves relative colour relationships — a region
    that's 20% darker than its neighbour stays 20% darker. Helps vtracer
    separate layers that are too similar without shifting hues.
    factor: 1.0 = no change, 1.1-1.4 = mild to strong
    """
    rgb    = img.convert('RGB')
    result = ImageEnhance.Contrast(rgb).enhance(factor)
    print(f'[engine] global_contrast: {factor}x', flush=True)
    if img.mode == 'RGBA':
        out = result.convert('RGBA'); out.putalpha(img.getchannel('A')); return out
    return result


# ---------------------------------------------------------------------------
# Pre-processing: Colour Quantise
# ---------------------------------------------------------------------------

def _colour_quantise(img, n_colors=32):
    """
    K-means style palette reduction via Pillow median-cut.
    Each colour is a genuine cluster from the image — no channel banding.
    Cleaner than posterize for illustrations and photos alike.
    """
    rgb      = img.convert('RGB')
    q        = rgb.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT, dither=0)
    result   = q.convert('RGB')
    print(f'[engine] colour_quantise: {n_colors} colours', flush=True)
    if img.mode == 'RGBA':
        out = result.convert('RGBA'); out.putalpha(img.getchannel('A')); return out
    return result


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _remove_short_paths(svg, min_size=2.0):
    def is_tiny(d):
        nums = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)
        if len(nums) < 4: return True
        coords = [float(n) for n in nums]
        xs, ys = coords[0::2], coords[1::2]
        return (max(xs)-min(xs)) < min_size and (max(ys)-min(ys)) < min_size
    return re.sub(r'<path[^>]*d="([^"]+)"[^/]*/?>',
                  lambda m: '' if is_tiny(m.group(1)) else m.group(0), svg)


def _smart_path_retention(svg, proximity_px=8.0):
    """Keep tiny paths near same-colour neighbours; discard isolated noise."""
    fill_re = re.compile(r'fill="(#[0-9a-fA-F]{3,6})"')
    num_re  = re.compile(r'[-+]?[0-9]*\.?[0-9]+')

    def bbox(d):
        nums = [float(x) for x in num_re.findall(d)]
        if len(nums) < 4: return None
        xs, ys = nums[0::2], nums[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    paths = []
    for m in re.finditer(r'<path\b[^>]*/?>|<path\b[^>]*>.*?</path>', svg, re.DOTALL):
        el  = m.group(0)
        dm  = re.search(r'd="([^"]+)"', el)
        fm  = fill_re.search(el)
        if not dm: continue
        bb = bbox(dm.group(1))
        if bb is None: continue
        w, h = bb[2]-bb[0], bb[3]-bb[1]
        paths.append({
            'el': el, 'fill': fm.group(1).lower() if fm else None,
            'cx': (bb[0]+bb[2])/2, 'cy': (bb[1]+bb[3])/2,
            'tiny': w < proximity_px and h < proximity_px,
        })

    keep = set()
    for i, p in enumerate(paths):
        if not p['tiny']:
            keep.add(i); continue
        for j, q in enumerate(paths):
            if i == j: continue
            if q['fill'] != p['fill']: continue
            if math.hypot(p['cx']-q['cx'], p['cy']-q['cy']) <= proximity_px*3:
                keep.add(i); break

    if len(keep) == len(paths):
        return svg

    idx = [0]
    def repl(m):
        k = idx[0]; idx[0] += 1
        return paths[k]['el'] if k in keep else ''
    result = re.sub(r'<path\b[^>]*/?>|<path\b[^>]*>.*?</path>', repl, svg, flags=re.DOTALL)
    print(f'[engine] smart_path_retention: kept {len(keep)}/{len(paths)}', flush=True)
    return result


def _rdp_dist(p, a, b):
    if a == b: return math.hypot(p[0]-a[0], p[1]-a[1])
    dx, dy = b[0]-a[0], b[1]-a[1]
    return abs(dy*p[0]-dx*p[1]+b[0]*a[1]-b[1]*a[0]) / math.hypot(dx, dy)

def _rdp(pts, eps):
    if len(pts) < 3: return pts
    d, idx = max((_rdp_dist(pts[i], pts[0], pts[-1]), i) for i in range(1, len(pts)-1))
    if d > eps: return _rdp(pts[:idx+1], eps)[:-1] + _rdp(pts[idx:], eps)
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
                    for pt in s[1:]: result.append(f'{pt[0]:.2f},{pt[1]:.2f}')
            line_pts = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t in ('M','m'):
                flush(); result.append(t); i += 1
                if i+1 < len(tokens):
                    x,y = float(tokens[i]), float(tokens[i+1])
                    result.append(f'{x:.2f},{y:.2f}'); line_pts=[(x,y)]; i+=2
            elif t == 'L':
                i += 1
                while i+1<len(tokens) and tokens[i] not in 'MmLlCcSsQqZz':
                    line_pts.append((float(tokens[i]),float(tokens[i+1]))); i+=2
            elif t in ('C','c','S','s','Q','q'):
                flush(); result.append(t); i+=1
                while i<len(tokens) and tokens[i] not in 'MmLlCcSsQqZz':
                    result.append(tokens[i]); i+=1
            elif t in ('Z','z'):
                flush(); result.append(t); i+=1
            else:
                result.append(t); i+=1
        flush()
        return ' '.join(result)
    return re.sub(r'd="([^"]+)"', lambda m: f'd="{simplify_path(m.group(1))}"', svg)


def _thin_path_stroke(svg, max_size=6.0, stroke_width=0.5):
    """Add hairline stroke to thin paths so they stay visible at all zoom levels."""
    fill_re = re.compile(r'fill="(#[0-9a-fA-F]{3,6})"')
    num_re  = re.compile(r'[-+]?[0-9]*\.?[0-9]+')
    def add_stroke(m):
        el = m.group(0)
        dm = re.search(r'd="([^"]+)"', el)
        if not dm: return el
        nums = [float(x) for x in num_re.findall(dm.group(1))]
        if len(nums) < 4: return el
        xs,ys = nums[0::2], nums[1::2]
        if (max(xs)-min(xs)) >= max_size and (max(ys)-min(ys)) >= max_size:
            return el
        fm = fill_re.search(el)
        if not fm: return el
        el = el.rstrip('/>').rstrip()
        el += f' stroke="{fm.group(1)}" stroke-width="{stroke_width}"/>'
        return el
    return re.sub(r'<path\b[^>]*/?>',  add_stroke, svg)


def _gap_filler(svg, stroke_width=1.5):
    stroke_paths = []
    fill_re = re.compile(r'fill="(#[0-9a-fA-F]{3,6})"')
    for m in re.finditer(r'<path\b[^>]*/?>|<path\b[^>]*>.*?</path>', svg, re.DOTALL):
        el = m.group(0); fm = fill_re.search(el)
        if not fm: continue
        color = fm.group(1)
        stroked = fill_re.sub(
            f'fill="none" stroke="{color}" stroke-width="{stroke_width}" '
            f'stroke-linejoin="round" stroke-linecap="round"', el, count=1)
        stroke_paths.append('    ' + stroked.strip())
    if not stroke_paths: return svg
    layer = '  <g id="gap-filler">\n' + '\n'.join(stroke_paths) + '\n  </g>'
    m = re.search(r'<svg[^>]+>', svg)
    if not m: return svg
    svg = svg[:m.end()] + '\n' + layer + svg[m.end():]
    print(f'[engine] gap_filler: {len(stroke_paths)} paths', flush=True)
    return svg


def _group_by_color(svg):
    m = re.search(r'(<svg[^>]+>)', svg)
    if not m: return svg
    svg_open  = m.group(1)
    elements  = re.findall(r'(<rect\b[^>]*/?>|<path\b[^>]*/?>|<path\b[^>]*>.*?</path>)', svg, re.DOTALL)
    if not elements: return svg
    fill_re = re.compile(r'\s*fill="(#[0-9a-fA-F]{3,6})"')
    bg, by_color, order = [], {}, []
    for el in elements:
        if el.strip().startswith('<rect'): bg.append(el); continue
        fm    = fill_re.search(el)
        color = ('#' + fm.group(1).lstrip('#').lower()) if fm else '__none__'
        clean = fill_re.sub('', el, count=1)
        if color not in by_color: by_color[color]=[]; order.append(color)
        by_color[color].append(clean)
    lines = [svg_open]
    for el in bg: lines.append('  '+el)
    for color in order:
        if color == '__none__':
            for p in by_color[color]: lines.append('  '+p.strip())
        else:
            lines.append(f'  <g fill="{color}">')
            for p in by_color[color]: lines.append('    '+p.strip())
            lines.append('  </g>')
    lines.append('</svg>')
    print(f'[engine] group_by_color: {len([c for c in order if c!="__none__"])} groups', flush=True)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def vectorize(
    image_data,
    colormode              = 'color',

    # Pre-processing
    unsharp_mask           = True,
    unsharp_radius         = 1.0,
    unsharp_percent        = 100,
    unsharp_threshold      = 3,

    clahe                  = False,
    clahe_clip             = 2.0,
    clahe_tile             = 8,

    lab_shadow_lift        = False,
    lab_shadow_strength    = 0.4,
    lab_shadow_end         = 100,

    saturation_boost       = False,
    saturation_factor      = 1.3,

    global_contrast        = False,
    global_contrast_factor = 1.2,

    colour_quantise        = False,
    colour_quantise_n      = 32,

    posterize_bits         = 0,

    # vtracer
    filter_speckle         = 4,
    color_precision        = 6,
    layer_difference       = 16,
    corner_threshold       = 60,
    length_threshold       = 4.0,
    splice_threshold       = 45,

    # Post-processing
    smart_path_retention   = True,
    smart_proximity        = 8.0,
    simplify_epsilon       = 0.1,
    thin_path_stroke       = True,
    thin_path_max_size     = 6.0,
    thin_stroke_width      = 0.5,
    gap_fill               = True,
):
    img = Image.open(io.BytesIO(image_data)).convert('RGBA')
    img = _resize(img)

    # ── Pre-processing ────────────────────────────────────────────────────
    if unsharp_mask:
        img = _unsharp_mask(img, radius=unsharp_radius,
                            percent=unsharp_percent, threshold=unsharp_threshold)
    if clahe:
        img = _clahe(img, clip_limit=clahe_clip, tile_size=clahe_tile)
    if lab_shadow_lift:
        img = _lab_shadow_lift(img, strength=lab_shadow_strength, shadow_end=lab_shadow_end)
    if saturation_boost:
        img = _saturation_boost(img, factor=saturation_factor)
    if global_contrast:
        img = _global_contrast(img, factor=global_contrast_factor)
    if colour_quantise:
        img = _colour_quantise(img, n_colors=colour_quantise_n)

    rgb = img.convert('RGB')
    if posterize_bits >= 2:
        rgb = ImageOps.posterize(rgb, posterize_bits)
        print(f'[engine] posterize: {posterize_bits} bits', flush=True)

    if img.mode == 'RGBA':
        out = rgb.convert('RGBA'); out.putalpha(img.getchannel('A')); processed = out
    else:
        processed = rgb

    # ── vtracer ───────────────────────────────────────────────────────────
    inp = vtr_out = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            processed.save(f.name, format='PNG'); inp = f.name
        vtr_out = inp.replace('.png', '.svg')
        print(f'[engine] vtracer: colormode={colormode} fs={filter_speckle} '
              f'cp={color_precision} ld={layer_difference} ct={corner_threshold} '
              f'lt={length_threshold} st={splice_threshold}', flush=True)
        vtracer.convert_image_to_svg_py(
            inp, vtr_out,
            colormode=colormode, hierarchical='stacked', mode='spline',
            filter_speckle=filter_speckle, color_precision=color_precision,
            layer_difference=layer_difference, corner_threshold=corner_threshold,
            length_threshold=length_threshold, max_iterations=10,
            splice_threshold=splice_threshold, path_precision=8,
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
        if inp and os.path.exists(inp): os.unlink(inp)
        if vtr_out and os.path.exists(vtr_out): os.unlink(vtr_out)
