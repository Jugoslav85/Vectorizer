"""
Vectorizer engine — vtracer backend.

Resize strategy (adaptive by colour complexity):
  Simple images (<=8 unique colours)  -> 0.6MP  -- logos, icons, flat art
  Normal images                        -> 1.0MP  -- illustrations, lineart

Pipelines:
  LINEART -- GaussianBlur -> MedianFilter -> contrast boost -> Otsu threshold -> morph close -> vtracer bw
  COLOR   -- GaussianBlur -> MedianFilter -> UnsharpMask -> posterize -> vtracer color

Post-processing:
  Short path removal -> RDP simplification (only if path count > 150)
"""

import io
import re
import tempfile
import os
import math
import vtracer

from PIL import Image, ImageFilter, ImageOps, ImageEnhance


# -- Adaptive resize ----------------------------------------------------------

TARGET_SIMPLE = 600_000    # 0.6MP -- simple/flat images
TARGET_NORMAL = 1_000_000  # 1.0MP -- everything else
MIN_UPSCALE   = 300_000    # upscale anything below 0.3MP


def _count_unique_colours(img):
    small = img.convert('RGB').resize((64, 64), Image.LANCZOS)
    quantised = small.quantize(colors=64, dither=0)
    return len(set(quantised.getdata()))


def _resize(img, target_px):
    w, h = img.size
    pixels = w * h
    if pixels < MIN_UPSCALE:
        scale = math.sqrt(target_px / pixels)
        nw, nh = int(w * scale), int(h * scale)
        print(f'[engine] upscaled {w}x{h} -> {nw}x{nh}', flush=True)
        return img.resize((nw, nh), Image.LANCZOS)
    if pixels > target_px:
        scale = math.sqrt(target_px / pixels)
        nw, nh = int(w * scale), int(h * scale)
        print(f'[engine] downscaled {w}x{h} -> {nw}x{nh} (target {target_px//1000}k px)', flush=True)
        return img.resize((nw, nh), Image.LANCZOS)
    print(f'[engine] size ok {w}x{h} ({pixels/1_000_000:.2f}MP)', flush=True)
    return img


# -- Otsu threshold -----------------------------------------------------------

def _otsu_threshold(gray):
    hist  = gray.histogram()
    total = sum(hist)
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


# -- Morphological close ------------------------------------------------------

def _morph_close(img, size=3):
    """Dilate then erode -- joins small gaps in lineart strokes."""
    size = size if size % 2 == 1 else size + 1
    dilated = img.filter(ImageFilter.MaxFilter(size))
    return dilated.filter(ImageFilter.MinFilter(size))


# -- SVG path simplification (RDP) --------------------------------------------

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


# -- Short path removal -------------------------------------------------------

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


# -- Auto-detect mode ---------------------------------------------------------

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


# -- Main entry point ---------------------------------------------------------

def vectorize(
    image_data,
    # Pre-processing
    blur_radius       = 0.8,
    median_size       = 3,
    morph_close_size  = 3,
    unsharp_radius    = 0.5,
    unsharp_percent   = 90,
    unsharp_threshold = 4,
    posterize_bits    = 6,
    # Engine
    engine_mode       = 'auto',
    # vtracer
    color_precision   = 6,
    layer_difference  = 4,
    filter_speckle    = 6,
    corner_threshold  = 48,
    splice_threshold  = 70,
    # Post-processing
    simplify          = True,
    simplify_epsilon  = 0.3,
    **kwargs,
):
    img = Image.open(io.BytesIO(image_data)).convert('RGBA')

    # Adaptive resize
    unique_cols = _count_unique_colours(img)
    target_px   = TARGET_SIMPLE if unique_cols <= 8 else TARGET_NORMAL
    print(f'[engine] {unique_cols} unique colours -> target {target_px//1000}k px', flush=True)
    img = _resize(img, target_px)
    w, h = img.size

    # Mode selection
    mode = engine_mode
    if mode == 'auto':
        mode = _detect_mode(img)
    print(f'[engine] mode={mode}', flush=True)

    if mode == 'lineart':
        rgb = img.convert('RGB')

        if blur_radius > 0:
            rgb = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        if median_size > 1:
            ms  = median_size if median_size % 2 == 1 else median_size + 1
            rgb = rgb.filter(ImageFilter.MedianFilter(size=ms))

        rgb    = ImageEnhance.Contrast(rgb).enhance(2.5)
        rgb    = ImageEnhance.Sharpness(rgb).enhance(2.0)
        gray   = rgb.convert('L')
        binary = _otsu_threshold(gray)

        if morph_close_size > 1:
            binary = _morph_close(binary, size=morph_close_size)

        processed = binary.convert('RGB')

        vtracer_kwargs = dict(
            colormode        = 'bw',
            mode             = 'spline',
            filter_speckle   = filter_speckle,
            corner_threshold = corner_threshold,
            length_threshold = 4.0,
            splice_threshold = splice_threshold,
            path_precision   = 2,
            max_iterations   = 1,
        )
        print(
            f'[engine] lineart pipeline: blur={blur_radius} median={median_size} '
            f'morph={morph_close_size} corner={corner_threshold} splice={splice_threshold}',
            flush=True,
        )

    else:  # color
        rgb = img.convert('RGB')

        if blur_radius > 0:
            rgb = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        if median_size > 1:
            ms  = median_size if median_size % 2 == 1 else median_size + 1
            rgb = rgb.filter(ImageFilter.MedianFilter(size=ms))

        if unsharp_percent > 0:
            rgb = rgb.filter(ImageFilter.UnsharpMask(
                radius=unsharp_radius,
                percent=unsharp_percent,
                threshold=unsharp_threshold,
            ))

        rgb = ImageOps.posterize(rgb, posterize_bits)

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
            length_threshold = 4.0,
            splice_threshold = splice_threshold,
            path_precision   = 2,
            max_iterations   = 1,
        )
        print(
            f'[engine] color pipeline: blur={blur_radius} median={median_size} '
            f'unsharp={unsharp_percent}% posterize={posterize_bits}bits '
            f'cp={color_precision} ld={layer_difference} '
            f'corner={corner_threshold} splice={splice_threshold}',
            flush=True,
        )

    # Run vtracer
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
