"""
Run this directly on your machine: python3 test_vtracer.py
It will tell us exactly what vtracer version you have and test both API methods.
"""
import vtracer, io
from PIL import Image

print("vtracer version:", getattr(vtracer, '__version__', 'unknown'))
print("vtracer file:", vtracer.__file__)
print()

# Load the gradient image
img_path = '/Users/' # <-- doesnt matter, we use a synthetic gradient

# Create a synthetic gradient in memory — dark blue to cyan
width, height = 400, 566
img = Image.new('RGB', (width, height))
pixels_rgb = []
for y in range(height):
    for x in range(width):
        # left=dark blue, right=cyan, top=teal
        r = int(x / width * 10)
        g = int(55 + y / height * 200)
        b = int(100 + x / width * 155)
        pixels_rgb.append((r, g, b))
img.putdata(pixels_rgb)
img_rgba = img.convert('RGBA')

print("=== Test 1: convert_pixels_to_svg with RGBA ===")
pixels = list(img_rgba.getdata())
svg1 = vtracer.convert_pixels_to_svg(
    pixels, (width, height),
    colormode='color', hierarchical='stacked', mode='spline',
    filter_speckle=1, color_precision=8, layer_difference=2,
    corner_threshold=60, length_threshold=4.0, max_iterations=20,
)
print("Paths:", svg1.count('<path'))

print()
print("=== Test 2: convert_raw_image_to_svg with PNG bytes ===")
buf = io.BytesIO()
img.save(buf, 'PNG')
raw = buf.getvalue()
svg2 = vtracer.convert_raw_image_to_svg(
    raw, img_format='png',
    colormode='color', hierarchical='stacked', mode='spline',
    filter_speckle=1, color_precision=8, layer_difference=2,
    corner_threshold=60, length_threshold=4.0, max_iterations=20,
)
print("Paths:", svg2.count('<path'))

print()
print("=== Test 3: pixels_to_svg with RGB tuples (no alpha) ===")
pixels_rgb_list = [(r,g,b,255) for r,g,b in pixels_rgb]
svg3 = vtracer.convert_pixels_to_svg(
    pixels_rgb_list, (width, height),
    colormode='color', hierarchical='stacked', mode='spline',
    filter_speckle=1, color_precision=8, layer_difference=2,
    corner_threshold=60, length_threshold=4.0, max_iterations=20,
)
print("Paths:", svg3.count('<path'))
