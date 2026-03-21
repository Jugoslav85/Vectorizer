import io
import vtracer
from PIL import Image

def vectorize(image_data: bytes, **kwargs) -> str:
    img = Image.open(io.BytesIO(image_data)).convert("RGBA")
    w, h = img.size
    pixels = list(img.getdata())
    print(f"[vtracer] {w}x{h} → tracing at full resolution", flush=True)
    svg = vtracer.convert_pixels_to_svg(pixels, (w, h), **kwargs)
    print(f"[vtracer] {svg.count('<path')} paths", flush=True)
    return svg
