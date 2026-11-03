import os
import pymupdf

OUTPUT_DIR = os.path.join(os.getcwd(), "diagrams")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def render_svg_to_png_and_jpg(svg_code: str, base_name: str, dpi: int = 300):
    svg_path = os.path.join(OUTPUT_DIR, f"{base_name}.svg")
    png_path = os.path.join(OUTPUT_DIR, f"{base_name}.png")
    jpg_path = os.path.join(OUTPUT_DIR, f"{base_name}.jpg")
    
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    
    doc = pymupdf.open(stream=svg_code.encode("utf-8"), filetype="svg")
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    pix.save(png_path)
    # also save jpg
    pix_rgb = pymupdf.Pixmap(pix, 0) if pix.alpha else pix
    pix_rgb.save(jpg_path)
    
    print(f"Generated: {base_name}.png ({pix.width}x{pix.height}) and {base_name}.jpg")
    return png_path, jpg_path

print("Rendering helper initialized.")
