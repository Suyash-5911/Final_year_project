import pymupdf

svg_code = """<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
  <rect width="200" height="100" fill="#2563eb" rx="10"/>
  <text x="100" y="55" fill="white" font-size="16" text-anchor="middle" font-family="Arial">Test Box</text>
</svg>"""

doc = pymupdf.open(stream=svg_code.encode("utf-8"), filetype="svg")
page = doc[0]
pix = page.get_pixmap(dpi=300)
pix.save("test.png")
print("Rendered test.png successfully!", pix.width, pix.height)
