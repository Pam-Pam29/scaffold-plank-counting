"""Convert the 5 test images (jpeg/jpg/avif) into images/img1.png ... img5.png (sorted by file name).
Usage: python src/prepare_images.py /path/to/folder/with/the/5/originals
Needs Pillow >= 11.3 for AVIF support."""
import sys, glob, os
from PIL import Image
src = sys.argv[1]
files = sorted(glob.glob(os.path.join(src, "*")))[:5]
os.makedirs("images", exist_ok=True)
for i, f in enumerate(files, 1):
    Image.open(f).convert("RGB").save(f"images/img{i}.png"); print(f, "->", f"images/img{i}.png")
