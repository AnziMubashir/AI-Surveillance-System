# image_convert.py

from PIL import Image
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python image_convert.py <image_path>")
    sys.exit(1)

input_path = sys.argv[1]
try:
    img = Image.open(input_path).convert("RGB")
    img.save(input_path)
    print(f"[+] Converted and saved: {input_path}")
except Exception as e:
    print(f"[!] Failed to convert {input_path}: {e}")
