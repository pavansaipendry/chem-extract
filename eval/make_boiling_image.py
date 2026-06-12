"""Test note for chemistry-grounded reconstruction. Mixes:
 - a unit chemistry can resolve with near-certainty (boiling water at 100 -> °C)
 - a genuinely ambiguous quantity (5 of NaCl -> mL? g?) that needs 2 versions
 - a reflux clue (ethanol reflux -> ~78 °C)
"""
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

LINES = [
    "Prep notes - batch 3",
    "Brought water to boiling at 100, added sample.",
    "Refluxed in ethanol for 40 min.",
    "Added 5 of NaCl to the flask.",
    "Cooled to 25 and filtered.",
]
OUT = Path(__file__).parent / "images" / "boiling_note.png"
FONT = "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf"


def main():
    random.seed(3)
    w, h = 1120, 500
    paper = Image.new("RGB", (w, h), (247, 245, 237))
    d = ImageDraw.Draw(paper)
    for y in range(105, h, 80):
        d.line([(40, y), (w - 40, y)], fill=(196, 205, 218), width=2)
    font = ImageFont.truetype(FONT, 39)
    y = 52
    for line in LINES:
        d.text((58 + random.randint(-5, 5), y + random.randint(-3, 3)), line,
               font=font, fill=(26, 28, 64))
        y += 80
    page = paper.rotate(-1.3, expand=True, fillcolor=(222, 217, 207), resample=Image.BICUBIC)
    arr = np.asarray(page).astype(np.int16) + np.random.normal(0, 6, np.asarray(page).shape).astype(np.int16)
    page = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    OUT.parent.mkdir(exist_ok=True)
    page.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
