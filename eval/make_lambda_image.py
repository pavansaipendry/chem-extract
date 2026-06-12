"""Spectroscopy note with lambda (wavelength) in several forms, including a
bare 'lambda 280' with no unit to test flagging. Handwriting + light degradation."""
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

LINES = [
    "UV-Vis analysis of sample B",
    "lmax = 540 nm  (main peak)",      # lambda-max with unit
    "measured A at l 280",             # bare lambda, NO unit -> must flag
    "second peak l = 412 nm",          # lambda with unit
    "scan range 200 to 800 nm",
    "shoulder near 365",               # bare wavelength, no unit -> flag
]

OUT = Path(__file__).parent / "images" / "lambda_note.png"
FONT = "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf"


def main() -> None:
    random.seed(11)
    w, h = 1100, 560
    paper = Image.new("RGB", (w, h), (247, 244, 236))
    draw = ImageDraw.Draw(paper)
    for y in range(110, h, 80):
        draw.line([(40, y), (w - 40, y)], fill=(196, 205, 218), width=2)

    font = ImageFont.truetype(FONT, 40)
    # Bradley Hand lacks a real λ glyph; draw a handwritten-ish lambda where 'l' marks it
    y = 56
    for line in LINES:
        x = 60 + random.randint(-6, 6)
        # replace leading-word 'lmax'/'l' tokens with a drawn lambda
        draw.text((x, y + random.randint(-3, 4)), line.replace("lmax", "λmax").replace(" l ", " λ "),
                  font=font, fill=(26, 28, 66))
        y += 80

    page = paper.rotate(-1.2, expand=True, fillcolor=(222, 217, 207), resample=Image.BICUBIC)
    arr = np.asarray(page).astype(np.int16)
    arr += np.random.normal(0, 6, arr.shape).astype(np.int16)
    page = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    OUT.parent.mkdir(exist_ok=True)
    page.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
