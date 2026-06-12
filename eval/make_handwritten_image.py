"""Render a handwriting-style lab note with photo-like degradation:
per-word jitter and rotation, uneven baselines, off-white paper, ruled lines,
gaussian noise, slight blur, and a global page tilt.
"""
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

LINES = [
    "Run #7 - recrystallization",
    "dissolved 2.5 g KNO3 in 40 ml water",
    "heated to 85, stirred 15 min",
    "added 0.5 HCl dropwise -> pH 3",
    "cooled in ice bath to 5 C, 45 min",
    "yield: 1.8",
]

OUT = Path(__file__).parent / "images" / "handwritten_note.png"
FONT = "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf"


def main() -> None:
    random.seed(7)
    w, h = 1100, 620
    paper = Image.new("RGB", (w, h), (246, 243, 234))
    draw = ImageDraw.Draw(paper)

    # ruled notebook lines
    for y in range(110, h, 80):
        draw.line([(40, y), (w - 40, y)], fill=(190, 200, 215), width=2)

    font = ImageFont.truetype(FONT, 38)
    y = 60
    for line in LINES:
        x = 60 + random.randint(-8, 8)
        for word in line.split(" "):
            # each word on its own tilted canvas for a hand-jitter feel
            wob = Image.new("RGBA", (420, 90), (0, 0, 0, 0))
            ImageDraw.Draw(wob).text(
                (8, 8), word, font=font,
                fill=(28, 30, 70 + random.randint(-20, 20)),
            )
            wob = wob.rotate(random.uniform(-2.5, 2.5), expand=False,
                             resample=Image.BICUBIC)
            paper.paste(wob, (x, y + random.randint(-4, 5)), wob)
            x += int(ImageDraw.Draw(paper).textlength(word + "  ", font=font))
        y += 82

    # global tilt + photo noise + blur
    page = paper.rotate(-1.6, expand=True, fillcolor=(220, 215, 205),
                        resample=Image.BICUBIC)
    arr = np.asarray(page).astype(np.int16)
    arr += np.random.normal(0, 7, arr.shape).astype(np.int16)
    page = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    page = page.filter(ImageFilter.GaussianBlur(0.7))

    OUT.parent.mkdir(exist_ok=True)
    page.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
