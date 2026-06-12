"""Render the worked-example text as a test image (printed-style for POC)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TEXT = [
    "Experiment 14 — NaCl addition trial",
    "",
    "Heated the mixture to 70 and held for 30 min.",
    "Added 5 of NaCl solution. pH adjusted to 7.",
    "Final volume 250 mL. Cooled to 25 C.",
]

OUT = Path(__file__).parent / "images" / "worked_example.png"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    img = Image.new("RGB", (900, 320), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 30)
    except OSError:
        font = ImageFont.load_default(size=30)
    y = 40
    for line in TEXT:
        draw.text((50, y), line, fill="black", font=font)
        y += 50
    img.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
