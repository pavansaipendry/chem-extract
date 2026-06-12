#!/usr/bin/env python3
"""
Synthetic handwritten chemistry lab-notes dataset generator.

Produces image + ground-truth-label pairs for training/testing a
document-extraction system whose core challenge is AMBIGUITY:
- units dropped ("70" instead of "70 degC")
- bare quantities ("5 of NaCl" -- mL? g? mol?)
- OCR-style noise / messy handwriting
Each image ships with a JSON label describing every field, whether a
unit was present in the written form, and what the *true* value is.
"""
import os, json, random, math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

random.seed(42)

FONT_DIR = "fonts"
OUT_DIR  = "chem_notes_dataset"
IMG_DIR  = os.path.join(OUT_DIR, "images")
LBL_DIR  = os.path.join(OUT_DIR, "labels")
for d in (IMG_DIR, LBL_DIR):
    os.makedirs(d, exist_ok=True)

FONTS = [os.path.join(FONT_DIR, f) for f in os.listdir(FONT_DIR)
         if f.endswith(".ttf") and os.path.getsize(os.path.join(FONT_DIR, f)) > 1000]

# ---- chemistry content building blocks -------------------------------------
REAGENTS  = ["NaCl", "HCl", "NaOH", "H2SO4", "EtOH", "MeOH", "K2CO3", "NaHCO3",
             "AcOH", "Et3N", "DCM", "THF", "DMF", "acetone", "NaBH4", "LiAlH4",
             "Pd/C", "AgNO3", "CuSO4", "KMnO4", "aniline", "phenol", "benzaldehyde"]
SOLVENTS  = ["ethanol", "methanol", "water", "DCM", "THF", "acetone", "toluene", "DMSO"]
VERBS     = ["Heated", "Cooled", "Stirred", "Refluxed", "Added", "Dissolved",
             "Titrated", "Quenched", "Filtered", "Distilled"]

# Each quantity type carries the "true" unit and whether writers tend to drop it.
TEMP_UNITS = ["degC", "C", "°C", ""]        # "" == dropped
VOL_UNITS  = ["mL", "ml", "L", ""]
MASS_UNITS = ["g", "mg", "kg", ""]
MOL_UNITS  = ["mol", "mmol", "M", ""]
TIME_UNITS = ["min", "h", "hr", "sec", ""]

def maybe_drop(unit_choices, drop_prob=0.45):
    """Pick a written unit; with drop_prob, drop it entirely (the hard case)."""
    if random.random() < drop_prob:
        return ""        # dropped -> ambiguous
    u = random.choice([u for u in unit_choices if u != ""])
    return u

def line_temp():
    val  = random.choice([0, 4, 25, 40, 50, 60, 70, 78, 80, 95, 100, 110, 150, -10, -78])
    written_u = maybe_drop(TEMP_UNITS)
    txt = f"{random.choice(['Heated','Cooled','Held'])} to {val}{(' '+written_u) if written_u else ''}"
    field = {"type":"temperature","raw_value":str(val),
             "written_unit": written_u if written_u else None,
             "true_unit":"celsius","unit_present": bool(written_u),
             "ambiguous": not bool(written_u)}
    return txt, field

def line_volume():
    val = random.choice([1,2,5,10,15,20,25,50,100,250])
    written_u = maybe_drop(VOL_UNITS)
    reag = random.choice(REAGENTS)
    txt = f"Added {val}{(' '+written_u) if written_u else ''} of {reag}"
    field = {"type":"volume","raw_value":str(val),
             "written_unit": written_u if written_u else None,
             "true_unit":"milliliter","reagent":reag,
             "unit_present": bool(written_u),"ambiguous": not bool(written_u)}
    return txt, field

def line_mass():
    val = random.choice([0.5,1,1.2,2,2.5,5,10,25,50,0.05,0.25])
    written_u = maybe_drop(MASS_UNITS)
    reag = random.choice(REAGENTS)
    txt = f"Weighed {val}{(' '+written_u) if written_u else ''} {reag}"
    field = {"type":"mass","raw_value":str(val),
             "written_unit": written_u if written_u else None,
             "true_unit":"gram","reagent":reag,
             "unit_present": bool(written_u),"ambiguous": not bool(written_u)}
    return txt, field

def line_conc():
    val = random.choice([0.1,0.5,1,2,6,12,0.05,0.25])
    written_u = maybe_drop(MOL_UNITS)
    reag = random.choice(["HCl","NaOH","H2SO4","NaCl","AgNO3","NaOH"])
    txt = f"{val}{(' '+written_u) if written_u else ''} {reag} soln"
    field = {"type":"concentration","raw_value":str(val),
             "written_unit": written_u if written_u else None,
             "true_unit":"molar","reagent":reag,
             "unit_present": bool(written_u),"ambiguous": not bool(written_u)}
    return txt, field

def line_time():
    val = random.choice([5,10,15,20,30,45,60,90,120,2,3,4])
    written_u = maybe_drop(TIME_UNITS)
    txt = f"{random.choice(['Stirred','Refluxed','Heated'])} for {val}{(' '+written_u) if written_u else ''}"
    field = {"type":"time","raw_value":str(val),
             "written_unit": written_u if written_u else None,
             "true_unit":"minute","unit_present": bool(written_u),
             "ambiguous": not bool(written_u)}
    return txt, field

def line_ph():
    val = random.choice([1,2,3,4,5,6,7,8,9,10,11,12,13])
    txt = f"adjusted pH to {val}"
    field = {"type":"ph","raw_value":str(val),"written_unit":None,
             "true_unit":"dimensionless","unit_present":True,"ambiguous":False}
    return txt, field

def line_yield():
    val = random.choice([42,55,63,71,78,85,90,93])
    has_pct = random.random() > 0.4
    txt = f"yield {val}{'%' if has_pct else ''}"
    field = {"type":"yield","raw_value":str(val),
             "written_unit": "%" if has_pct else None,
             "true_unit":"percent","unit_present":has_pct,"ambiguous": not has_pct}
    return txt, field

GENERATORS = [line_temp, line_volume, line_mass, line_conc, line_time, line_ph, line_yield]

# ---- rendering --------------------------------------------------------------
PAPER = (252, 250, 242)
INK_COLORS = [(20,24,60),(28,30,40),(10,20,90),(40,30,30)]  # blue/black-ish pens

def make_paper(w, h):
    img = Image.new("RGB", (w, h), PAPER)
    d = ImageDraw.Draw(img)
    # faint ruled lines
    for y in range(60, h, 46):
        d.line([(20,y),(w-20,y)], fill=(210,220,230), width=1)
    # margin line
    d.line([(70,20),(70,h-20)], fill=(230,190,190), width=1)
    return img

def jitter_text(draw, xy, text, font, fill):
    """Draw text char-by-char with slight baseline/rotation jitter for realism."""
    x, y = xy
    for ch in text:
        dy = random.randint(-2, 2)
        draw.text((x, y+dy), ch, font=font, fill=fill)
        w = draw.textlength(ch, font=font)
        x += w + random.uniform(-0.5, 1.2)
    return x

def render_note(idx):
    w, h = 760, 520
    img = make_paper(w, h)
    draw = ImageDraw.Draw(img)
    font_path = random.choice(FONTS)
    base_size = random.choice([30, 34, 38])
    ink = random.choice(INK_COLORS)

    title_font = ImageFont.truetype(font_path, base_size+6)
    body_font  = ImageFont.truetype(font_path, base_size)

    # header
    expt = random.randint(1, 99)
    jitter_text(draw, (85, 26), f"Expt {expt}: {random.choice(['Synthesis','Titration','Extraction','Reflux','Prep'])}",
                title_font, ink)

    n_lines = random.randint(4, 7)
    chosen = random.sample(GENERATORS, k=min(n_lines, len(GENERATORS)))
    # pad if we want more lines than unique generators
    while len(chosen) < n_lines:
        chosen.append(random.choice(GENERATORS))

    fields = []
    y = 92
    for i, gen in enumerate(chosen):
        txt, field = gen()
        prefix = f"{i+1}) "
        jitter_text(draw, (88, y), prefix + txt, body_font, ink)
        field["line"] = i+1
        field["written_text"] = txt
        fields.append(field)
        y += random.randint(48, 60)

    # slight blur + noise to mimic scan
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(0.4))
    # rotate a touch
    if random.random() > 0.4:
        img = img.rotate(random.uniform(-1.5, 1.5), expand=False, fillcolor=PAPER)

    label = {
        "image_id": f"chem_{idx:04d}",
        "experiment": f"Expt {expt}",
        "font": os.path.basename(font_path),
        "n_fields": len(fields),
        "n_ambiguous": sum(1 for f in fields if f["ambiguous"]),
        "fields": fields,
    }
    return img, label

# ---- main -------------------------------------------------------------------
N = 200
manifest = []
amb_total = 0
field_total = 0
for i in range(N):
    img, label = render_note(i)
    img.save(os.path.join(IMG_DIR, f"chem_{i:04d}.png"))
    with open(os.path.join(LBL_DIR, f"chem_{i:04d}.json"), "w") as f:
        json.dump(label, f, indent=2)
    amb_total += label["n_ambiguous"]
    field_total += label["n_fields"]
    manifest.append({"image": f"images/chem_{i:04d}.png",
                     "label": f"labels/chem_{i:04d}.json",
                     "n_fields": label["n_fields"],
                     "n_ambiguous": label["n_ambiguous"]})

with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Generated {N} images")
print(f"Total fields: {field_total}, ambiguous (unit dropped/uncertain): {amb_total} "
      f"({100*amb_total/field_total:.1f}%)")
print(f"Fonts used: {[os.path.basename(x) for x in FONTS]}")
