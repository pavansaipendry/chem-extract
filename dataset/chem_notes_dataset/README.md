# Synthetic Handwritten Chemistry Notes Dataset

200 handwriting-rendered chemistry lab-note images, each paired with a
ground-truth JSON label. Purpose-built to test/train a document-extraction
system whose core challenge is **ambiguity** -- chemists who omit units
("70" instead of "70 degC"), bare quantities ("5 of NaCl" -- mL? g? mol?),
and messy handwriting.

## Layout
```
images/        chem_0000.png ... chem_0199.png   (handwritten note images)
labels/        chem_0000.json ...                (ground-truth per image)
manifest.json  index of all image/label pairs
README.md      this file
```

## Label schema (per field)
| key            | meaning                                                        |
|----------------|----------------------------------------------------------------|
| type           | temperature / volume / mass / concentration / time / ph / yield|
| raw_value      | the number as written                                          |
| written_unit   | unit actually written, or null if omitted                      |
| true_unit      | the correct/intended unit (ground truth)                       |
| unit_present   | was a unit written? (false = the hard case)                    |
| ambiguous      | true when a unit is missing/uncertain -> must be flagged       |
| written_text   | the full line as rendered in the image                         |

## Stats
- 200 images, ~1080 labeled fields
- ~38% of fields are ambiguous (unit dropped) by design
- 6 handwriting fonts, ruled-paper texture, scan blur + page rotation

## Suggested eval metrics for your pipeline
1. **Extraction accuracy** -- raw_value read correctly.
2. **Flag recall** -- of fields where ambiguous=true, what % did the
   system flag for review instead of silently guessing? (most important)
3. **Unflagged-output accuracy** -- of fields passed through WITHOUT a
   flag, what % are fully correct (value + unit)? Target >= 99%.

## Note
This is SYNTHETIC data (handwriting fonts, not real human writing) for
fast bootstrapping -- the same approach DECIMER used to beat data scarcity.
For production, supplement with 50-100 hand-labeled real lab pages from
your actual target users.
