"""Hand-drawn structure recognition — recognition-first, OCSR demoted to a cross-check.

Why this shape (learned the hard way): pixel->SMILES OCSR is the *fragile* leg.
On real lab-notebook pages it fails for two independent reasons — ruled paper
corrupts the image, and shorthand "cartoon" depictions (a crown ether drawn as a
dashed macrocycle with a metal ion floating inside) are simply not in any OCSR
model's training distribution. Cleaning pixels fixes the first, never the second.

So the answer must NOT flow through OCSR. Every drawn structure on a notebook
page is, in practice, a *named, known reagent labelled right next to the drawing*
("LiTFSI", "12-crown-4", "diglyme"). Resolving a name to a correct structure is
basically solved and fully deterministic. That is the reliable leg.

Three legs, read independently, converged on InChIKey:

    Leg A  drawing --Claude VLM recognizer--> (name, SMILES)   [PRIMARY: names it]
    Leg B  name    --local/OPSIN/PubChem-----> SMILES          [ANSWER KEY: trusted]
    Leg C  crop    --DECIMER hand-drawn------> SMILES          [CROSS-CHECK: may abstain]

The trusted structure is Leg B's resolution of the recognized name. Leg A's own
SMILES and Leg C are *confirmations*: when they agree (same InChIKey / skeleton)
trust rises; when they disagree or can't be obtained, we FLAG for a human instead
of asserting a molecule nobody verified. Ruled lines and cartoon notation only
ever touch the cross-check leg, never the answer.

CLI:
  python -m chemextract.structures name "12-crown-4"        # answer-key leg only
  python -m chemextract.structures recognize page.png       # full page, all legs
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from pydantic import BaseModel, Field

# NOTE: MODEL and _image_block are imported lazily inside recognize_structures()
# (not at module top) so this module doesn't pull in extract.py — that keeps
# schema.py able to import these result models without an import cycle.

# --- Curated fallbacks for trivial names/abbreviations OPSIN can't parse ------
# OPSIN only handles systematic IUPAC names; notebooks are full of jargon. A tiny
# local table keeps the common electrochemistry reagents resolvable fully offline
# (no network, no Java) — the rest fall through to OPSIN then PubChem.
LOCAL_NAMES: dict[str, str] = {
    "litfsi": "[Li+].[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "tfsi": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "12-crown-4": "C1COCCOCCOCCO1",
    "15-crown-5": "C1COCCOCCOCCOCCO1",
    "18-crown-6": "C1COCCOCCOCCOCCOCCO1",
    "diglyme": "COCCOCCOC",
    "monoglyme": "COCCOC",
    "glyme": "COCCOC",
    "dme": "COCCOC",
    "tetraglyme": "COCCOCCOCCOCCOC",
}


# --- Result models -----------------------------------------------------------

class RecognizedStructure(BaseModel):
    """Leg A: one drawn structure as read by the Claude vision recognizer."""
    label: Optional[str] = Field(
        description="the name/caption written next to the drawing, verbatim, or null"
    )
    name: Optional[str] = Field(
        description="the molecule you identify this drawing as (common or IUPAC), or null if you cannot"
    )
    smiles: Optional[str] = Field(
        description="your best SMILES for the drawing, or null. Your own independent read — it is cross-checked, not trusted blindly."
    )
    is_reaction: bool = Field(
        description="true if this depicts a reaction/scheme (arrows, + e-, ->) rather than a single molecule"
    )
    confidence: float = Field(description="0-1, how sure you are of the identification")
    reasoning: str = Field(description="what the drawing shows and why you identified it so")


class RecognizerOutput(BaseModel):
    structures: list[RecognizedStructure] = Field(
        description="every distinct chemical-structure DRAWING on the page (not text mentions)"
    )


class NameResolution(BaseModel):
    """Leg B: a name resolved to a structure deterministically — the answer key."""
    name: str
    smiles: Optional[str] = None
    source: str = Field(description="local | opsin | pubchem | none")
    inchikey: Optional[str] = None


class OCSRResult(BaseModel):
    """Leg C: a drawing read by the OCSR model — the leg under test (may abstain)."""
    smiles: Optional[str] = None
    confidence: Optional[float] = None
    inchikey: Optional[str] = None
    error: Optional[str] = None


class StructureVerdict(BaseModel):
    """Converged result for one drawn structure."""
    label: Optional[str]
    name: Optional[str]
    structure_smiles: Optional[str] = Field(description="the trusted structure (name-leg resolution, or VLM fallback)")
    inchikey: Optional[str] = None
    svg: Optional[str] = Field(default=None, description="2D structure drawing (SVG) rendered from structure_smiles")
    recognized: RecognizedStructure
    name_resolution: NameResolution
    ocsr: Optional[OCSRResult] = None
    confirmations: int = Field(description="how many independent legs agree with the trusted structure")
    flags: list[str] = []
    needs_review: bool = False
    note: str = ""


# --- RDKit helpers -----------------------------------------------------------

def _rdkit():
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    return Chem


def to_inchikey(smiles: str | None) -> Optional[str]:
    """Canonical InChIKey, or None if RDKit can't parse the SMILES — or isn't
    installed. Without RDKit the cross-check degrades to 'indeterminate' (so
    structures are recognised by name but flagged, not silently trusted), rather
    than crashing the pipeline for users who only installed the core deps."""
    if not smiles:
        return None
    try:
        Chem = _rdkit()
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


def _core_key(smiles: str | None) -> Optional[str]:
    """Connectivity skeleton with counter-ion stripped and charges neutralised.
    A drawn salt and a named anion (or a Li salt vs the bare anion) collapse to
    the same key, so 'same molecule, paired differently' isn't called a mismatch."""
    if not smiles:
        return None
    try:
        Chem = _rdkit()
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.FragmentParent(mol)        # drop salts/counter-ions
        mol = rdMolStandardize.Uncharger().uncharge(mol)  # neutralise
        return Chem.MolToInchiKey(mol).split("-")[0]
    except Exception:
        return None


def compare(smiles_a: str | None, smiles_b: str | None) -> str:
    """exact (full InChIKey) > skeleton (same core ignoring salt/charge) > mismatch."""
    key_a, key_b = to_inchikey(smiles_a), to_inchikey(smiles_b)
    if not key_a or not key_b:
        return "indeterminate"
    if key_a == key_b:
        return "exact"
    core = _core_key(smiles_a)
    if core and core == _core_key(smiles_b):
        return "skeleton"
    return "mismatch"


def render_svg(smiles: str | None, width: int = 380, height: int = 260) -> Optional[str]:
    """Render a SMILES to a 2D structure drawing (SVG) — the diagram itself, not
    just the string. The SMILES already came out of Leg B / the recognizer, so
    this is a clean, canonical depiction of the resolved molecule. Returns None
    if there's no SMILES, RDKit can't parse it, or RDKit isn't installed, so the
    UI simply shows no drawing rather than erroring."""
    if not smiles:
        return None
    try:
        Chem = _rdkit()
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    d = rdMolDraw2D.MolDraw2DSVG(width, height)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    return svg[svg.find("<svg"):]  # drop the <?xml?> prolog so it embeds in HTML


def render_png(smiles: str | None, width: int = 380, height: int = 260) -> Optional[bytes]:
    """Render a SMILES to a PNG (raster) of the 2D structure, or None. The layout
    pass needs the canonical drawing as an *image* it can compare to the page, and
    the vision API takes raster bytes, not SVG. Best-effort: needs RDKit's Cairo
    backend; if it (or RDKit) is missing we return None and the caller falls back
    to the page image plus the text manifest."""
    if not smiles:
        return None
    try:
        Chem = _rdkit()
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        d = rdMolDraw2D.MolDraw2DCairo(width, height)
        d.DrawMolecule(mol)
        d.FinishDrawing()
        return d.GetDrawingText()  # PNG bytes
    except Exception:  # Cairo backend not built into this RDKit wheel
        return None


# --- Leg B: name -> SMILES (local -> complex -> OPSIN -> PubChem) -------------

# Metal cations we know how to compose into a coordination/host-guest complex.
_METAL_CATIONS: dict[str, str] = {
    "lithium": "[Li+]", "li": "[Li+]",
    "sodium": "[Na+]", "na": "[Na+]",
    "potassium": "[K+]", "k": "[K+]",
    "magnesium": "[Mg+2]", "mg": "[Mg+2]",
    "calcium": "[Ca+2]", "ca": "[Ca+2]",
}


def _resolve_complex(name: str) -> Optional[str]:
    """Compose a coordination / host-guest complex SMILES when the name pairs a
    known metal cation with a known ligand — e.g. 'lithium 12-crown-4 complex' or
    '[Li(12-crown-4)]+' -> '[Li+].C1COCCOCCOCCO1'. Returns None if it isn't such
    a pairing. This lets the answer key resolve the crown-ether complex on the
    page instead of leaving it merely flagged."""
    low = name.strip().lower()
    ligand = next(
        (smi for key, smi in LOCAL_NAMES.items()
         if key not in _METAL_CATIONS and key in low),
        None,
    )
    if not ligand:
        return None
    # longest metal token first, word-bounded so 'li' doesn't match inside a word
    metal = next(
        (_METAL_CATIONS[m] for m in sorted(_METAL_CATIONS, key=len, reverse=True)
         if re.search(rf"\b{re.escape(m)}\b", low)),
        None,
    )
    return f"{metal}.{ligand}" if metal else None


def _pubchem_smiles(name: str) -> Optional[str]:
    enc = urllib.parse.quote(name)
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{enc}"
        "/property/SMILES,InChIKey/JSON"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.load(r)
        return data["PropertyTable"]["Properties"][0].get("SMILES")
    except Exception:
        return None


def name_to_smiles(name: str) -> NameResolution:
    """Resolve a chemical name/abbreviation to SMILES, cheapest+most reliable
    source first: curated table > OPSIN (systematic) > PubChem (broad). This is
    the answer key, so order is deliberate."""
    key = name.strip().lower()
    if key in LOCAL_NAMES:
        smi = LOCAL_NAMES[key]
        return NameResolution(name=name, smiles=smi, source="local", inchikey=to_inchikey(smi))

    composed = _resolve_complex(name)
    if composed:
        return NameResolution(name=name, smiles=composed, source="composed", inchikey=to_inchikey(composed))

    try:
        from py2opsin import py2opsin
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            smi = py2opsin(name)
        if smi:
            smi = smi.split("\n")[0].strip()  # OPSIN can emit one line per name
            return NameResolution(name=name, smiles=smi, source="opsin", inchikey=to_inchikey(smi))
    except Exception:
        pass

    smi = _pubchem_smiles(name)
    if smi:
        return NameResolution(name=name, smiles=smi, source="pubchem", inchikey=to_inchikey(smi))

    return NameResolution(name=name, source="none")


# --- Leg A: Claude VLM recognizer (the primary leg) --------------------------

RECOGNIZER_SYSTEM = """You are the structure-recognition component of a chemistry \
document extraction system. You are shown a page (often a handwritten lab notebook) \
that may contain drawn chemical structures.

Find every distinct chemical-structure DRAWING on the page and identify it. A drawing \
is a depicted molecule or reaction — skeletal formulas, structural formulas, and \
hand-drawn shorthand "cartoons" (e.g. a crown ether drawn as a dashed ring with a metal \
ion in the middle, or a bracketed complex like [Li(12-crown-4)]+). Do NOT report plain \
text mentions of a chemical as structures — only actual drawings.

Rules — these matter:
1. The name or caption written next to a drawing is strong evidence. Read it VERBATIM \
into `label`. Most notebook structures are common, named reagents (LiTFSI, 12-crown-4, \
diglyme, TFSI-) drawn as shorthand — recognise them by name even when the drawing is a \
cartoon, not a proper skeletal structure.
2. Set `name` to the molecule you identify (its SMILES will be re-derived from the name \
deterministically and cross-checked against your `smiles`). If you genuinely cannot \
identify it, set name null and lower confidence — never invent a molecule.
3. Give your own best `smiles` for the drawing independently. It is a cross-check, not \
the final answer; an honest null beats a confident wrong SMILES.
4. If the drawing is a REACTION or scheme (arrows, "+ e-", "->"), set is_reaction=true \
and identify the species you can; do not force it into one molecule.
5. `confidence` is how sure you are of the identification (0-1)."""


def recognize_structures(
    image_path: str | Path, transcript: str | None = None, client: Anthropic | None = None
) -> list[RecognizedStructure]:
    """Leg A: Claude locates and identifies every drawn structure on the page.

    Takes the whole page — no segmentation model on the critical path, so ruled
    lines never get a chance to corrupt the read."""
    from .extract import MODEL, _image_block   # lazy: avoids a schema<->extract cycle
    client = client or Anthropic()
    user_text = "Identify every drawn chemical structure on this page."
    if transcript:
        user_text += f"\n\nVerbatim transcript of the page (context for the labels):\n{transcript}"
    response = client.messages.parse(
        model=MODEL,
        max_tokens=8000,
        system=RECOGNIZER_SYSTEM,
        messages=[{
            "role": "user",
            "content": [_image_block(image_path), {"type": "text", "text": user_text}],
        }],
        output_format=RecognizerOutput,
    )
    return response.parsed_output.structures


# --- Leg C: DECIMER hand-drawn OCSR (cross-check only, loaded lazily) ---------

_DECIMER = None


def _get_decimer():
    """Import DECIMER on first use only — it drags in TensorFlow and a model
    checkpoint, so importing this module stays cheap if OCSR is never run."""
    global _DECIMER
    if _DECIMER is None:
        from DECIMER import predict_SMILES  # package name: decimer
        _DECIMER = predict_SMILES
    return _DECIMER


def ocsr_decimer(crop_path: str | Path) -> OCSRResult:
    """Read a single cropped structure into SMILES with DECIMER's hand-drawn model.
    This is the leg under test — it abstains (returns error) rather than throwing,
    so a failure here flags the structure instead of sinking the pipeline."""
    try:
        smi = _get_decimer()(str(crop_path))
        return OCSRResult(smiles=smi or None, inchikey=to_inchikey(smi))
    except Exception as e:
        return OCSRResult(error=f"{type(e).__name__}: {e}")


# --- Convergence: combine the legs into one verdict --------------------------

def converge(
    rec: RecognizedStructure, ocsr: OCSRResult | None = None
) -> StructureVerdict:
    """Recognition-first convergence. The trusted structure is the name-leg
    resolution of the recognised name; the VLM's own SMILES and DECIMER are
    independent confirmations that raise trust or, on disagreement/absence, flag."""
    name_res = name_to_smiles(rec.name) if rec.name else NameResolution(name="", source="none")
    # if the identified name didn't resolve, try the verbatim label on the page
    # (e.g. name='lithium 12-crown-4 complex' fails but label='[Li(12-crown-4)]+' resolves)
    if name_res.source == "none" and rec.label:
        alt = name_to_smiles(rec.label)
        if alt.smiles:
            name_res = alt

    flags: list[str] = []
    confirmations = 0

    # The answer: name-leg resolution if we have one, else fall back to the VLM's
    # own SMILES (and flag, because nothing independent confirmed it).
    if name_res.smiles:
        trusted = name_res.smiles
    else:
        trusted = rec.smiles
        if rec.name:
            flags.append("name_unresolved")   # named it, but no answer key for that name
        else:
            flags.append("unidentified")       # couldn't even name it

    # Confirmation 1: does the VLM's independent SMILES agree with the answer key?
    if name_res.smiles and rec.smiles:
        verdict = compare(rec.smiles, name_res.smiles)
        if verdict in ("exact", "skeleton"):
            confirmations += 1
        elif verdict == "mismatch":
            flags.append("vlm_drawing_mismatch")  # the name and the drawing disagree

    # Confirmation 2: DECIMER (only if it was run and produced something)
    if ocsr is not None:
        if ocsr.error or not ocsr.smiles:
            flags.append("ocsr_abstained")
        else:
            verdict = compare(ocsr.smiles, trusted)
            if verdict in ("exact", "skeleton"):
                confirmations += 1
            elif verdict == "mismatch":
                flags.append("ocsr_disagree")

    if rec.is_reaction:
        flags.append("is_reaction")            # a scheme, not a single molecule — hand to L4/human
    if rec.confidence < 0.6:
        flags.append("low_confidence")

    needs_review = bool(flags) or confirmations == 0

    disagree = "vlm_drawing_mismatch" in flags or "ocsr_disagree" in flags
    if rec.is_reaction:
        note = "A reaction scheme, not a single molecule — handed to experiment understanding / a human."
    elif not trusted:
        note = "Could not resolve a structure from the drawing or its label — flagged."
    elif disagree:
        note = "Drawing and label/OCSR DISAGREE — routed to a human, not asserted."
    elif "name_unresolved" in flags:
        note = "Identified, but the name didn't resolve to a deterministic answer key; using the drawing's own reading — review."
    elif confirmations >= 1 and not flags:
        note = "Name and drawing agree on the same molecule — trusted."
    elif name_res.smiles and confirmations == 0:
        note = "Resolved from the label, but no independent read confirmed the drawing — review."
    else:
        note = "Flagged for review."

    return StructureVerdict(
        label=rec.label,
        name=rec.name,
        structure_smiles=trusted,
        inchikey=to_inchikey(trusted),
        svg=render_svg(trusted),
        recognized=rec,
        name_resolution=name_res,
        ocsr=ocsr,
        confirmations=confirmations,
        flags=flags,
        needs_review=needs_review,
        note=note,
    )


def analyze_structures(
    image_path: str | Path,
    transcript: str | None = None,
    run_ocsr: bool = False,
    client: Anthropic | None = None,
) -> list[StructureVerdict]:
    """Full page -> per-structure verdicts. OCSR (DECIMER) is off by default; the
    name + VLM legs already carry the answer and need no heavy deps."""
    recognized = recognize_structures(image_path, transcript=transcript, client=client)
    verdicts = []
    for rec in recognized:
        ocsr = ocsr_decimer(image_path) if (run_ocsr and not rec.is_reaction) else None
        verdicts.append(converge(rec, ocsr=ocsr))
    return verdicts


# --- CLI ---------------------------------------------------------------------

def _main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Recognition-first hand-drawn structure analysis")
    sub = p.add_subparsers(dest="cmd", required=True)

    pn = sub.add_parser("name", help="answer-key leg only: resolve a name to SMILES")
    pn.add_argument("name")

    pr = sub.add_parser("recognize", help="full page: recognise + resolve + (optional) OCSR")
    pr.add_argument("image")
    pr.add_argument("--ocsr", action="store_true", help="also run the DECIMER cross-check")

    args = p.parse_args()
    if args.cmd == "name":
        print(name_to_smiles(args.name).model_dump_json(indent=2))
    else:
        out = analyze_structures(args.image, run_ocsr=args.ocsr)
        print(json.dumps([json.loads(v.model_dump_json()) for v in out], indent=2))


if __name__ == "__main__":
    _main()
