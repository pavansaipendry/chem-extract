"""OCSR ↔ name cross-check for hand-drawn chemical structures.

A general vision LLM can't be trusted to turn a hand-drawn molecule into a
correct SMILES. So we read the same structure two *independent* ways and
compare:

    drawing  --MolScribe (OCSR)-->  SMILES_A      (reads the picture)
    label    --OPSIN/PubChem----->  SMILES_B      (reads the caption)

If the two agree on heavy-atom connectivity (InChIKey), we trust the structure.
If they disagree, we flag it for a human instead of asserting a molecule nobody
verified. Same self-consistency idea the rest of the pipeline uses, applied
across modalities.

The name leg is deterministic and cheap; it's the trustworthy "answer key".
The OCSR leg is the unreliable one we're checking. MolScribe loads lazily — you
only pay the model download/load if you actually call the OCSR path.

Status / known limitation: the name leg (OPSIN/PubChem/local) is solid. The
OCSR leg works on clean skeletal structures but is unreliable on real
hand-drawn lab-notebook drawings — ruled paper and shorthand "cartoon"
notation (e.g. a crown ether drawn as a dashed macrocycle with O's and a metal
ion floating inside) make MolScribe return garbage. The cross-check correctly
*flags* that disagreement rather than asserting it, but turning the OCSR leg
into something reliable on this input style is still open work (line removal,
region pre-classification, and/or a vision-LLM reader as an alternative leg).
"""
from __future__ import annotations

import sys
import types
import urllib.parse
import urllib.request
import json
import warnings
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# --- Curated fallbacks for trivial names/abbreviations OPSIN doesn't parse ---
# OPSIN only handles systematic IUPAC names; lab notebooks are full of jargon
# ("LiTFSI", "12-crown-4"). PubChem covers most of it, but a tiny local table
# keeps the common electrochemistry reagents resolvable fully offline.
LOCAL_NAMES: dict[str, str] = {
    "litfsi": "[Li+].[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "tfsi": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "12-crown-4": "C1COCCOCCOCCO1",
    "15-crown-5": "C1COCCOCCOCCOCCO1",
    "18-crown-6": "C1COCCOCCOCCOCCOCCO1",
    "diglyme": "COCCOCCOC",
    "monoglyme": "COCCOC",
    "dme": "COCCOC",
}


# --- Result models -----------------------------------------------------------

class NameResolution(BaseModel):
    """The trustworthy leg: a label resolved to a structure deterministically."""
    name: str
    smiles: Optional[str] = None
    source: str = Field(description="local | opsin | pubchem | none")
    inchikey: Optional[str] = None


class OCSRResult(BaseModel):
    """The leg under test: a drawing read by the OCSR model."""
    smiles: Optional[str] = None
    confidence: Optional[float] = None
    inchikey: Optional[str] = None
    error: Optional[str] = None


class CrossCheck(BaseModel):
    """Verdict from comparing the two legs."""
    label: str
    crop: str
    ocsr: OCSRResult
    name: NameResolution
    match: str = Field(description="exact | skeleton | mismatch | indeterminate")
    agree: bool = Field(description="True if the drawing and the label describe the same molecule")
    note: str = ""


# --- RDKit helpers -----------------------------------------------------------

def _rdkit():
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    return Chem


def to_inchikey(smiles: str | None) -> Optional[str]:
    """Canonical InChIKey, or None if RDKit can't parse the SMILES."""
    if not smiles:
        return None
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


def _core_key(smiles: str | None) -> Optional[str]:
    """Connectivity skeleton with the counter-ion stripped and charges
    neutralised. A drawn salt and a named anion (or a Li salt vs the bare
    anion) collapse to the same key here, so we can recognise "same molecule,
    just protonated/paired differently" rather than calling it a mismatch."""
    if not smiles:
        return None
    Chem = _rdkit()
    from rdkit.Chem.MolStandardize import rdMolStandardize
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.FragmentParent(mol)        # drop salts/counter-ions
        mol = rdMolStandardize.Uncharger().uncharge(mol)  # neutralise
        return Chem.MolToInchiKey(mol).split("-")[0]
    except Exception:
        return None


def compare(ocsr_smiles: str | None, name_smiles: str | None) -> str:
    """exact (full InChIKey) > skeleton (same core ignoring salt/charge) > mismatch."""
    ocsr_key, name_key = to_inchikey(ocsr_smiles), to_inchikey(name_smiles)
    if not ocsr_key or not name_key:
        return "indeterminate"
    if ocsr_key == name_key:
        return "exact"
    core = _core_key(ocsr_smiles)
    if core and core == _core_key(name_smiles):
        return "skeleton"
    return "mismatch"


# --- Name leg: label -> SMILES (local -> OPSIN -> PubChem) --------------------

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
    """Resolve a chemical name/abbreviation to SMILES, trying the cheapest,
    most reliable source first. This is the answer key the OCSR is checked
    against, so order matters: curated > OPSIN (systematic) > PubChem (broad)."""
    key = name.strip().lower()
    if key in LOCAL_NAMES:
        smi = LOCAL_NAMES[key]
        return NameResolution(name=name, smiles=smi, source="local", inchikey=to_inchikey(smi))

    try:
        from py2opsin import py2opsin
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            smi = py2opsin(name)
        if smi:
            return NameResolution(name=name, smiles=smi, source="opsin", inchikey=to_inchikey(smi))
    except Exception:
        pass

    smi = _pubchem_smiles(name)
    if smi:
        return NameResolution(name=name, smiles=smi, source="pubchem", inchikey=to_inchikey(smi))

    return NameResolution(name=name, source="none")


# --- OCSR leg: drawing -> SMILES (MolScribe, loaded lazily) -------------------

_MOLSCRIBE = None  # cached singleton; the checkpoint is ~1 GB to load


class _SerialPool:
    """Drop-in for multiprocessing.Pool that runs in-process. MolScribe
    parallelises its graph→SMILES post-step with a Pool; on macOS that uses
    'spawn', and the fresh workers re-import onmt→torchtext *without* our
    runtime shim (it lives only in this process's sys.modules). Running serial
    keeps it in-process where the shim is. Post-processing a couple of crops is
    cheap, so we lose nothing."""
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def map(self, fn, it, chunksize=None): return [fn(x) for x in it]
    def starmap(self, fn, it, chunksize=None): return [fn(*x) for x in it]
    def imap(self, fn, it, chunksize=None): return [fn(x) for x in it]
    def close(self): pass
    def join(self): pass
    def terminate(self): pass


class _SerialMP:
    Pool = _SerialPool


def _install_torchtext_shim() -> None:
    """OpenNMT-py 2.2.0's package init imports the legacy ``torchtext.data``
    API (Field/Vocab) that no longer exists for torch 2.x. MolScribe only uses
    OpenNMT's pure-PyTorch decoder modules, never the inputters, so we stub the
    names that ``import onmt`` touches at load time. No real torchtext needed."""
    if "torchtext" in sys.modules:
        return
    tt = types.ModuleType("torchtext")
    data = types.ModuleType("torchtext.data")
    vocab = types.ModuleType("torchtext.vocab")
    # Any non-dunder attribute (Field, Vocab, Iterator, batch, …) returns a
    # fresh dummy so every `from torchtext.x import Y` onmt does at load time
    # resolves. Dunders must still raise so normal module machinery (__file__,
    # __path__, import system) keeps working. onmt's inputters are never
    # executed, so the stubs are never actually called.
    def _stub_getattr(name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return type(name, (), {})
    data.__getattr__ = _stub_getattr
    vocab.__getattr__ = _stub_getattr
    tt.data = data
    tt.vocab = vocab
    sys.modules["torchtext"] = tt
    sys.modules["torchtext.data"] = data
    sys.modules["torchtext.vocab"] = vocab


def _get_molscribe():
    """Load MolScribe once, fetching the checkpoint from the Hub on first use."""
    global _MOLSCRIBE
    if _MOLSCRIBE is not None:
        return _MOLSCRIBE

    _install_torchtext_shim()
    import torch
    from huggingface_hub import hf_hub_download
    from molscribe import MolScribe
    import molscribe.chemistry as _chem
    _chem.multiprocessing = _SerialMP  # keep post-processing in-process (see _SerialPool)

    # torch>=2.6 defaults torch.load(weights_only=True); the checkpoint stores
    # an argparse.Namespace under 'args', so force the full unpickler.
    _orig_load = torch.load
    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)
    torch.load = _load
    try:
        ckpt = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")
        _MOLSCRIBE = MolScribe(ckpt, device=torch.device("cpu"))
    finally:
        torch.load = _orig_load
    return _MOLSCRIBE


def ocsr(image_path: str | Path) -> OCSRResult:
    """Read a single cropped structure image into a SMILES with MolScribe."""
    try:
        model = _get_molscribe()
        out = model.predict_image_file(str(image_path), return_confidence=True)
        smi = out.get("smiles")
        return OCSRResult(
            smiles=smi,
            confidence=out.get("confidence"),
            inchikey=to_inchikey(smi),
        )
    except Exception as e:
        return OCSRResult(error=f"{type(e).__name__}: {e}")


# --- The cross-check ---------------------------------------------------------

def cross_check(crop_path: str | Path, label: str) -> CrossCheck:
    """Read a hand-drawn structure two independent ways and compare.

    `crop_path` is a tight crop of one drawn structure; `label` is the caption
    a human (or the text-extraction pass) read next to it, e.g. "12-crown-4".
    """
    name = name_to_smiles(label)
    structure = ocsr(crop_path)
    verdict = compare(structure.smiles, name.smiles)
    agree = verdict in ("exact", "skeleton")

    if structure.error:
        note = f"OCSR failed — {structure.error}"
    elif name.source == "none":
        note = f"Could not resolve label '{label}' to a structure; nothing to check against."
    elif verdict == "exact":
        note = "Drawing and label resolve to the same molecule."
    elif verdict == "skeleton":
        note = ("Same heavy-atom connectivity; differ only in charge/stereo "
                "(expected for a salt drawn with its counter-ion).")
    else:
        note = "Drawing and label DISAGREE — route to a human, do not assert."

    return CrossCheck(
        label=label, crop=str(crop_path),
        ocsr=structure, name=name, match=verdict, agree=agree, note=note,
    )


# --- CLI ---------------------------------------------------------------------

def _main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="MolScribe ↔ OPSIN/PubChem structure cross-check")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("check", help="cross-check one cropped structure against a label")
    pc.add_argument("crop", help="path to a cropped structure image")
    pc.add_argument("--label", required=True, help="the caption next to it, e.g. '12-crown-4'")

    pn = sub.add_parser("name", help="resolve a name to SMILES (answer-key leg only)")
    pn.add_argument("name")

    args = p.parse_args()
    if args.cmd == "name":
        print(name_to_smiles(args.name).model_dump_json(indent=2))
    else:
        print(cross_check(args.crop, args.label).model_dump_json(indent=2))


if __name__ == "__main__":
    _main()
