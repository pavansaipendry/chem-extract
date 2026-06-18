"""Deterministic pieces of the structure module — no API key, no DECIMER model.

The Claude recognizer (Leg A) and DECIMER (Leg C) are exercised manually with
credentials/model present. Here we pin down what must be correct without either:
the name-resolution answer key (Leg B) and the recognition-first convergence that
turns three independent reads into one trusted-or-flagged verdict.

Cases mirror the real Page 57 structures: LiTFSI (skeletal, confirmable), the
12-crown-4 cartoon (resolvable from its label but not independently confirmable),
the [Li(12-crown-4)]+ + e- reaction scheme, and a name/drawing mismatch.
"""
from chemextract.structures import (
    LOCAL_NAMES,
    NameResolution,
    RecognizedStructure,
    compare,
    converge,
    name_to_smiles,
    render_svg,
    to_inchikey,
)

CROWN12 = "C1COCCOCCOCCO1"
CROWN18 = "C1COCCOCCOCCOCCOCCO1"
LITFSI = "[Li+].[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F"
TFSI_ANION = "FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F"


def rec(name=None, smiles=None, label=None, is_reaction=False, confidence=0.9):
    return RecognizedStructure(
        label=label, name=name, smiles=smiles,
        is_reaction=is_reaction, confidence=confidence, reasoning="test",
    )


# --- Leg B: the answer key ---------------------------------------------------

def test_local_names_resolve_offline():
    r = name_to_smiles("12-crown-4")
    assert r.source == "local"
    assert r.inchikey == "XQQZRZQVBFHBHL-UHFFFAOYSA-N"


def test_local_table_is_valid_chemistry():
    # Every curated SMILES must parse, or the answer key itself lies.
    for name, smi in LOCAL_NAMES.items():
        assert to_inchikey(smi) is not None, name


def test_compare_levels():
    assert compare(CROWN12, CROWN12) == "exact"
    # bare anion vs the Li salt: same core once salt-stripped + neutralised
    assert compare(TFSI_ANION, LITFSI) == "skeleton"
    assert compare(CROWN12, TFSI_ANION) == "mismatch"
    assert compare(None, CROWN12) == "indeterminate"


# --- Recognition-first convergence ------------------------------------------

def test_trusts_when_name_and_drawing_agree():
    # LiTFSI: recognizer names it AND its own SMILES matches the answer key.
    v = converge(rec(name="LiTFSI", smiles=LITFSI, label="LiTFSI"))
    assert v.structure_smiles == LITFSI
    assert v.confirmations >= 1
    assert v.flags == []
    assert v.needs_review is False


def test_cartoon_resolves_from_label_but_flags_for_review():
    # 12-crown-4 drawn as a dashed cartoon: recognizer reads the LABEL but can't
    # give a SMILES from the drawing. We still resolve it from the name, but with
    # nothing confirming the drawing it must be flagged, never silently trusted.
    v = converge(rec(name="12-crown-4", smiles=None, label="12-crown-4", confidence=0.8))
    assert v.structure_smiles == LOCAL_NAMES["12-crown-4"]   # answer key carried it
    assert v.confirmations == 0
    assert v.needs_review is True
    assert "review" in v.note.lower()


def test_flags_when_drawing_contradicts_label():
    # Labelled 12-crown-4 but the drawing reads as 18-crown-6 → disagreement, flag.
    v = converge(rec(name="12-crown-4", smiles=CROWN18, label="12-crown-4"))
    assert "vlm_drawing_mismatch" in v.flags
    assert v.needs_review is True


def test_reaction_scheme_is_flagged_not_forced():
    # [Li(12-crown-4)]+ + e- -> Li + 12-crown-4 : a scheme, hand to L4/human.
    v = converge(rec(name="lithium 12-crown-4 complex", is_reaction=True, confidence=0.7))
    assert "is_reaction" in v.flags
    assert v.needs_review is True


def test_unidentified_drawing_is_flagged_with_no_structure():
    v = converge(rec(name=None, smiles=None, confidence=0.3))
    assert v.structure_smiles is None
    assert "unidentified" in v.flags
    assert v.needs_review is True


def test_unresolvable_name_falls_back_to_vlm_smiles_and_flags(monkeypatch):
    # Recognizer names something the answer key can't resolve: fall back to the
    # VLM's own SMILES but flag it, because nothing independent confirmed it.
    monkeypatch.setattr(
        "chemextract.structures.name_to_smiles",
        lambda n: NameResolution(name=n, source="none"),
    )
    v = converge(rec(name="madeupol", smiles=CROWN12))
    assert v.structure_smiles == CROWN12
    assert "name_unresolved" in v.flags
    assert v.needs_review is True


# --- coordination / host-guest complex resolution ---------------------------

def test_lithium_crown_complex_resolves_from_descriptive_name():
    # The real Page 57 case: recognizer named it 'lithium 12-crown-4 complex',
    # which isn't a systematic name. The answer key should still compose it.
    r = name_to_smiles("lithium 12-crown-4 complex")
    assert r.source == "composed"
    assert r.inchikey == to_inchikey("[Li+].C1COCCOCCOCCO1")


def test_bracket_label_resolves_complex():
    r = name_to_smiles("[Li(12-crown-4)]+")
    assert r.smiles is not None
    assert r.inchikey == to_inchikey("[Li+].C1COCCOCCOCCO1")


def test_complex_now_converges_trusted_not_flagged():
    # With the complex resolvable, the recognizer's own SMILES confirms it →
    # trusted, no review (previously it was flagged name_unresolved).
    v = converge(rec(
        name="lithium 12-crown-4 complex",
        smiles="[Li+].C1COCCOCCOCCO1",
        label="[Li(12-crown-4)]+",
    ))
    assert v.confirmations >= 1
    assert "name_unresolved" not in v.flags
    assert v.needs_review is False
    assert "trusted" in v.note.lower()


def test_name_unresolved_note_is_not_misleading():
    # When the name truly can't resolve, the note must say so — NOT claim a
    # "discrepancy between legs" (the old wording bug).
    v = converge(rec(name="totally-made-up-ligand", smiles=CROWN12))
    assert "name_unresolved" in v.flags
    assert "discrepancy" not in v.note.lower()
    assert "review" in v.note.lower()


def test_plain_crown_still_resolves_locally_not_as_complex():
    # No metal in the name → must NOT be treated as a complex.
    r = name_to_smiles("12-crown-4")
    assert r.source == "local"


# --- rendering the structure to a drawing -----------------------------------

def test_render_svg_produces_a_drawing():
    svg = render_svg(CROWN12)
    assert svg is not None
    assert svg.lstrip().startswith("<svg")   # no <?xml?> prolog, embeds in HTML
    assert "</svg>" in svg


def test_render_svg_handles_nothing_and_garbage():
    assert render_svg(None) is None
    assert render_svg("") is None
    assert render_svg("not a smiles ###") is None


def test_converge_attaches_the_drawing():
    # a trusted structure carries its rendered SVG so the UI can show the diagram
    v = converge(rec(name="12-crown-4", smiles=CROWN12, label="12-crown-4"))
    assert v.svg and "<svg" in v.svg
    # a reaction (no single molecule) has no drawing
    r = converge(rec(name=None, smiles=None, is_reaction=True))
    assert r.svg is None


# --- DECIMER cross-check leg (passed in, no model needed for the logic) ------

def test_ocsr_confirmation_raises_trust():
    from chemextract.structures import OCSRResult
    v = converge(rec(name="LiTFSI", smiles=LITFSI), ocsr=OCSRResult(smiles=LITFSI))
    assert v.confirmations >= 2          # VLM smiles + OCSR both agree
    assert v.needs_review is False


def test_ocsr_abstain_is_flagged_but_not_fatal():
    from chemextract.structures import OCSRResult
    v = converge(rec(name="LiTFSI", smiles=LITFSI), ocsr=OCSRResult(error="garbage on ruled paper"))
    assert "ocsr_abstained" in v.flags   # ruled lines broke the cross-check...
    assert v.structure_smiles == LITFSI  # ...but the answer is unharmed


def test_ocsr_disagreement_is_flagged():
    from chemextract.structures import OCSRResult
    v = converge(rec(name="12-crown-4", smiles=CROWN12), ocsr=OCSRResult(smiles=CROWN18))
    assert "ocsr_disagree" in v.flags
    assert v.needs_review is True
