"""Cross-check logic that needs neither the MolScribe model nor the network.

The OCSR leg (MolScribe) is exercised manually with the model loaded; here we
pin down the deterministic pieces: name resolution and the InChIKey compare.
"""
from chemextract.structures import (
    LOCAL_NAMES,
    compare,
    name_to_smiles,
    to_inchikey,
)


def test_local_names_resolve_offline():
    r = name_to_smiles("12-crown-4")
    assert r.source == "local"
    assert r.inchikey == "XQQZRZQVBFHBHL-UHFFFAOYSA-N"


def test_abbreviation_and_iupac_agree():
    # The whole point: two independent name routes land on the same molecule.
    abbrev = name_to_smiles("LiTFSI")                 # local table
    iupac = name_to_smiles("lithium bis(trifluoromethanesulfonyl)imide")  # OPSIN
    assert abbrev.inchikey == iupac.inchikey


def test_local_table_is_valid_chemistry():
    # Every curated SMILES must be parseable, or the answer key lies.
    for name, smi in LOCAL_NAMES.items():
        assert to_inchikey(smi) is not None, name


def test_compare_levels():
    crown = "C1COCCOCCOCCO1"
    assert compare(crown, crown) == "exact"
    # bare anion vs the Li salt: same core once salt-stripped + neutralised
    anion = "FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F"
    li_salt = "[Li+].[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F"
    assert compare(anion, li_salt) == "skeleton"
    # different molecules
    assert compare(crown, anion) == "mismatch"
    # nothing to compare against
    assert compare(None, crown) == "indeterminate"
