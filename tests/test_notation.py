"""Faithful scientific-notation / special-symbol handling (Level 2), deterministic.

The forms here are the ones Page 57 actually uses and that transcription mangles:
E-notation (1.5E-4), ×10ⁿ typography, cm⁻¹ / cm² super-/subscripts, °C, 2θ, λ.
"""
from chemextract.notation import (
    ascii_units,
    normalize_sci,
    notation_features,
    notation_flags,
    parse_number,
)


# --- parse every typographic form to the same number -------------------------

def test_parse_e_notation():
    assert parse_number("1.5E-4") == 1.5e-4
    assert parse_number("8.4e-6") == 8.4e-6
    assert parse_number("5.8E-5") == 5.8e-5


def test_parse_times_ten_typography():
    assert parse_number("1.5×10⁻⁴") == 1.5e-4
    assert parse_number("1.5 x 10^-4") == 1.5e-4
    assert parse_number("8.4·10⁻⁶") == 8.4e-6


def test_parse_unicode_minus_and_commas():
    assert parse_number("−5.8e-5") == -5.8e-5     # U+2212 minus
    assert parse_number("1,000") == 1000.0


def test_parse_number_inside_text_and_garbage():
    assert parse_number("2.1°") == 2.1
    assert parse_number("pH") is None


# --- normalise unicode so unit tables can match ------------------------------

def test_ascii_units_super_and_subscripts():
    assert ascii_units("cm⁻¹") == "cm-1"
    assert ascii_units("cm²") == "cm2"
    assert ascii_units("H₂O") == "H2O"
    assert ascii_units("mol·dm⁻³") == "mol·dm-3"


def test_normalize_sci_canonical_form():
    assert normalize_sci("1.5×10⁻⁴") == "1.5e-4"
    assert normalize_sci("8.4 · 10^-6") == "8.4e-6"
    assert normalize_sci("0.81") == "0.81"        # not sci notation → unchanged


# --- feature detection (fidelity telemetry) ----------------------------------

def test_notation_features():
    assert "greek" in notation_features("λmax = 540 nm")
    assert "superscript" in notation_features("2θ = 2.1° ; ν̄ in cm⁻¹")
    assert "subscript" in notation_features("H₂O < 1 ppm")
    assert "degree" in notation_features("heated to 85 °C")
    assert "sci_notation" in notation_features("1.5×10⁻⁴ A")


# --- the dangerous case: a dropped exponent ----------------------------------

def test_flags_dropped_exponent():
    # quote shows the power of ten, captured value lost it → 9-order error.
    assert notation_flags("1.5", "J = 1.5×10⁻⁴ A") == ["exponent_dropped"]


def test_no_flag_when_exponent_preserved():
    assert notation_flags("1.5e-4", "J = 1.5×10⁻⁴ A") == []


def test_no_flag_when_quote_has_no_exponent():
    assert notation_flags("0.81", "Q = 0.81 C") == []
    assert notation_flags("85", "heated to 85 °C") == []


def test_no_false_positive_on_calculation_line():
    # Regression: a calc-line quote holds several numbers; '5400' must NOT borrow
    # the exponent from a different number (1.5E-4) earlier in the same quote.
    assert notation_flags("5400", "Q = 1.5E-4 A · 5400 s") == []
    assert notation_flags("96485", "0.81 C / 96485 C/mol") == []
    # but the value that really did drop its exponent is still caught
    assert notation_flags("1.5", "Q = 1.5E-4 A · 5400 s") == ["exponent_dropped"]


def test_spaced_e_notation_counts_as_having_exponent():
    assert notation_flags("8.4 E-6", "8.4 E-6 mol") == []
