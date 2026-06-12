"""The worked example from the spec, as literal test cases:

  "Heated the mixture to 70 and held for 30 min.
   Added 5 of NaCl solution. pH adjusted to 7."
"""
from chemextract.schema import QuantityType, RawObservation
from chemextract.validate import (
    parse_value,
    plausible_units,
    to_canonical,
    validate_observation,
)


def obs(qtype, value, unit, quote="", confidence=0.95):
    return RawObservation(
        quantity_type=qtype, value=value, unit=unit, quote=quote, confidence=confidence
    )


# --- Worked example ---

def test_temp_70_no_unit_is_flagged():
    flags = validate_observation(
        obs(QuantityType.TEMPERATURE, "70", None, "Heated the mixture to 70")
    )
    assert flags == ["missing_unit"]


def test_30_min_passes_clean():
    flags = validate_observation(
        obs(QuantityType.TIME, "30", "min", "held for 30 min")
    )
    assert flags == []


def test_nacl_5_no_unit_is_flagged():
    flags = validate_observation(
        obs(QuantityType.VOLUME, "5", None, "Added 5 of NaCl solution")
    )
    assert flags == ["missing_unit"]


def test_ph_7_passes_clean_unitless():
    flags = validate_observation(obs(QuantityType.PH, "7", None, "pH adjusted to 7"))
    assert flags == []


# --- Plausibility / unit disambiguation ---

def test_plausible_units_empty_for_untabled_types():
    # OTHER/YIELD have no plausibility table — must return {} (not KeyError),
    # so the inference pass can still run on flagged bare quantities.
    assert plausible_units(QuantityType.OTHER, 100.0) == {}
    assert plausible_units(QuantityType.YIELD, 55.0) == {}


def test_70_kelvin_is_implausible_70_celsius_is_plausible():
    candidates = plausible_units(QuantityType.TEMPERATURE, 70.0)
    assert candidates["k"] is False          # 70 K = -203 °C, not "heating"
    assert candidates["c"] is True
    assert candidates["f"] is True           # 70 °F = 21 °C is in range (context must decide)


def test_ocr_misread_700_celsius_is_implausible():
    flags = validate_observation(obs(QuantityType.TEMPERATURE, "700", "C"))
    assert flags == ["implausible_value"]


def test_ph_15_out_of_range():
    flags = validate_observation(obs(QuantityType.PH, "15", None))
    assert flags == ["out_of_range_ph"]


def test_unknown_unit_flagged():
    flags = validate_observation(obs(QuantityType.VOLUME, "5", "blorps"))
    assert flags == ["unknown_unit"]


def test_other_with_missing_unit_is_flagged():
    # "Added 100 of benzaldehyde" — model couldn't type it, no unit written.
    # Most ambiguous case there is; must be flagged, never guessed.
    flags = validate_observation(obs(QuantityType.OTHER, "100", None, "Added 100 of benzaldehyde"))
    assert flags == ["missing_unit"]


def test_yield_with_missing_unit_is_flagged():
    flags = validate_observation(obs(QuantityType.YIELD, "55", None, "yield 55"))
    assert flags == ["missing_unit"]


def test_yield_with_percent_is_clean():
    flags = validate_observation(obs(QuantityType.YIELD, "93", "%", "yield 93%"))
    assert flags == []


def test_other_with_unit_present_is_accepted():
    # unit present but unrecognized type — accept rather than over-flag
    flags = validate_observation(obs(QuantityType.OTHER, "5", "drops", "5 drops"))
    assert flags == []


def test_unparseable_value_flagged():
    flags = validate_observation(obs(QuantityType.MASS, "5O", "g"))  # letter O, OCR error
    assert flags == ["unparseable_value"]


# --- Conversions ---

def test_fahrenheit_conversion():
    assert abs(to_canonical(QuantityType.TEMPERATURE, 212.0, "F") - 100.0) < 1e-9


def test_kelvin_conversion():
    assert abs(to_canonical(QuantityType.TEMPERATURE, 273.15, "K") - 0.0) < 1e-9


# --- extended chemistry quantities ---

def test_wavelength_nm_clean():
    assert validate_observation(obs(QuantityType.WAVELENGTH, "540", "nm", "λmax = 540 nm")) == []


def test_wavelength_ocr_misread_meters_implausible():
    # "540 m" instead of "540 nm" — 540 m = 5.4e11 nm, way out of optical range
    assert validate_observation(obs(QuantityType.WAVELENGTH, "540", "m")) == ["implausible_value"]


def test_wavelength_missing_unit_flagged():
    assert validate_observation(obs(QuantityType.WAVELENGTH, "280", None, "λ 280")) == ["missing_unit"]


def test_absorbance_unitless_clean():
    assert validate_observation(obs(QuantityType.ABSORBANCE, "0.85", None, "A = 0.85")) == []


def test_absorbance_out_of_range():
    assert validate_observation(obs(QuantityType.ABSORBANCE, "50", None)) == ["out_of_range_absorbance"]


def test_pka_unitless_clean():
    assert validate_observation(obs(QuantityType.PKA, "4.76", None, "pKa = 4.76")) == []


def test_equilibrium_constant_any_magnitude_accepted():
    assert validate_observation(obs(QuantityType.EQUILIBRIUM_CONSTANT, "1.8e-5", None, "Ka = 1.8e-5")) == []


def test_refractive_index_range():
    assert validate_observation(obs(QuantityType.REFRACTIVE_INDEX, "1.333", None)) == []
    assert validate_observation(obs(QuantityType.REFRACTIVE_INDEX, "9", None)) == ["out_of_range_refractive_index"]


def test_energy_negative_enthalpy_clean():
    # exothermic ΔH is negative — must be allowed
    assert validate_observation(obs(QuantityType.ENERGY, "-92", "kJ/mol", "ΔH = -92 kJ/mol")) == []


def test_molar_mass_clean():
    assert validate_observation(obs(QuantityType.MOLAR_MASS, "180", "g/mol", "M = 180 g/mol")) == []


def test_symbol_ambiguity():
    from chemextract.symbols import is_ambiguous_symbol, quantities_for_symbol
    assert is_ambiguous_symbol("M")      # molarity vs molar mass
    assert is_ambiguous_symbol("m")      # mass vs molality
    assert not is_ambiguous_symbol("λ")  # wavelength only
    assert QuantityType.WAVELENGTH in quantities_for_symbol("lambda")


def test_parse_value():
    assert parse_value("70") == 70.0
    assert parse_value("7.0") == 7.0
    assert parse_value("1,000") == 1000.0
    assert parse_value("70C") is None
