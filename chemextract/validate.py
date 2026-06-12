"""Deterministic validation layer. No LLM calls here — pure functions only.

Flags emitted:
  missing_unit              unit is null but this quantity type requires one
  unknown_unit              unit not in the whitelist for this quantity type
  unparseable_value         value is not a number
  implausible_value         value+unit converts to something outside sane bench ranges
  out_of_range_ph           pH outside 0-14
"""
import re
from typing import Optional

from .schema import QuantityType, RawObservation

# Genuinely unitless quantities — a missing unit here is correct, not a flag.
# Some still have a sane numeric range we can range-check (pH 0-14, etc.).
# This includes OTHER and YIELD logic elsewhere: a value the model couldn't
# confidently type, with no unit written, is the most ambiguous case there is —
# it must be flagged, never guessed ("yield 55", "Added 100 of benzaldehyde").
UNITLESS_RANGES: dict[QuantityType, tuple[float, float]] = {
    QuantityType.PH: (0.0, 14.0),
    QuantityType.ABSORBANCE: (0.0, 10.0),
    QuantityType.REFRACTIVE_INDEX: (1.0, 2.6),
    QuantityType.PKA: (-10.0, 60.0),
}
# Unitless but no universal range (spans many orders of magnitude) — just accept.
UNITLESS_NO_RANGE = {QuantityType.EQUILIBRIUM_CONSTANT}
UNIT_NOT_REQUIRED = set(UNITLESS_RANGES) | UNITLESS_NO_RANGE

# unit -> (canonical multiplier/converter). Canonical units:
# temperature: celsius, volume: mL, mass: g, time: s, concentration: as-is, pressure: kPa
UNIT_TABLES: dict[QuantityType, dict[str, callable]] = {
    QuantityType.TEMPERATURE: {
        "c": lambda v: v, "°c": lambda v: v, "celsius": lambda v: v, "degc": lambda v: v,
        "f": lambda v: (v - 32) * 5 / 9, "°f": lambda v: (v - 32) * 5 / 9,
        "fahrenheit": lambda v: (v - 32) * 5 / 9,
        "k": lambda v: v - 273.15, "kelvin": lambda v: v - 273.15,
    },
    QuantityType.VOLUME: {
        "ml": lambda v: v, "milliliter": lambda v: v, "milliliters": lambda v: v,
        "l": lambda v: v * 1000, "liter": lambda v: v * 1000, "liters": lambda v: v * 1000,
        "ul": lambda v: v / 1000, "µl": lambda v: v / 1000, "μl": lambda v: v / 1000,
    },
    QuantityType.MASS: {
        "g": lambda v: v, "gram": lambda v: v, "grams": lambda v: v,
        "mg": lambda v: v / 1000, "kg": lambda v: v * 1000,
        "ug": lambda v: v / 1e6, "µg": lambda v: v / 1e6, "μg": lambda v: v / 1e6,
    },
    QuantityType.TIME: {
        "s": lambda v: v, "sec": lambda v: v, "secs": lambda v: v,
        "second": lambda v: v, "seconds": lambda v: v,
        "min": lambda v: v * 60, "mins": lambda v: v * 60,
        "minute": lambda v: v * 60, "minutes": lambda v: v * 60,
        "h": lambda v: v * 3600, "hr": lambda v: v * 3600, "hrs": lambda v: v * 3600,
        "hour": lambda v: v * 3600, "hours": lambda v: v * 3600,
        "d": lambda v: v * 86400, "day": lambda v: v * 86400, "days": lambda v: v * 86400,
    },
    QuantityType.CONCENTRATION: {
        "m": lambda v: v, "mol/l": lambda v: v, "molar": lambda v: v,
        "mm": lambda v: v / 1000, "mmol/l": lambda v: v / 1000,
        "%": lambda v: v, "percent": lambda v: v,
        "% w/v": lambda v: v, "% v/v": lambda v: v, "% w/w": lambda v: v,
        "g/l": lambda v: v, "mg/ml": lambda v: v,
    },
    QuantityType.PRESSURE: {
        "kpa": lambda v: v, "pa": lambda v: v / 1000, "mpa": lambda v: v * 1000,
        "bar": lambda v: v * 100, "mbar": lambda v: v / 10,
        "atm": lambda v: v * 101.325, "psi": lambda v: v * 6.895,
        "torr": lambda v: v * 0.1333, "mmhg": lambda v: v * 0.1333,
    },
}

# Plausible bench-chemistry ranges in canonical units (deliberately generous —
# these catch unit confusion and OCR misreads, not unusual-but-real chemistry)
PLAUSIBLE_RANGES: dict[QuantityType, tuple[float, float]] = {
    QuantityType.TEMPERATURE: (-100.0, 400.0),     # °C
    QuantityType.VOLUME: (0.0001, 50000.0),        # mL
    QuantityType.MASS: (1e-6, 100000.0),           # g
    QuantityType.TIME: (0.01, 60 * 86400),         # s (up to 60 days)
    QuantityType.PH: (0.0, 14.0),
    QuantityType.CONCENTRATION: (0.0, 1000.0),     # loose; unit-dependent
    QuantityType.PRESSURE: (0.01, 100000.0),       # kPa
}

# --- extended chemistry quantities (spectroscopy, thermo, electrochem) ---
# canonical units: wavelength=nm, frequency=Hz, wavenumber=cm⁻¹, density=g/mL,
# amount=mol, molar_mass=g/mol, molality=mol/kg, energy=kJ/mol, potential=V,
# current=A, molar_absorptivity=M⁻¹cm⁻¹, specific_rotation=deg.
_lin = lambda k: (lambda v: v * k)  # noqa: E731  linear unit -> canonical multiplier

UNIT_TABLES.update({
    QuantityType.WAVELENGTH: {
        "nm": _lin(1), "µm": _lin(1000), "um": _lin(1000), "micron": _lin(1000),
        "å": _lin(0.1), "angstrom": _lin(0.1), "ang": _lin(0.1),
        "pm": _lin(1e-3), "mm": _lin(1e6), "cm": _lin(1e7), "m": _lin(1e9),
    },
    QuantityType.FREQUENCY: {
        "hz": _lin(1), "khz": _lin(1e3), "mhz": _lin(1e6),
        "ghz": _lin(1e9), "thz": _lin(1e12),
    },
    QuantityType.WAVENUMBER: {
        "cm-1": _lin(1), "cm^-1": _lin(1), "/cm": _lin(1), "cm⁻¹": _lin(1), "1/cm": _lin(1),
    },
    QuantityType.DENSITY: {
        "g/ml": _lin(1), "g/cm3": _lin(1), "g/cm³": _lin(1), "g/cc": _lin(1),
        "kg/l": _lin(1), "mg/ml": _lin(1e-3), "g/l": _lin(1e-3),
        "kg/m3": _lin(1e-3), "kg/m³": _lin(1e-3),
    },
    QuantityType.AMOUNT: {
        "mol": _lin(1), "mmol": _lin(1e-3), "µmol": _lin(1e-6), "umol": _lin(1e-6),
        "nmol": _lin(1e-9), "pmol": _lin(1e-12), "kmol": _lin(1e3), "mole": _lin(1),
    },
    QuantityType.MOLAR_MASS: {
        "g/mol": _lin(1), "kg/mol": _lin(1e3), "da": _lin(1), "dalton": _lin(1),
        "kda": _lin(1e3), "amu": _lin(1), "g·mol⁻¹": _lin(1),
    },
    QuantityType.MOLALITY: {
        "mol/kg": _lin(1), "m": _lin(1), "molal": _lin(1), "mmol/kg": _lin(1e-3),
    },
    QuantityType.ENERGY: {
        "kj/mol": _lin(1), "j/mol": _lin(1e-3), "kcal/mol": _lin(4.184),
        "cal/mol": _lin(4.184e-3), "ev": _lin(96.485), "kj": _lin(1),
        "j": _lin(1e-3), "kcal": _lin(4.184),
    },
    QuantityType.MOLAR_ABSORPTIVITY: {
        "m-1cm-1": _lin(1), "m⁻¹cm⁻¹": _lin(1), "l/mol/cm": _lin(1),
        "l·mol⁻¹·cm⁻¹": _lin(1), "lmol-1cm-1": _lin(1),
    },
    QuantityType.SPECIFIC_ROTATION: {"deg": _lin(1), "°": _lin(1), "degree": _lin(1)},
    QuantityType.POTENTIAL: {
        "v": _lin(1), "mv": _lin(1e-3), "kv": _lin(1e3), "µv": _lin(1e-6), "uv": _lin(1e-6),
    },
    QuantityType.CURRENT: {
        "a": _lin(1), "ma": _lin(1e-3), "µa": _lin(1e-6), "ua": _lin(1e-6),
        "na": _lin(1e-9), "ka": _lin(1e3), "amp": _lin(1),
    },
})

PLAUSIBLE_RANGES.update({
    QuantityType.WAVELENGTH: (0.1, 1e8),        # nm — X-ray to far-IR
    QuantityType.FREQUENCY: (1.0, 1e16),        # Hz
    QuantityType.WAVENUMBER: (1.0, 1e6),        # cm⁻¹
    QuantityType.DENSITY: (1e-5, 30.0),         # g/mL
    QuantityType.AMOUNT: (1e-12, 1e5),          # mol
    QuantityType.MOLAR_MASS: (1.0, 1e7),        # g/mol
    QuantityType.MOLALITY: (1e-4, 1e3),         # mol/kg
    QuantityType.ENERGY: (-1e5, 1e5),           # kJ/mol (negative = exothermic)
    QuantityType.MOLAR_ABSORPTIVITY: (0.0, 1e7),
    QuantityType.SPECIFIC_ROTATION: (-400.0, 400.0),
    QuantityType.POTENTIAL: (-15.0, 15.0),      # V
    QuantityType.CURRENT: (1e-12, 1e5),         # A
})

_NUM_RE = re.compile(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$")


def parse_value(value: str) -> Optional[float]:
    v = value.strip().replace(",", "")
    if _NUM_RE.match(v):
        return float(v)
    return None


def normalize_unit(unit: str) -> str:
    return unit.strip().lower().replace("degrees", "°").replace(" ", "") \
        if "°" in unit or "degrees" in unit.lower() else unit.strip().lower()


def to_canonical(qtype: QuantityType, value: float, unit: str) -> Optional[float]:
    """Convert value to the canonical unit for its type. None if unit unknown."""
    table = UNIT_TABLES.get(qtype)
    if not table:
        return None
    conv = table.get(normalize_unit(unit))
    return conv(value) if conv else None


def plausible_units(qtype: QuantityType, value: float) -> dict[str, bool]:
    """For a unitless value, which candidate units would put it in a sane range?
    Deterministic grounding for the inference pass (e.g. 70 K = -203 °C → False).
    Returns {} for quantity types we have no table for (OTHER, YIELD) — the
    inference pass then reasons from document context alone."""
    if qtype not in UNIT_TABLES or qtype not in PLAUSIBLE_RANGES:
        return {}
    lo, hi = PLAUSIBLE_RANGES[qtype]
    out = {}
    for unit in UNIT_TABLES.get(qtype, {}):
        canonical = to_canonical(qtype, value, unit)
        out[unit] = canonical is not None and lo <= canonical <= hi
    return out


def validate_observation(obs: RawObservation) -> list[str]:
    """Return the list of flags for one raw observation. Empty list = clean."""
    flags: list[str] = []

    num = parse_value(obs.value)
    if num is None:
        flags.append("unparseable_value")
        return flags  # range checks meaningless without a number

    # Unitless quantities (pH, absorbance, refractive index, pKa, K): no unit
    # required. Range-check the ones with a sane numeric range.
    if obs.quantity_type in UNIT_NOT_REQUIRED:
        rng = UNITLESS_RANGES.get(obs.quantity_type)
        if rng and not (rng[0] <= num <= rng[1]):
            flags.append(f"out_of_range_{obs.quantity_type.value}")
        return flags

    if obs.unit is None or not obs.unit.strip():
        flags.append("missing_unit")
        return flags

    # Unit is present. If we have a conversion table for this quantity type,
    # validate it (catches unknown units and OCR/unit-confusion misreads).
    # For types with no table (OTHER, YIELD), a written unit is accepted as-is —
    # we can't range-check a quantity we couldn't classify, and flagging it
    # would only add noise without improving safety.
    if obs.quantity_type in UNIT_TABLES:
        canonical = to_canonical(obs.quantity_type, num, obs.unit)
        if canonical is None:
            flags.append("unknown_unit")
        else:
            lo, hi = PLAUSIBLE_RANGES[obs.quantity_type]
            if not (lo <= canonical <= hi):
                flags.append("implausible_value")

    return flags
