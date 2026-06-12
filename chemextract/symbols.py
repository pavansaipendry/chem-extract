"""Chemistry quantity-symbol knowledge base.

The single source of truth for: which physical quantities the system knows,
the symbols chemists write for each, their valid units, and plausibility ranges.

Design principles:
  - This is NOT meant to enumerate "every symbol ever" — that is unbounded and
    many symbols are overloaded. It covers the standard *measurable* quantities.
  - Anything not here still falls through to QuantityType.OTHER and gets flagged
    for review — unknown symbols are handled safely, never silently dropped.
  - Overloaded symbols (M, m, T, E, n, k) map to MULTIPLE quantities. When a
    symbol is ambiguous and context can't resolve it, the right answer is to
    FLAG, not to guess one meaning.
"""
from .schema import QuantityType as Q

# symbol (as written) -> list of quantities it can denote.
# A list of length > 1 means the symbol is overloaded → ambiguous unless context
# (units, surrounding words) disambiguates.
SYMBOL_TO_QUANTITIES: dict[str, list[Q]] = {
    # --- length / spectroscopy ---
    "λ": [Q.WAVELENGTH], "lambda": [Q.WAVELENGTH], "λmax": [Q.WAVELENGTH],
    "ν": [Q.FREQUENCY, Q.WAVENUMBER], "nu": [Q.FREQUENCY, Q.WAVENUMBER],
    "ν̄": [Q.WAVENUMBER], "wavenumber": [Q.WAVENUMBER],
    "A": [Q.ABSORBANCE], "Abs": [Q.ABSORBANCE], "absorbance": [Q.ABSORBANCE],
    "ε": [Q.MOLAR_ABSORPTIVITY], "epsilon": [Q.MOLAR_ABSORPTIVITY],
    "n": [Q.AMOUNT, Q.REFRACTIVE_INDEX],  # moles vs refractive index — overloaded
    "[α]": [Q.SPECIFIC_ROTATION], "alpha": [Q.SPECIFIC_ROTATION],
    # --- state ---
    "T": [Q.TEMPERATURE],  # also transmittance in spectroscopy — context-dependent
    "P": [Q.PRESSURE], "p": [Q.PRESSURE],
    "V": [Q.VOLUME], "ρ": [Q.DENSITY], "rho": [Q.DENSITY], "d": [Q.DENSITY],
    "t": [Q.TIME],
    # --- amount / composition ---
    "M": [Q.CONCENTRATION, Q.MOLAR_MASS],  # molarity vs molar mass — overloaded
    "c": [Q.CONCENTRATION], "C": [Q.CONCENTRATION],
    "m": [Q.MASS, Q.MOLALITY],  # mass vs molality — overloaded
    "Mw": [Q.MOLAR_MASS], "MW": [Q.MOLAR_MASS], "Mr": [Q.MOLAR_MASS],
    "mol": [Q.AMOUNT], "n_mol": [Q.AMOUNT],
    # --- thermodynamics ---
    "ΔH": [Q.ENERGY], "dH": [Q.ENERGY], "ΔG": [Q.ENERGY], "dG": [Q.ENERGY],
    "ΔS": [Q.ENTROPY], "Ea": [Q.ENERGY], "Q_heat": [Q.ENERGY],
    # --- equilibrium / kinetics ---
    "K": [Q.EQUILIBRIUM_CONSTANT], "Keq": [Q.EQUILIBRIUM_CONSTANT],
    "Ka": [Q.EQUILIBRIUM_CONSTANT], "Kb": [Q.EQUILIBRIUM_CONSTANT],
    "Kw": [Q.EQUILIBRIUM_CONSTANT], "Ksp": [Q.EQUILIBRIUM_CONSTANT],
    "pKa": [Q.PKA], "pKb": [Q.PKA], "pH": [Q.PH], "pOH": [Q.PKA],
    "k": [Q.RATE_CONSTANT],  # also Boltzmann const — context
    # --- electrochemistry ---
    "E": [Q.POTENTIAL, Q.ENERGY],  # cell potential vs energy — overloaded
    "E°": [Q.POTENTIAL], "Ecell": [Q.POTENTIAL],
    "I": [Q.CURRENT], "i": [Q.CURRENT],
}


def quantities_for_symbol(symbol: str) -> list[Q]:
    return SYMBOL_TO_QUANTITIES.get(symbol.strip(), [])


def is_ambiguous_symbol(symbol: str) -> bool:
    return len(quantities_for_symbol(symbol)) > 1


# Human-readable legend injected into the extraction prompt so the vision model
# classifies into the richer taxonomy instead of defaulting to "other".
LEGEND = """Common chemistry quantity symbols (use to set quantity_type):
  λ / lambda / λmax = wavelength (nm, µm, Å)
  ν / nu = frequency (Hz) OR wavenumber (cm⁻¹) — pick by unit/context
  A / Abs = absorbance (unitless)        ε / epsilon = molar_absorptivity (M⁻¹cm⁻¹)
  T = temperature (°C/°F/K)              P / p = pressure (kPa/atm/bar/psi/mmHg/torr)
  V = volume                              ρ / rho / d = density (g/mL, g/cm³, kg/m³)
  n = amount (mol) OR refractive_index — ambiguous, judge by context
  M = concentration/molarity (mol/L) OR molar_mass (g/mol) — ambiguous
  c / C = concentration                   m = mass OR molality — ambiguous
  Mw / MW / Mr = molar_mass (g/mol)       mol = amount (mol, mmol, µmol)
  ΔH / ΔG / Ea = energy (kJ/mol, kcal/mol)   ΔS = entropy (J/mol·K)
  K / Keq / Ka / Kb / Kw / Ksp = equilibrium_constant   pKa / pKb / pOH = pka (unitless)
  pH = ph (unitless)                      k = rate_constant
  E / E° / Ecell = potential (V, mV) [E can also be energy]   I / i = current (A, mA)
  [α] / alpha = specific_rotation (degrees)

When a symbol is overloaded (M, m, n, T, E) and the unit/context does NOT clearly
resolve which quantity it is, still extract the value verbatim and choose your
best quantity_type, but use a LOWER confidence — the validation layer will flag it."""
