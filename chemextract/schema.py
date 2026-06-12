"""Data models for chemistry document extraction.

Core principle: distinguish "what the document says" (value, unit, quote —
always verbatim) from "what we infer it means" (the `inferred` sub-object,
which never overwrites the raw extraction).
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class QuantityType(str, Enum):
    # core bench quantities
    TEMPERATURE = "temperature"
    VOLUME = "volume"
    MASS = "mass"
    TIME = "time"
    PH = "ph"
    CONCENTRATION = "concentration"
    PRESSURE = "pressure"
    YIELD = "yield"
    # spectroscopy / analytical
    WAVELENGTH = "wavelength"          # λ
    FREQUENCY = "frequency"            # ν
    WAVENUMBER = "wavenumber"          # ν̄ (cm⁻¹)
    ABSORBANCE = "absorbance"          # A (unitless)
    MOLAR_ABSORPTIVITY = "molar_absorptivity"  # ε
    SPECIFIC_ROTATION = "specific_rotation"    # [α]
    REFRACTIVE_INDEX = "refractive_index"      # n (unitless)
    # amount / composition
    AMOUNT = "amount"                 # n (mol)
    MOLAR_MASS = "molar_mass"         # M / Mw (g/mol)
    MOLALITY = "molality"             # m (mol/kg)
    DENSITY = "density"               # ρ (g/mL)
    # thermodynamics / kinetics / electrochem
    ENERGY = "energy"                 # ΔH, ΔG, Ea (kJ/mol)
    ENTROPY = "entropy"               # ΔS (J/mol·K)
    EQUILIBRIUM_CONSTANT = "equilibrium_constant"  # K, Ka (unitless)
    PKA = "pka"                       # pKa/pKb/pOH (unitless)
    RATE_CONSTANT = "rate_constant"   # k
    POTENTIAL = "potential"           # E (V)
    CURRENT = "current"               # I (A)
    OTHER = "other"


# --- What the vision LLM returns (raw extraction, schema-enforced) ---

class RawObservation(BaseModel):
    """One physical quantity exactly as written on the page."""
    quantity_type: QuantityType
    value: str = Field(description="The numeric value verbatim, e.g. '70', '7.0'")
    unit: Optional[str] = Field(
        description="The unit verbatim if visible, otherwise null. Never invent a unit."
    )
    quote: str = Field(
        description="The exact source text this value came from, verbatim."
    )
    confidence: float = Field(description="Self-reported read confidence, 0-1")


class RawExtraction(BaseModel):
    transcript: str = Field(description="Full verbatim transcript of the page")
    observations: list[RawObservation]


# --- What the inference pass returns (clearly labeled as inference) ---

class InferredUnit(BaseModel):
    unit: Optional[str] = Field(
        description="Most likely intended unit, or null if no defensible inference"
    )
    confidence: float
    reasoning: str = Field(description="Human-readable justification")


# --- Final pipeline output per field ---

class Observation(BaseModel):
    quantity_type: QuantityType
    value: str
    unit: Optional[str]
    quote: str
    confidence: float
    flags: list[str] = []
    inferred: Optional[InferredUnit] = None
    needs_review: bool = False


# --- Reconstruction: the corrected, fully-specified rewrite of the document ---

class Interpretation(BaseModel):
    """One candidate reading of a doubtful span — a complete, valid version."""
    text: str = Field(description="the span rewritten under this interpretation, fully specified")
    confidence: float
    reasoning: str = Field(description="chemistry-grounded justification, citing the physical fact used")


class Ambiguity(BaseModel):
    """A span the system could not resolve with confidence — offered to the human
    as 2+ competing versions to choose between, never silently picked."""
    span: str = Field(description="the doubtful text exactly as written")
    interpretations: list[Interpretation] = Field(description="competing versions, most likely first")
    resolved: bool = Field(description="True if one reading clearly dominates on chemistry grounds")


class Reconstruction(BaseModel):
    canonical_text: str = Field(
        description="best single fully-specified rewrite; inferred insertions marked, e.g. 70[°C]")
    confidence: float
    ambiguities: list[Ambiguity] = []


class DocumentExtraction(BaseModel):
    source_image: str
    transcript: str
    observations: list[Observation]
    reconstruction: Optional[Reconstruction] = None
