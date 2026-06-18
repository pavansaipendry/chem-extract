"""Level 2: faithful handling of scientific notation and special symbols.

The symbols that get mangled by OCR/transcription are exactly the ones that carry
meaning: a dropped exponent turns 1.5×10⁻⁴ A into 1.5 A (nine orders of magnitude),
a lost ° turns 85 °C into a bare 85, cm⁻¹ becomes cm1. This module is deterministic
(no LLM): it PARSES every typographic form to the same number, normalises unicode
super/subscripts to ASCII so unit tables match, and FLAGS when a captured value
looks like it lost notation that its source quote still shows.

Nothing here rewrites the verbatim capture — it parses and checks alongside it.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

# unicode super/subscript digits (and signs) -> ASCII
_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"
_SUBSCRIPTS = "₀₁₂₃₄₅₆₇₈₉₊₋"
_ASCII_FOR_SUPER = "0123456789+-"
_ASCII_FOR_SUB = "0123456789+-"
_SUPER_MAP = {ord(s): a for s, a in zip(_SUPERSCRIPTS, _ASCII_FOR_SUPER)}
_SUB_MAP = {ord(s): a for s, a in zip(_SUBSCRIPTS, _ASCII_FOR_SUB)}

# typographic minus / multiplication signs -> ASCII
_DASHES = "−–—‐‑"   # − – — ‐ ‑
_TIMES = "×·∗⋅"          # × · ∗ ⋅

GREEK = set("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")


def _to_ascii_signs(text: str) -> str:
    out = []
    for ch in text:
        if ch in _DASHES:
            out.append("-")
        elif ch in _TIMES:
            out.append("x")
        else:
            out.append(ch)
    return "".join(out)


def ascii_units(text: str) -> str:
    """Map unicode super/subscripts to ASCII so 'cm⁻¹'->'cm-1', 'cm²'->'cm2',
    'H₂O'->'H2O'. Lets unit whitelists match without enumerating every glyph.
    Non-destructive: callers keep the original verbatim string."""
    return text.translate(_SUPER_MAP).translate(_SUB_MAP)


def normalize_sci(text: str) -> str:
    """Canonicalise any scientific-notation form to ASCII 'mantissaE±exp':
    '1.5×10⁻⁴' / '1.5 x 10^-4' / '1.5·10-4' / '1.5e-4' -> '1.5e-4'. Returns the
    input (sign/super normalised) unchanged if it isn't sci notation."""
    t = _to_ascii_signs(ascii_units(text)).strip()
    # mantissa (x|*)? 10 ^? exponent   →   mantissaeexponent
    m = re.search(
        r"([+-]?\d+(?:\.\d+)?)\s*[x*]?\s*10\s*\^?\s*([+-]?\d+)",
        t,
    )
    if m:
        return f"{m.group(1)}e{m.group(2)}"
    return t


_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def parse_number(text: str) -> Optional[float]:
    """Parse a numeric token in any typographic form to a float:
    '1.5E-4', '1.5×10⁻⁴', '1.5 x 10^-4', '8.4·10⁻⁶', '−5.8e-5', '1,000'.
    Returns None if there is no parseable number."""
    if text is None:
        return None
    t = normalize_sci(text).replace(",", "").strip()
    if _NUM_RE.match(t):
        return float(t)
    # last resort: a bare number sitting inside other text
    m = re.search(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", t)
    return float(m.group()) if m else None


def notation_features(text: str) -> set[str]:
    """Which special-notation features a string carries — useful to assert that a
    transcript preserved them (telemetry / fidelity checks)."""
    feats: set[str] = set()
    if any(c in GREEK for c in text):
        feats.add("greek")
    if any(c in _SUPERSCRIPTS for c in text):
        feats.add("superscript")
    if any(c in _SUBSCRIPTS for c in text):
        feats.add("subscript")
    if "°" in text or "º" in text:   # degree sign U+00B0 or masculine ordinal U+00BA
        feats.add("degree")
    if any(c in _DASHES for c in text):
        feats.add("unicode_minus")
    if re.search(r"\d\s*[x*×·⋅]\s*10", text) or re.search(r"[eE][+-]?\d", text):
        feats.add("sci_notation")
    return feats


# a power-of-ten exponent: ×10ⁿ / 10^n / E±n, tolerant of spaces ("8.4 E-6")
_EXP = r"(?:x?\s*10\s*\^?\s*[+-]?\s*\d+|[eE]\s*[+-]?\s*\d+)"


def _has_scientific_exponent(text: str) -> bool:
    """True if the text clearly expresses a power-of-ten (×10ⁿ, 10^n, or E±n)."""
    t = _to_ascii_signs(ascii_units(text))
    return bool(re.search(r"\d\s*" + _EXP, t))


def notation_flags(value: str, quote: str) -> list[str]:
    """Deterministic fidelity check: did the captured `value` drop an exponent its
    source `quote` still shows? Flags, never fixes.

    Precise by construction — it only fires when the value string appears in the
    quote IMMEDIATELY followed by a power-of-ten it omitted (e.g. quote
    '1.5×10⁻⁴ A', value '1.5'). That avoids false positives on calculation lines
    where the quote holds several numbers (e.g. value '5400' in
    '1.5E-4 A · 5400 s' is not flagged, because 5400 has no exponent after it)."""
    flags: list[str] = []
    if _has_scientific_exponent(value):
        return flags  # value already carries its exponent — nothing dropped
    q = _to_ascii_signs(ascii_units(quote))
    if re.search(re.escape(value.strip()) + r"\s*" + _EXP, q):
        flags.append("exponent_dropped")
    return flags
