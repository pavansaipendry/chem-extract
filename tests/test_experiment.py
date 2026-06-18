"""Deterministic calculation checking — the trustworthy half of Level 4.

No LLM, no API key. We feed the verifier the exact Faraday's-law chain written on
Page 57 (Li electrodeposition) and confirm it re-derives every step, then prove it
FLAGS a wrong stated result rather than accepting it.

    I = J·A      = 5e-4 A/cm² · 0.3 cm²   = 1.5e-4 A
    Q = I·t      = 1.5e-4 A · 5400 s      = 0.81 C
    n = Q/(z·F)  = 0.81 C / (1 · 96485)   = 8.4e-6 mol
    m = n·M      = 8.4e-6 mol · 6.94 g/mol = 5.8e-5 g
"""
from chemextract.experiment import (
    CalcInput,
    Calculation,
    ExperimentRecord,
    verify_calculation,
    verify_experiment,
)


def calc(relation, inputs, stated, quote="from page 57"):
    return Calculation(
        relation=relation,
        description=relation,
        inputs=[CalcInput(name=n, value=v, unit=u) for n, v, u in inputs],
        stated_result=stated,
        quote=quote,
    )


# --- The Page 57 chain re-derives correctly ----------------------------------

def test_current_from_density():
    c = verify_calculation(calc(
        "current_from_density",
        [("current_density_A_cm2", 5e-4, "A/cm2"), ("area_cm2", 0.3, "cm2")],
        stated=1.5e-4,
    ))
    assert c.agree is True
    assert abs(c.recomputed - 1.5e-4) < 1e-9


def test_faraday_charge():
    c = verify_calculation(calc(
        "faraday_charge",
        [("current_A", 1.5e-4, "A"), ("time_s", 5400, "s")],
        stated=0.81,
    ))
    assert c.agree is True
    assert abs(c.recomputed - 0.81) < 1e-6


def test_faraday_moles():
    c = verify_calculation(calc(
        "faraday_moles",
        [("charge_C", 0.81, "C"), ("electrons", 1, "")],
        stated=8.4e-6,
    ))
    assert c.agree is True                 # 0.81/96485 = 8.395e-6 ≈ 8.4e-6
    assert abs(c.recomputed - 8.4e-6) < 1e-7


def test_mass_from_moles():
    c = verify_calculation(calc(
        "mass_from_moles",
        [("moles", 8.4e-6, "mol"), ("molar_mass_g_mol", 6.94, "g/mol")],
        stated=5.8e-5,
    ))
    assert c.agree is True                 # 8.4e-6·6.94 = 5.83e-5 ≈ 5.8e-5


# --- It flags errors instead of accepting them -------------------------------

def test_wrong_stated_result_is_flagged():
    # A decimal slip: Q stated as 8.1 C instead of 0.81 C.
    c = verify_calculation(calc(
        "faraday_charge",
        [("current_A", 1.5e-4, "A"), ("time_s", 5400, "s")],
        stated=8.1,
    ))
    assert c.agree is False
    assert "calculation_mismatch" in c.flags
    assert c.rel_error > 0.03


def test_dropped_electron_count_is_caught():
    # n computed as Q/F (forgetting z): for z=1 it's fine, so simulate a 2-electron
    # process stated as if 1-electron — the recompute with electrons=2 disagrees
    # with a 1-electron stated value.
    c = verify_calculation(calc(
        "faraday_moles",
        [("charge_C", 0.81, "C"), ("electrons", 2, "")],
        stated=8.4e-6,                      # the z=1 number
    ))
    assert c.agree is False
    assert "calculation_mismatch" in c.flags


def test_unknown_relation_flagged():
    c = verify_calculation(calc("nernst_equation", [("E0", 0.0, "V")], stated=0.0))
    assert "unknown_relation" in c.flags


def test_missing_inputs_flagged():
    c = verify_calculation(calc("faraday_charge", [("current_A", 1.5e-4, "A")], stated=0.81))
    assert "missing_inputs" in c.flags


def test_no_stated_result_reports_recompute():
    c = verify_calculation(calc(
        "faraday_charge",
        [("current_A", 1.5e-4, "A"), ("time_s", 5400, "s")],
        stated=None,
    ))
    assert "no_stated_result" in c.flags
    assert abs(c.recomputed - 0.81) < 1e-6


def test_verify_experiment_runs_whole_chain():
    record = ExperimentRecord(
        goal="screen electrolyte for stable Li plating",
        calculations=[
            calc("current_from_density",
                 [("current_density_A_cm2", 5e-4, "A/cm2"), ("area_cm2", 0.3, "cm2")], 1.5e-4),
            calc("faraday_charge",
                 [("current_A", 1.5e-4, "A"), ("time_s", 5400, "s")], 0.81),
            calc("faraday_moles",
                 [("charge_C", 0.81, "C"), ("electrons", 1, "")], 8.4e-6),
            calc("mass_from_moles",
                 [("moles", 8.4e-6, "mol"), ("molar_mass_g_mol", 6.94, "g/mol")], 5.8e-5),
        ],
    )
    checks = verify_experiment(record)
    assert len(checks) == 4
    assert all(c.agree for c in checks)     # the whole page checks out
