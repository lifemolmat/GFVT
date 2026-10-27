"""Progress observes the true solver grid without changing equilibria."""

import pickle

import numpy as np
import pytest

from gfvt_rods import PD


def model():
    return PD(1e-9, 2.5e-9, 66.4, 35, "good", 3, [.05, .4],
              progress=False, verbose=False)


def test_progress_refined_grid_and_unchanged_scientific_results():
    events = []
    options = dict(L_rod=40e-9, dep_model="phs", coexistence="II")
    baseline = model().GFVT_rods(.5, **options)
    observed = model().GFVT_rods(.5, progress_callback=events.append, **options)
    assert events[0]["stage"] == "critical_points"
    trace = [e for e in events if e["stage"] == "tracing"]
    total = trace[0]["total"]
    assert total > observed.N_steps
    assert [e["completed"] for e in trace] == list(range(total+1))
    assert all(e["total"] == total for e in trace)
    assert events[-2]["stage"] == "finalizing"
    assert events[-1] == dict(stage="complete", completed=total, total=total)
    for key in ("II", "IN", "NN"):
        assert len(baseline.rod_branches[key]) == len(observed.rod_branches[key])
        for a, b in zip(baseline.rod_branches[key], observed.rod_branches[key]):
            for field in ("phi_R", "phi", "alpha", "S", "mu", "pressure", "stable"):
                np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert baseline.rod_critical_points == observed.rod_critical_points


def test_progress_failure_has_no_complete_and_keeps_original_state(monkeypatch):
    from gfvt_rods.rod_phase import _Equilibria
    pd = model()
    before = pickle.dumps(pd)
    events = []
    def fail(*args):
        raise RuntimeError("synthetic equilibrium failure")
    monkeypatch.setattr(_Equilibria, "pairs", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        pd.GFVT_rods(0, L_rod=20e-9, coexistence="IN", progress_callback=events.append)
    assert not any(e["stage"] == "complete" for e in events)
    assert pickle.dumps(pd) == before


def test_invalid_or_broken_observer_does_not_damage_a_result():
    pd = model()
    with pytest.raises(TypeError, match="callable"):
        pd.GFVT_rods(0, progress_callback=True)
    def fail(_):
        raise ValueError("observer failure")
    with pytest.warns(RuntimeWarning, match="observer failure"):
        pd.GFVT_rods(0, L_rod=20e-9, coexistence="IN", progress_callback=fail)
    assert pd.rod_branches["IN"]
