"""Public spherocylinder API, coexistence landmarks and plotting contracts."""

import pickle

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gfvt_rods import PD
from gfvt_rods.rod_phase import _Equilibria


def _model(q=2.5, steps=5, **options):
    return PD(1e-9, q*1e-9, 66.4, 35, "good", steps, [.05, .4],
              progress=False, verbose=False, **options)


@pytest.fixture(scope="module")
def phs_long():
    return _model().GFVT_rods(1, L_rod=40e-9, dep_model="phs", coexistence="all")


@pytest.fixture(scope="module")
def int_long():
    return _model().GFVT_rods(1, L_rod=40e-9, coexistence="all")


@pytest.fixture(scope="module")
def phs_short():
    return _model(.3, 7).GFVT_rods(.09, L_rod=40e-9, dep_model="phs", coexistence="all")


@pytest.fixture(scope="module")
def numerical_in():
    return _model(1).GFVT_rods(.5, L_rod=20e-9, dep_model="phs",
                               coexistence="IN", odf="numerical")


def _at_reservoir(model, kind, reservoir):
    rows = [(branch, row) for branch in model.rod_branches[kind]
            for row in np.flatnonzero(np.isclose(branch.phi_R, reservoir,
                                                 rtol=1e-8, atol=1e-10))]
    assert rows, f"No {kind} coexistence recorded at phi_R={reservoir}."
    return rows


def test_constructor_radii_set_diameter_depletant_size_and_actual_rod_volume(phs_long):
    pd = phs_long
    assert pd.D == 2*pd.Rpr_in
    assert pd.sigma == 2*pd.Rpol_in
    assert pd.q == pytest.approx(2.5)
    assert pd.L == pd.L_rod == 40e-9
    assert pd.L_over_D == 20
    volume = np.pi*pd.D**2*pd.L_rod/4 + np.pi*pd.D**3/6
    assert pd.vbsa == pytest.approx(volume*6.022e23)
    assert pd.phix_bsa == pytest.approx(1/(volume*6.022e23))
    assert pd.rod_theory == "book2011"


def test_length_alias_and_default_depletant_are_explicit():
    for length in ({"L": 0}, {"L_rod": 0}, {"L": 0, "L_rod": 0}):
        pd = _model().GFVT_rods(0, coexistence="NN", **length)
        assert pd.L_rod == 0
        assert pd.dep_model == "int"
        assert pd.odf == "gaussian"
    with pytest.raises(ValueError, match="conflicting"):
        _model().GFVT_rods(L_rod=1e-9, L=2e-9)
    with pytest.raises(TypeError):
        _model().GFVT_rods(D=2e-9)


@pytest.mark.parametrize("options", [
    {"L_rod": -1}, {"L_rod": np.nan}, {"L_rod": np.inf},
    {"L_rod": [1]}, {"PhiP_max": -1}, {"PhiP_max": np.nan},
    {"coexistence": "solid"}, {"coexistence": None},
    {"dep_model": "ideal"}, {"odf": "trial"},
    {"n_orient": 8}, {"n_azimuth": 8},
])
def test_invalid_options_fail_before_publishing_results(options):
    pd = _model()
    before = pickle.dumps(pd)
    with pytest.raises(ValueError):
        pd.GFVT_rods(**options)
    assert pickle.dumps(pd) == before


def test_interacting_closure_rejects_conflicting_solvent_or_exponent():
    with pytest.raises(ValueError, match="good-solvent"):
        PD(1e-9, 2.5e-9, 66.4, 35, "theta", 5, [.05, .4],
           progress=False, verbose=False).GFVT_rods()
    with pytest.raises(ValueError, match="good-solvent"):
        _model(nu=.588).GFVT_rods()


def test_phs_isotropic_critical_and_iin_triple_benchmarks(phs_long):
    critical = phs_long.rod_critical_points["II"]
    np.testing.assert_allclose([critical["phi"], critical["phi_R"]],
                               [.04380575226079822, .4120264189197381], atol=2e-10)
    assert critical["stable"]
    triple, = [p for p in phs_long.rod_triple_points if p["kind"] == "IIN"]
    assert triple["phi_R"] == pytest.approx(.5599172772162843, abs=2e-9)
    np.testing.assert_allclose(triple["phi"],
                               [.0028512738625367, .123492538952032, .193935515840400],
                               atol=2e-8)
    for kind in ("II", "IN"):
        assert _at_reservoir(phs_long, kind, triple["phi_R"])


@pytest.mark.parametrize("fixture", ["phs_long", "int_long", "phs_short", "numerical_in"])
def test_branch_units_coexistence_and_continuity(fixture, request):
    pd = request.getfixturevalue(fixture)
    pressure_scale = pd.phix_bsa*8.314*(273.15 + pd.TC)
    for kind, branches in pd.rod_branches.items():
        for branch in branches:
            assert branch.phase_labels == tuple(kind)
            assert branch.phi.shape == (len(branch.phi_R), 2)
            assert np.all(np.diff(branch.phi_R) < 0)
            assert np.all(np.isfinite(branch.phi) & (branch.phi > 0) & (branch.phi < 1))
            assert np.all(branch.phi[:, 0] <= branch.phi[:, 1])
            np.testing.assert_allclose(branch.mu[:, 0], branch.mu[:, 1], atol=1e-8, rtol=0)
            np.testing.assert_allclose(branch.pressure[:, 0], branch.pressure[:, 1],
                                       atol=pressure_scale*1e-8, rtol=0)
            for side, column in (("Sup", 0), ("Drop", 1)):
                values = branch.values
                np.testing.assert_allclose(values[f"Pr_{side}_p"], branch.phi[:, column]*pd.phix_bsa)
                np.testing.assert_allclose(values[f"Pol_{side}_p"],
                                           branch.alpha[:, column]*branch.phi_R*pd.phix_peg)
                np.testing.assert_allclose(values[f"Pi_{side}"], branch.pressure[:, column])
                np.testing.assert_allclose(values[f"P0_{side}"] + values[f"Pp_{side}"],
                                           values[f"Pi_{side}"])
                np.testing.assert_allclose(values[f"Mu0_{side}"] + values[f"MuR_{side}"],
                                           values[f"Mu_{side}"])


def test_interacting_ii_is_retained_as_metastable_against_in(int_long):
    assert int_long.dep_model == "int"
    assert int_long.rod_branches["II"]
    assert not any(np.any(branch.stable) for branch in int_long.rod_branches["II"])
    assert any(np.any(branch.stable) for branch in int_long.rod_branches["IN"])
    fig, ax = plt.subplots()
    try:
        int_long.molplot_phiR("polymers ", "blue", branch="II", ax=ax)
        assert not ax.lines
        int_long.molplot_phiR("polymers ", "blue", branch="II", ax=ax,
                              show_metastable=True)
        assert ax.lines and all(line.get_linestyle() == "--" for line in ax.lines)
    finally:
        plt.close(fig)


def test_short_range_phs_has_nematic_critical_and_inn_triple(phs_short):
    assert phs_short.rod_branches["NN"]
    critical = phs_short.rod_critical_points["NN"]
    assert critical["phi"] == pytest.approx(.452694407, abs=2e-5)
    assert critical["phi_R"] == pytest.approx(.047672304, abs=2e-7)
    assert critical["S"] > .9
    assert critical["stable"]
    triple, = [p for p in phs_short.rod_triple_points if p["kind"] == "INN"]
    assert triple["phi_R"] == pytest.approx(.0557491050806855, abs=2e-8)
    np.testing.assert_allclose(triple["phi"],
                               [.12967107094, .19306564198, .61790461817], atol=3e-7)
    assert triple["phi_R"] > critical["phi_R"]
    assert np.all(np.diff(triple["phi"]) > 0)
    assert triple["S"][0] == 0
    assert np.all(triple["S"][1:] > .9)


def test_absent_branch_is_empty_and_numerical_failure_is_atomic(phs_long, monkeypatch):
    absent = _model().GFVT_rods(0, L_rod=0, coexistence="NN", dep_model="phs")
    assert absent.rod_branches["NN"] == []
    fig, ax = plt.subplots()
    try:
        assert absent.molplot_phiR("spheres ", "blue", branch="NN", ax=ax) is ax
        assert not ax.lines
    finally:
        plt.close(fig)
    pd = pickle.loads(pickle.dumps(phs_long))
    before = pickle.dumps(pd)

    def fail(*args, **kwargs):
        raise RuntimeError("Injected continuation failure")

    monkeypatch.setattr(_Equilibria, "pairs", fail)
    with pytest.raises(RuntimeError, match="Injected"):
        pd.GFVT_rods(.5, L_rod=40e-9, coexistence="IN", dep_model="phs")
    assert pickle.dumps(pd) == before


def test_serialization_and_sphere_reset_clear_rod_metadata(phs_long):
    pd = pickle.loads(pickle.dumps(phs_long))
    np.testing.assert_array_equal(pd.rod_branches["II"][0].phi,
                                  phs_long.rod_branches["II"][0].phi)
    assert pd.rod_orientation_grid == (128, 256)
    with pytest.raises(ValueError, match="branches"):
        pd.to_legacy_list()
    pd.GFVT_full(PhiP_max=10)
    assert pd.geometry == "spheres"
    for name in ("rod_branches", "rod_critical_points", "rod_triple_points",
                 "rod_orientation_grid", "L_over_D", "L_rod", "sigma", "D", "L"):
        assert not hasattr(pd, name), name
    assert len(pd.to_legacy_list()) == 32


@pytest.mark.slow
def test_full_numerical_odf_includes_bare_and_depleted_endpoints(numerical_in):
    bare = _at_reservoir(numerical_in, "IN", 0)
    depleted = _at_reservoir(numerical_in, "IN", .5)
    assert len(bare) == len(depleted) == 1
    bare_branch, bare_row = bare[0]
    np.testing.assert_allclose(bare_branch.phi[bare_row], [.22893104, .25376535],
                               atol=2e-7)
    branch, row = depleted[0]
    np.testing.assert_allclose(branch.phi[row], [.00032759348958, .46899260813944],
                               atol=2e-7)
    assert branch.S[row, 0] == 0
    assert .9 < branch.S[row, 1] < 1
    assert numerical_in.odf == "numerical"


@pytest.mark.parametrize("method,options", [
    ("molplot", {}), ("molplot", {"mass": True}),
    ("molplot_phi", {}), ("molplot_phiR", {}),
])
def test_multibranch_plots_are_unit_consistent_and_do_not_mutate(phs_long, method, options):
    pd = phs_long
    before = pickle.dumps(pd)
    fig, ax = plt.subplots()
    try:
        assert getattr(pd, method)("rods ", "blue", branch="II", ax=ax,
                                    n_tie=1, show_metastable=True, **options) is ax
        branch = pd.rod_branches["II"][0]
        stable = branch.stable
        line = ax.lines[0]
        x, y = branch.phi[:, 0], branch.alpha[:, 0]*branch.phi_R
        if method == "molplot":
            x, y = x*pd.phix_bsa, y*pd.phix_peg
            if options.get("mass"):
                x, y = x*pd.mbsa, y*pd.mpeg
        elif method == "molplot_phiR":
            y = branch.phi_R
        np.testing.assert_allclose(line.get_xdata()[stable], x[stable])
        np.testing.assert_allclose(line.get_ydata()[stable], y[stable])
        assert np.all(np.isnan(line.get_xdata()[~stable]))
        assert len(ax.collections) == 2  # II critical and IIN triple markers.
        assert pickle.dumps(pd) == before
    finally:
        plt.close(fig)


def test_fixed_book_rods_reject_empirical_plot_scaling(phs_long):
    with pytest.raises(ValueError, match="fixed rod dimensions"):
        phs_long.molplot_scale("rods ", "blue", s=1)


@pytest.mark.parametrize("with_status", [False, True])
def test_plotting_distinguishes_uncalculated_from_physically_absent_branch(with_status):
    pd = _model()
    pd.geometry = "spherocylinders"
    pd.rod_branches = {"II": [], "IN": [], "NN": []}
    pd.rod_requested_coexistence = "IN"
    if with_status:
        pd.rod_branch_status = {"IN": "absent"}
    with pytest.raises(ValueError, match="No NN branches were calculated"):
        pd.molplot_phiR("rods ", "blue", branch="NN")
    fig, ax = plt.subplots()
    try:
        assert pd.molplot_phiR("rods ", "blue", branch="IN", ax=ax) is ax
        assert not ax.lines
    finally:
        plt.close(fig)


@pytest.mark.parametrize("q,maximum,kind,reservoir,fractions", [
    (2.5, 1., "IIN", .5599172772162843,
     [.0028512738625367, .123492538952032, .193935515840400]),
    (.3, .09, "INN", .0557491050806855,
     [.12967107094, .19306564198, .61790461817]),
])
def test_in_only_retains_triple_landmarks_on_two_point_output_grid(
        q, maximum, kind, reservoir, fractions):
    pd = _model(q, steps=2).GFVT_rods(
        maximum, L_rod=40e-9, dep_model="phs", coexistence="IN")
    assert pd.rod_critical_points == {}
    assert pd.rod_branches["II"] == []
    assert pd.rod_branches["NN"] == []
    assert pd.rod_requested_coexistence == "IN"
    triple, = pd.rod_triple_points
    assert triple["kind"] == kind
    assert triple["phi_R"] == pytest.approx(reservoir, abs=2e-8)
    np.testing.assert_allclose(triple["phi"], fractions, atol=3e-7)
    rows = _at_reservoir(pd, "IN", triple["phi_R"])
    assert len(rows) == 2
    assert all(branch.stable[row] for branch, row in rows)
    assert np.ptp(triple["mu"]) < 1e-8
    pressure_scale = pd.phix_bsa*8.314*(273.15 + pd.TC)
    assert np.ptp(triple["pressure"])/pressure_scale < 1e-8


def test_in_only_does_not_require_a_same_phase_critical_point(monkeypatch):
    monkeypatch.setattr(_Equilibria, "critical_i", lambda self: None)
    pd = _model(1., steps=2).GFVT_rods(
        .5, L_rod=20e-9, dep_model="phs", coexistence="IN")
    assert pd.rod_critical_points == {}
    assert pd.rod_triple_points == []
    assert _at_reservoir(pd, "IN", 0.)
    assert _at_reservoir(pd, "IN", .5)


def test_large_polymer_ii_critical_uses_requested_reservoir_range():
    # PEG100K-sized q=14.8 and a 10 nm rod (L/D=4). All of the old
    # reservoir seeds were <=1 and missed this independently refined root.
    pd = PD(1e-9, 14.8e-9, 16.5, 100., "good", 3, [.04, .4],
            progress=False, verbose=False)
    pd.GFVT_rods(10., L_rod=8e-9, coexistence="II")
    cp = pd.rod_critical_points["II"]
    np.testing.assert_allclose([cp["phi"], cp["phi_R"]],
                               [.03991730933202692, 9.362326276590201], atol=2e-8)
    assert cp["stable"]
    assert any(np.any(branch.stable) for branch in pd.rod_branches["II"])
    for branch in pd.rod_branches["II"]:
        np.testing.assert_allclose(branch.mu[:, 0], branch.mu[:, 1], atol=1e-8, rtol=0)


def test_large_polymer_below_critical_range_has_ii_empty_and_in_present():
    pd = PD(1e-9, 14.8e-9, 16.5, 100., "good", 2, [.04, .4],
            progress=False, verbose=False)
    pd.GFVT_rods(1., L_rod=8e-9, coexistence="all")
    assert pd.rod_critical_points == {}
    assert pd.rod_branches["II"] == []
    branch, = pd.rod_branches["IN"]
    assert np.all(branch.stable)
    np.testing.assert_allclose(branch.phi[0], [.4155026260744962, .45272287986132026],
                               atol=2e-8)
