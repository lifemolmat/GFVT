"""Independent thermodynamic and book-figure checks for the rod backend."""

import numpy as np
import pytest
from scipy.optimize import root
from scipy.special import expit

from gfvt_rods.rod_thermo import NoOrderedPhase, RodConvergenceError, RodEOS


def _coexistence(eos, phi_R, guess):
    reservoir = eos.at(phi_R)

    def residual(z):
        phases = 0.95*expit(z)
        left = reservoir.evaluate(phases[0], "I")
        right = reservoir.evaluate(phases[1], "N")
        return [left.mu - right.mu, left.pressure - right.pressure]

    guess = np.asarray(guess)
    fit = root(residual, np.log(guess/(0.95 - guess)), tol=1e-10)
    assert fit.success
    assert np.max(abs(fit.fun)) < 1e-8
    phases = 0.95*expit(fit.x)
    return phases, reservoir.evaluate(phases[1], "N")


def test_geometry_and_exact_cylindrical_interacting_depletion():
    eos = RodEOS(2., 40., 2.5)
    assert eos.gamma == 21
    assert eos.q == 2.5
    assert eos.ratio == pytest.approx((3*21 - 1)/(2*2.5**3))
    qx, dpi = eos.depletion(0.6)
    x = 1.071*2.5/np.sqrt(1 + 3.95*0.6**1.54)
    expected = np.sqrt(1 + 2*x - 0.51215*x**1.6 + 1.214*x**1.77) - 1
    assert qx == pytest.approx(expected, rel=1e-13)
    assert dpi == pytest.approx(eos.ratio*(1 + 3.7422*0.6**1.31))


@pytest.mark.parametrize("model", ["phs", "int"])
@pytest.mark.parametrize("phase", ["I", "N"])
def test_envelope_derivatives_and_pressure_identity(model, phase):
    reservoir = RodEOS(1., 20., 0.5, model).at(0.2)
    phi, step = 0.3, 2e-6
    state = reservoir.evaluate(phi, phase)
    derivative = (reservoir.evaluate(phi + step, phase).omega
                  - reservoir.evaluate(phi - step, phase).omega)/(2*step)
    assert state.mu == pytest.approx(derivative, rel=3e-9)
    assert state.pressure == pytest.approx(phi*state.mu - state.omega, rel=1e-13)


def test_phs_integral_and_reservoir_partition_identity():
    reservoir = RodEOS(1., 20., 0.5, "phs").at(0.2)
    state = reservoir.evaluate(0.3, "N")
    bare = RodEOS(1., 20., 0.5, "phs").at(0)
    same_odf_reference = bare._make_state(
        0.3, "N", state.entropy, state.rho, state.S, state.kappa)
    assert state.omega == pytest.approx(
        same_odf_reference.omega - 0.2*reservoir.eos.ratio*state.alpha)
    h = 1e-6
    low = reservoir.eos.at(0.2 - h).evaluate(0.3, "N")
    high = reservoir.eos.at(0.2 + h).evaluate(0.3, "N")
    assert (high.omega - low.omega)/(2*h) == pytest.approx(
        -state.alpha*state.dpi, rel=2e-8)


def test_gaussian_whole_potential_ordering_and_no_spurious_low_density_branch():
    eos = RodEOS(1., 10., 0.5, "phs")
    bare = eos.at(0).evaluate(0.3, "N")
    depleted = eos.at(0.3).evaluate(0.3, "N")
    assert depleted.kappa > bare.kappa
    assert depleted.orientation_residual < 1e-9
    with pytest.raises(NoOrderedPhase):
        eos.at(0).evaluate(0.01, "N")
    with pytest.raises(NoOrderedPhase):
        RodEOS(1., 0., 0.5).at(1).evaluate(0.4, "N")


def test_fig_6_6_gaussian_benchmark():
    phases, state = _coexistence(RodEOS(1., 10., 0.5, "phs"),
                                 0.3, [0.146, 0.368])
    np.testing.assert_allclose(phases, [0.14629757, 0.36802503], atol=1e-8)
    assert state.S == pytest.approx(0.9615106189, abs=1e-9)


@pytest.mark.slow
def test_fig_6_6_numerical_odf_and_quadrature_refinement():
    coarse = RodEOS(1., 10., 0.5, "phs", "numerical")
    fine = RodEOS(1., 10., 0.5, "phs", "numerical",
                  n_orient=256, n_azimuth=512)
    phases, state = _coexistence(coarse, 0.3, [0.1414, 0.363])
    refined, refined_state = _coexistence(fine, 0.3, phases)
    np.testing.assert_allclose(refined, [0.14143608, 0.36297045], atol=1e-8)
    np.testing.assert_allclose(phases, refined, atol=1e-5)
    assert state.S == pytest.approx(refined_state.S, abs=1e-5)
    assert refined_state.S == pytest.approx(0.95484761, abs=1e-8)
    assert state.orientation_residual < 1e-10
    assert refined_state.orientation_residual < 1e-10
    assert np.isnan(state.kappa)
    assert state.odf_p.shape == (128,)
    assert not state.odf_p.flags.writeable


def test_numerical_absent_branch_and_interacting_envelope():
    eos = RodEOS(1., 10., 0.5, "phs", "numerical")
    with pytest.raises(NoOrderedPhase):
        eos.at(0).evaluate(0.01, "N")
    reservoir = RodEOS(1., 20., 0.5, "int", "numerical").at(0.2)
    phi, step = 0.3, 2e-6
    state = reservoir.evaluate(phi, "N")
    hi = reservoir.evaluate(phi + step, "N", seed=state)
    lo = reservoir.evaluate(phi - step, "N", seed=state)
    assert max(s.orientation_residual for s in (state, hi, lo)) < 1e-10
    assert state.mu == pytest.approx((hi.omega - lo.omega)/(2*step), rel=1e-8)
    assert state.pressure == pytest.approx(phi*state.mu - state.omega, rel=1e-13)


def test_grossly_unresolved_numerical_nematic_raises_resolution_error():
    eos = RodEOS(1., 20., 0.15, "phs", "numerical")
    with pytest.raises(RodConvergenceError, match=r"Unresolved.*n_orient=128"):
        eos.at(0.055749105080685535).evaluate(0.6179046181684946, "N")


def test_fig_6_10_interacting_benchmark_and_polymer_quadrature():
    eos = RodEOS(1., 20., 1.25, "int")
    phases, _ = _coexistence(eos, 0.6, [0.008, 0.275])
    np.testing.assert_allclose(phases, [0.008018695, 0.274775276], atol=2e-8)
    fine = RodEOS(1., 20., 1.25, "int", n_poly=192)
    refined, _ = _coexistence(fine, 0.6, phases)
    np.testing.assert_allclose(phases, refined, atol=1e-9)


def test_isotropic_critical_slopes_and_sphere_limit():
    reservoir = RodEOS(1., 20., 1.25, "phs").at(0.4120264189197381)
    np.testing.assert_allclose(
        reservoir.isotropic_slopes(0.04380575226079822), 0., atol=2e-8)
    sphere = RodEOS(1., 0., 0.5, "phs").at(0).evaluate(0.2)
    y = 0.2/0.8
    assert sphere.pressure == pytest.approx(y + 3*y*y + 3*y**3)


@pytest.mark.parametrize("kwargs", [
    {"D": 0}, {"L": -1}, {"Rpol": 0}, {"dep_model": "ideal"},
    {"odf": "trial"}, {"n_orient": 8}, {"n_poly": 3.5},
])
def test_validation(kwargs):
    args = dict(D=1, L=10, Rpol=0.5)
    args.update(kwargs)
    with pytest.raises(ValueError):
        RodEOS(**args)
