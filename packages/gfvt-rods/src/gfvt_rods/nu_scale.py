"""Spherical GFVT with a droplet-side density-dependent polymer exponent.

The phenomenological law is one-sided and smooth,

    nu_eff(phi) = nu * (1 + s * H_w(phi - phi_c)),

where H_w(x)=0 for x<=0 and x*exp(-w/x) for x>0.  The dilute fluid and crystal
use the base nu exactly, while the dense fluid approaches the requested linear
law away from phi_c.  H_w and all derivatives vanish at the join, keeping the
semigrand free energy smooth.  All density derivatives below include the
dependence of both depletion thickness and polymer osmotic pressure on nu.
"""

from copy import copy, deepcopy
from math import factorial
from numbers import Real

import numpy as np
from scipy.optimize import brentq, least_squares
from tqdm import tqdm

from ._quadrature import _T, _W
from .polymer import PolymerScaling
from .smooth_exclusion import smooth_ramp
from .spheres import SphereEOS, critical_endpoint as constant_nu_endpoint


def _real(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or (positive and value <= 0)):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}real scalar.")
    return float(value)


# Truncated Taylor jets.  Coefficient k is f^(k)/k!, so products are ordinary
# convolutions.  This gives stable, exact chain rules through the critical
# third derivative without finite-difference step choices.
def _align(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim < b.ndim:
        a = a.reshape(a.shape + (1,) * (b.ndim - a.ndim))
    elif b.ndim < a.ndim:
        b = b.reshape(b.shape + (1,) * (a.ndim - b.ndim))
    return np.broadcast_arrays(a, b)


def _jconst(value, order):
    value = np.asarray(value)
    out = np.zeros((order + 1,) + value.shape, dtype=np.result_type(value, float))
    out[0] = value
    return out


def _jvar(value, order):
    out = _jconst(value, order)
    if order:
        out[1] = 1
    return out


def _jadd(a, b):
    a, b = _align(a, b)
    return a + b


def _jmul(a, b):
    a, b = _align(a, b)
    order = a.shape[0] - 1
    return np.stack([sum(a[k] * b[n - k] for k in range(n + 1))
                     for n in range(order + 1)])


def _jinv(a):
    a = np.asarray(a)
    order = a.shape[0] - 1
    out = np.empty_like(a)
    out[0] = 1 / a[0]
    for n in range(1, order + 1):
        out[n] = -sum(a[k] * out[n - k] for k in range(1, n + 1)) / a[0]
    return out


def _jdiv(a, b):
    return _jmul(a, _jinv(b))


def _jexp(a):
    a = np.asarray(a)
    order = a.shape[0] - 1
    out = np.empty_like(a)
    out[0] = np.exp(a[0])
    for n in range(1, order + 1):
        out[n] = sum(k * a[k] * out[n - k] for k in range(1, n + 1)) / n
    return out


def _jlog(a):
    a = np.asarray(a)
    order = a.shape[0] - 1
    out = np.empty_like(a)
    out[0] = np.log(a[0])
    for n in range(1, order + 1):
        remainder = sum(a[k] * (n - k) * out[n - k] for k in range(1, n))
        out[n] = (n * a[n] - remainder) / (n * a[0])
    return out


def _jpow(a, exponent):
    return _jexp(_jmul(_jconst(exponent, a.shape[0] - 1), _jlog(a)))


def _free_volume_jet(phi, qx):
    one = _jconst(1.0, phi.shape[0] - 1)
    gap = _jadd(one, -phi)
    y = _jdiv(phi, gap)
    qx2, qx3 = _jmul(qx, qx), _jmul(_jmul(qx, qx), qx)
    a = _jadd(_jadd(_jmul(_jconst(3.0, phi.shape[0] - 1), qx),
                    _jmul(_jconst(3.0, phi.shape[0] - 1), qx2)), qx3)
    b = _jadd(_jmul(_jconst(4.5, phi.shape[0] - 1), qx2),
              _jmul(_jconst(3.0, phi.shape[0] - 1), qx3))
    c = _jmul(_jconst(3.0, phi.shape[0] - 1), qx3)
    y2, y3 = _jmul(y, y), _jmul(_jmul(y, y), y)
    exponent = -_jadd(_jadd(_jmul(a, y), _jmul(b, y2)), _jmul(c, y3))
    return _jmul(gap, _jexp(exponent))


class DensityScaledNuEOS:
    """GFVT EOS for smooth droplet-side ``nu_eff(phi)`` scaling."""

    def __init__(self, q, solvent, nu, s, phi_c, mu_constant=0, *, width=0.1):
        self.q = _real(q, "q", positive=True)
        self.s = _real(s, "s")
        if self.s < 0:
            raise ValueError("s must be non-negative.")
        self.phi_c = _real(phi_c, "phi_c", positive=True)
        if self.phi_c >= 1:
            raise ValueError("phi_c must be less than 1.")
        if solvent != "good":
            raise ValueError("Density-dependent nu is currently defined for solvent='good'.")
        self.solvent = solvent
        self.nu = float(nu)
        self.width = _real(width, "width", positive=True)
        PolymerScaling(solvent, self.nu)
        self.mu_constant = float(mu_constant)
        self.base = SphereEOS(self.q, solvent, self.mu_constant, nu=self.nu)
        limits = self.effective_nu(np.array([0.0, 1.0]))
        if limits[0] < 0.5 or limits[1] >= 1:
            raise ValueError(
                "s makes nu_eff leave 0.5 <= nu_eff < 1 over 0 <= phi_BSA <= 1."
            )

    @staticmethod
    def eta_from_phi(phi):
        return np.asarray(phi)

    def effective_nu(self, phi):
        ramp = smooth_ramp(np.asarray(phi) - self.phi_c, self.width, order=0)[0]
        return self.nu * (1 + self.s * ramp)

    def at(self, phi_R):
        return _DensityScaledNuReservoir(self, phi_R)


class _DensityScaledNuReservoir:
    def __init__(self, eos, phi_R):
        self.eos = eos
        self.phi_R = _real(phi_R, "phi_R", positive=True)
        self.grid = self.phi_R * _T**4
        self.weights = _W * (4 * self.phi_R * _T**3)

    def _polymer_jets(self, phi, order, reservoir, *, scaled=True):
        phi_value = float(phi)
        phi = _jvar(phi_value, order)
        if scaled:
            ramp_derivatives = smooth_ramp(
                phi_value - self.eos.phi_c, self.eos.width, order=order,
            )
            ramp = np.array([
                ramp_derivatives[k] / factorial(k) for k in range(order + 1)
            ])
            nu = _jmul(
                _jconst(self.eos.nu, order),
                _jadd(_jconst(1.0, order), _jmul(_jconst(self.eos.s, order), ramp)),
            )
        else:
            nu = _jconst(self.eos.nu, order)
        gamma = _jdiv(nu, _jadd(_jmul(_jconst(3.0, order), nu), _jconst(-1.0, order)))
        log_r = np.log(np.asarray(reservoir))
        gamma_grid = gamma.reshape(gamma.shape + (1,) * log_r.ndim)
        log_r_jet = _jconst(log_r, order)
        r2g = _jexp(_jmul(_jmul(_jconst(2.0, order), gamma_grid), log_r_jet))
        denominator = _jadd(_jconst(1.0, order), _jmul(_jconst(3.95, order), r2g))
        qx = _jmul(_jconst(0.865 * self.eos.q**0.88, order), _jpow(denominator, -0.44))
        exponent = _jadd(_jmul(_jconst(3.0, order), gamma_grid), _jconst(-1.0, order))
        rpower = _jexp(_jmul(exponent, log_r_jet))
        dpi = _jdiv(
            _jadd(_jconst(1.0, order),
                  _jmul(_jmul(_jconst(4.86, order), gamma_grid), rpower)),
            _jconst(self.eos.q**3, order),
        )
        return phi, qx, dpi

    def _reference_jet(self, phi, order, solid=False):
        p = _jvar(phi, order)
        one = _jconst(1.0, order)
        if solid:
            gap = _jadd(one, _jmul(_jconst(-1 / 0.741, order), p))
            logarithm = _jlog(_jdiv(p, gap))
            bracket = _jadd(_jconst(self.eos.mu_constant + 2.1178, order),
                            _jmul(_jconst(3.0, order), logarithm))
        else:
            gap = _jadd(one, -p)
            excess = _jdiv(
                _jadd(_jmul(_jconst(4.0, order), p),
                      _jmul(_jconst(-3.0, order), _jmul(p, p))),
                _jmul(gap, gap),
            )
            bracket = _jadd(
                _jadd(_jconst(self.eos.mu_constant - 1.0, order), _jlog(p)),
                excess,
            )
        return _jmul(p, bracket)

    def free_derivatives(self, phi, solid=False, order=4):
        if order not in range(5):
            raise ValueError("order must be between 0 and 4.")
        values = np.asarray(phi)
        if values.ndim:
            return np.stack([self.free_derivatives(x, solid=solid, order=order)
                             for x in values.flat], axis=1).reshape((order + 1,) + values.shape)
        value = float(values)
        p, qx, dpi = self._polymer_jets(value, order, self.grid, scaled=not solid)
        alpha = _free_volume_jet(p.reshape(p.shape + (1,)), qx)
        integrand = _jmul(alpha, dpi)
        integral = np.sum(self.weights * integrand, axis=-1)
        coefficients = self._reference_jet(value, order, solid=solid) - integral
        return np.array([factorial(k) * coefficients[k] for k in range(order + 1)])

    def evaluate(self, phi, solid=False):
        derivatives = self.free_derivatives(phi, solid=solid, order=1)
        phi = np.asarray(phi)
        return np.array([derivatives[0], derivatives[1], phi * derivatives[1] - derivatives[0]])

    def slopes(self, phi):
        derivatives = self.free_derivatives(phi, order=3)
        phi = np.asarray(phi)
        return np.array([phi * derivatives[2], derivatives[2] + phi * derivatives[3]])

    def _endpoint_jets(self, phi, order, *, solid=False):
        p, qx, dpi = self._polymer_jets(
            phi, order, np.asarray(self.phi_R), scaled=not solid,
        )
        alpha = _free_volume_jet(p, qx)
        return alpha, qx, dpi

    def partition(self, phi, solid=False):
        values = np.asarray(phi)
        if values.ndim:
            return np.stack([self.partition(x, solid=solid) for x in values.flat], axis=1).reshape((3,) + values.shape)
        alpha = self._endpoint_jets(float(values), 1, solid=solid)[0]
        a, derivative = alpha[0], alpha[1]
        return np.array([a, -derivative, a - float(values) * derivative])

    def qx_at(self, phi):
        values = np.asarray(phi)
        if values.ndim:
            return np.array([self.qx_at(x) for x in values.flat]).reshape(values.shape)
        return float(self._endpoint_jets(float(values), 0)[1][0])

    def dpi_at(self, phi):
        values = np.asarray(phi)
        if values.ndim:
            return np.array([self.dpi_at(x) for x in values.flat]).reshape(values.shape)
        return float(self._endpoint_jets(float(values), 0)[2][0])


def _critical(q, nu, s, width, guess):
    def residual(values):
        phi_c, phi_R = values
        return DensityScaledNuEOS(q, "good", nu, s, phi_c, width=width).at(phi_R).slopes(phi_c)

    solution = least_squares(
        residual, guess, bounds=([1e-5, 1e-8], [0.6, 500]),
        x_scale="jac", gtol=1e-12, ftol=1e-12, xtol=1e-12, max_nfev=400,
    )
    if (not solution.success or np.any(solution.active_mask)
            or np.max(np.abs(solution.fun)) > 2e-8):
        raise RuntimeError(f"Could not resolve the density-scaled-nu critical point at q={q:g}.")
    return np.asarray(solution.x, dtype=float)


def _triple(eos, critical, guess):
    def residual(values):
        gas, liquid, solid, phi_R = values
        state = eos.at(phi_R)
        vg, vl, vs = state.evaluate(gas), state.evaluate(liquid), state.evaluate(solid, solid=True)
        return [vg[1] - vl[1], vl[1] - vs[1], vg[2] - vl[2], vl[2] - vs[2]]

    phi_c = critical[0]
    solution = least_squares(
        residual, guess,
        bounds=([1e-10, phi_c, phi_c, 1e-8],
                [phi_c, 0.741 - 1e-8, 0.741 - 1e-8, 500]),
        x_scale="jac", gtol=1e-12, ftol=1e-12, xtol=1e-12, max_nfev=700,
    )
    values = np.asarray(solution.x, dtype=float)
    if (not solution.success or np.any(solution.active_mask)
            or np.max(np.abs(solution.fun)) > 2e-8
            or not (values[0] < phi_c < values[1] < values[2] < 0.741)):
        raise RuntimeError(f"Could not resolve the density-scaled-nu triple point at q={eos.q:g}.")
    return values


def _critical_endpoint(nu, s, width):
    start = constant_nu_endpoint("good", nu=nu)

    def residual(values):
        fluid, solid, phi_R, q = values
        state = DensityScaledNuEOS(q, "good", nu, s, fluid, width=width).at(phi_R)
        return np.r_[state.slopes(fluid),
                     state.evaluate(fluid)[1:] - state.evaluate(solid, solid=True)[1:]]

    solution = least_squares(
        residual,
        [start["phi_fluid"], start["phi_solid"], start["phi_R"], start["q"]],
        bounds=([1e-5, 0.45, 1e-8, 0.02], [0.6, 0.741 - 1e-8, 500, 10]),
        x_scale="jac", gtol=1e-11, ftol=1e-11, xtol=1e-11, max_nfev=1200,
    )
    fluid, solid, phi_R, q = map(float, solution.x)
    if (not solution.success or np.any(solution.active_mask)
            or np.max(np.abs(solution.fun)) > 3e-8 or fluid >= solid):
        raise RuntimeError("Could not resolve the density-scaled-nu critical endpoint.")
    return {"phi_fluid": fluid, "phi_solid": solid, "phi_R": phi_R, "q": q}


def _spinodals(state, phi_c):
    floor, ceiling = 1e-150, 1 - 1e-7
    # Only Omega'' is needed for dPi/dphi.  Avoid evaluating Omega''' at the
    # extremely dilute bracket, where the ideal-gas term can overflow even
    # though phi*Omega'' remains finite.
    slope = lambda value: value * state.free_derivatives(value, order=2)[2]
    try:
        log_gas = brentq(lambda log_phi: slope(np.exp(log_phi)),
                         np.log(floor), np.log(phi_c), xtol=1e-12)
        return (np.exp(log_gas),
                brentq(slope, phi_c, ceiling, xtol=1e-14))
    except ValueError:
        # Retain a small fallback scan for a model with additional extrema;
        # the ordinary two-spinodal case avoids hundreds of expensive
        # third-derivative evaluations.
        grid = np.unique(np.r_[np.geomspace(floor, phi_c, 24),
                               np.linspace(phi_c, ceiling, 36)])
        slopes = np.array([slope(value) for value in grid])
        changes = np.flatnonzero(slopes[:-1] * slopes[1:] < 0)
        roots = [brentq(slope, grid[i], grid[i + 1], xtol=1e-14) for i in changes]
        below = [root for root in roots if root < phi_c]
        above = [root for root in roots if root > phi_c]
        if not below or not above:
            raise RuntimeError("Could not bracket both fluid spinodals.")
        return below[-1], above[0]


def _pair(state, phi_c, previous):
    spin_gas, spin_liquid = _spinodals(state, phi_c)
    floor, ceiling = 1e-150, 1 - 1e-7
    lower, upper = np.log([floor, spin_liquid]), np.log([spin_gas, ceiling])
    midpoint = (spin_gas + spin_liquid) / 2
    half_gap = np.sqrt(3) * (spin_liquid - spin_gas) / 2
    seed = [max(midpoint - half_gap, spin_gas / 2),
            min(midpoint + half_gap, (1 + spin_liquid) / 2)]

    def residual(log_phi):
        gas, liquid = np.exp(log_phi)
        return (state.evaluate(gas)[1:] - state.evaluate(liquid)[1:]) / (liquid - gas)

    for guess in (previous, seed):
        if guess is None:
            continue
        solution = least_squares(
            residual, np.clip(np.log(guess), lower + 1e-10, upper - 1e-10),
            bounds=(lower, upper), x_scale="jac", gtol=1e-11, ftol=1e-11,
            xtol=1e-11, max_nfev=250,
        )
        if solution.success and not np.any(solution.active_mask) and np.max(np.abs(solution.fun)) < 2e-8:
            pair = np.exp(solution.x)
            curvatures = [phi * state.free_derivatives(phi, order=2)[2] for phi in pair]
            if np.all(np.asarray(curvatures) > 0):
                return pair
    raise RuntimeError(f"Could not resolve fluid coexistence at phi_R={state.phi_R:.10g}.")


def density_scaled_qxf(q, phi_R, phi, *, nu, s, phi_c, width=0.1):
    """Endpoint depletion thickness and pressure derivative for diagnostics."""
    eos = DensityScaledNuEOS(q, "good", nu, s, phi_c, width=width)
    values = np.broadcast_arrays(np.asarray(phi_R), np.asarray(phi))
    qx = np.empty(values[0].shape)
    dpi = np.empty(values[0].shape)
    for index in np.ndindex(values[0].shape):
        state = eos.at(float(values[0][index]))
        qx[index], dpi[index] = state.qx_at(float(values[1][index])), state.dpi_at(float(values[1][index]))
    if qx.ndim == 0:
        return float(qx), float(dpi)
    return qx, dpi


def calculate_nu_scale(pd, s, PhiP_max, *, nu, width=0.1, critical=None,
                       progress=None, verbose=None):
    """Calculate and install a complete density-scaled-nu phase diagram."""
    s = _real(s, "s")
    if s < 0:
        raise ValueError("s must be non-negative.")
    width = _real(width, "width", positive=True)
    if s == 0:
        pd.nu = nu
        pd.GFVT_full(PhiP_max=PhiP_max, critical=critical, progress=progress, verbose=verbose)
        pd.model_kind, pd.nu_scale_law = "polymer_scaling", "constant"
        return pd
    PhiP_max = _real(PhiP_max, "PhiP_max", positive=True)
    if pd.N_steps < 2:
        raise ValueError("N_steps must be at least 2.")
    progress = pd.progress if progress is None else progress
    verbose = pd.verbose if verbose is None else verbose
    critical_scan = pd.critical if critical is None else critical

    reference = copy(pd)
    reference.nu = nu
    landmarks = reference._calculate_gfvt(
        PhiP_max_override=None, critical=False,
        progress=False, verbose=False, _reference_only=True,
    )
    phi_c, phi_R_c = _critical(pd.q, nu, s, width, landmarks["outguess_c"])
    temperature = 273.15 + pd.TC
    wavelength = 6.626e-34 / np.sqrt(
        2 * np.pi * pd.mbsa * 1000 * 1.38e-23 * temperature / 6.022e23
    )
    mu_constant = np.log(wavelength**3 / pd.vbsa)
    eos = DensityScaledNuEOS(pd.q, "good", nu, s, phi_c, mu_constant, width=width)
    triple_guess = landmarks["outguess_tp"]
    triple = _triple(eos, (phi_c, phi_R_c), triple_guess)
    if PhiP_max < phi_R_c:
        raise ValueError(
            f"PhiP_max={PhiP_max:g} must be >= the recalculated critical reservoir "
            f"concentration {phi_R_c:.10g}."
        )

    cp_list, tp_list, endpoint = [], [], None
    if critical_scan:
        endpoint = _critical_endpoint(nu, s, width)
        cp_rows, tp_rows = [], []
        previous_c = np.array([phi_c, phi_R_c])
        previous_t = triple.copy()
        previous_q = pd.q
        for qscan in tqdm(np.flip(np.linspace(endpoint["q"], 10, pd.N_steps)),
                          desc="Density-scaled nu critical/triple lines", disable=not progress):
            Rpol = qscan * pd.Rpr_in
            phix_peg = 1 / (4 / 3 * np.pi * Rpol**3 * 6.022e23)
            scale = (qscan / previous_q)**(1 / PolymerScaling("good", nu).gamma)
            if np.isclose(qscan, endpoint["q"]):
                cpoint = np.array([endpoint["phi_fluid"], endpoint["phi_R"]])
            else:
                cpoint = _critical(qscan, nu, s, width,
                                   [previous_c[0], previous_c[1] * scale])
            eos_scan = DensityScaledNuEOS(qscan, "good", nu, s, cpoint[0], width=width)
            cp_alpha = eos_scan.at(cpoint[1]).partition(cpoint[0])[0]
            cp_rows.append([qscan, pd.Rpr_in, Rpol, phix_peg, cpoint[0], cpoint[1],
                            cpoint[0] * pd.phix_bsa, cp_alpha * cpoint[1] * phix_peg])
            if qscan > endpoint["q"] * (1 + 1e-10):
                tguess = previous_t.copy()
                tguess[3] *= scale
                tpoint = _triple(eos_scan, cpoint, tguess)
                state = eos_scan.at(tpoint[3])
                pol = [state.partition(phi, solid=(index == 2))[0] * tpoint[3] * phix_peg
                       for index, phi in enumerate(tpoint[:3])]
                tp_rows.append([qscan, pd.Rpr_in, Rpol, phix_peg, *tpoint,
                                *(tpoint[:3] * pd.phix_bsa), *pol])
                previous_t = tpoint
            previous_c, previous_q = cpoint, qscan
        cp_list = np.asarray(cp_rows)
        tp_list = np.asarray(tp_rows).reshape((-1, 14))

    phi_R_values = np.linspace(phi_R_c, PhiP_max, pd.N_steps)
    result = {name: np.empty(pd.N_steps) for name in pd.RETURN_FIELDS
              if name not in ("cp", "tp", "cp_list", "tp_list", "params")}
    phi_drop, phi_sup = np.empty(pd.N_steps), np.empty(pd.N_steps)
    nu_drop, nu_sup = np.empty(pd.N_steps), np.empty(pd.N_steps)
    current, pair = phi_R_c, None
    step_seed = max(phi_R_c, 1e-3) * 0.01
    min_step = max(phi_R_c, 1.0) * 1e-10
    for k, target in enumerate(tqdm(phi_R_values, desc="Density-scaled nu binodal", disable=not progress)):
        attempts = 0
        while current < target:
            step = min(target - current, max(step_seed, (current - phi_R_c) / 2))
            while True:
                attempts += 1
                next_phi_R = min(target, current + step)
                try:
                    next_pair = _pair(eos.at(next_phi_R), phi_c, pair)
                except (ValueError, RuntimeError) as exc:
                    step /= 2
                    if attempts > 180 or step < min_step or current + step == current:
                        raise RuntimeError(
                            f"Density-scaled-nu continuation failed at phi_R={next_phi_R:.10g}."
                        ) from exc
                else:
                    current, pair = next_phi_R, next_pair
                    break
        gas, liquid = (phi_c, phi_c) if pair is None else pair
        state = eos.at(target)
        j = pd.N_steps - 1 - k
        result["phi_R"][j] = target
        phi_sup[j], phi_drop[j] = gas, liquid
        nu_sup[j], nu_drop[j] = eos.effective_nu(gas), eos.effective_nu(liquid)
        for phase, phi in (("Sup", gas), ("Drop", liquid)):
            free, mu, pressure = state.evaluate(phi)
            alpha, g, h = state.partition(phi)
            mu0, p0 = eos.base.reference(phi)
            values = {
                f"Pr_{phase}_p": phi * pd.phix_bsa,
                f"Pol_{phase}_p": alpha * target * pd.phix_peg,
                f"a_{phase}_p": alpha,
                f"qx_{phase}_p": state.qx_at(phi),
                f"dPidPhiP_{phase}_p": state.dpi_at(phi),
                f"Pi_{phase}": pressure * pd.phix_bsa * 8.314 * temperature,
                f"P0_{phase}": p0 * pd.phix_bsa * 8.314 * temperature,
                f"Pp_{phase}": (pressure - p0) * pd.phix_bsa * 8.314 * temperature,
                f"Mu_{phase}": mu,
                f"Mu0_{phase}": mu0,
                f"MuR_{phase}": mu - mu0,
                f"g_{phase}": g,
                f"h_{phase}": h,
            }
            for name, value in values.items():
                result[name][j] = value

    cp_state = eos.at(phi_R_c)
    cp = [phi_c * pd.phix_bsa, cp_state.partition(phi_c)[0] * phi_R_c * pd.phix_peg]
    tp_state = eos.at(triple[3])
    tp = [*(triple[:3] * pd.phix_bsa),
          *[tp_state.partition(phi, solid=(index == 2))[0] * triple[3] * pd.phix_peg
            for index, phi in enumerate(triple[:3])]]

    pd._clear_scaled_rod_metadata()
    pd.geometry, pd.model_kind = "spheres", "density_scaled_nu"
    pd.nu, pd.s = nu, s
    pd.nu_scale_law, pd.nu_scale_width = "smooth_droplet_side", width
    pd.reference_cp_phi = phi_c
    pd.critical_kind = "smooth"
    pd.PhiP_min, pd.PhiP_max = phi_R_c, PhiP_max
    pd.outguess_c, pd.outguess_tp = [phi_c, phi_R_c], list(triple)
    pd.Phi_Drop_p, pd.Phi_Sup_p = phi_drop, phi_sup
    pd.nu_Drop_p, pd.nu_Sup_p = nu_drop, nu_sup
    pd.gamma_Drop_p = nu_drop / (3 * nu_drop - 1)
    pd.gamma_Sup_p = nu_sup / (3 * nu_sup - 1)
    pd.q_Drop_p = np.full(pd.N_steps, pd.q)
    pd.q_Sup_p = np.full(pd.N_steps, pd.q)
    for name, value in landmarks.items():
        setattr(pd, "reference_" + name, deepcopy(value))
    if endpoint is not None:
        pd.critical_endpoint = endpoint
    result.update(cp=cp, tp=tp, cp_list=cp_list, tp_list=tp_list, params=pd._build_params())
    pd._assign_return_values([result[name] for name in pd.RETURN_FIELDS])
    if verbose:
        print(f"Density-scaled nu: s={s:g}, critical phi={phi_c:.8g}, phi_R={phi_R_c:.8g}; "
              f"triple phi_R={triple[3]:.8g}")
    return pd
