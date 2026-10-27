"""SPT/GFVT thermodynamics of spherocylinders and polymeric depletants.

The reference is Lekkerkerker and Tuinier, *Colloids and the Depletion
Interaction* (2011), Chapter 6, Eqs. 6.27--6.53.  In particular ``int``
uses the cylindrical depletion-layer interpolation of Eq. 6.50, rather
than a spherical curvature approximation.  Length L excludes the caps.

Free-energy density and pressure are in kT/v_rod; chemical potentials
are in kT.  Both orientation models minimize the complete semi-grand
potential at the specified reservoir concentration.
"""

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp


class NoOrderedPhase(ValueError):
    """No interior, prolate nematic minimum exists at this state point."""


class RodConvergenceError(RuntimeError):
    """The orientation calculation failed to reach a stationary solution."""


@dataclass(frozen=True)
class RodState:
    omega: float
    mu: float
    pressure: float
    alpha: float
    rho: float
    S: float
    kappa: float
    entropy: float
    mu0: float
    pressure0: float
    qx: float
    dpi: float
    g: float
    h: float
    orientation_residual: float = 0.0
    odf_p: Optional[np.ndarray] = None
    phi: float = np.nan
    phi_R: float = np.nan
    phase: str = "I"


def _scalar(value, name, *, positive=False):
    if (not np.isscalar(value) or not np.isreal(value)
            or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
        requirement = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {requirement} real scalar.")
    return float(value)


def _order(value, name, minimum):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return int(value)


@lru_cache(maxsize=12)
def _quadrature(n):
    x, w = leggauss(n)
    x, w = (x + 1) / 2, w / 2
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


@lru_cache(maxsize=8)
def _orientation_grid(n_orient, n_azimuth):
    x, w = _quadrature(n_orient)
    u, wu = _quadrature(n_azimuth)
    xx = x[:, None] * x[None, :]
    ss = np.outer(np.sqrt(1 - x*x), np.sqrt(1 - x*x))
    kernel = np.zeros((n_orient, n_orient))
    for angle, weight in zip(np.pi*u, wu):
        dot = xx + ss*np.cos(angle)
        kernel += weight*np.sqrt(np.maximum(0, 1 - dot*dot))
    kernel.setflags(write=False)
    return x, w, kernel


class RodEOS:
    """Hard-spherocylinder reference plus PHS or good-solvent polymers.

    D, L and Rpol must use one consistent length unit.  ``q`` is the
    nominal size ratio 2*Rpol/D; depletion-layer curvature does not alter
    the polymer reference volume.  ``odf='numerical'`` resolves the full
    axisymmetric orientational distribution rather than a trial function.
    Strongly aligned states, especially nematic--nematic coexistence,
    require angular-grid refinement using n_orient and n_azimuth.  An
    Euler-equation residual checks stationarity on that grid, not its
    continuum accuracy; grossly unresolved distributions raise an error.
    """

    def __init__(self, D, L, Rpol, dep_model="int", odf="gaussian", *,
                 n_orient=128, n_azimuth=256, n_poly=96):
        self.D = _scalar(D, "D", positive=True)
        self.L = _scalar(L, "L")
        self.Rpol = _scalar(Rpol, "Rpol", positive=True)
        if dep_model not in ("phs", "int"):
            raise ValueError("dep_model must be 'phs' or 'int'.")
        if odf not in ("gaussian", "numerical"):
            raise ValueError("odf must be 'gaussian' or 'numerical'.")
        self.dep_model, self.odf = dep_model, odf
        self.n_orient = _order(n_orient, "n_orient", 16)
        self.n_azimuth = _order(n_azimuth, "n_azimuth", 16)
        self.n_poly = _order(n_poly, "n_poly", 8)
        self.gamma = 1 + self.L/self.D
        self.q = 2*self.Rpol/self.D
        self.volume = np.pi*self.D**2*self.L/4 + np.pi*self.D**3/6
        self.polymer_volume = 4*np.pi*self.Rpol**3/3
        self.ratio = self.volume/self.polymer_volume
        den = 3*self.gamma - 1
        self.A1 = 3*(self.gamma - 1)**2/den
        self.B0 = 12*self.gamma*(2*self.gamma - 1)/den**2
        self.B1 = 12*self.gamma*(self.gamma - 1)**2/den**2
        self.A, self.B = 3 + self.A1, self.B0 + self.B1
        self._reservoirs = OrderedDict()

    def depletion(self, phi_R):
        """Return effective insertion size and rod-scaled compressibility."""
        t = np.asarray(phi_R, dtype=float)
        if np.any(~np.isfinite(t)) or np.any(t < 0):
            raise ValueError("phi_R must be finite and non-negative.")
        if self.dep_model == "phs":
            qx = np.full_like(t, self.q)
            dpi = np.full_like(t, self.ratio)
        else:
            x = 1.071*self.q/np.sqrt(1 + 3.95*t**1.54)
            # Rationalized sqrt(...)-1 preserves small depletion layers.
            delta = 2*x - 0.51215*x**1.6 + 1.214*x**1.77
            qx = delta/(np.sqrt(1 + delta) + 1)
            dpi = self.ratio*(1 + 3.7422*t**1.31)
        return qx, dpi

    def coefficients(self, qx):
        """Return a, b0, b1, c0, c1, with b=b0+b1*rho."""
        qx = np.asarray(qx)
        g, den = self.gamma, 3*self.gamma - 1
        a = (6*g*qx + 3*(g + 1)*qx**2 + 2*qx**3)/den
        b0 = 0.5*(6*g/den)**2*qx**2 + 6/den*qx**3
        b1 = 6*(g - 1)**2/den**2*qx**3
        c0, c1 = 2*self.B0/den*qx**3, 2*self.B1/den*qx**3
        return a, b0, b1, c0, c1

    def at(self, phi_R):
        phi_R = _scalar(phi_R, "phi_R")
        if phi_R in self._reservoirs:
            self._reservoirs.move_to_end(phi_R)
            return self._reservoirs[phi_R]
        reservoir = RodReservoir(self, phi_R)
        self._reservoirs[phi_R] = reservoir
        if len(self._reservoirs) > 12:
            self._reservoirs.popitem(last=False)
        return reservoir


class RodReservoir:
    """Thermodynamics at one fixed reservoir polymer concentration.

    ``qx``/``dpi`` and each returned state's ``alpha``/``g``/``h`` are
    endpoint quantities. ``weights`` already include d(phi_R)*dpi for
    evaluating the semi-grand integral.
    """

    def __init__(self, eos, phi_R):
        self.eos, self.phi_R = eos, phi_R
        if eos.dep_model == "phs":
            self.nodes_qx = np.array([eos.q])
            self.weights = np.array([phi_R*eos.ratio])
        else:
            u, w = _quadrature(eos.n_poly)
            nodes = phi_R*u**4
            self.nodes_qx, dpi = eos.depletion(nodes)
            self.weights = w*(4*phi_R*u**3)*dpi
        self.coefficients = eos.coefficients(self.nodes_qx)
        qx, dpi = eos.depletion(phi_R)
        self.qx, self.dpi = float(qx), float(dpi)
        self.endpoint_coefficients = eos.coefficients(self.qx)
        self._states = OrderedDict()
        self._last_ordered = None

    @staticmethod
    def _parts(phi, rho, coefficients):
        y = phi/(1 - phi)
        a, b0, b1, c0, c1 = coefficients
        b, c = b0 + b1*rho, c0 + c1*rho
        beta = np.exp(-(a*y + b*y*y + c*y**3))
        Q1 = a + 2*b*y + 3*c*y*y
        alpha = (1 - phi)*beta
        g = beta*(1 + (1 + y)*Q1)
        h = beta*(1 + y*Q1)
        return y, alpha, g, h, beta, Q1, b, c

    def _orientation_terms(self, phi):
        y = phi/(1 - phi)
        a, b0, b1, c0, c1 = self.coefficients
        C = phi*(self.eos.A1*y + 0.5*self.eos.B1*y*y)
        E = b1*y*y + c1*y**3
        alpha0 = (1 - phi)*np.exp(-(a*y + b0*y*y + c0*y**3))
        return C, E, alpha0

    def _gaussian_minima(self, phi):
        if self.eos.L == 0:
            raise NoOrderedPhase("A sphere has no distinct nematic branch.")
        C, E, alpha0 = self._orientation_terms(phi)

        def derivative(rho):
            return (-2*phi/rho + C
                    + np.dot(self.weights, E*alpha0*np.exp(-E*rho)))

        if self.phi_R == 0:
            rho = 2*phi/C if C else np.inf
            roots = [rho] if 0 < rho < 1 else []
        else:
            grid = np.unique(np.r_[np.geomspace(1e-12, 0.1, 42),
                                   np.linspace(0.1, 1, 65)])
            values = (-2*phi/grid + C
                      + np.einsum("ij,j->i", np.exp(-grid[:, None]*E),
                                  self.weights*E*alpha0, optimize=False))
            roots = []
            for left, right, f_left, f_right in zip(
                    grid[:-1], grid[1:], values[:-1], values[1:]):
                if f_left < 0 <= f_right:
                    root = brentq(derivative, left, right,
                                  xtol=1e-14, rtol=1e-13)
                    if root < 1 - 1e-9:
                        roots.append(root)
        if not roots:
            raise NoOrderedPhase(
                f"No ordered Gaussian minimum at phi={phi:g}, phi_R={self.phi_R:g}.")
        minima = []
        for rho in roots:
            kappa = 16/(np.pi*rho*rho)
            entropy = np.log(kappa) - 1
            # This is the book's large-kappa Gaussian convention (Eq. 6.25).
            S = 1 - 3/kappa
            residual = abs(derivative(rho))/max(phi, 1e-300)
            minima.append(self._make_state(phi, "N", entropy, rho, S, kappa,
                                           residual))
        return sorted(minima, key=lambda state: state.omega)

    def _numerical_minimum(self, phi, seed):
        if self.eos.L == 0:
            raise NoOrderedPhase("A sphere has no distinct nematic branch.")
        x, w, kernel = _orientation_grid(self.eos.n_orient,
                                         self.eos.n_azimuth)
        C, E, alpha0 = self._orientation_terms(phi)
        theta2 = np.arccos(x)**2

        def product(mass):
            # Avoid multithreaded BLAS overhead for this small matrix.
            return np.einsum("ij,j->i", kernel, mass, optimize=False)

        def normalize(lp):
            return lp - logsumexp(lp, b=w)

        def statistics(lp):
            mass = w*np.exp(lp)
            kp = product(mass)
            rho = 4/np.pi*np.dot(mass, kp)
            alpha = alpha0*np.exp(-E*rho)
            J = C + np.dot(self.weights, E*alpha)
            target = normalize(-8*J/(np.pi*phi)*kp)
            return mass, rho, target

        def descend(lp):
            lp = normalize(lp)
            for _ in range(3000):
                mass, rho, target = statistics(lp)
                residual = float(np.max(np.abs(target - lp)))
                if residual < 1e-10:
                    return lp, mass, rho, residual
                lp = normalize(0.8*lp + 0.2*target)
            # A variational fallback helps close to an orientational spinodal.
            # The softmax enforces positive, normalized orientation masses.
            def objective(z):
                logp = normalize(z)
                mass = w*np.exp(logp)
                kp = product(mass)
                rho = 4/np.pi*np.dot(mass, kp)
                alpha = alpha0*np.exp(-E*rho)
                entropy = np.dot(mass, logp)
                J = C + np.dot(self.weights, E*alpha)
                functional = phi*entropy + C*rho - np.dot(self.weights, alpha)
                variation = phi*(logp + 1) + 8*J/np.pi*kp
                gradient = mass*(variation - np.dot(mass, variation))
                return functional, gradient

            fit = minimize(objective, lp, jac=True, method="L-BFGS-B",
                           options={"maxiter": 750, "gtol": 1e-13,
                                    "ftol": 1e-15, "maxls": 40})
            lp = normalize(fit.x)
            # Polish the variational minimum to the Euler-equation tolerance.
            # Fixed-point and variational iterations share one 5000-step
            # budget per seed; a stalled solve never relaxes its tolerance.
            for _ in range(5000 - 3000 - fit.nit):
                mass, rho, target = statistics(lp)
                residual = float(np.max(np.abs(target - lp)))
                if residual < 1e-10:
                    return lp, mass, rho, residual
                lp = normalize(0.9*lp + 0.1*target)
            return None

        seeds = []
        for candidate in (seed, self._last_ordered):
            if isinstance(candidate, RodState):
                if candidate.odf_p is not None:
                    candidate = candidate.odf_p
                elif np.isfinite(candidate.kappa) and candidate.kappa > 0:
                    candidate = candidate.kappa
                else:
                    continue
            if candidate is None:
                continue
            if np.isscalar(candidate):
                if np.isfinite(candidate) and candidate > 0:
                    seeds.append(-float(candidate)*theta2/2)
            else:
                candidate = np.asarray(candidate, dtype=float)
                if (candidate.shape == x.shape and np.all(candidate >= 0)
                        and np.any(candidate > 0) and np.all(np.isfinite(candidate))):
                    seeds.append(np.log(np.maximum(candidate, 1e-300)))
        try:
            gaussian = self._gaussian_minima(phi)
            seeds.extend(-min(state.kappa, 1e5)*theta2/2 for state in gaussian)
        except NoOrderedPhase:
            pass
        # Multiple initial distributions distinguish an absent ordered branch
        # from an iteration that happened to start in the isotropic basin.
        seeds.extend(-kappa*theta2/2 for kappa in (5., 30., 160.))
        solutions, failed = [], False
        for initial in seeds:
            solution = descend(initial)
            if solution is None:
                failed = True
                continue
            lp, mass, rho, residual = solution
            S = float(np.dot(mass, (3*x*x - 1)/2))
            if S <= 1e-4 or rho >= 1 - 1e-7:
                continue
            if any(abs(S - old.S) < 1e-6 for old in solutions):
                continue
            p = np.exp(lp)
            p.setflags(write=False)
            solutions.append(self._make_state(
                phi, "N", float(np.dot(mass, lp)), float(rho), S, np.nan,
                residual, p))
        if not solutions:
            if failed:
                raise RodConvergenceError(
                    f"Numerical ODF did not converge at phi={phi:g}, "
                    f"phi_R={self.phi_R:g}.")
            raise NoOrderedPhase(
                f"No ordered numerical ODF at phi={phi:g}, phi_R={self.phi_R:g}.")
        best = min(solutions, key=lambda state: state.omega)
        peak_mass = float(np.max(w*best.odf_p))
        if peak_mass > 0.5:
            raise RodConvergenceError(
                f"Unresolved numerical ODF at phi={phi:g}, "
                f"phi_R={self.phi_R:g}: peak quadrature mass={peak_mass:.3g} "
                f"with n_orient={self.eos.n_orient}, "
                f"n_azimuth={self.eos.n_azimuth}. Increase the angular "
                "resolution (n_orient and n_azimuth) and check refinement.")
        self._last_ordered = best
        return best

    def _make_state(self, phi, phase, entropy, rho, S, kappa,
                    residual=0., odf_p=None):
        eos = self.eos
        y, alpha, g, h, _, _, _, _ = self._parts(phi, rho, self.coefficients)
        A, B = 3 + eos.A1*rho, eos.B0 + eos.B1*rho
        logy = np.log(phi) - np.log1p(-phi)
        reference = phi*(-1 + entropy + logy + A*y + 0.5*B*y*y)
        mu0 = (logy + entropy + (1 + 2*A)*y
               + (A + 1.5*B)*y*y + B*y**3)
        pressure0 = y + A*y*y + B*y**3
        omega = reference - np.dot(self.weights, alpha)
        mu = mu0 + np.dot(self.weights, g)
        pressure = pressure0 + np.dot(self.weights, h)
        _, al_end, g_end, h_end, _, _, _, _ = self._parts(
            phi, rho, self.endpoint_coefficients)
        return RodState(
            omega=float(omega), mu=float(mu), pressure=float(pressure),
            alpha=float(al_end), rho=float(rho), S=float(S),
            kappa=float(kappa), entropy=float(entropy), mu0=float(mu0),
            pressure0=float(pressure0), qx=self.qx, dpi=self.dpi,
            g=float(g_end), h=float(h_end),
            orientation_residual=float(residual), odf_p=odf_p,
            phi=phi, phi_R=self.phi_R, phase=phase)

    def evaluate(self, phi, phase="I", seed=None):
        phi = _scalar(phi, "phi", positive=True)
        if phi >= 1:
            raise ValueError("phi must be strictly between zero and one.")
        if phase not in ("I", "N"):
            raise ValueError("phase must be 'I' or 'N'.")
        key = (phi, phase)
        if key in self._states:
            self._states.move_to_end(key)
            return self._states[key]
        if phase == "I":
            state = self._make_state(phi, phase, 0., 1., 0., 0.)
        elif self.eos.odf == "gaussian":
            state = self._gaussian_minima(phi)[0]
        else:
            state = self._numerical_minimum(phi, seed)
        self._states[key] = state
        if len(self._states) > (64 if self.eos.odf == "numerical" else 256):
            self._states.popitem(last=False)
        return state

    def ordered_minima(self, phi):
        """Return all Gaussian minima, or the resolved numerical minimum."""
        phi = _scalar(phi, "phi", positive=True)
        if phi >= 1:
            raise ValueError("phi must be strictly between zero and one.")
        if self.eos.odf == "gaussian":
            return tuple(self._gaussian_minima(phi))
        return (self.evaluate(phi, "N"),)

    def isotropic_slopes(self, phi):
        """Return dP/dy and d²P/dy² for analytic II critical-point solves."""
        phi = _scalar(phi, "phi", positive=True)
        if phi >= 1:
            raise ValueError("phi must be strictly between zero and one.")
        y, _, _, _, beta, Q1, b, c = self._parts(
            phi, 1., self.coefficients)
        Q2 = 2*b + 6*c*y
        beta2 = beta*(Q1*Q1 - Q2)
        beta3 = beta*(-Q1**3 + 3*Q1*Q2 - 6*c)
        A, B = self.eos.A, self.eos.B
        return np.array([
            1 + 2*A*y + 3*B*y*y - y*np.dot(self.weights, beta2),
            2*A + 6*B*y - np.dot(self.weights, beta2 + y*beta3)])

    slopes = isotropic_slopes

    def partition(self, phi, phase="I", seed=None):
        state = self.evaluate(phi, phase, seed)
        return np.array([state.alpha, state.g, state.h])
