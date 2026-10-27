"""Interaction free energy relative to isolated proteins at fixed polymer mu."""

import numpy as np

from ._quadrature import _T, _W
from .electrostatics import ElectrostaticEOS
from .spheres import SphereEOS


def interaction_free_energy(pd, phi, phi_R):
    r"""Return spherical-fluid GFVT interaction free energies per BSA, in kBT.

    With w = Omega*v_BSA/(V*kBT), define
        a_int = [w(phi)-w(0)-f_id(phi)-phi*mu_insert_inf]/phi.
    f_id uses the same one-body convention as the model's reference EOS;
    mu_insert_inf = -integral alpha_phi(0, r) dPi_R(r).
    Thus ideal translational entropy, reservoir background and isolated
    insertion work cancel. The result includes the reference repulsion and
    the collective polymer contribution, not a pair binding constant or the
    free-energy drop of an entire sample on demixing. Negative means favorable.

    phi and phi_R broadcast; phi=0 returns the analytic isolated limit, zero.
    The reservoir quadrature and polymer closures match the binodal solver.
    At fixed phi_R, d(phi*a_int)/dphi is the interaction chemical potential
    relative to ideal proteins plus their isolated polymer insertion cost.
    """
    kind = getattr(pd, "model_kind", None)
    if (getattr(pd, "geometry", "spheres") != "spheres"
            or kind in ("density_scaled_nu", "scaled_exclusion_layer", "polymer_scaling")):
        raise ValueError("Interaction diagnostic requires unscaled spherical GFVT/GFVT_es.")
    phi, phi_R = np.broadcast_arrays(np.asarray(phi, dtype=float),
                                     np.asarray(phi_R, dtype=float))
    nu = getattr(pd, "nu", None)
    crossover_options = dict(kuhn_crossover=getattr(pd, "kuhn_crossover", None),
                             protein_radius=getattr(pd, "Rpr_in", None))
    if kind == "electrostatic":
        m = pd.es_m
        eos = ElectrostaticEOS(pd.q, m, depletion_exponent=pd.es_depletion_exponent,
                               nu=nu, **crossover_options)
        depletion = eos.depletion
    else:
        m = 1.0
        eos = SphereEOS(pd.q, pd.solvent, nu=nu, **crossover_options)
        depletion = lambda r: eos.polymer.qxf(pd.q, r)
    if (not np.all(np.isfinite(phi)) or not np.all(np.isfinite(phi_R))
            or np.any(phi < 0) or np.any(m * phi >= 1) or np.any(phi_R < 0)):
        raise ValueError("Require finite phi_R >= 0 and 0 <= m*phi < 1.")

    reservoir_grid = phi_R[..., None] * _T**4
    qx, dpi = depletion(reservoir_grid)
    weights = (4 * phi_R[..., None] * _T**3) * _W * dpi
    p = phi[..., None]
    x, y = p / (1 - p), m * p / (1 - m * p)
    a, b, c = 3*qx + 3*qx**2, 4.5*qx**2, qx**3
    log_alpha = np.log1p(-p) - (a*x + b*x**2 + c*(y + 3*y**2 + 3*y**3))
    # alpha = 1 - A*phi + C2*phi**2 + C3*phi**3 + ... .
    # The cubic term has m-dependent linear slope in GFVT_es, so the isolated
    # reference must use -alpha_phi(0), not the geometric (1+qx)**3 there.
    A = 1 + a + m*c
    L2 = 0.5 + a + b + 4*m**2*c
    L3 = 1/3 + a + 2*b + 10*m**3*c
    C2 = A**2/2 - L2
    C3 = -L3 + A*L2 - A**3/6
    residual_per_phi = np.divide(
        np.expm1(log_alpha) + A*p, p,
        out=np.zeros_like(log_alpha), where=p > 0,
    )
    dilute = (A*p < 1e-4) & (m*p < 1e-4)
    residual_per_phi = np.where(dilute, C2*p + C3*p**2, residual_per_phi)
    polymer = -np.sum(weights * residual_per_phi, axis=-1)
    eta = m * phi
    reference = (4*eta - 3*eta**2) / (1 - eta)**2
    return {
        "F_interaction_per_BSA_kT": reference + polymer,
        "F_reference_interaction_per_BSA_kT": reference,
        "F_polymer_interaction_per_BSA_kT": polymer,
        "mu_polymer_isolated_kT": np.sum(weights * A, axis=-1),
    }
