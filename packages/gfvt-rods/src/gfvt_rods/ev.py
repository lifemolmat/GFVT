"""Excluded-volume approximation in volume-fraction and concentration coordinates."""

import numpy as np

from .pd import alpha_spt, qxf_external


def _ev_volume_fractions(Rpr_in, Rpol_in, PhiP_min, PhiP_max, HS=True):
    if not all(np.isfinite(r) and r > 0 for r in (Rpr_in, Rpol_in)):
        raise ValueError("Protein and polymer radii must be positive and finite (metres).")
    if not (np.isfinite(PhiP_min) and np.isfinite(PhiP_max) and 0 < PhiP_min < PhiP_max):
        raise ValueError("The EV range must satisfy 0 < PhiP_min < PhiP_max.")

    q = Rpol_in / Rpr_in
    P = np.arange(PhiP_min, PhiP_max, 1e-3)
    log_phi_protein = 3.65 * P**0.8 - P * (1 + q)**3
    if HS:
        log_phi_protein += np.heaviside(1 - P, 0.5) * np.log(P)
    phi_protein = np.exp(log_phi_protein)
    if np.any(~np.isfinite(phi_protein) | (phi_protein >= 1)):
        raise ValueError("The EV approximation gives protein volume fractions >= 1; adjust the P range.")

    phi_R = P * q**3
    q_eff, _ = qxf_external(q, phi_R, "good")
    # Free volume must be evaluated before converting either axis to concentration.
    phi_polymer = phi_R * alpha_spt(phi_protein, q_eff)
    return phi_protein, phi_polymer


def EV_plot(Rpr_in, Rpol_in, N_steps, PhiP_min, PhiP_max, pcolor, ax=None):
    """Plot the good-solvent EV curve on dimensionless volume-fraction axes.

    Radii are in metres. The range is in P = phi_R / q**3, as in EV_plot_q.
    N_steps is retained for compatibility; the legacy P spacing is 0.001.
    Return the supplied axes, or the current pyplot axes when ax is omitted.
    """
    import matplotlib.pyplot as plt

    phi_protein, phi_polymer = _ev_volume_fractions(Rpr_in, Rpol_in, PhiP_min, PhiP_max)
    if ax is None:
        ax = plt.gca()
    ax.plot(phi_protein, phi_polymer, "--", color=pcolor)
    return ax


def EV_molplot(
    Rpr_in,
    Rpol_in,
    N_steps,
    PhiP_min,
    PhiP_max,
    plabel,
    pcolor,
    mbsa=0,
    mpeg=0,
    ax=None,
    mass=False,
    HS=True,
):
    """Plot the EV curve in mM (mass=False) or g/L (mass=True) with a dashed line.

    Uses the same good-solvent curve and P range as EV_plot. Radii are in metres;
    mbsa (protein) and mpeg are molecular weights in kDa, required for mass=True.
    Like molplot, plabel and pcolor set the legend label and color, and ax selects
    existing axes. Axis labels, limits, and legend placement remain caller-owned.
    N_steps retains the legacy interface; the P grid spacing is 0.001.
    HS=False omits the low-P correction, matching EV_plot_q_noHS.
    Return the supplied axes, or the current pyplot axes when ax is omitted.
    """
    import matplotlib.pyplot as plt

    if mass and not all(np.isfinite(m) and m > 0 for m in (mbsa, mpeg)):
        raise ValueError("mass=True requires positive mbsa and mpeg molecular weights in kDa.")
    phi_protein, phi_polymer = _ev_volume_fractions(Rpr_in, Rpol_in, PhiP_min, PhiP_max, HS)
    protein_mM = phi_protein / (4 / 3 * np.pi * Rpr_in**3 * 6.022e23)
    polymer_mM = phi_polymer / (4 / 3 * np.pi * Rpol_in**3 * 6.022e23)
    if mass:
        protein_mM = protein_mM * mbsa
        polymer_mM = polymer_mM * mpeg
    if ax is None:
        ax = plt.gca()
    ax.plot(protein_mM, polymer_mM, "--", label=plabel, color=pcolor)
    return ax
