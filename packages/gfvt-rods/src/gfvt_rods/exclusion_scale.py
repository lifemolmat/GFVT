"""PD integration for spherical GFVT with a scaled PEG exclusion layer."""

from copy import copy, deepcopy
from numbers import Real

import numpy as np
from tqdm import tqdm

from .common_tangent import _coexistence
from .exclusion import ExclusionLayerEOS


def calculate_exclusion_scale(pd, s=0, PhiP_max=1, *, width=0.1, critical=None,
                              progress=None, verbose=None):
    if not isinstance(s, Real) or not np.isfinite(s) or s < 0:
        raise ValueError("s must be a finite non-negative real scalar.")
    s = float(s)
    if not isinstance(width, Real) or not np.isfinite(width) or width < 0:
        raise ValueError("width must be a finite non-negative real scalar; use 0 for the sharp law.")
    width = float(width)
    if s == 0:
        pd.GFVT_full(PhiP_max=PhiP_max, critical=critical,
                     progress=progress, verbose=verbose)
        pd.model_kind = "scaled_exclusion_layer"
        pd.exclusion_width = width
        pd.exclusion_law = "smooth_exponential" if width > 0 else "sharp"
        return pd
    if width > 0:
        return _calculate_smooth(pd, s, PhiP_max, width=width, critical=critical,
                                 progress=progress, verbose=verbose)
    if not isinstance(PhiP_max, Real) or not np.isfinite(PhiP_max) or PhiP_max <= 0:
        raise ValueError("PhiP_max must be a finite positive real scalar.")
    if pd.N_steps < 2:
        raise ValueError("N_steps must be at least 2.")
    PhiP_max = float(PhiP_max)
    progress = pd.progress if progress is None else progress
    verbose = pd.verbose if verbose is None else verbose

    # Reuse the ordinary landmarks without computing a second binodal. Work on
    # a shallow copy so a failed solve leaves the previous PD result intact.
    reference = copy(pd)
    landmarks = reference._calculate_gfvt(
        PhiP_max_override=PhiP_max, critical=critical, progress=progress,
        verbose=False, _reference_only=True,
    )
    phi_c, phi_R_c = landmarks["outguess_c"]
    eos = ExclusionLayerEOS(reference.q, pd.solvent, s, phi_c, nu=getattr(pd, "nu", None))
    phix_bsa, phix_peg = reference.phix_bsa, reference.phix_peg
    T = 273.15 + pd.TC
    Lam = 6.626e-34 / np.sqrt(2 * np.pi * pd.mbsa * 1000 * 1.38e-23 * T / 6.022e23)
    mu_constant = np.log(Lam**3 / reference.vbsa)
    pressure_units = phix_bsa * 8.314 * T
    result = {name: np.empty(pd.N_steps) for name in pd.RETURN_FIELDS
              if name not in ("cp", "tp", "cp_list", "tp_list", "params")}
    phi_drop, phi_sup = np.empty(pd.N_steps), np.empty(pd.N_steps)
    boundary, fractions = np.empty(pd.N_steps, bool), np.empty(pd.N_steps)
    previous = None
    if verbose:
        print(f"Scaled PEG exclusion: s={s:g}, reference phi_c={phi_c:.8g}, "
              f"onset phi_R={phi_R_c:.8g}; particle sizes fixed")
    for k, phi_R in enumerate(tqdm(np.linspace(phi_R_c, PhiP_max, pd.N_steps),
                                  desc="Scaled exclusion binodal", disable=not progress)):
        reservoir = eos.at(phi_R)
        if k == 0 or phi_R == phi_R_c:
            gas, liquid, at_corner, fraction = phi_c, phi_c, True, 0.0
        else:
            try:
                gas, liquid, at_corner, fraction = _coexistence(reservoir, previous)
            except (ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Could not resolve scaled-exclusion coexistence at "
                    f"phi_R={phi_R:.10g}, s={s:g}: {exc}"
                ) from exc
        previous = (gas, liquid)
        j = pd.N_steps - 1 - k
        result["phi_R"][j] = phi_R
        phi_sup[j], phi_drop[j] = gas, liquid
        boundary[j], fractions[j] = at_corner, fraction
        for phase, phi in (("Sup", gas), ("Drop", liquid)):
            if phase == "Sup":
                _, mu, pressure = reservoir.gas.evaluate(phi)
                alpha, g, h = reservoir.gas.partition(phi)
                qx = reservoir.qx
            elif at_corner:
                # At the cusp, the supporting slope is a subgradient between
                # the unmodified and modified one-sided derivatives.
                _, mu, pressure = (
                    (1 - fraction) * reservoir.gas.evaluate(phi_c)
                    + fraction * reservoir.evaluate(phi_c)
                )
                alpha, g, h = (
                    (1 - fraction) * reservoir.gas.partition(phi_c)
                    + fraction * reservoir.partition(phi_c)
                )
                qx = reservoir.qx_at(phi_c)
            else:
                _, mu, pressure = reservoir.evaluate(phi)
                alpha, g, h = reservoir.partition(phi)
                qx = reservoir.qx_at(phi)
            mu0, p0 = eos.base.reference(phi)
            for key, value in {
                f"Pr_{phase}_p": phi * phix_bsa,
                f"Pol_{phase}_p": alpha * phi_R * phix_peg,
                f"a_{phase}_p": alpha,
                f"qx_{phase}_p": qx,
                f"dPidPhiP_{phase}_p": reservoir.dpi,
                f"Pi_{phase}": pressure * pressure_units,
                f"P0_{phase}": p0 * pressure_units,
                f"Pp_{phase}": (pressure - p0) * pressure_units,
                f"Mu_{phase}": mu + mu_constant,
                f"Mu0_{phase}": mu0 + mu_constant,
                f"MuR_{phase}": mu - mu0,
                f"g_{phase}": g,
                f"h_{phase}": h,
            }.items():
                result[key][j] = value

    pd._clear_scaled_rod_metadata()
    for name in ("D", "L", "aspect_ratio", "depletion_model"):
        pd.__dict__.pop(name, None)
    pd.geometry = "spheres"
    pd.model_kind = "scaled_exclusion_layer"
    pd.exclusion_width, pd.exclusion_law = 0.0, "sharp"
    pd.s = s
    pd.q, pd.vbsa, pd.vpeg = reference.q, reference.vbsa, reference.vpeg
    pd.phix_bsa, pd.phix_peg = phix_bsa, phix_peg
    pd.PhiP_min, pd.PhiP_max = phi_R_c, PhiP_max
    pd.outguess_c, pd.outguess_tp = list(landmarks["outguess_c"]), None
    for name, value in landmarks.items():
        setattr(pd, "reference_" + name, deepcopy(value))
    pd.reference_cp_phi = phi_c
    pd.onset = list(landmarks["cp"])
    pd.critical_kind = "one-sided onset at the imposed exclusion threshold"
    pd.Phi_Drop_p, pd.Phi_Sup_p = phi_drop, phi_sup
    pd.boundary_Drop_p, pd.corner_fraction_p = boundary, fractions
    pd.q_Drop_p = np.full(pd.N_steps, pd.q)
    pd.q_Sup_p = np.full(pd.N_steps, pd.q)
    result.update(cp=None, tp=None, cp_list=[], tp_list=[], params=pd._build_params())
    pd._assign_return_values([result[name] for name in pd.RETURN_FIELDS])
    return pd


def _calculate_smooth(pd, s, PhiP_max, *, width, critical, progress, verbose):
    """Continue the smooth fluid branch, publishing only a completed result."""
    from .smooth_exclusion import SmoothExclusionLayerEOS, stable_pair

    if not isinstance(PhiP_max, Real) or not np.isfinite(PhiP_max) or PhiP_max <= 0:
        raise ValueError("PhiP_max must be a finite positive real scalar.")
    if pd.N_steps < 2:
        raise ValueError("N_steps must be at least 2.")
    PhiP_max = float(PhiP_max)
    progress = pd.progress if progress is None else progress
    verbose = pd.verbose if verbose is None else verbose
    reference = copy(pd)
    landmarks = reference._calculate_gfvt(
        PhiP_max_override=PhiP_max, critical=critical, progress=progress,
        verbose=False, _reference_only=True,
    )
    phi_c, phi_R_c = landmarks["outguess_c"]
    eos = SmoothExclusionLayerEOS(reference.q, pd.solvent, s, phi_c, width=width,
                                 nu=getattr(pd, "nu", None))
    # The correction and every derivative vanish at phi_c. The reference
    # critical point is therefore also a critical point of the smooth EOS.
    derivatives = eos.at(phi_R_c).free_derivatives(phi_c)
    if np.max(np.abs(derivatives[2:4])) > 1e-6 or derivatives[4] <= 0:
        raise RuntimeError("The reference endpoint does not satisfy the smooth critical conditions.")
    phix_bsa, phix_peg = reference.phix_bsa, reference.phix_peg
    T = 273.15 + pd.TC
    Lam = 6.626e-34 / np.sqrt(2 * np.pi * pd.mbsa * 1000 * 1.38e-23 * T / 6.022e23)
    mu_constant = np.log(Lam**3 / reference.vbsa)
    pressure_units = phix_bsa * 8.314 * T
    result = {name: np.empty(pd.N_steps) for name in pd.RETURN_FIELDS
              if name not in ("cp", "tp", "cp_list", "tp_list", "params")}
    phi_drop, phi_sup = np.empty(pd.N_steps), np.empty(pd.N_steps)
    breaks = np.zeros(pd.N_steps, dtype=bool)
    current, pair = phi_R_c, None
    seed_step = max(phi_R_c, 1e-3) * 0.01
    min_step = max(phi_R_c, 1.0) * 1e-10
    transition_resolution = max(phi_R_c, 1.0) * 1e-6
    if verbose:
        print(f"Smooth PEG exclusion: s={s:g}, width={width:g}; "
              f"critical phi={phi_c:.8g}, phi_R={phi_R_c:.8g}")

    for k, target in enumerate(tqdm(np.linspace(phi_R_c, PhiP_max, pd.N_steps),
                                   desc="Smooth exclusion binodal", disable=not progress)):
        j = pd.N_steps - 1 - k
        attempts = 0
        while current < target:
            step = min(target - current, max(seed_step, (current - phi_R_c) / 2))
            while True:
                attempts += 1
                if attempts > 250:
                    raise RuntimeError(f"Smooth-exclusion continuation stalled at phi_R={current:.10g}.")
                next_phi_R = min(target, current + step)
                try:
                    next_pair, switched = stable_pair(eos.at(next_phi_R), pair)
                    # Resolve any additional transition before flagging a
                    # finite jump. Smoothing the EOS does not remove genuine
                    # crossings between competing dense-fluid free energies.
                    old_dense = phi_c if pair is None else pair[1]
                    jumping = switched and abs(next_pair[1] - old_dense) > 1e-5
                    if jumping and step > transition_resolution:
                        step /= 2
                        continue
                except (RuntimeError, ValueError) as exc:
                    step /= 2
                    if step < min_step or current + step == current:
                        raise RuntimeError(
                            f"Could not resolve smooth-exclusion coexistence at phi_R={next_phi_R:.10g}, "
                            f"s={s:g}, width={width:g}: {exc}"
                        ) from exc
                else:
                    current, pair = next_phi_R, next_pair
                    breaks[j] |= jumping
                    break
        gas, liquid = (phi_c, phi_c) if pair is None else pair
        state = eos.at(target)
        result["phi_R"][j] = target
        phi_sup[j], phi_drop[j] = gas, liquid
        for phase, phi in (("Sup", gas), ("Drop", liquid)):
            _, mu, pressure = state.evaluate(phi)
            alpha, g, h = state.partition(phi)
            mu0, p0 = eos.base.reference(phi)
            for key, value in {
                f"Pr_{phase}_p": phi * phix_bsa,
                f"Pol_{phase}_p": alpha * target * phix_peg,
                f"a_{phase}_p": alpha,
                f"qx_{phase}_p": state.qx_at(phi),
                f"dPidPhiP_{phase}_p": state.dpi,
                f"Pi_{phase}": pressure * pressure_units,
                f"P0_{phase}": p0 * pressure_units,
                f"Pp_{phase}": (pressure - p0) * pressure_units,
                f"Mu_{phase}": mu + mu_constant,
                f"Mu0_{phase}": mu0 + mu_constant,
                f"MuR_{phase}": mu - mu0,
                f"g_{phase}": g,
                f"h_{phase}": h,
            }.items():
                result[key][j] = value

    pd._clear_scaled_rod_metadata()
    for name in ("D", "L", "aspect_ratio", "depletion_model"):
        pd.__dict__.pop(name, None)
    pd.geometry, pd.model_kind = "spheres", "scaled_exclusion_layer"
    pd.s, pd.exclusion_width, pd.exclusion_law = s, width, "smooth_exponential"
    pd.q, pd.vbsa, pd.vpeg = reference.q, reference.vbsa, reference.vpeg
    pd.phix_bsa, pd.phix_peg = phix_bsa, phix_peg
    pd.PhiP_min, pd.PhiP_max = phi_R_c, PhiP_max
    pd.outguess_c, pd.outguess_tp = list(landmarks["outguess_c"]), None
    for name, value in landmarks.items():
        setattr(pd, "reference_" + name, deepcopy(value))
    pd.reference_cp_phi = phi_c
    pd.critical_kind = "smooth"
    pd.Phi_Drop_p, pd.Phi_Sup_p = phi_drop, phi_sup
    pd.coexistence_breaks_p = breaks
    pd.q_Drop_p = np.full(pd.N_steps, pd.q)
    pd.q_Sup_p = np.full(pd.N_steps, pd.q)
    # At each scanned q the law is anchored at that q's ordinary critical
    # density, so those critical coordinates remain exact. Crystal/triple
    # results remain reference-only because no modified crystal EOS is used.
    result.update(cp=deepcopy(landmarks["cp"]), tp=None,
                  cp_list=deepcopy(landmarks["cp_list"]), tp_list=[], params=pd._build_params())
    pd._assign_return_values([result[name] for name in pd.RETURN_FIELDS])
    return pd
