"""Plot independently continued spherocylinder coexistence branches.

Coordinates are calculated from the occupied rod volume and the reservoir
partition coefficient. Separate branches and stability gaps are never joined.
"""

from numbers import Integral

import numpy as np


ROD_BRANCH_COLORS = {"II": "#0072B2", "IN": "#D55E00", "NN": "#CC79A7"}


def _coordinates(pd, phi, phi_R, alpha, coordinates, mass, mbsa, mpeg, peg_scale):
    phi = np.asarray(phi, dtype=float)
    reservoir = np.asarray(phi_R, dtype=float)
    if coordinates == "phiR":
        return phi, np.broadcast_to(reservoir, phi.shape)
    sample_polymer = np.asarray(alpha, dtype=float) * reservoir
    if coordinates == "phi":
        return phi, sample_polymer * peg_scale
    x = phi * pd.phix_bsa
    y = sample_polymer * pd.phix_peg
    if mass:
        x, y = x * mbsa, y * mpeg
    return x, y


def _matching_alpha(branches, phi, phi_R):
    """Look up a calculated landmark; do not invent a partition coefficient."""
    for branch in branches:
        reservoir = np.asarray(branch.phi_R, dtype=float)
        fractions = np.asarray(branch.phi, dtype=float)
        rows = np.flatnonzero(np.isclose(reservoir, phi_R, rtol=1e-6, atol=1e-10))
        for row in rows:
            sides = np.flatnonzero(np.isclose(fractions[row], phi, rtol=1e-5, atol=1e-8))
            if sides.size:
                return float(np.asarray(branch.alpha)[row, sides[0]])
    return np.nan


def plot_rod_branches(
    pd, plabel, pcolor, *, coordinates, ax=None, mass=False, mbsa=0,
    mpeg=0, gamma=None, branch=None, show_metastable=False, n_tie=0,
    crit_color=False, show_legend=False,
):
    """Plot book-model rod results using the public PD plotting conventions.

    branch selects II, IN, NN, or all. Solid lines are stable coexistence;
    show_metastable adds dashed continuations. n_tie is the stride between
    displayed tie lines, with zero disabling ordinary tie lines.
    """
    if not hasattr(pd, "rod_branches"):
        raise RuntimeError("Run GFVT_rods() before plotting rod branches.")
    if not isinstance(n_tie, Integral) or n_tie < 0:
        raise ValueError("n_tie must be a non-negative integer stride.")
    status = getattr(pd, "rod_branch_status", None)
    requested = getattr(pd, "rod_requested_coexistence", "all")
    if status is not None:
        modes = tuple(kind for kind in status if kind in pd.rod_branches)
    elif requested.upper() != "ALL":
        modes = (requested.upper(),)
    else:
        modes = tuple(pd.rod_branches)
    calculated_modes = modes
    if branch is not None:
        if not isinstance(branch, str) or branch.upper() not in ("II", "IN", "NN", "ALL"):
            raise ValueError("branch must be II, IN, NN, all, or None.")
        if branch.upper() != "ALL":
            modes = (branch.upper(),)
            if modes[0] not in calculated_modes:
                raise ValueError(f"No {modes[0]} branches were calculated; rerun GFVT_rods with that coexistence selection.")
    ax = pd._get_axis(ax)
    mbsa, mpeg = pd._plot_masses(mbsa, mpeg)
    gamma = pd._axis_gamma(gamma) if coordinates == "phi" else None
    peg_scale = 1 if gamma is None else pd.q ** (-1 / gamma)
    selected = [item for kind in modes for item in pd.rod_branches[kind]]
    source = f"{getattr(pd, 'dep_model', 'int')}, {getattr(pd, 'odf', 'gaussian')}"
    labels_used = set()

    def label_once(label):
        if label in labels_used:
            return "_nolegend_"
        labels_used.add(label)
        return label

    for kind in modes:
        for item in pd.rod_branches[kind]:
            phi_R = np.asarray(item.phi_R, dtype=float)
            fractions = np.asarray(item.phi, dtype=float)
            partition = np.asarray(item.alpha, dtype=float)
            stable = np.asarray(item.stable, dtype=bool)
            if fractions.shape != (phi_R.size, 2) or partition.shape != fractions.shape or stable.shape != phi_R.shape:
                raise ValueError("Rod branch arrays must have phi/alpha shape (n, 2) and phi_R/stable shape (n,).")
            x, y = _coordinates(pd, fractions, phi_R[:, None], partition,
                                coordinates, mass, mbsa, mpeg, peg_scale)
            valid = np.all(np.isfinite(x) & np.isfinite(y), axis=1)
            styles = [(stable & valid, "-", 1.0, f"{plabel}{kind} ({source})")]
            if show_metastable:
                styles.append((~stable & valid, "--", 0.55, f"{plabel}{kind} metastable ({source})"))
            for mask, linestyle, opacity, label in styles:
                if not np.any(mask):
                    continue
                for side in (0, 1):
                    ax.plot(np.where(mask, x[:, side], np.nan),
                            np.where(mask, y[:, side], np.nan),
                            linestyle, color=pcolor, alpha=opacity,
                            label=label_once(label) if side == 0 else "_nolegend_")
            if n_tie:
                visible = valid & (stable | bool(show_metastable))
                for row in np.flatnonzero(visible)[::n_tie]:
                    ax.plot(x[row], y[row], "--", color=pcolor,
                            alpha=0.5 if stable[row] else 0.25)

    marker_color = crit_color or pcolor
    for kind, point in getattr(pd, "rod_critical_points", {}).items():
        if kind not in modes or (not point.get("stable", True) and not show_metastable):
            continue
        phi, phi_R = float(point["phi"]), float(point["phi_R"])
        alpha = point.get("alpha", _matching_alpha(selected, phi, phi_R))
        x, y = _coordinates(pd, phi, phi_R, alpha, coordinates, mass, mbsa, mpeg, peg_scale)
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        stable = point.get("stable", True)
        ax.scatter(x, y, marker="*", s=180, edgecolors=marker_color,
                   facecolors=marker_color if stable else "none",
                   label=label_once(f"{plabel}{kind} critical"))

    for point in getattr(pd, "rod_triple_points", []):
        if not point.get("stable", True) and not show_metastable:
            continue
        kind = point.get("kind", "")
        phase_kinds = kind.replace("-", "").upper()
        pairs = {phase_kinds[i] + phase_kinds[j]
                 for i in range(len(phase_kinds)) for j in range(i + 1, len(phase_kinds))}
        if pairs and not pairs.intersection(modes):
            continue
        fractions = np.asarray(point["phi"], dtype=float)
        phi_R = float(point["phi_R"])
        partition = point.get("alpha")
        if partition is None:
            partition = [_matching_alpha(selected, phi, phi_R) for phi in fractions]
        x, y = _coordinates(pd, fractions, phi_R, partition, coordinates,
                            mass, mbsa, mpeg, peg_scale)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            continue
        ax.scatter(x[valid], y[valid], marker="^", s=75, color=marker_color,
                   label=label_once(f"{plabel}{kind} triple"))
        if np.all(valid):
            ax.plot(x, y, ":", color=marker_color, alpha=0.7)
    pd._maybe_legend(ax, show_legend)
    return ax


def plot_rod_composition(
    model, *, ax=None, coordinates="reservoir", linestyle="-", color="black",
    branch_colors=None, branch_linestyles=None, show_metastable=False,
    marker=None, markevery=None,
):
    """Draw calculated rod coexistence without running a solver.

    coordinates selects reservoir fraction, system fraction, system mM, or
    system g/L ('reservoir', 'system', 'mM', 'g/L'). System coordinates use
    each phase's partition factor, phi_p=alpha*phi_p^R. mM uses the stored
    geometric reference volumes; g/L additionally uses each result's rod and
    depletant masses in kDa (mM * kDa = g/L).
    branch_colors and branch_linestyles optionally override color and style
    by coexistence kind. Their defaults use the common color and linestyle.
    marker/markevery add open markers to the coexistence curves, for example
    to distinguish a numerical ODF while retaining branch-specific styles.
    """
    coordinate_names = {"reservoir": "phiR", "system": "phi", "mM": "mM", "g/L": "mM"}
    if coordinates not in coordinate_names:
        raise ValueError("coordinates must be 'reservoir', 'system', 'mM', or 'g/L'.")
    if not hasattr(model, "rod_branches"):
        raise RuntimeError("Run GFVT_rods() before plotting rod branches.")
    ax = model._get_axis(ax)

    def convert(phi, phi_R, alpha):
        return _coordinates(model, phi, phi_R, alpha, coordinate_names[coordinates],
                            coordinates == "g/L", model.mbsa, model.mpeg, 1)

    def branch_color(kind):
        return color if branch_colors is None else branch_colors[kind]

    def branch_style(kind):
        return linestyle if branch_linestyles is None else branch_linestyles.get(kind, linestyle)

    for kind, branches in model.rod_branches.items():
        for branch in branches:
            x, y = convert(branch.phi, branch.phi_R[:, None], branch.alpha)
            stable = np.asarray(branch.stable, dtype=bool)
            for side in (0, 1):
                ax.plot(np.where(stable, x[:, side], np.nan),
                        np.where(stable, y[:, side], np.nan),
                        color=branch_color(kind), linestyle=branch_style(kind), linewidth=1.7,
                        marker=marker, markevery=markevery, markerfacecolor="white",
                        markersize=4, markeredgewidth=0.9)
                if show_metastable:
                    ax.plot(np.where(~stable, x[:, side], np.nan),
                            np.where(~stable, y[:, side], np.nan),
                            color=branch_color(kind), linestyle=branch_style(kind),
                            linewidth=1.0, alpha=0.3, marker=marker,
                            markevery=markevery, markerfacecolor="white", markersize=4)
    for kind, point in model.rod_critical_points.items():
        if point.get("stable", True) or show_metastable:
            x, y = convert(point["phi"], point["phi_R"], point["alpha"])
            ax.plot(x, y, "o", color=branch_color(kind), markersize=4,
                    alpha=1.0 if point.get("stable", True) else 0.3)
    for point in model.rod_triple_points:
        if not point.get("stable", True) and not show_metastable:
            continue
        x, y = convert(point["phi"], point["phi_R"], point["alpha"])
        # Three system compositions form a tie triangle. In reservoir
        # coordinates adjacent segments retain the distinct branch colors.
        pairs = ((0, 1), (1, 2)) if coordinates == "reservoir" else ((0, 1), (1, 2), (0, 2))
        for left, right in pairs:
            kind = point["kind"][left] + point["kind"][right]
            if kind in model.rod_branch_status:
                ax.plot(x[[left, right]], y[[left, right]],
                        color=branch_color(kind), linestyle=branch_style(kind),
                        linewidth=1.0, alpha=0.8)
    return ax


def plot_rod_orientation_comparison(
    models, *, limits=None, show_metastable=False, title="Orientation comparison",
):
    """Compare Gaussian/numerical results with an order-parameter inset.

    models maps 'gaussian'/'numerical' to already calculated PD objects.
    limits optionally contains 'fig66' and 'fig66_order' axis settings.
    Returns (figure, main_axis, order_axis); does not call pyplot.show().
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not models or any(name not in ("gaussian", "numerical") for name in models):
        raise ValueError("models must map gaussian/numerical names to calculated PD objects.")
    limits = {} if limits is None else limits
    styles = {"gaussian": "-", "numerical": "--"}
    order_styles = {"gaussian": "-", "numerical": ":"}
    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    for orientation, model in models.items():
        plot_rod_composition(model, ax=ax, linestyle=styles[orientation],
                             show_metastable=show_metastable)
    reference = next(iter(models.values()))
    depletant = {"phs": "PHS", "int": "Interacting polymers"}[reference.dep_model]
    ax.set(
        xlabel=r"Rod volume fraction $\phi$",
        ylabel=r"Reservoir depletant fraction $\phi_p^{R}$",
        title=(rf"{title}: {depletant}, $L/D={reference.L_over_D:g}$, $q={reference.q:g}$"
               "\n" r"Nominal size ratio $q=\sigma/D=2R_g/D$"),
        **limits.get("fig66", {}),
    )
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(
        handles=[Line2D([], [], color="black", linestyle=styles[name],
                        label=name.capitalize()) for name in models],
        loc="upper left", frameon=False,
    )
    order_ax = ax.inset_axes([0.65, 0.12, 0.30, 0.32])
    for orientation, model in models.items():
        for branch in model.rod_branches["IN"]:
            order_ax.plot(branch.phi_R,
                          np.where(branch.stable, branch.S[:, 1], np.nan),
                          color="black", linestyle=order_styles[orientation])
    order_ax.set(xlabel=r"$\phi_p^{R}$", ylabel=r"$S_N$",
                 **limits.get("fig66_order", {}))
    order_ax.tick_params(direction="in", top=True, right=True, labelsize=8)
    order_ax.xaxis.label.set_size(9)
    order_ax.yaxis.label.set_size(9)
    fig.tight_layout()
    return fig, ax, order_ax


def plot_rod_depletant_comparison(
    models, *, limits=None, branch_colors=None, show_metastable=False,
    title="Depletant-model comparison",
):
    """Plot reservoir fractions, system fractions, and system mM in three rows.

    models maps nominal q to {'phs': PD, 'int': PD}. Columns follow its order.
    limits uses fig610_q*, fig610_system_q*, and fig610_mM_q* keys, matching
    the notebook controls. Returns (figure, axes), with axes shape (3, n_q).
    Calculated results are reused; no thermodynamic solver is called.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    styles = {"phs": "-", "int": "--"}
    labels = {"phs": "PHS", "int": "Interacting polymers"}
    if not models or any(not group or any(name not in styles for name in group)
                         for group in models.values()):
        raise ValueError("models must map q to nonempty phs/int model dictionaries.")
    limits = {} if limits is None else limits
    colors = ROD_BRANCH_COLORS if branch_colors is None else branch_colors
    rows = (
        ("reservoir", "fig610_q", r"Rod volume fraction $\phi$",
         r"Reservoir depletant fraction $\phi_p^{R}$"),
        ("system", "fig610_system_q", r"Rod volume fraction $\phi$",
         r"System depletant fraction $\phi_p=\alpha\phi_p^{R}$"),
        ("mM", "fig610_mM_q", "Rod concentration (mM)",
         "System depletant concentration (mM)"),
    )
    fig, axes = plt.subplots(3, len(models), figsize=(5.8*len(models), 12.3),
                             squeeze=False)
    for column, (q, group) in enumerate(models.items()):
        for depletant, model in group.items():
            for row, (coordinates, key, xlabel, ylabel) in enumerate(rows):
                ax = axes[row, column]
                plot_rod_composition(
                    model, ax=ax, coordinates=coordinates, linestyle=styles[depletant],
                    branch_colors=colors, show_metastable=show_metastable,
                )
                ax.set(xlabel=xlabel, ylabel=ylabel,
                       **limits.get(f"{key}{q:g}", limits.get("custom", {})))
                ax.tick_params(direction="in", top=True, right=True)
        reference = next(iter(group.values()))
        axes[0, column].set_title(rf"$L/D={reference.L_over_D:g}$, $q={q:g}$")
    visible_kinds = {
        kind for group in models.values() for model in group.values()
        for kind, branches in model.rod_branches.items()
        if any(np.any(branch.stable) or (show_metastable and len(branch.phi_R))
               for branch in branches)
    }
    depletants = tuple(dict.fromkeys(name for group in models.values() for name in group))
    fig.legend(
        handles=[Line2D([], [], color=colors[kind], label=kind, linewidth=2)
                 for kind in ("II", "IN", "NN") if kind in visible_kinds]
                + [Line2D([], [], color="black", linestyle=styles[name],
                          label=labels[name]) for name in depletants],
        loc="upper center", bbox_to_anchor=(0.5, 0.945),
        ncol=len(visible_kinds) + len(depletants), frameon=False,
    )
    fig.suptitle(title + "\n" r"Nominal size ratio $q=\sigma/D=2R_g/D$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig, axes


def plot_rod_results(
    models, *, limits=None, result_colors=None, branch_linestyles=None,
    show_metastable=False, title="Rods + depletants",
):
    """Overlay labeled PD results in a 2x2 grid of concentration panels.

    models maps display labels to already calculated PD objects. result_colors
    maps those labels to colors; each result keeps its color in every panel
    and branch. branch_linestyles maps II/IN/NN to line styles, defaulting to
    solid/dashed/dotted. Limits use reservoir/system/mM/g/L keys.
    The top row contains reservoir and system fractions; the bottom row
    contains system mM and g/L. Each model retains its own partition factors,
    reference volumes, and molecular weights. No solver is called.
    Returns (figure, axes), with axes shape (2, 2).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not models:
        raise ValueError("models must contain at least one labeled PD result.")
    if result_colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        result_colors = {label: cycle[i % len(cycle)] for i, label in enumerate(models)}
    missing = [label for label in models if label not in result_colors]
    if missing:
        raise ValueError(f"result_colors is missing selected results: {missing}")
    styles = {"II": "-", "IN": "--", "NN": ":"}
    if branch_linestyles is not None:
        if set(branch_linestyles) - set(styles):
            raise ValueError("branch_linestyles keys must be II, IN, or NN.")
        styles.update(branch_linestyles)
    limits = {} if limits is None else limits
    panels = (
        ("reservoir", "Reservoir fraction", r"Rod volume fraction $\phi$",
         r"Reservoir depletant fraction $\phi_p^{R}$"),
        ("system", "System fraction", r"Rod volume fraction $\phi$",
         r"System depletant fraction $\phi_p=\alpha\phi_p^{R}$"),
        ("mM", "Concentrations in mM", "Rod concentration (mM)",
         "System depletant concentration (mM)"),
        ("g/L", "Concentrations in g/L", "Rod concentration (g/L)",
         "System depletant concentration (g/L)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 10.4))
    handles = []
    for label, model in models.items():
        color = result_colors[label]
        for ax, (coordinates, heading, xlabel, ylabel) in zip(axes.flat, panels):
            plot_rod_composition(
                model, ax=ax, coordinates=coordinates, color=color,
                branch_linestyles=styles, show_metastable=show_metastable,
            )
        for kind in ("II", "IN", "NN"):
            if any(np.any(branch.stable) or (show_metastable and len(branch.phi_R))
                   for branch in model.rod_branches[kind]):
                handles.append(Line2D([], [], color=color, linestyle=styles[kind],
                                      linewidth=1.7, label=f"{label} ({kind})"))
    for ax, (coordinates, heading, xlabel, ylabel) in zip(axes.flat, panels):
        ax.set(title=heading, xlabel=xlabel, ylabel=ylabel,
               **limits.get(coordinates, {}))
        ax.tick_params(direction="in", top=True, right=True)
    legend_rows = 1
    if handles:
        columns = min(4, len(handles))
        legend_rows = int(np.ceil(len(handles)/columns))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925),
                   ncol=columns, frameon=False, fontsize=9)
    fig.suptitle(title + "\n" r"Nominal size ratio $q=\sigma/D=2R_g/D$", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, .96-.04*(legend_rows-1)])
    return fig, axes
