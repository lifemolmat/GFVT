"""Notebook controls for calculating and comparing rod + PEG equilibria.

Import explicitly from this module; ipywidgets/ipympl remain optional for
non-notebook use of gfvt. The canvas has its own manager and does not change
the process-wide Matplotlib backend. Solvers run on one background thread;
all trait and canvas updates occur on the notebook's asyncio event loop.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from html import escape
from pathlib import Path
import textwrap
from unittest.mock import patch

import ipywidgets as widgets
import matplotlib
import numpy as np
# ipympl 0.9 imports call matplotlib.use() unconditionally, which would switch
# the notebook backend and close existing pyplot figures. Only suppress that
# import-time side effect; our canvas gets its own explicit manager below.
with patch.object(matplotlib, "use"):
    from ipympl.backend_nbagg import Canvas, FigureManager
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .pd import PD
from .rod_plot import plot_rod_composition


ROD_RESULT_PALETTE = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#0CC2F5", "#E9B600", "#E00DB2",
)
BRANCH_STYLES = {"II": "-", "IN": "--", "NN": ":"}
PANELS = {
    "reservoir": ("Reservoir volume fractions", r"Rod volume fraction $\phi$",
                  r"Reservoir PEG fraction $\phi_p^R$"),
    "system": ("System volume fractions", r"Rod volume fraction $\phi$",
               r"System PEG fraction $\phi_p=\alpha\phi_p^R$"),
    "mM": ("Concentrations in mM", "Rod concentration (mM)",
           "System PEG concentration (mM)"),
    "g/L": ("Concentrations in g/L", "Rod concentration (g/L)",
            "System PEG concentration (g/L)"),
}


def _field(kind, description, value, **kwargs):
    return kind(description=description, value=value,
                style={"description_width": "165px"},
                layout=widgets.Layout(width="325px"), **kwargs)


def _finite(value, name, *, allow_zero=False):
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}.")
    return float(value)


@dataclass
class _DiagramCard:
    id: str
    name: object
    peg: object
    rod_radius: object
    length: object
    peg_radius: object
    rod_mass: object
    maximum: object
    coexistence: object
    dep_model: object
    calculate_odfs: dict
    calculate_button: object
    color: object
    visible: dict
    bars: dict
    stages: dict
    activity: dict
    status: object
    heading: object
    selection_label: object
    box: object
    selection_box: object
    snapshot: object = None
    running_snapshot: object = None


class RodPhaseDiagramWidget(widgets.VBox):
    """Calculate selected ODFs per diagram and plot cached results interactively.

    peg_presets maps labels such as 'PEG35K' to (mass_kDa, radius_m).
    Supply the notebook's mpeg*/rpeg*K values rather than duplicating presets.
    N_steps/n_orient/n_azimuth are shared controls. limits uses the existing
    reservoir/system/mM/g/L mapping, with xlim/ylim pairs and None for auto.

    results maps stable card IDs to {'gaussian': PD, 'numerical': PD}; cards
    exposes their input controls. await calculate(card_id) is also available
    for scripted notebook use. figure and axes expose the current plot;
    save_figure() returns its PNG and EPS paths. No calculation runs on init.
    Each card's calculate_odfs checkboxes choose Gaussian, numerical, or both;
    only Gaussian is checked initially. Plot visibility is independent.
    """

    def __init__(self, peg_presets, *, N_steps=61, n_orient=128,
                 n_azimuth=256, limits=None, output_dir="figures/rods"):
        super().__init__()
        if not peg_presets:
            raise ValueError("Supply at least one PEG preset.")
        self.peg_presets = {
            str(label): (_finite(mass, "PEG molecular weight"),
                         _finite(radius, "PEG radius"))
            for label, (mass, radius) in peg_presets.items()
        }
        self.cards = []
        self.results = {}
        self.axes = {}
        self.task = None
        self.busy = False
        self._disposed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gfvt-rods")
        self._views = {}
        self._rendered_views = {}
        self._has_results_plot = False
        self._limits = {
            key: dict(xlim=(None, None), ylim=(None, None)) for key in PANELS
        }
        for key, settings in (limits or {}).items():
            if key in self._limits:
                self._limits[key].update(deepcopy(settings))

        self.N_steps = _field(widgets.IntText, "rod_N_steps", int(N_steps))
        self.n_orient = _field(widgets.IntText, "rod_n_orient", int(n_orient))
        self.n_azimuth = _field(widgets.IntText, "rod_n_azimuth", int(n_azimuth))
        for control in (self.N_steps, self.n_orient, self.n_azimuth):
            control.observe(self._parameters_changed, names="value")
        self._card_box = widgets.VBox()
        self._selection_box = widgets.VBox()
        self.add_button = widgets.Button(description="Add phase diagram", icon="plus",
                                         layout=widgets.Layout(width="190px"))
        self.add_button.on_click(lambda _: self.add_phase_diagram())
        self.layout_selector = widgets.ToggleButtons(
            options=[("All four (2×2)", "all"), ("mM and g/L (1×2)", "concentrations")],
            value="all")
        self.layout_selector.observe(lambda _: self.render(), names="value")

        self.limit_fields = {}
        limit_rows = []
        for key, (title, _, _) in PANELS.items():
            fields = {}
            for axis in ("x", "y"):
                for index, end in enumerate(("min", "max")):
                    bound = self._limits[key][axis + "lim"][index]
                    fields[axis + end] = widgets.Text(
                        description=axis + end,
                        value="" if bound is None else f"{bound:g}", placeholder="auto",
                        continuous_update=False, style={"description_width": "40px"},
                        layout=widgets.Layout(width="155px"))
            self.limit_fields[key] = fields
            limit_rows.append(widgets.VBox([widgets.HTML(f"<b>{title}</b>"),
                                            widgets.HBox(list(fields.values()))]))
        self.apply_limits_button = widgets.Button(description="Apply Limits")
        self.reset_view_button = widgets.Button(description="Reset View")
        self.apply_limits_button.on_click(lambda _: self.apply_limits())
        self.reset_view_button.on_click(lambda _: self.reset_view())
        self.plot_status = widgets.HTML()
        limits_box = widgets.Accordion(children=[widgets.VBox([
            *limit_rows, widgets.HBox([self.apply_limits_button, self.reset_view_button]),
            widgets.HTML("Blank bounds mean automatic. Apply Limits sets the view; Reset View restores these bounds.")])])
        limits_box.set_title(0, "Axis limits")
        limits_box.selected_index = None

        self.figure = Figure(figsize=(11.8, 10))
        self.canvas = Canvas(self.figure)
        self._manager = FigureManager(self.canvas, id(self))
        self.canvas.header_visible = False
        self.canvas.toolbar_visible = True
        self.canvas.layout.width = "100%"
        self.output_directory = widgets.Text(description="Output directory", value=str(output_dir),
                                             style={"description_width": "110px"},
                                             layout=widgets.Layout(width="440px"))
        self.filename_base = widgets.Text(description="Filename base", value="rods_phase_diagrams",
                                          style={"description_width": "110px"},
                                          layout=widgets.Layout(width="400px"))
        self.save_button = widgets.Button(description="Save PNG + EPS", icon="save", disabled=True,
                                          layout=widgets.Layout(width="180px"))
        self.save_button.on_click(self._save_clicked)
        self.save_status = widgets.HTML()
        self.children = (
            widgets.HTML("<h3>Calculate Rods + PEG phase diagrams</h3>"
                         "Dimensions: nm. Molecular weights: kDa. L is the cylinder length, excluding caps."),
            widgets.HBox([self.N_steps, self.n_orient, self.n_azimuth],
                         layout=widgets.Layout(flex_flow="row wrap")),
            self._card_box, self.add_button,
            widgets.HTML("<h3>Plot calculated phase diagrams</h3>"
                         "Color identifies the diagram; II solid, IN dashed, NN dotted. "
                         "Open markers identify numerical ODF results. Use the toolbar to zoom or pan."),
            self._selection_box, self.layout_selector, limits_box, self.plot_status,
            self.canvas,
            widgets.HBox([self.output_directory, self.filename_base],
                         layout=widgets.Layout(flex_flow="row wrap")),
            self.save_button, self.save_status,
        )
        self.add_phase_diagram()

    def add_phase_diagram(self):
        """Add one independent calculation card and return its controls."""
        number = len(self.cards) + 1
        card_id = f"diagram-{number}"
        preset = "PEG35K" if "PEG35K" in self.peg_presets else next(iter(self.peg_presets))
        name = _field(widgets.Text, "Phase diagram name", f"Diagram {number}", continuous_update=False)
        peg = _field(widgets.Dropdown, "PEG molecular weight", preset,
                     options=[(f"{label} ({mass:g} kDa)", label)
                              for label, (mass, _) in self.peg_presets.items()])
        controls = dict(
            rod_radius=_field(widgets.FloatText, "Rod radius (nm)", 1.),
            length=_field(widgets.FloatText, "Rod cylinder L (nm)", 8.),
            peg_radius=_field(widgets.FloatText, "PEG radius R (nm)", self.peg_presets[preset][1]*1e9),
            rod_mass=_field(widgets.FloatText, "Rod MW (kDa)", 16.5),
            maximum=_field(widgets.FloatText, "PhiP_max (reservoir)", 20.),
            coexistence=_field(widgets.Dropdown, "Coexistence", "all", options=["all", "II", "IN", "NN"]),
            dep_model=_field(widgets.Dropdown, "PEG model", "int",
                             options=[("Interacting PEG (int)", "int"), ("PHS (phs)", "phs")]),
        )
        button = widgets.Button(description="Calculate phase diagram", icon="calculator",
                                disabled=self.busy, layout=widgets.Layout(width="240px"))
        calculate_odfs = {
            odf: widgets.Checkbox(description=label, value=odf == "gaussian", indent=False,
                                  layout=widgets.Layout(width="175px"))
            for odf, label in (("gaussian", "Gaussian"), ("numerical", "Numerical ODF"))
        }
        color = widgets.ColorPicker(description="Color", value=ROD_RESULT_PALETTE[(number-1) % len(ROD_RESULT_PALETTE)],
                                    concise=True, layout=widgets.Layout(width="155px"))
        visible = {odf: widgets.Checkbox(description=odf.capitalize(), value=True, disabled=True,
                                        indent=False, layout=widgets.Layout(width="135px"))
                   for odf in ("gaussian", "numerical")}
        bars = {odf: widgets.IntProgress(min=0, max=1, value=0,
                                         layout=widgets.Layout(width="440px", display="none"))
                for odf in visible}
        stages = {odf: widgets.HTML(f"{odf.capitalize()}: not calculated") for odf in visible}
        activity = {odf: widgets.HTML(layout=widgets.Layout(display="none")) for odf in visible}
        status, heading, selection_label = widgets.HTML(), widgets.HTML(), widgets.HTML()
        selection_label.layout.min_width = "230px"
        selection_box = widgets.HBox([selection_label, color, *visible.values()],
                                     layout=widgets.Layout(flex_flow="row wrap", align_items="center"))
        box = widgets.VBox(
            [heading, widgets.HBox([name, peg], layout=widgets.Layout(flex_flow="row wrap")),
             widgets.HBox(list(controls.values()), layout=widgets.Layout(flex_flow="row wrap")),
             widgets.HBox([widgets.HTML("<b>Calculate:</b>"), *calculate_odfs.values()],
                          layout=widgets.Layout(flex_flow="row wrap", align_items="center")),
             button, status,
             *[widgets.VBox([stages[odf], activity[odf], bars[odf]]) for odf in visible]],
            layout=widgets.Layout(border="1px solid #ccc", padding="12px", margin="6px 0"))
        card = _DiagramCard(card_id, name, peg, **controls, calculate_odfs=calculate_odfs,
                            calculate_button=button, color=color,
                            visible=visible, bars=bars, stages=stages, activity=activity,
                            status=status, heading=heading, selection_label=selection_label,
                            box=box, selection_box=selection_box)
        self.cards.append(card)
        self.results[card_id] = {}
        self._card_box.children = tuple(c.box for c in self.cards)
        self._selection_box.children = tuple(c.selection_box for c in self.cards)
        peg.observe(lambda change: self._peg_changed(card, change["new"]), names="value")
        name.observe(lambda _: self._rename(card), names="value")
        for control in controls.values():
            control.observe(self._parameters_changed, names="value")
        button.on_click(lambda _: self._start_calculation(card))
        color.observe(lambda _: self.render(), names="value")
        for control in visible.values():
            control.observe(lambda _: self.render(), names="value")
        self._rename(card)
        return card

    def _card(self, card_id):
        return next(card for card in self.cards if card.id == card_id)

    def _name(self, card, peg=None):
        base = card.name.value.strip() or card.id.replace("-", " ").capitalize()
        tag = card.peg.value if peg is None else peg
        return base if tag in base else f"{base} + {tag}"

    def _rename(self, card):
        card.heading.value = f"<b>{escape(self._name(card))}</b>"
        shown_peg = card.snapshot["peg"] if card.snapshot is not None else card.peg.value
        card.selection_label.value = escape(self._name(card, shown_peg))
        self.render()

    def _peg_changed(self, card, peg):
        card.peg_radius.value = self.peg_presets[peg][1]*1e9
        self._parameters_changed()
        self._rename(card)

    def _snapshot(self, card):
        orders = {}
        for key, control, minimum in (("N_steps", self.N_steps, 2),
                                       ("n_orient", self.n_orient, 16),
                                       ("n_azimuth", self.n_azimuth, 16)):
            if control.value < minimum:
                raise ValueError(f"{key} must be at least {minimum}.")
            orders[key] = int(control.value)
        return dict(
            peg=card.peg.value, peg_mass=self.peg_presets[card.peg.value][0],
            rod_radius=_finite(card.rod_radius.value, "Rod radius")*1e-9,
            L_rod=_finite(card.length.value, "Rod length", allow_zero=True)*1e-9,
            peg_radius=_finite(card.peg_radius.value, "PEG radius")*1e-9,
            rod_mass=_finite(card.rod_mass.value, "Rod molecular weight"),
            PhiP_max=_finite(card.maximum.value, "PhiP_max", allow_zero=True),
            coexistence=card.coexistence.value, dep_model=card.dep_model.value, **orders)

    def _stale(self, card):
        if card.snapshot is None:
            return False
        try:
            return self._snapshot(card) != card.snapshot
        except ValueError:
            return True

    def _parameters_changed(self, _=None):
        for card in self.cards:
            if self._stale(card):
                card.status.value = "<b>Inputs changed: recalculate to update results. Plots retain calculated parameters.</b>"
            elif card.status.value.startswith("<b>Inputs changed:"):
                card.status.value = "Inputs match the calculated results."

    def _start_calculation(self, card):
        try:
            loop = asyncio.get_running_loop()
            if not self.busy:
                self.task = loop.create_task(self.calculate(card.id))
        except RuntimeError:
            card.status.value = "Run this widget in a live Jupyter kernel with asyncio support."

    @staticmethod
    def _solve(snapshot, odf, callback):
        model = PD(snapshot["rod_radius"], snapshot["peg_radius"],
                   snapshot["rod_mass"], snapshot["peg_mass"], "good",
                   snapshot["N_steps"], [.04, .4], progress=False, verbose=False)
        return model.GFVT_rods(
            snapshot["PhiP_max"], L_rod=snapshot["L_rod"],
            coexistence=snapshot["coexistence"], dep_model=snapshot["dep_model"], odf=odf,
            n_orient=snapshot["n_orient"], n_azimuth=snapshot["n_azimuth"],
            progress_callback=callback)

    async def calculate(self, card_id):
        """Calculate one card's selected ODFs without blocking the notebook loop."""
        if self._disposed:
            raise RuntimeError("This widget is closed.")
        if self.busy:
            raise RuntimeError("A phase diagram is already being calculated.")
        card = self._card(card_id)
        methods = tuple(odf for odf, checkbox in card.calculate_odfs.items() if checkbox.value)
        if not methods:
            card.status.value = "<b>Select Gaussian, Numerical ODF, or both before calculating.</b>"
            return
        try:
            snapshot = self._snapshot(card)
        except ValueError as error:
            card.status.value = f"<b>Invalid inputs:</b> {escape(str(error))}"
            return
        loop = asyncio.get_running_loop()
        card.running_snapshot = snapshot
        self.busy = True
        for item in self.cards:
            item.calculate_button.disabled = True
        for odf in card.visible:
            card.stages[odf].value = (f"{odf.capitalize()}: waiting" if odf in methods else
                                      f"{odf.capitalize()}: not selected for this calculation")
            card.bars[odf].value = 0
            card.bars[odf].bar_style = "info"
            card.bars[odf].layout.display = "none"
            card.activity[odf].layout.display = "none"
        card.status.value = "Calculating " + " and ".join(
            "Gaussian" if odf == "gaussian" else "numerical ODF" for odf in methods) + "…"
        successful, failures = {}, {}
        try:
            for odf in methods:
                def callback(event, orientation=odf):
                    if not self._disposed:
                        loop.call_soon_threadsafe(self._on_progress, card.id, orientation, event)
                try:
                    model = await loop.run_in_executor(
                        self._executor, partial(self._solve, snapshot, odf, callback))
                except Exception as error:
                    failures[odf] = f"{type(error).__name__}: {error}"
                    if not self._disposed:
                        self._on_progress(card.id, odf, dict(stage="failed", message=failures[odf]))
                    continue
                if self._disposed:
                    return
                successful[odf] = model
                # Once a new method succeeds, never mix it with an old method.
                self.results[card.id] = successful.copy()
                card.snapshot = snapshot.copy()
                for method, checkbox in card.visible.items():
                    checkbox.disabled = method not in successful
                self._rename(card)
            if not self._disposed:
                if failures:
                    prefix = ("Successful methods retained; " if successful else
                              "Calculation failed; previous results retained; " if self.results[card.id] else
                              "Calculation failed; no results calculated; ")
                    card.status.value = escape(prefix) + "<br>" + "<br>".join(
                        f"<b>{odf.capitalize()}:</b> {escape(message)}" for odf, message in failures.items())
                else:
                    counts = []
                    for odf, model in successful.items():
                        kinds = ", ".join(f"{kind}: {sum(int(np.count_nonzero(b.stable)) for b in group)} stable points"
                                          for kind, group in model.rod_branches.items())
                        counts.append(f"{odf.capitalize()} — {kinds}")
                    card.status.value = "<b>Calculation complete.</b><br>" + "<br>".join(map(escape, counts))
                if self._stale(card):
                    card.status.value += "<br><b>Inputs changed during calculation; recalculate to apply them.</b>"
        finally:
            self.busy = False
            card.running_snapshot = None
            if not self._disposed:
                for item in self.cards:
                    item.calculate_button.disabled = False

    def _on_progress(self, card_id, odf, event):
        if self._disposed:
            return
        card = self._card(card_id)
        bar, activity = card.bars[odf], card.activity[odf]
        stage = event["stage"]
        peg = (card.running_snapshot or card.snapshot or {"peg": card.peg.value})["peg"]
        label = f"{escape(self._name(card, peg))} — {odf.capitalize()}: "
        if stage in ("critical_points", "finalizing"):
            text = "locating critical points" if stage == "critical_points" else "finalizing branches and stability"
            activity.value = '<progress aria-label="Calculation in progress" style="width:440px"></progress>'
            activity.layout.display = ""
            bar.layout.display = "none"
        elif stage in ("tracing", "complete"):
            activity.layout.display = "none"
            bar.layout.display = ""
            total = max(1, int(event["total"]))
            bar.value = min(bar.value, total)
            bar.max = total
            bar.value = int(event["completed"])
            bar.bar_style = "success" if stage == "complete" else "info"
            text = (f"complete ({bar.value}/{total} reservoir points)" if stage == "complete"
                    else f"reservoir points {bar.value}/{total}")
        else:
            activity.layout.display = "none"
            bar.layout.display = ""
            bar.bar_style = "danger"
            text = "failed — " + escape(event.get("message", "unknown error"))
        card.stages[odf].value = label + text

    def apply_limits(self):
        """Validate all bound fields before replacing the last valid view."""
        settings = {}
        try:
            for key, fields in self.limit_fields.items():
                settings[key] = {}
                for axis in ("x", "y"):
                    bounds = []
                    for end in ("min", "max"):
                        raw = fields[axis + end].value.strip()
                        number = None if raw.lower() in ("", "auto", "none") else float(raw)
                        if number is not None and not np.isfinite(number):
                            raise ValueError("Axis bounds must be finite or blank.")
                        bounds.append(number)
                    if None not in bounds and bounds[0] >= bounds[1]:
                        raise ValueError(f"{key}: {axis}min must be smaller than {axis}max.")
                    settings[key][axis + "lim"] = tuple(bounds)
        except ValueError as error:
            self.plot_status.value = f"<b>Invalid limits:</b> {escape(str(error))}"
            return False
        self._limits = settings
        self.plot_status.value = ""
        self.reset_view()
        return True

    def reset_view(self):
        """Restore applied bounds (automatic wherever a field was blank)."""
        self._views.clear()
        self.render(preserve_view=False)

    def render(self, *, preserve_view=True):
        """Redraw cached results only, preserving per-coordinate zoom."""
        if self._disposed:
            return
        if preserve_view and self._has_results_plot:
            for key, ax in self.axes.items():
                view = dict(xlim=ax.get_xlim(), ylim=ax.get_ylim())
                previous = self._rendered_views.get(key)
                if key in self._views or (previous is not None and any(
                        not np.allclose(view[axis], previous[axis], rtol=1e-12, atol=1e-14)
                        for axis in ("xlim", "ylim"))):
                    self._views[key] = view
        self.figure.clear()
        keys = tuple(PANELS) if self.layout_selector.value == "all" else ("mM", "g/L")
        rows = 2 if len(keys) == 4 else 1
        axes = self.figure.subplots(rows, 2, squeeze=False)
        self.axes = dict(zip(keys, axes.flat))
        legend_handles, selected = [], []
        for card in self.cards:
            models = [(odf, model) for odf, model in self.results[card.id].items()
                      if card.visible[odf].value]
            if not models:
                continue
            peg = card.snapshot["peg"]
            card.selection_label.value = escape(self._name(card, peg))
            model = models[0][1]
            label = (f"{self._name(card, peg)} [{model.dep_model}]: "
                     f"D={model.D*1e9:g} nm, L={model.L_rod*1e9:g} nm, "
                     f"R={model.Rpol_in*1e9:g} nm, q={model.q:g}")
            legend_handles.append(Line2D([], [], color=card.color.value, linewidth=2,
                                         label=textwrap.fill(label, width=105)))
            for odf, model in models:
                selected.append((card, odf, model))
                count = max((len(b.phi_R) for group in model.rod_branches.values() for b in group), default=1)
                for key, ax in self.axes.items():
                    plot_rod_composition(
                        model, ax=ax, coordinates=key, color=card.color.value,
                        branch_linestyles=BRANCH_STYLES,
                        marker="o" if odf == "numerical" else None,
                        markevery=max(1, count//18) if odf == "numerical" else None)
        for key, ax in self.axes.items():
            title, xlabel, ylabel = PANELS[key]
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
            ax.tick_params(direction="in", top=True, right=True)
            ax.set(**self._limits[key])
            if key in self._views:
                ax.set(**self._views[key])
            if not selected:
                ax.text(.5, .5, "Calculate a phase diagram and select results to display",
                        transform=ax.transAxes, ha="center", va="center", fontsize=9, wrap=True)
            elif not any(np.any(np.isfinite(line.get_ydata())) for line in ax.lines):
                ax.text(.5, .5, "No stable coexistence in the requested range",
                        transform=ax.transAxes, ha="center", va="center", fontsize=9)
        legend_lines = sum(h.get_label().count("\n") + 1 for h in legend_handles)
        legend_height = .25*legend_lines + .55 if legend_handles else .2
        height = (8.9 if rows == 2 else 4.9) + legend_height
        self.figure.set_size_inches(11.8, height, forward=True)
        self.figure.suptitle("Custom Rods + PEG\n"
                             r"$D=2r_{\rm rod}$; $L$: cylinder length; $R=R_g$; $q=2R/D$",
                             fontsize=13, y=1-.12/height)
        if legend_handles:
            self.figure.legend(handles=legend_handles, loc="lower center", ncol=1,
                               bbox_to_anchor=(.5, .02/height), frameon=False, fontsize=9)
            visible_kinds = {kind for _, _, model in selected
                             for kind, group in model.rod_branches.items()
                             if any(np.any(b.stable) for b in group)}
            style_handles = [Line2D([], [], color="black", linestyle=style, label=kind)
                             for kind, style in BRANCH_STYLES.items() if kind in visible_kinds]
            orientations = {odf for _, odf, _ in selected}
            for odf in ("gaussian", "numerical"):
                if odf in orientations:
                    style_handles.append(Line2D(
                        [], [], color="black", marker="o" if odf == "numerical" else None,
                        markerfacecolor="white", markersize=4, label=f"{odf.capitalize()} ODF"))
            self.figure.legend(handles=style_handles, loc="lower center", ncol=len(style_handles), frameon=False,
                               bbox_to_anchor=(.5, (legend_height-.25)/height), fontsize=9)
        self.figure.subplots_adjust(left=.09, right=.97, top=1-.9/height,
                                    bottom=(legend_height+.5)/height, hspace=.35, wspace=.3)
        self._has_results_plot = bool(selected)
        self._rendered_views = {key: dict(xlim=ax.get_xlim(), ylim=ax.get_ylim())
                                for key, ax in self.axes.items()}
        self.save_button.disabled = not bool(selected)
        self.canvas.draw_idle()

    def save_figure(self):
        """Save the current view, including legends; return (png_path, eps_path)."""
        if self.save_button.disabled:
            raise ValueError("Select a calculated result before saving.")
        base = self.filename_base.value.strip()
        if not base or Path(base).name != base or base in (".", ".."):
            raise ValueError("Filename base must be a name without directory separators.")
        directory = Path(self.output_directory.value or "figures/rods").expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        png, eps = (directory / f"{base}_{stamp}.{extension}" for extension in ("png", "eps"))
        self.figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        # EPS has no transparency: composite line opacity onto the white page,
        # then restore the live canvas. This also preserves triple tie colors.
        changed = []
        try:
            for line in self.figure.findobj(Line2D):
                alpha = line.get_alpha()
                if alpha is not None and alpha < 1:
                    old = (line, alpha, line.get_color(), line.get_markerfacecolor(), line.get_markeredgecolor())
                    changed.append(old)
                    for getter, setter in ((line.get_color, line.set_color),
                                           (line.get_markerfacecolor, line.set_markerfacecolor),
                                           (line.get_markeredgecolor, line.set_markeredgecolor)):
                        rgba = to_rgba(getter())
                        if rgba[3] != 0:
                            setter(tuple(1-alpha*rgba[3]*(1-c) for c in rgba[:3]))
                    line.set_alpha(1)
            self.figure.savefig(eps, format="eps", bbox_inches="tight", facecolor="white")
        finally:
            for line, alpha, color, face, edge in changed:
                line.set_alpha(alpha)
                line.set_color(color)
                line.set_markerfacecolor(face)
                line.set_markeredgecolor(edge)
        self.save_status.value = "<b>Saved:</b><br>" + "<br>".join(escape(str(p)) for p in (png, eps))
        return png, eps

    def _save_clicked(self, _):
        try:
            self.save_figure()
        except Exception as error:
            self.save_status.value = f"<b>Save failed:</b> {escape(str(error))}"

    def close(self):
        """Release the canvas/worker when rerunning the notebook launch cell."""
        if getattr(self, "_disposed", True):
            return
        self._disposed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._manager.destroy()
        super().close()
