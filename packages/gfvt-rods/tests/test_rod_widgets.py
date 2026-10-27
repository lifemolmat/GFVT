"""Notebook UI state, scientific conversion boundaries and live progress."""

import asyncio
from copy import deepcopy
import json
import importlib.util
import pickle
import subprocess
import sys
import threading
from pathlib import Path

import matplotlib
import numpy as np
import pytest

pytest.importorskip("ipywidgets")
if importlib.util.find_spec("ipympl") is None:
    pytest.skip("ipympl is optional", allow_module_level=True)

from gfvt_rods import PD
from gfvt_rods.rod_widgets import RodPhaseDiagramWidget, ROD_RESULT_PALETTE


PRESETS = {"PEG4K": (4, 1.9e-9), "PEG6K": (6, 2.45e-9),
           "PEG8K": (8, 3.1e-9), "PEG20K": (20, 5.4e-9),
           "PEG35K": (35, 7.8e-9), "PEG100K": (100, 14.8e-9)}


def test_first_import_preserves_backend_and_existing_figures():
    code = '''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
figure = plt.figure()
before = matplotlib.get_backend()
from gfvt_rods.rod_widgets import RodPhaseDiagramWidget
assert matplotlib.get_backend() == before
assert plt.fignum_exists(figure.number)
widget = RodPhaseDiagramWidget({"PEG35K": (35, 7.8e-9)})
assert matplotlib.get_backend() == before
assert plt.fignum_exists(figure.number)
widget.close()
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.fixture
def ui(tmp_path):
    widget = RodPhaseDiagramWidget(PRESETS, output_dir=tmp_path)
    yield widget
    widget.close()


@pytest.fixture(scope="module")
def model():
    return PD(1e-9, 7.8e-9, 16.5, 35, "good", 3, [.04, .4],
              progress=False, verbose=False).GFVT_rods(5, L_rod=8e-9, coexistence="all")


def publish(ui, model, odfs=("gaussian", "numerical")):
    card = ui.cards[0]
    card.snapshot = ui._snapshot(card)
    for odf in odfs:
        result = deepcopy(model)
        result.odf = odf
        ui.results[card.id][odf] = result
        card.visible[odf].disabled = False
    ui.render()
    return card


def test_defaults_presets_units_and_backend(ui):
    backend = matplotlib.get_backend()
    card = ui.cards[0]
    assert len(ui.cards) == 1 and ui.results[card.id] == {}
    assert card.peg.value == "PEG35K"
    assert card.rod_radius.value == 1 and card.length.value == 8
    assert card.rod_mass.value == 16.5 and card.maximum.value == 20
    assert card.coexistence.value == "all" and card.dep_model.value == "int"
    assert card.calculate_odfs["gaussian"].value
    assert not card.calculate_odfs["numerical"].value
    assert (ui.N_steps.value, ui.n_orient.value, ui.n_azimuth.value) == (61, 128, 256)
    card.peg.value = "PEG100K"
    assert card.peg_radius.value == pytest.approx(14.8)
    card.peg_radius.value = 12.3
    snapshot = ui._snapshot(card)
    assert snapshot["peg_radius"] == pytest.approx(12.3e-9)
    assert snapshot["peg_mass"] == 100
    assert snapshot["rod_radius"] == 1e-9
    assert snapshot["L_rod"] == 8e-9
    assert "PEG100K" in ui._name(card)
    card.peg.value = "PEG4K"
    assert card.peg_radius.value == pytest.approx(1.9)
    assert list(ui.axes) == ["reservoir", "system", "mM", "g/L"]
    assert all(f.value == "" for fields in ui.limit_fields.values() for f in fields.values())
    assert matplotlib.get_backend() == backend
    assert ui.canvas.toolbar is not None


def test_colors_names_and_selections_do_not_recalculate(ui, model, monkeypatch):
    def forbidden(*args):
        pytest.fail("A presentation change called the solver")
    monkeypatch.setattr(ui, "_solve", forbidden)
    card = publish(ui, model)
    original = pickle.dumps(ui.results[card.id])
    colors = [card.color.value]
    for _ in range(7):
        colors.append(ui.add_phase_diagram().color.value)
    assert colors == list(ROD_RESULT_PALETTE) + [ROD_RESULT_PALETTE[0]]
    card.color.value = "#123456"
    card.name.value = "My rod"
    assert "My rod + PEG35K" in ui.figure.legends[0].get_texts()[0].get_text()
    card.visible["numerical"].value = False
    assert "Numerical ODF" not in [t.get_text() for t in ui.figure.legends[1].get_texts()]
    assert all(line.get_marker() != "o" or len(line.get_xdata()) == 1
               for line in ui.axes["mM"].lines)
    card.visible["numerical"].value = True
    assert any(line.get_marker() == "o" and len(line.get_xdata()) > 1
               for line in ui.axes["mM"].lines)
    assert card.color.value == "#123456"
    assert original == pickle.dumps(ui.results[card.id])


def test_units_styles_zoom_layout_and_parameter_legends(ui, model):
    card = publish(ui, model)
    result = ui.results[card.id]["gaussian"]
    first = next(b for kind in ("II", "IN", "NN") for b in result.rod_branches[kind])
    expected_x = np.where(first.stable, first.phi[:, 0]*result.phix_bsa, np.nan)
    expected_y = np.where(first.stable, first.alpha[:, 0]*first.phi_R*result.phix_peg, np.nan)
    np.testing.assert_allclose(ui.axes["mM"].lines[0].get_xdata(), expected_x)
    np.testing.assert_allclose(ui.axes["mM"].lines[0].get_ydata(), expected_y)
    np.testing.assert_allclose(ui.axes["g/L"].lines[0].get_xdata(), expected_x*16.5)
    np.testing.assert_allclose(ui.axes["g/L"].lines[0].get_ydata(), expected_y*35)
    text = " ".join(t.get_text() for legend in ui.figure.legends for t in legend.get_texts())
    for token in ("PEG35K", "D=2 nm", "L=8 nm", "R=7.8 nm", "q=7.8", "[int]", "Numerical ODF"):
        assert token in text
    assert not any("depletant" in ax.get_ylabel().lower() for ax in ui.axes.values())
    assert {line.get_linestyle() for line in ui.axes["mM"].lines} >= {"-", "--"}
    ui.axes["mM"].set_xlim(.1, 1.4)
    ui.axes["mM"].set_ylim(.02, .3)
    card.color.value = "#abcdef"
    assert ui.axes["mM"].get_xlim() == pytest.approx((.1, 1.4))
    ui.layout_selector.value = "concentrations"
    assert list(ui.axes) == ["mM", "g/L"]
    assert len(ui.figure.axes) == 2
    assert ui.axes["mM"].get_ylim() == pytest.approx((.02, .3))
    ui.layout_selector.value = "all"
    assert len(ui.figure.axes) == 4
    assert ui.axes["mM"].get_xlim() == pytest.approx((.1, 1.4))
    card.peg.value = "PEG100K"
    # Editing inputs must never relabel a stored PEG35K calculation as PEG100K.
    text = " ".join(t.get_text() for t in ui.figure.legends[0].get_texts())
    assert "PEG35K" in text and "PEG100K" not in text
    assert "Inputs changed" in card.status.value


def test_axis_validation_keeps_last_valid_view(ui, model):
    publish(ui, model)
    fields = ui.limit_fields["mM"]
    fields["xmin"].value, fields["xmax"].value = "0.2", "2.0"
    fields["ymax"].value = "3"
    assert ui.apply_limits()
    assert ui.axes["mM"].get_xlim() == pytest.approx((.2, 2))
    fields["xmin"].value = "nan"
    assert not ui.apply_limits()
    assert ui.axes["mM"].get_xlim() == pytest.approx((.2, 2))
    fields["xmin"].value = "4"
    assert not ui.apply_limits()
    fields["xmin"].value = ""
    fields["xmax"].value = "auto"
    assert ui.apply_limits()
    assert ui.axes["mM"].get_xlim() != pytest.approx((.2, 2))


def test_automatic_limits_include_added_results_until_user_zooms(ui, model):
    publish(ui, model)
    original_max = ui.axes["g/L"].get_xlim()[1]
    second = ui.add_phase_diagram()
    second.rod_mass.value = 165
    second.snapshot = ui._snapshot(second)
    heavy = deepcopy(model)
    heavy.mbsa = 165
    ui.results[second.id] = {"gaussian": heavy}
    second.visible["gaussian"].disabled = False
    ui.render()
    assert ui.axes["g/L"].get_xlim()[1] > 5*original_max
    ui.axes["g/L"].set_xlim(0, 50)
    second.visible["gaussian"].value = False
    assert ui.axes["g/L"].get_xlim() == pytest.approx((0, 50))


def test_export_preserves_current_view_legends_and_models(ui, model):
    publish(ui, model)
    ui.axes["mM"].set_xlim(.1, 1.2)
    ui.axes["mM"].set_ylim(.01, .2)
    bounds = [(ax.get_xlim(), ax.get_ylim()) for ax in ui.axes.values()]
    raw = pickle.dumps(ui.results)
    png, eps = ui.save_figure()
    assert png.read_bytes().startswith(b"\x89PNG")
    assert eps.read_bytes().startswith(b"%!PS-Adobe")
    assert png.stat().st_size > 10000 and eps.stat().st_size > 10000
    assert [(ax.get_xlim(), ax.get_ylim()) for ax in ui.axes.values()] == bounds
    assert raw == pickle.dumps(ui.results)
    assert "D=2 nm" in ui.figure.legends[0].get_texts()[0].get_text()
    ui.filename_base.value = "../invalid"
    with pytest.raises(ValueError, match="without directory"):
        ui.save_figure()


def test_background_progress_snapshots_partial_success_and_no_duplicate_jobs(ui, model, monkeypatch):
    calls, thread_ids = [], []
    started, release = threading.Event(), threading.Event()
    main_thread = threading.get_ident()

    def solve(snapshot, odf, callback):
        calls.append((snapshot.copy(), odf))
        thread_ids.append(threading.get_ident())
        callback(dict(stage="critical_points", completed=None, total=None))
        started.set()
        assert release.wait(5)
        callback(dict(stage="tracing", completed=0, total=7))
        callback(dict(stage="tracing", completed=7, total=7))
        callback(dict(stage="finalizing", completed=None, total=None))
        if odf == "numerical":
            raise RuntimeError("ODF did not converge")
        callback(dict(stage="complete", completed=7, total=7))
        return model

    monkeypatch.setattr(ui, "_solve", solve)
    card = ui.cards[0]
    card.calculate_odfs["numerical"].value = True

    async def run():
        task = asyncio.create_task(ui.calculate(card.id))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(.01)
        assert ui.busy and card.calculate_button.disabled
        assert card.activity["gaussian"].layout.display != "none"
        with pytest.raises(RuntimeError, match="already"):
            await ui.calculate(card.id)
        card.length.value = 17
        card.peg.value = "PEG100K"
        # The methods requested at click time remain fixed during a run.
        card.calculate_odfs["numerical"].value = False
        ui.layout_selector.value = "concentrations"
        added = ui.add_phase_diagram()
        assert added.calculate_button.disabled
        release.set()
        await task

    asyncio.run(run())
    assert [odf for _, odf in calls] == ["gaussian", "numerical"]
    assert all(s["L_rod"] == 8e-9 for s, _ in calls)
    assert all(thread_id != main_thread for thread_id in thread_ids)
    assert set(ui.results[card.id]) == {"gaussian"}
    assert card.visible["numerical"].disabled
    assert "ODF did not converge" in card.status.value
    assert "Inputs changed during" in card.status.value
    assert "PEG35K" in card.stages["gaussian"].value
    assert "PEG100K" not in card.stages["gaussian"].value
    assert card.bars["gaussian"].bar_style == "success"
    assert card.bars["numerical"].bar_style == "danger"
    assert not ui.busy and not card.calculate_button.disabled


def test_failed_recalculation_keeps_previous_and_invalid_input_does_not_solve(ui, model, monkeypatch):
    card = publish(ui, model)
    original = pickle.dumps(ui.results)
    def fail(*args):
        raise RuntimeError("intentional numerical failure")
    monkeypatch.setattr(ui, "_solve", fail)
    asyncio.run(ui.calculate(card.id))
    assert pickle.dumps(ui.results) == original
    assert "previous results retained" in card.status.value
    assert not card.visible["numerical"].disabled
    monkeypatch.setattr(ui, "_solve", lambda *a: pytest.fail("invalid inputs reached solver"))
    ui.n_orient.value = 1
    asyncio.run(ui.calculate(card.id))
    assert "Invalid inputs" in card.status.value
    assert pickle.dumps(ui.results) == original


def test_successful_recalculation_never_combines_old_and_new_odfs(ui, model, monkeypatch):
    card = publish(ui, model)
    card.calculate_odfs["numerical"].value = True
    old_numerical = ui.results[card.id]["numerical"]
    def solve(snapshot, odf, callback):
        if odf == "numerical":
            assert ui.results[card.id] == {"gaussian": model}
            raise RuntimeError("no numerical solution")
        return model
    monkeypatch.setattr(ui, "_solve", solve)
    asyncio.run(ui.calculate(card.id))
    assert old_numerical not in ui.results[card.id].values()
    assert set(ui.results[card.id]) == {"gaussian"}


def test_real_zero_peg_both_odfs_and_progress(ui):
    card = ui.cards[0]
    card.calculate_odfs["numerical"].value = True
    ui.N_steps.value = 2
    card.maximum.value = 0
    card.coexistence.value = "IN"
    asyncio.run(ui.calculate(card.id))
    assert set(ui.results[card.id]) == {"gaussian", "numerical"}, card.status.value
    for odf, result in ui.results[card.id].items():
        assert result.odf == odf and result.rod_branches["IN"]
        assert card.bars[odf].bar_style == "success"
        assert card.bars[odf].value == card.bars[odf].max
        for branch in result.rod_branches["IN"]:
            assert np.ptp(branch.mu, axis=1).max() < 1e-8
            assert np.ptp(branch.pressure, axis=1).max()/result.phix_bsa/8.314/(273.15+result.TC) < 1e-8


@pytest.mark.parametrize("methods", [("gaussian",), ("numerical",), ("gaussian", "numerical"), ()])
def test_only_selected_calculation_methods_run(ui, model, monkeypatch, methods):
    card = ui.cards[0]
    calls = []
    def solve(snapshot, odf, callback):
        calls.append(odf)
        callback(dict(stage="complete", completed=1, total=1))
        return model
    monkeypatch.setattr(ui, "_solve", solve)
    for odf, checkbox in card.calculate_odfs.items():
        checkbox.value = odf in methods
    asyncio.run(ui.calculate(card.id))
    assert calls == list(methods)
    assert set(ui.results[card.id]) == set(methods)
    if not methods:
        assert "Select Gaussian" in card.status.value and not ui.busy
    for odf in set(card.visible) - set(methods):
        assert card.visible[odf].disabled
        assert card.bars[odf].bar_style != "success"


def test_new_cards_default_to_gaussian_and_selection_does_not_solve(ui, model, monkeypatch):
    card = publish(ui, model)
    before = pickle.dumps(ui.results[card.id])
    monkeypatch.setattr(ui, "_solve", lambda *a: pytest.fail("Selecting a method ran the solver"))
    card.calculate_odfs["numerical"].value = True
    card.calculate_odfs["gaussian"].value = False
    assert before == pickle.dumps(ui.results[card.id])
    assert all(checkbox.value for checkbox in card.visible.values())
    second = ui.add_phase_diagram()
    assert second.calculate_odfs["gaussian"].value
    assert not second.calculate_odfs["numerical"].value


def test_notebook_cell_catalog_and_placement():
    path = Path(__file__).parents[1] / "notebooks/gfvt_rods_widget.ipynb"
    nb = json.loads(path.read_text())
    assert len(nb["cells"]) == 2
    assert all(cell["cell_type"] == "code" for cell in nb["cells"])
    initialization = "".join(nb["cells"][0]["source"])
    launch = "".join(nb["cells"][1]["source"])
    assert "import gfvt_rods.rod_widgets as rod_widgets" in initialization
    assert "sys.path" not in initialization
    assert "RodPhaseDiagramWidget" in launch
    for label in ("4", "6", "8", "20", "35", "100"):
        assert f'"PEG{label}K": (mpeg{label}, rpeg{label}K)' in initialization
