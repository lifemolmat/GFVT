import pickle
from pathlib import Path

import matplotlib
import numpy as np

from gfvt import PD

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PICKLE_PATH = ROOT / "pickles" / "260424_output.pkl"
DATA_PATH = ROOT / "data" / "PEG_4K_BSA_Data.csv"


def _assert_same_value(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        np.testing.assert_allclose(left, right, rtol=0, atol=0)
    elif isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_same_value(left_item, right_item)
    elif isinstance(left, float) or isinstance(right, float):
        np.testing.assert_allclose(left, right, rtol=0, atol=0)
    else:
        assert left == right


def test_return_fields_match_notebook_gfvt_contract():
    assert PD.RETURN_FIELDS == (
        "phi_R",
        "Pr_Drop_p",
        "Pol_Drop_p",
        "a_Drop_p",
        "Pr_Sup_p",
        "Pol_Sup_p",
        "a_Sup_p",
        "qx_Drop_p",
        "dPidPhiP_Drop_p",
        "qx_Sup_p",
        "dPidPhiP_Sup_p",
        "Pi_Drop",
        "P0_Drop",
        "Pp_Drop",
        "Mu_Drop",
        "Mu0_Drop",
        "MuR_Drop",
        "Pi_Sup",
        "P0_Sup",
        "Pp_Sup",
        "Mu_Sup",
        "Mu0_Sup",
        "MuR_Sup",
        "cp",
        "cp_list",
        "tp",
        "tp_list",
        "g_Drop",
        "g_Sup",
        "h_Drop",
        "h_Sup",
        "params",
    )


def test_from_legacy_list_round_trips_existing_pickle_output():
    with PICKLE_PATH.open("rb") as handle:
        loaded = pickle.load(handle)

    legacy = loaded["gfvt4K_200mM_Z0"]
    pd = PD.from_legacy_list(legacy)
    round_trip = pd.to_legacy_list()

    assert len(round_trip) == 32
    for left, right in zip(round_trip, legacy):
        _assert_same_value(left, right)


def test_from_legacy_list_exposes_params_and_plot_aliases():
    with PICKLE_PATH.open("rb") as handle:
        loaded = pickle.load(handle)

    legacy = loaded["gfvt4K_200mM_Z0"]
    pd = PD.from_legacy_list(legacy)

    assert pd.Rpr_in == legacy[-1][0]
    assert pd.Rpol_in == legacy[-1][1]
    assert pd.mbsa == legacy[-1][2]
    assert pd.mpeg == legacy[-1][3]
    assert pd.solvent == legacy[-1][4]
    assert pd.N_steps == legacy[-1][5]
    assert pd.PhiP_min == legacy[-1][6]
    assert pd.PhiP_max == legacy[-1][7]
    assert pd.outguess_c == legacy[-1][8]
    assert pd.outguess_tp == legacy[-1][9]
    assert pd.guess_b == legacy[-1][10]
    assert pd.TC == legacy[-1][11]
    assert pd.phix_bsa == legacy[-1][12]
    assert pd.phix_peg == legacy[-1][13]
    assert pd.BSA_Drop is pd.Pr_Drop_p
    assert pd.PEG_Drop is pd.Pol_Drop_p
    assert pd.BSA_Sup is pd.Pr_Sup_p
    assert pd.PEG_Sup is pd.Pol_Sup_p


def test_constructor_makes_legacy_guess_b_explicit():
    pd = PD(
        3.0e-9,
        1.9e-9,
        66.4,
        4,
        "good",
        10,
        [0.24, 0.7],
        progress=False,
        verbose=False,
    )

    assert pd.guess_b == [0.0001, 0.5]
    assert pd.params[10] == [0.0001, 0.5]


def test_gfvt_small_run_matches_legacy_critical_and_triple_points():
    with PICKLE_PATH.open("rb") as handle:
        loaded = pickle.load(handle)

    legacy = loaded["gfvt4K_200mM_Z0"]
    pd = PD(
        3.0e-9,
        1.9e-9,
        66.4,
        4,
        "good",
        3,
        [0.24, 0.7],
        progress=False,
        verbose=False,
    )

    pd.GFVT()

    assert len(pd.to_legacy_list()) == 32
    np.testing.assert_allclose(pd.cp, legacy[23], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pd.tp, legacy[25], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pd.PhiP_min, legacy[-1][6], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pd.PhiP_max, legacy[-1][7], rtol=1e-12, atol=1e-12)


def test_pd_plotting_methods_return_axes():
    with PICKLE_PATH.open("rb") as handle:
        loaded = pickle.load(handle)
    pd = PD.from_legacy_list(loaded["gfvt4K_200mM_Z0"])

    plot_calls = [
        lambda ax: pd.molplot("PEG 4K, ", "orange", ax=ax),
        lambda ax: pd.molplot_phi("PEG 4K, ", "orange", ax=ax),
        lambda ax: pd.molplot_phiR("PEG 4K, ", "orange", crit_color="red", ax=ax),
        lambda ax: pd.molplot("PEG 4K, ", "orange", ax=ax, plim=np.inf),
        lambda ax: pd.molplot_phi_analytical("PEG 4K, ", "orange", crit_color="red", ax=ax),
        lambda ax: pd.molplot_analytical("PEG 4K, ", "orange", crit_color="red", ax=ax),
    ]

    for plot_call in plot_calls:
        fig, ax = plt.subplots()
        returned = plot_call(ax)
        assert returned is ax
        assert len(ax.lines) > 0 or len(ax.collections) > 0
        plt.close(fig)


def test_pd_data_plotting_methods_return_axes():
    with PICKLE_PATH.open("rb") as handle:
        loaded = pickle.load(handle)
    pd = PD.from_legacy_list(loaded["gfvt4K_200mM_Z0"])
    data = np.loadtxt(DATA_PATH, delimiter=",")

    fig, ax = plt.subplots()
    returned = PD.molplot_dat_err_csv(data, "PEG 4K", "orange", ax=ax)
    assert returned is ax
    assert len(ax.lines) > 0 or len(ax.collections) > 0
    plt.close(fig)

    fig, ax = plt.subplots()
    returned = pd.molplot_dat_err_csv_phi(data, "PEG 4K", "orange", ax=ax)
    assert returned is ax
    assert len(ax.lines) > 0 or len(ax.collections) > 0
    plt.close(fig)
