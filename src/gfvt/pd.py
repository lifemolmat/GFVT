"""Phase diagram object for GFVT calculations.

This module intentionally keeps the first migration conservative: the math in
``PD.GFVT`` mirrors the notebook implementation, while the result is exposed as
named attributes and can still be converted back to the legacy list format.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm


PARAM_FIELDS = (
    "Rpr_in",
    "Rpol_in",
    "mbsa",
    "mpeg",
    "solvent",
    "N_steps",
    "PhiP_min",
    "PhiP_max",
    "outguess_c",
    "outguess_tp",
    "guess_b",
    "TC",
    "phix_bsa",
    "phix_peg",
)

RETURN_FIELDS = (
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


class PD:
    """GFVT phase diagram with notebook-compatible state and outputs."""

    PARAM_FIELDS = PARAM_FIELDS
    RETURN_FIELDS = RETURN_FIELDS

    def __init__(
        self,
        Rpr_in: float,
        Rpol_in: float,
        mbsa: float,
        mpeg: float,
        solvent: str,
        N_steps: int,
        guess_c: Iterable[float],
        *,
        guess_b: Iterable[float] = (0.0001, 0.5),
        critical: bool = False,
        TC: float = 22,
        progress: bool = True,
        verbose: bool = True,
    ) -> None:
        self.Rpr_in = Rpr_in
        self.Rpol_in = Rpol_in
        self.mbsa = mbsa
        self.mpeg = mpeg
        self.solvent = solvent
        self.N_steps = int(N_steps)
        self.guess_c = list(guess_c)
        self.guess_b = list(guess_b)
        self.critical = critical
        self.TC = TC
        self.progress = progress
        self.verbose = verbose

        self.vbsa = 4 / 3 * np.pi * self.Rpr_in**3 * 6.022e23
        self.vpeg = 4 / 3 * np.pi * self.Rpol_in**3 * 6.022e23
        self.q = self.Rpol_in / self.Rpr_in
        self.phix_bsa = 1 / self.vbsa
        self.phix_peg = 1 / self.vpeg

        self.PhiP_min = None
        self.PhiP_max = None
        self.outguess_c = None
        self.outguess_tp = None

        for field in RETURN_FIELDS:
            setattr(self, field, None)
        self.params = self._build_params()
        self._sync_aliases()

    @classmethod
    def from_legacy_list(cls, values: list[Any]) -> "PD":
        """Build a ``PD`` object from the notebook's 32-item return list."""
        if len(values) != len(RETURN_FIELDS):
            raise ValueError(
                f"Expected {len(RETURN_FIELDS)} legacy values, got {len(values)}."
            )

        params = values[-1]
        if len(params) != len(PARAM_FIELDS):
            raise ValueError(
                f"Expected {len(PARAM_FIELDS)} params values, got {len(params)}."
            )

        (
            Rpr_in,
            Rpol_in,
            mbsa,
            mpeg,
            solvent,
            N_steps,
            PhiP_min,
            PhiP_max,
            outguess_c,
            outguess_tp,
            guess_b,
            TC,
            phix_bsa,
            phix_peg,
        ) = params

        pd = cls(
            Rpr_in,
            Rpol_in,
            mbsa,
            mpeg,
            solvent,
            N_steps,
            outguess_c,
            guess_b=guess_b,
            TC=TC,
            progress=False,
            verbose=False,
        )
        pd.PhiP_min = PhiP_min
        pd.PhiP_max = PhiP_max
        pd.outguess_c = outguess_c
        pd.outguess_tp = outguess_tp
        pd.phix_bsa = phix_bsa
        pd.phix_peg = phix_peg
        pd._assign_return_values(values)
        return pd

    def GFVT(
        self,
        *,
        PhiP_max_override: float | None = None,
        critical: bool | None = None,
        progress: bool | None = None,
        verbose: bool | None = None,
    ) -> "PD":
        """Run the notebook GFVT calculation and store results on this object."""
        critical = self.critical if critical is None else critical
        progress = self.progress if progress is None else progress
        verbose = self.verbose if verbose is None else verbose

        Rpr_in = self.Rpr_in
        Rpol_in = self.Rpol_in
        mbsa = self.mbsa
        mpeg = self.mpeg
        solvent = self.solvent
        N_steps = self.N_steps
        guess_c = self.guess_c
        guess_b = self.guess_b

        vbsa = self.vbsa
        vpeg = self.vpeg
        q = self.q

        if verbose:
            print(f"q = {q:.2f}, Rbsa = {Rpr_in * 1e9:.2f} nm, Rp = {Rpol_in * 1e9:.2f} nm")

        phix_peg = self.phix_peg
        phix_bsa = self.phix_bsa

        hp = 6.626e-34
        kb = 1.38e-23
        Nav = 6.022e23
        Rb = 8.314
        TC = self.TC
        T = 273.15 + TC
        Lam = hp / (2 * np.pi * mbsa * 1000 * kb * T / Nav) ** (1 / 2)

        def qxf(q, Phi_pol, solvent):
            if solvent == "good":
                dPidPhiP = (1 + 3.73 * Phi_pol**1.31) / q**3
                qx = 0.865 * (q / (1 + 3.95 * Phi_pol**1.54) ** (1 / 2)) ** 0.88
            elif solvent == "theta":
                dPidPhiP = (1 + 12.3 * Phi_pol**2) / q**3
                qx = 0.938 * (q / (1 + 6.02 * Phi_pol**2) ** (1 / 2)) ** 0.9
            elif solvent == "RG":
                dPidPhiP = (1 + 3.73 * Phi_pol**1.31) / q**3
                dq = q / (1 + 3.95 * Phi_pol**1.54) ** (1 / 2)
                qx = (1 + 3 * dq + 2.73 * dq**2 - 0.0975 * dq**3) ** (1 / 3) - 1
            else:
                dPidPhiP = 1
                qx = q
            return qx, dPidPhiP

        def Qsf(Phi, Phi_pol, q, solvent):
            y = Phi / (1 - Phi)
            qx, dPidPhiP = qxf(q, Phi_pol, solvent)
            a = 3 * qx + 3 * qx**2 + qx**3
            b = 9 / 2 * qx**2 + 3 * qx**3
            c = 3 * qx**3
            return a, b, c, y, dPidPhiP

        def beta_0(Phi, Phi_pol, q, solvent):
            a, b, c, y, dPidPhiP = Qsf(Phi, Phi_pol, q, solvent)
            Qs = a * y + b * y**2 + c * y**3
            return np.exp(-Qs)

        def beta_1(Phi, Phi_pol, q, solvent):
            beta0 = beta_0(Phi, Phi_pol, q, solvent)
            a, b, c, y, dPidPhiP = Qsf(Phi, Phi_pol, q, solvent)
            Q1 = a + 2 * b * y + 3 * c * y**2
            return -beta0 * Q1

        def beta_2(Phi, Phi_pol, q, solvent):
            beta0 = beta_0(Phi, Phi_pol, q, solvent)
            beta1 = beta_1(Phi, Phi_pol, q, solvent)
            a, b, c, y, dPidPhiP = Qsf(Phi, Phi_pol, q, solvent)
            Q1 = a + 2 * b * y + 3 * c * y**2
            Q2 = 2 * b + 6 * c * y
            return -beta0 * Q2 - beta1 * Q1

        def beta_3(Phi, Phi_pol, q, solvent):
            beta0 = beta_0(Phi, Phi_pol, q, solvent)
            beta1 = beta_1(Phi, Phi_pol, q, solvent)
            beta2 = beta_2(Phi, Phi_pol, q, solvent)
            a, b, c, y, dPidPhiP = Qsf(Phi, Phi_pol, q, solvent)
            Q1 = a + 2 * b * y + 3 * c * y**2
            Q2 = 2 * b + 6 * c * y
            Q3 = 6 * c
            return -beta0 * Q3 - 2 * beta1 * Q2 - beta2 * Q1

        def alpha(Phi, Phi_pol, q, solvent):
            beta0 = beta_0(Phi, Phi_pol, q, solvent)
            return (1 - Phi) * beta0

        def Mu0f(Phi, Lam, vc):
            return np.log(Lam**3 / vc) + np.log(Phi) + (3 - Phi) / (1 - Phi) ** 3 - 3

        def Mu0s(Phi, Lam, vc):
            Phicp = 0.741
            return np.log(Lam**3 / vc) + 2.1178 + 3 * np.log(Phi / (1 - Phi / Phicp)) + 3 / (
                1 - Phi / Phicp
            )

        def P0f(Phi):
            return (Phi + Phi**2 + Phi**3 - Phi**4) / (1 - Phi) ** 3

        def P0s(Phi):
            Phicp = 0.741
            return 3 * Phi / (1 - Phi / Phicp)

        def P0f1(Phi):
            y = Phi / (1 - Phi)
            return (1 + y) ** -2 + 8 * y + 6 * y**2

        def P0f2(Phi):
            y = Phi / (1 - Phi)
            return -2 * (1 + y) ** -3 + 8 + 12 * y

        def _grid_to(Phi_pol, step=1e-4, min_nodes=2):
            n_int = max(int(np.ceil(Phi_pol / step)), 1)
            n = max(n_int + 1, min_nodes)
            return np.linspace(0.0, float(Phi_pol), n)

        def Muf(Phi, Phi_pol, q, solvent):
            Phi_poln = _grid_to(Phi_pol, 1e-3)
            a, b, c, y, dPidPhiP = Qsf(Phi, Phi_poln, q, solvent)
            Qs = a * y + b * y**2 + c * y**3
            g = np.exp(-Qs) * (1 + (1 + y) * (a + 2 * y * b + 3 * c * y**2))
            M = g * dPidPhiP
            int_Mu = np.trapz(M, Phi_poln)
            Mu0 = Mu0f(Phi, Lam, vbsa)
            out = int_Mu + Mu0
            return out.item()

        def Pif(Phi, Phi_pol, q, solvent):
            Phi_poln = _grid_to(Phi_pol, 1e-3)
            a, b, c, y, dPidPhiP = Qsf(Phi, Phi_poln, q, solvent)
            Qs = a * y + b * y**2 + c * y**3
            h = np.exp(-Qs) * (1 + a * y + 2 * y**2 * b + 3 * c * y**3)
            M = h * dPidPhiP
            int_Pi = np.trapz(M, Phi_poln)
            P0 = P0f(Phi)
            out = P0 + int_Pi
            return out.item()

        def Pif1(Phi, Phi_pol, q, solvent):
            Phi_poln = _grid_to(Phi_pol, 1e-3)
            a, b, c, y, dPidPhiP = Qsf(Phi, Phi_poln, q, solvent)
            beta2 = beta_2(Phi, Phi_poln, q, solvent)
            M = -y * beta2 * dPidPhiP
            int_Pi = np.trapz(M, Phi_poln)
            P01 = P0f1(Phi)
            out = P01 + int_Pi
            return out.item()

        def Pif2(Phi, Phi_pol, q, solvent):
            Phi_poln = _grid_to(Phi_pol, 1e-3)
            a, b, c, y, dPidPhiP = Qsf(Phi, Phi_poln, q, solvent)
            beta2 = beta_2(Phi, Phi_poln, q, solvent)
            beta3 = beta_3(Phi, Phi_poln, q, solvent)
            M = -(beta2 + y * beta3) * dPidPhiP
            int_Pi = np.trapz(M, Phi_poln)
            P02 = P0f2(Phi)
            out = P02 + int_Pi
            return out.item()

        def Mus(Phi, Phi_pol, q, solvent):
            Phi_poln = _grid_to(Phi_pol, 1e-3)
            a, b, c, y, dPidPhiP = Qsf(Phi, Phi_poln, q, solvent)
            Qs = a * y + b * y**2 + c * y**3
            g = np.exp(-Qs) * (1 + (1 + y) * (a + 2 * y * b + 3 * c * y**2))
            M = g * dPidPhiP
            int_Mu = np.trapz(M, Phi_poln)
            Mu0 = Mu0s(Phi, Lam, vbsa)
            out = int_Mu + Mu0
            return out.item()

        def Pis(Phi, Phi_pol, q, solvent):
            Phi_poln = _grid_to(Phi_pol, 1e-3)
            a, b, c, y, dPidPhiP = Qsf(Phi, Phi_poln, q, solvent)
            Qs = a * y + b * y**2 + c * y**3
            h = np.exp(-Qs) * (1 + a * y + 2 * y**2 * b + 3 * c * y**3)
            M = h * dPidPhiP
            int_Pi = np.trapz(M, Phi_poln)
            P0 = P0s(Phi)
            out = P0 + int_Pi
            return out.item()

        def hfunc(Phi, Phi_pol, q, solvent):
            a, b, c, y, dPidPhiP = Qsf(Phi, Phi_pol, q, solvent)
            Qs = a * y + b * y**2 + c * y**3
            h = np.exp(-Qs) * (1 + a * y + 2 * y**2 * b + 3 * c * y**3)
            return h

        def gfunc(Phi, Phi_pol, q, solvent):
            a, b, c, y, dPidPhiP = Qsf(Phi, Phi_pol, q, solvent)
            Qs = a * y + b * y**2 + c * y**3
            g = np.exp(-Qs) * (1 + (1 + y) * (a + 2 * y * b + 3 * c * y**2))
            return g

        array_names = (
            "Pr_Drop_p",
            "Pol_Drop_p",
            "a_Drop_p",
            "qx_Drop_p",
            "dPidPhiP_Drop_p",
            "Pr_Sup_p",
            "Pol_Sup_p",
            "a_Sup_p",
            "qx_Sup_p",
            "dPidPhiP_Sup_p",
            "phi_R",
            "P0_Drop",
            "P0_Sup",
            "Pi_Drop",
            "Pi_Sup",
            "Pp_Drop",
            "Pp_Sup",
            "Mu_Drop",
            "Mu_Sup",
            "MuR_Drop",
            "MuR_Sup",
            "Mu0_Drop",
            "Mu0_Sup",
            "g_Drop",
            "g_Sup",
            "h_Drop",
            "h_Sup",
        )
        arrays = {name: np.empty(N_steps) for name in array_names}
        (
            Pr_Drop_p,
            Pol_Drop_p,
            a_Drop_p,
            qx_Drop_p,
            dPidPhiP_Drop_p,
            Pr_Sup_p,
            Pol_Sup_p,
            a_Sup_p,
            qx_Sup_p,
            dPidPhiP_Sup_p,
            phi_R,
            P0_Drop,
            P0_Sup,
            Pi_Drop,
            Pi_Sup,
            Pp_Drop,
            Pp_Sup,
            Mu_Drop,
            Mu_Sup,
            MuR_Drop,
            MuR_Sup,
            Mu0_Drop,
            Mu0_Sup,
            g_Drop,
            g_Sup,
            h_Drop,
            h_Sup,
        ) = (arrays[name] for name in array_names)
        
        cp_list = np.empty((N_steps, 8))
        tp_list = np.empty((0, 14))

        def fcrit(variabs):
            fout1 = Pif1(variabs[0], variabs[1], q, solvent)
            fout2 = Pif2(variabs[0], variabs[1], q, solvent)
            return [fout1, fout2]

        sol = least_squares(
            fcrit,
            (guess_c[0], guess_c[1]),
            bounds=((1e-16, 1e-16), (1, 500)),
            jac="2-point",
            gtol=3e-16,
            ftol=3e-16,
            xtol=3e-16,
        )

        crit1 = float(sol.x[0])
        crit2 = float(sol.x[1])
        outguess_c = [crit1, crit2]
        Pr_Crit = crit1 * phix_bsa
        Pol_Crit = alpha(crit1, crit2, q, solvent) * crit2 * phix_peg
        cp = [Pr_Crit, Pol_Crit]

        if verbose:
            print(f"BSA_crit = {crit1:.4f}, PEG_crit = {crit2:.4f}")
            print(f"[BSA]_crit = {Pr_Crit:.2f} mM, [PEG]_crit = {Pol_Crit:.2f} mM\n")

        def ftp(variabs):
            f1 = Muf(variabs[0], variabs[3], q, solvent) - Muf(variabs[1], variabs[3], q, solvent)
            f2 = Muf(variabs[1], variabs[3], q, solvent) - Mus(variabs[2], variabs[3], q, solvent)
            f3 = Pif(variabs[0], variabs[3], q, solvent) - Pif(variabs[1], variabs[3], q, solvent)
            f4 = Pif(variabs[1], variabs[3], q, solvent) - Pis(variabs[2], variabs[3], q, solvent)
            return [f1, f2, f3, f4]

        tp_sol = least_squares(
            ftp,
            (1e-4, outguess_c[0] + 0.4, 0.7, outguess_c[1]),
            bounds=((1e-16, outguess_c[0], outguess_c[0], 1e-16), (outguess_c[0], 1, 1, 500)),
            jac="2-point",
            gtol=3e-16,
            ftol=3e-16,
            xtol=3e-16,
        )

        tp1 = float(tp_sol.x[0])
        tp2 = float(tp_sol.x[1])
        tp3 = float(tp_sol.x[2])
        tp4 = float(tp_sol.x[3])
        outguess_tp = [tp1, tp2, tp3, tp4]
        g1 = tp1
        g2 = tp3
        Pr_tp1 = tp1 * phix_bsa
        Pr_tp2 = tp2 * phix_bsa
        Pr_tp3 = tp3 * phix_bsa
        Pol_tp1 = alpha(tp1, tp4, q, solvent) * tp4 * phix_peg
        Pol_tp2 = alpha(tp2, tp4, q, solvent) * tp4 * phix_peg
        Pol_tp3 = alpha(tp3, tp4, q, solvent) * tp4 * phix_peg
        tp = [Pr_tp1, Pr_tp2, Pr_tp3, Pol_tp1, Pol_tp2, Pol_tp3]

        if verbose:
            print(f"BSA_tp_G = {tp1:.4f}, BSA_tp_L = {tp2:.4f}, BSA_tp_S = {tp3:.4f}, PEG_tp = {tp4:.4f}")
            print(
                f"[BSA]_tp_G = {Pr_tp1:.2f} mM, [BSA]_tp_L = {Pr_tp2:.2f} mM, "
                f"[BSA]_tp_S = {Pr_tp3:.2f} mM, [PEG]_tp_G = {Pol_tp1:.2f} mM, "
                f"[PEG]_tp_L = {Pol_tp2:.2f} mM, [PEG]_tp_S = {Pol_tp3:.2f} mM\n"
            )

        PhiP_min = crit2
        PhiP_max = tp4 if PhiP_max_override is None else PhiP_max_override

        if critical:
            if solvent == "good":
                qrcep = 0.388
            elif solvent == "theta":
                qrcep = 0.337
            elif solvent == "RG":
                qrcep = 0.388
            else:
                qrcep = 0.1

            Qlist = np.flip(np.linspace(qrcep, 10, N_steps))

            for idx, qr in tqdm(
                enumerate(Qlist),
                desc="Calculating Critical and Triple Lines",
                total=len(Qlist),
                unit="step",
                disable=not progress,
            ):
                Rpol_qr = qr * Rpr_in
                phix_peg_qr = 1 / (4 / 3 * np.pi * Rpol_qr**3 * 6.022e23)

                def fcrit(variabs):
                    fout1 = Pif1(variabs[0], variabs[1], qr, solvent)
                    fout2 = Pif2(variabs[0], variabs[1], qr, solvent)
                    return [fout1, fout2]

                sol = least_squares(
                    fcrit,
                    (crit1, crit2),
                    bounds=((1e-16, 1e-16), (1, 500)),
                    jac="2-point",
                    gtol=3e-16,
                    ftol=3e-16,
                    xtol=3e-16,
                )

                crit1 = float(sol.x[0])
                crit2 = float(sol.x[1])
                Pr_Crit_t = crit1 * phix_bsa
                Pol_Crit_t = alpha(crit1, crit2, qr, solvent) * crit2 * phix_peg_qr
                cp_list[idx] = [qr, Rpr_in, Rpol_qr, phix_peg_qr, crit1, crit2, Pr_Crit_t, Pol_Crit_t]

                if qr > qrcep:

                    def ftp(variabs):
                        f1 = Muf(variabs[0], variabs[3], qr, solvent) - Muf(
                            variabs[1], variabs[3], qr, solvent
                        )
                        f2 = Muf(variabs[1], variabs[3], qr, solvent) - Mus(
                            variabs[2], variabs[3], qr, solvent
                        )
                        f3 = Pif(variabs[0], variabs[3], qr, solvent) - Pif(
                            variabs[1], variabs[3], qr, solvent
                        )
                        f4 = Pif(variabs[1], variabs[3], qr, solvent) - Pis(
                            variabs[2], variabs[3], qr, solvent
                        )
                        return [f1, f2, f3, f4]

                    tp_sol = least_squares(
                        ftp,
                        (tp1, tp2, tp3, tp4),
                        bounds=((1e-16, crit1, crit1, 1e-16), (crit1, 1, 1, 500)),
                        jac="2-point",
                        gtol=3e-16,
                        ftol=3e-16,
                        xtol=3e-16,
                    )

                    tp1 = float(tp_sol.x[0])
                    tp2 = float(tp_sol.x[1])
                    tp3 = float(tp_sol.x[2])
                    tp4 = float(tp_sol.x[3])
                    Pr_tp1 = tp1 * phix_bsa
                    Pr_tp2 = tp2 * phix_bsa
                    Pr_tp3 = tp3 * phix_bsa
                    Pol_tp1 = alpha(tp1, tp4, qr, solvent) * tp4 * phix_peg_qr
                    Pol_tp2 = alpha(tp2, tp4, qr, solvent) * tp4 * phix_peg_qr
                    Pol_tp3 = alpha(tp3, tp4, qr, solvent) * tp4 * phix_peg_qr
                    tp_list = np.vstack(
                        [
                            tp_list,
                            [
                                qr,
                                Rpr_in,
                                Rpol_qr,
                                phix_peg_qr,
                                tp1,
                                tp2,
                                tp3,
                                tp4,
                                Pr_tp1,
                                Pr_tp2,
                                Pr_tp3,
                                Pol_tp1,
                                Pol_tp2,
                                Pol_tp3,
                            ],
                        ]
                    )
        else:
            cp_list = []
            tp_list = []

        PhiP = np.linspace(PhiP_min, PhiP_max, N_steps)
        j = 0

        for i in tqdm(np.flip(PhiP), desc="Calculating Binodal", unit="step", disable=not progress):

            def f(variabs):
                f1 = Muf(variabs[0], i, q, solvent) - Muf(variabs[1], i, q, solvent)
                f2 = Pif(variabs[0], i, q, solvent) - Pif(variabs[1], i, q, solvent)
                return [f1, f2]

            sol = least_squares(
                f,
                (g1, g2),
                bounds=((1e-16, 1e-16), (0.9999999999999999999, 0.9999999999999999999)),
                jac="2-point",
                gtol=3e-16,
                ftol=3e-16,
                xtol=3e-16,
            )

            g1 = float(sol.x[0])
            g2 = float(sol.x[1])
            Pr_Sup_p[j] = float(g1) * phix_bsa
            Pr_Drop_p[j] = float(g2) * phix_bsa

            a1, b1, c1, y1, _ = Qsf(g1, i, q, solvent)
            a2, b2, c2, y2, _ = Qsf(g2, i, q, solvent)
            alpha1 = (1 - g1) * np.exp(-(a1 * y1 + b1 * y1**2 + c1 * y1**3))
            alpha2 = (1 - g2) * np.exp(-(a2 * y2 + b2 * y2**2 + c2 * y2**3))

            Pol_Sup_p[j] = alpha1 * i * phix_peg
            Pol_Drop_p[j] = alpha2 * i * phix_peg
            a_Sup_p[j] = alpha1
            a_Drop_p[j] = alpha2
            phi_R[j] = i

            qx_Sup_p[j], dPidPhiP_Sup_p = qxf(q, i, solvent)
            qx_Drop_p[j], dPidPhiP_Drop_p = qxf(q, i, solvent)

            pi_drop = Pif(g2, i, q, solvent)
            pi_sup = Pif(g1, i, q, solvent)
            mu_drop = Muf(g2, i, q, solvent)
            mu_sup = Muf(g1, i, q, solvent)
            p0_drop = P0f(g2)
            p0_sup = P0f(g1)
            mu0_drop = Mu0f(g2, Lam, vbsa)
            mu0_sup = Mu0f(g1, Lam, vbsa)
            pressure_scale = phix_bsa * Rb * T

            Pi_Drop[j] = pi_drop * pressure_scale
            Pi_Sup[j] = pi_sup * pressure_scale
            Mu_Drop[j] = mu_drop
            Mu_Sup[j] = mu_sup
            P0_Drop[j] = p0_drop * pressure_scale
            P0_Sup[j] = p0_sup * pressure_scale
            Pp_Drop[j] = (pi_drop - p0_drop) * pressure_scale
            Pp_Sup[j] = (pi_sup - p0_sup) * pressure_scale
            Mu0_Drop[j] = mu0_drop
            Mu0_Sup[j] = mu0_sup
            MuR_Drop[j] = mu_drop - mu0_drop
            MuR_Sup[j] = mu_sup - mu0_sup
            g_Drop[j] = gfunc(g2, i, q, solvent)
            g_Sup[j] = gfunc(g1, i, q, solvent)
            h_Drop[j] = hfunc(g2, i, q, solvent)
            h_Sup[j] = hfunc(g1, i, q, solvent)
            j += 1

        params = [
            Rpr_in,
            Rpol_in,
            mbsa,
            mpeg,
            solvent,
            N_steps,
            PhiP_min,
            PhiP_max,
            outguess_c,
            outguess_tp,
            guess_b,
            TC,
            phix_bsa,
            phix_peg,
        ]

        self.vbsa = vbsa
        self.vpeg = vpeg
        self.q = q
        self.phix_bsa = phix_bsa
        self.phix_peg = phix_peg
        self.PhiP_min = PhiP_min
        self.PhiP_max = PhiP_max
        self.outguess_c = outguess_c
        self.outguess_tp = outguess_tp

        self._assign_return_values(
            [
                phi_R,
                Pr_Drop_p,
                Pol_Drop_p,
                a_Drop_p,
                Pr_Sup_p,
                Pol_Sup_p,
                a_Sup_p,
                qx_Drop_p,
                dPidPhiP_Drop_p,
                qx_Sup_p,
                dPidPhiP_Sup_p,
                Pi_Drop,
                P0_Drop,
                Pp_Drop,
                Mu_Drop,
                Mu0_Drop,
                MuR_Drop,
                Pi_Sup,
                P0_Sup,
                Pp_Sup,
                Mu_Sup,
                Mu0_Sup,
                MuR_Sup,
                cp,
                cp_list,
                tp,
                tp_list,
                g_Drop,
                g_Sup,
                h_Drop,
                h_Sup,
                params,
            ]
        )
        return self

    def GFVT_full(
        self,
        PhiP_max: float = 1,
        *,
        critical: bool | None = None,
        progress: bool | None = None,
        verbose: bool | None = None,
    ) -> "PD":
        """Run the notebook GFVT_full variant with a caller-specified PhiP_max."""
        return self.GFVT(
            PhiP_max_override=PhiP_max,
            critical=critical,
            progress=progress,
            verbose=verbose,
        )

    @staticmethod
    def molplot_dat_err_csv(
        data,
        plabel,
        pcolor,
        mbsa=0,
        mpeg=0,
        n_tie=False,
        plot_tot=False,
        ax=None,
        show_legend=False,
    ):
        """Plot experimental binodal CSV data with error bars."""
        ax = PD._get_axis(ax)
        (
            BN_BSA_tot,
            BN_BSA_tot_err,
            BN_BSA_sup,
            BN_BSA_sup_err,
            BN_BSA_drop,
            BN_BSA_drop_err,
            BN_PEG_tot,
            BN_PEG_tot_err,
            BN_PEG_sup,
            BN_PEG_sup_err,
            BN_PEG_drop,
            BN_PEG_drop_err,
            CP_BSA,
            CP_BSA_err,
            CP_PEG,
            CP_PEG_err,
        ) = PD._parse_binodal_data(data, mbsa=mbsa, mpeg=mpeg)

        PD._plot_data_errorbars(
            ax,
            plabel,
            pcolor,
            BN_BSA_tot,
            BN_BSA_tot_err,
            BN_BSA_sup,
            BN_BSA_sup_err,
            BN_BSA_drop,
            BN_BSA_drop_err,
            BN_PEG_tot,
            BN_PEG_tot_err,
            BN_PEG_sup,
            BN_PEG_sup_err,
            BN_PEG_drop,
            BN_PEG_drop_err,
            CP_BSA,
            CP_BSA_err,
            CP_PEG,
            CP_PEG_err,
            n_tie=n_tie,
            plot_tot=plot_tot,
        )
        PD._maybe_legend(ax, show_legend)
        return ax

    def molplot_dat_err_csv_phi(
        self,
        data,
        plabel,
        pcolor,
        ax=None,
        mbsa=0,
        mpeg=0,
        n_tie=False,
        plot_tot=False,
        gamma=1 / (3 - 1 / 0.6379),
        show_legend=False,
    ):
        """Plot experimental CSV data in volume-fraction coordinates."""
        self._require_results()
        ax = self._get_axis(ax)
        q = self.Rpol_in / self.Rpr_in
        qq = 1 if gamma == 0 else q ** -(1 / gamma)
        mbsa = self.mbsa if mbsa == 0 else mbsa
        mpeg = self.mpeg if mpeg == 0 else mpeg

        (
            BN_BSA_tot,
            BN_BSA_tot_err,
            BN_BSA_sup,
            BN_BSA_sup_err,
            BN_BSA_drop,
            BN_BSA_drop_err,
            BN_PEG_tot,
            BN_PEG_tot_err,
            BN_PEG_sup,
            BN_PEG_sup_err,
            BN_PEG_drop,
            BN_PEG_drop_err,
            CP_BSA,
            CP_BSA_err,
            CP_PEG,
            CP_PEG_err,
        ) = self._parse_binodal_data(
            data,
            mbsa=mbsa,
            mpeg=mpeg,
            phix_bsa=self.phix_bsa,
            phix_peg=self.phix_peg,
            peg_scale=qq,
        )

        self._plot_data_errorbars(
            ax,
            plabel,
            pcolor,
            BN_BSA_tot,
            BN_BSA_tot_err,
            BN_BSA_sup,
            BN_BSA_sup_err,
            BN_BSA_drop,
            BN_BSA_drop_err,
            BN_PEG_tot,
            BN_PEG_tot_err,
            BN_PEG_sup,
            BN_PEG_sup_err,
            BN_PEG_drop,
            BN_PEG_drop_err,
            CP_BSA,
            CP_BSA_err,
            CP_PEG,
            CP_PEG_err,
            n_tie=n_tie,
            plot_tot=plot_tot,
        )
        self._maybe_legend(ax, show_legend)
        return ax

    def molplot(
        self,
        plabel,
        pcolor,
        crit_color=False,
        FVT_color=False,
        n_tie=0,
        mbsa=0,
        mpeg=0,
        ax=None,
        show_legend=False,
        mass=False,
        plim=500,
    ):
        """Plot the GFVT binodal in concentration or mass coordinates."""
        self._require_results()
        self._reject_fvt_overlay(FVT_color)
        ax = self._get_axis(ax)
        mbsa, mpeg = self._plot_masses(mbsa, mpeg)
        full_label = self._full_label(plabel)

        if mass:
            b_bsa_A = self.Pr_Drop_p * mbsa
            b_bsa_B = self.Pr_Sup_p * mbsa
            b_PEG_A = self.Pol_Drop_p * mpeg
            b_PEG_B = self.Pol_Sup_p * mpeg
            c_bsa = self.cp[0] * mbsa
            c_PEG = self.cp[1] * mpeg
        else:
            b_bsa_A = self.Pr_Drop_p
            b_bsa_B = self.Pr_Sup_p
            b_PEG_A = self.Pol_Drop_p
            b_PEG_B = self.Pol_Sup_p
            c_bsa, c_PEG = self.cp

        idx = np.abs(np.asarray(self.Pol_Sup_p) - plim).argmin()
        self._plot_binodal_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, full_label)
        ax.plot(b_bsa_A[idx:], b_PEG_A[idx:], "-", color=pcolor, linewidth=10, alpha=0.4)
        ax.plot(b_bsa_B[idx:], b_PEG_B[idx:], "-", color=pcolor, linewidth=10, alpha=0.4)
        self._plot_tie_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, n_tie)
        self._plot_critical_lines(ax, crit_color, full_label, mass=mass, mbsa=mbsa, mpeg=mpeg)
        ax.scatter(c_bsa, c_PEG, s=300, color=pcolor, marker="*", label=full_label + " critical point")
        self._maybe_legend(ax, show_legend)
        return ax

    def molplot_phi(
        self,
        plabel,
        pcolor,
        ax=None,
        crit_color=False,
        FVT_color=False,
        n_tie=0,
        mbsa=0,
        mpeg=0,
        gamma=1 / (3 - 1 / 0.6379),
        show_legend=False,
        mass=False,
    ):
        """Plot the GFVT binodal in volume-fraction coordinates."""
        self._require_results()
        self._reject_fvt_overlay(FVT_color)
        ax = self._get_axis(ax)
        mbsa, mpeg = self._plot_masses(mbsa, mpeg)
        full_label = self._full_label(plabel)
        q = self.Rpol_in / self.Rpr_in
        qq = q ** -(1 / gamma)

        if mass:
            b_bsa_A = self.Pr_Drop_p * mbsa
            b_bsa_B = self.Pr_Sup_p * mbsa
            b_PEG_A = qq * self.Pol_Drop_p * mpeg
            b_PEG_B = qq * self.Pol_Sup_p * mpeg
            c_bsa = self.cp[0] * mbsa
            c_PEG = qq * self.cp[1] * mpeg
        else:
            b_bsa_A = self.Pr_Drop_p / self.phix_bsa
            b_bsa_B = self.Pr_Sup_p / self.phix_bsa
            b_PEG_A = qq * self.Pol_Drop_p / self.phix_peg
            b_PEG_B = qq * self.Pol_Sup_p / self.phix_peg
            c_bsa = self.cp[0] / self.phix_bsa
            c_PEG = qq * self.cp[1] / self.phix_peg

        self._plot_binodal_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, full_label)
        self._plot_tie_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, n_tie)
        self._plot_phi_critical_lines(ax, crit_color, full_label, gamma=gamma)
        ax.scatter(c_bsa, c_PEG, s=300, color=pcolor, marker="*", label=full_label + " critical")
        self._maybe_legend(ax, show_legend)
        return ax

    def molplot_phiR(
        self,
        plabel,
        pcolor,
        crit_color=False,
        FVT_color=False,
        n_tie=0,
        ax=None,
        show_legend=False,
    ):
        """Plot the GFVT binodal against reservoir polymer volume fraction."""
        self._require_results()
        self._reject_fvt_overlay(FVT_color)
        ax = self._get_axis(ax)
        full_label = self._full_label(plabel)
        q = self.Rpol_in / self.Rpr_in
        gamma, _, qq, y_a_L, bsa_a_L, bsa_a_G = self._analytical_base_curves()
        color = crit_color or pcolor

        b_bsa_A = self.Pr_Drop_p / self.phix_bsa
        b_bsa_B = self.Pr_Sup_p / self.phix_bsa
        b_PEG_A = np.asarray(self.phi_R) * qq
        b_PEG_B = np.asarray(self.phi_R) * qq

        self._plot_binodal_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, full_label)
        self._plot_tie_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, n_tie)
        ax.scatter(self.outguess_c[0], qq * self.outguess_c[1], color=color, s=300, label=full_label + " critical point")
        ax.plot(bsa_a_L, y_a_L * qq, "--", color=color, label=full_label + " analytical Liquid")
        ax.plot(bsa_a_G, y_a_L * qq, "--", color=color, label=full_label + " analytical Gas")
        self._maybe_legend(ax, show_legend)
        return ax

    def molplot_phi_analytical(
        self,
        plabel,
        pcolor,
        ax=None,
        crit_color=False,
        FVT_color=False,
        n_tie=0,
        show_legend=False,
    ):
        """Plot GFVT volume-fraction binodal with analytical approximations."""
        self._require_results()
        self._reject_fvt_overlay(FVT_color)
        ax = self._get_axis(ax)
        full_label = self._full_label(plabel)
        color = crit_color or pcolor
        _, _, _, y_a_L, bsa_a_L, bsa_a_G = self._analytical_base_curves()

        b_bsa_A = self.Pr_Drop_p / self.phix_bsa
        b_bsa_B = self.Pr_Sup_p / self.phix_bsa
        b_PEG_A = self.Pol_Drop_p / self.phix_peg
        b_PEG_B = self.Pol_Sup_p / self.phix_peg
        phi_a_L = [self._alpha(Phi, Phi_pol) * Phi_pol for Phi_pol, Phi in zip(y_a_L, bsa_a_L)]
        phi_a_G = [self._alpha(Phi, Phi_pol) * Phi_pol for Phi_pol, Phi in zip(y_a_L, bsa_a_G)]

        self._plot_binodal_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, full_label)
        self._plot_tie_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, n_tie)
        ax.plot(bsa_a_L, phi_a_L, "--", color=color, label=full_label + " analytical Liquid")
        ax.plot(bsa_a_G, phi_a_G, "--", color=color, label=full_label + " analytical Gas")
        self._maybe_legend(ax, show_legend)
        return ax

    def molplot_analytical(
        self,
        plabel,
        pcolor,
        ax=None,
        crit_color=False,
        mbsa=0,
        mpeg=0,
        FVT_color=False,
        n_tie=0,
        show_legend=False,
        mass=False,
    ):
        """Plot GFVT concentration binodal with analytical approximations."""
        self._require_results()
        self._reject_fvt_overlay(FVT_color)
        ax = self._get_axis(ax)
        mbsa, mpeg = self._plot_masses(mbsa, mpeg)
        full_label = self._full_label(plabel)
        color = crit_color or pcolor
        _, _, _, y_a_L, phi_bsa_a_L, phi_bsa_a_G = self._analytical_base_curves()

        if mass:
            b_bsa_A = self.Pr_Drop_p * mbsa
            b_bsa_B = self.Pr_Sup_p * mbsa
            b_PEG_A = self.Pol_Drop_p * mpeg
            b_PEG_B = self.Pol_Sup_p * mpeg
            phi_a_L = [
                self._alpha(Phi, Phi_pol) * Phi_pol * self.phix_peg * mpeg
                for Phi_pol, Phi in zip(y_a_L, phi_bsa_a_L)
            ]
            phi_a_G = [
                self._alpha(Phi, Phi_pol) * Phi_pol * self.phix_peg * mpeg
                for Phi_pol, Phi in zip(y_a_L, phi_bsa_a_G)
            ]
            bsa_a_L = [Phi * self.phix_bsa * mbsa for Phi in phi_bsa_a_L]
            bsa_a_G = [Phi * self.phix_bsa * mbsa for Phi in phi_bsa_a_G]
        else:
            b_bsa_A = self.Pr_Drop_p
            b_bsa_B = self.Pr_Sup_p
            b_PEG_A = self.Pol_Drop_p
            b_PEG_B = self.Pol_Sup_p
            phi_a_L = [
                self._alpha(Phi, Phi_pol) * Phi_pol * self.phix_peg
                for Phi_pol, Phi in zip(y_a_L, phi_bsa_a_L)
            ]
            phi_a_G = [
                self._alpha(Phi, Phi_pol) * Phi_pol * self.phix_peg
                for Phi_pol, Phi in zip(y_a_L, phi_bsa_a_G)
            ]
            bsa_a_L = [Phi * self.phix_bsa for Phi in phi_bsa_a_L]
            bsa_a_G = [Phi * self.phix_bsa for Phi in phi_bsa_a_G]

        self._plot_binodal_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, full_label)
        self._plot_tie_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, n_tie)
        ax.plot(bsa_a_L, phi_a_L, "--", color=color, label=full_label + " analytical Liquid")
        ax.plot(bsa_a_G, phi_a_G, "--", color=color, label=full_label + " analytical Gas")
        self._maybe_legend(ax, show_legend)
        return ax

    def d_Rg(
        self,
        pcolor,
        plabel,
        ax=None,
        plim=100,
        *,
        mass=False,
        highlight=True,
        n_points=100000,
    ):
        """Plot the polymer depletion thickness approximation."""
        self._require_results()
        ax = self._get_axis(ax)
        q = self.Rpol_in / self.Rpr_in
        x_scale = self.phix_peg * (self.mpeg if mass else 1)
        phi_r = np.asarray(self.phi_R)
        phi_grid = np.linspace(0, 2000, n_points)
        phi_grid_approx = np.linspace(0, 2000, n_points)

        depletion = self.Rpr_in * 1e9 * 0.865 * q**0.88 / (1 + 3.95 * phi_grid**1.54) ** 0.44
        with np.errstate(divide="ignore", invalid="ignore"):
            depletion_approx = (
                self.Rpr_in
                * 1e9
                * 0.865
                * q**0.88
                / (3.95 * phi_grid_approx**1.54) ** 0.44
            )

        ax.plot(phi_grid * x_scale, depletion, label=plabel, color=pcolor)
        ax.plot(phi_grid_approx * x_scale, depletion_approx, color=pcolor, linestyle="--", linewidth=1)
        ax.scatter(phi_r[-1] * x_scale, self.Rpr_in * 1e9 * self.qx_Drop_p[-1], color=pcolor, s=150)

        idx = np.abs(np.asarray(self.Pol_Sup_p) - plim).argmin()
        start = idx if highlight else 0
        linewidth = 10 if highlight else 5
        ax.plot(
            phi_r[start:] * x_scale,
            self.Rpr_in * 1e9 * np.asarray(self.qx_Sup_p)[start:],
            color=pcolor,
            linewidth=linewidth,
            alpha=0.6,
        )
        return ax

    def pi_scale(
        self,
        pcolor,
        plabel,
        ax=None,
        plim=100,
        *,
        mass=False,
        reduced=False,
        gamma=0.77,
        highlight=True,
        secondary_axis=None,
        n_points=100000,
    ):
        """Plot the polymer osmotic pressure scaling curve."""
        self._require_results()
        ax = self._get_axis(ax)
        q = self.Rpol_in / self.Rpr_in
        x_scale = self.phix_peg * (self.mpeg if mass else 1)
        y_scale = q**-3 if (mass or reduced) else self.phix_peg
        phi_r = np.asarray(self.phi_R)
        phi_grid = np.linspace(0, 2000, n_points)
        total = phi_grid + 1.62 * phi_grid ** (3 * gamma)
        polymer = 1.62 * phi_grid ** (3 * gamma)

        ax.plot(phi_grid * x_scale, y_scale * total, label=plabel, color=pcolor)
        ax.plot(phi_grid * x_scale, y_scale * polymer, color=pcolor, linestyle="--", linewidth=1)

        if highlight:
            idx = np.abs(np.asarray(self.Pol_Sup_p) - plim).argmin()
            phi_highlight = phi_r[idx:]
            total_highlight = phi_highlight + 1.62 * phi_highlight ** (3 * gamma)
            ax.plot(phi_highlight * x_scale, y_scale * total_highlight, color=pcolor, linewidth=10, alpha=0.6)
            ax.scatter(
                phi_r[-1] * x_scale,
                y_scale * (phi_r[-1] + 1.62 * phi_r[-1] ** (3 * gamma)),
                color=pcolor,
                s=150,
                marker="o",
            )

        if secondary_axis is None:
            secondary_axis = not (mass or reduced)
        if secondary_axis:
            rb = 8.314
            temperature = 273.15 + self.TC

            def inverse(x):
                return 1000 * x / rb / temperature

            def forward(x):
                return x * rb * temperature / 1000

            secay = ax.secondary_yaxis("right", functions=(forward, inverse))
            secay.set_ylabel(r"$\Pi$ (kPa)", fontsize=28)
            secay.tick_params(axis="y", labelsize=24)
        return ax

    def a_plot(
        self,
        pcolor,
        plabel,
        ax=None,
        *,
        show_hard_sphere=True,
        linewidth=3,
    ):
        """Plot free volume fraction against protein concentration."""
        self._require_results()
        ax = self._get_axis(ax)
        q = self.Rpol_in / self.Rpr_in
        phi_bsa_sup = np.asarray(self.Pr_Sup_p) / self.phix_bsa

        ax.scatter(self.Pr_Drop_p[-1], self.a_Drop_p[-1], color=pcolor, s=150, marker="o")
        ax.plot(self.Pr_Drop_p, self.a_Drop_p, label=plabel, color=pcolor, linewidth=linewidth)
        ax.plot(self.Pr_Sup_p, self.a_Sup_p, color=pcolor, linewidth=linewidth)

        if show_hard_sphere:
            phi_bsa_range = np.linspace(0, 0.6, 1000)
            ax.plot(
                self.Pr_Sup_p,
                [1 - phi * (1 + qx) ** 3 for qx, phi in zip(self.qx_Sup_p, phi_bsa_sup)],
                color=pcolor,
                linewidth=2,
                linestyle="--",
            )
            ax.plot(
                phi_bsa_range * self.phix_bsa,
                [1 - phi * (1 + q) ** 3 for phi in phi_bsa_range],
                color=pcolor,
                linewidth=1,
                linestyle="--",
            )
        return ax

    def a_plot_peg(self, pcolor, plabel, ax=None, *, linewidth=3):
        """Plot free volume fraction against polymer concentration."""
        self._require_results()
        ax = self._get_axis(ax)
        ax.scatter(self.Pol_Drop_p[-1], self.a_Drop_p[-1], color=pcolor, s=150, marker="o")
        ax.plot(self.Pol_Drop_p, self.a_Drop_p, label=plabel, color=pcolor, linewidth=linewidth)
        ax.plot(self.Pol_Sup_p, self.a_Sup_p, color=pcolor, linewidth=linewidth)
        return ax

    def g_plot(self, pcolor, plabel, ax=None, *, gamma=0.77):
        """Plot the polymer contribution weighted by g."""
        self._require_results()
        ax = self._get_axis(ax)
        pressure = self.phix_peg * (np.asarray(self.phi_R) + 1.62 * np.asarray(self.phi_R) ** (3 * gamma))
        ax.semilogx(self.Pr_Drop_p, pressure * self.g_Drop, label=plabel, color=pcolor, linestyle="--", linewidth=1)
        ax.semilogx(self.Pr_Sup_p, pressure * self.g_Sup, color=pcolor, linestyle="--", linewidth=1)
        ax.scatter(self.Pr_Drop_p[-1], pressure[-1] * self.g_Drop[-1], color=pcolor, s=150, marker="o")
        return ax

    def h_plot(self, pcolor, plabel, ax=None):
        """Plot h against protein concentration."""
        self._require_results()
        ax = self._get_axis(ax)
        ax.scatter(self.Pr_Drop_p[-1], self.h_Drop[-1], color=pcolor, s=150, marker="o")
        ax.plot(self.Pr_Drop_p, self.h_Drop, label=plabel, color=pcolor)
        ax.plot(self.Pr_Sup_p, self.h_Sup, color=pcolor, linewidth=1, linestyle="--")
        return ax

    def mu_plot(
        self,
        pcolor,
        plabel,
        ax=None,
        plim=500,
        *,
        depletion=False,
        reference_band=False,
    ):
        """Plot chemical potential curves."""
        self._require_results()
        ax = self._get_axis(ax)
        idx = np.abs(np.asarray(self.Pol_Sup_p) - plim).argmin()

        if depletion:
            mu_drop = np.asarray(self.Mu_Drop) - np.asarray(self.Mu0_Drop)
            mu_sup = np.asarray(self.Mu_Sup) - np.asarray(self.Mu0_Sup)
            ax.scatter(self.Pr_Drop_p[-1], mu_drop[-1], color=pcolor, s=150, marker="o")
            ax.semilogx(self.Pr_Drop_p, mu_drop, label=plabel, color=pcolor)
            ax.semilogx(self.Pr_Sup_p, mu_sup, color=pcolor, linewidth=1, linestyle="--")
            if reference_band:
                tmpx = np.linspace(1e-3, 10, 1000)
                tmpy = [12 - 6 * np.log(tmp) for tmp in tmpx]
                ax.semilogx(tmpx, tmpy, color="gray", linewidth=200, alpha=0.05)
            return ax

        ax.plot(self.Pr_Sup_p, self.Mu_Sup, label=plabel + " total supernatant", color=pcolor)
        ax.plot(self.Pr_Drop_p, self.Mu_Drop, label=plabel + " total droplet", color=pcolor)
        ax.plot(self.Pr_Sup_p[idx:], self.Mu_Sup[idx:], color=pcolor, linewidth=10, alpha=0.6)
        ax.plot(self.Pr_Drop_p[idx:], self.Mu_Drop[idx:], color=pcolor, linewidth=10, alpha=0.6)
        ax.scatter(self.Pr_Sup_p[-1], self.Mu_Sup[-1], color=pcolor, marker="*", s=200)
        ax.plot(self.Pr_Sup_p, self.Mu0_Sup, label=plabel + " HS supernatant", color=pcolor, linestyle="--")
        ax.plot(self.Pr_Drop_p, self.Mu0_Drop, label=plabel + " HS droplet", color=pcolor, linestyle="--")
        return ax

    def Pi_Mu_a_plots(self, axes, pcolor, plabel, mpeg=None, *, verbose=False):
        """Plot pressure, chemical potential, and free-volume summary panels."""
        self._require_results()
        q = self.Rpol_in / self.Rpr_in
        if verbose:
            print(self.phix_bsa)
            print(self.phix_peg)
            print(f"rescale with q = {q:.2f}, phix_peg/q**3={self.phix_peg * q**3:.2f}")

        axes[0].plot(self.Pr_Drop_p, self.Pi_Drop / 1000, label="Droplet Total", color=pcolor)
        axes[0].plot(self.Pr_Drop_p, (self.Pi_Drop - self.P0_Drop) / 1000, label="Polymer", linestyle="--", color=pcolor)
        axes[0].plot(self.Pr_Drop_p, self.P0_Drop / 1000, label="Droplet HS", linestyle=":", color=pcolor)
        axes[0].plot(self.Pr_Sup_p, self.Pi_Sup / 1000, label="Supernatant Total", color=pcolor)
        axes[0].plot(self.Pr_Sup_p, (self.Pi_Sup - self.P0_Sup) / 1000, label="Polymer", linestyle="--", color=pcolor)
        axes[0].plot(self.Pr_Sup_p, self.P0_Sup / 1000, label="Sup HS", linestyle=":", color=pcolor)
        axes[0].set_title(r"$\Pi$ (kPa)")
        axes[0].set_xlabel(r"$[BSA]_{sup}$ (mM)")
        axes[0].set_ylabel(r"$\Pi$ (kPa)")
        axes[0].set_xlim(0, 7)
        axes[0].set_ylim(0, 300)

        axes[1].plot(self.Pr_Sup_p, self.Mu_Sup, label="Mu Total Supernatant", color=pcolor)
        axes[1].plot(self.Pr_Drop_p, self.Mu_Drop, label="Mu Total Drop", color=pcolor)
        axes[1].plot(self.Pr_Sup_p, self.MuR_Sup, label="Mu Depletion Supernatant", color=pcolor, linestyle="--")
        axes[1].plot(self.Pr_Drop_p, self.MuR_Drop, label="Mu Depletion Drop", color=pcolor, linestyle="--")
        axes[1].plot(self.Pr_Sup_p, self.Mu0_Sup, label="Mu HS Supernatant", color=pcolor, linestyle="-.")
        axes[1].plot(self.Pr_Drop_p, self.Mu0_Drop, label="Mu HS Drop", color=pcolor, linestyle="-.")
        axes[1].set_title("Chemical Potential")
        axes[1].set_xlabel(r"$[BSA]_{sup}$ (mM)")
        axes[1].set_ylabel(r"$\mu_{BSA}/kT$")
        axes[1].set_xlim(0, 8)
        axes[1].set_ylim(-100, 50)
        axes[1].set_xscale("log")

        q_bsa_drop = [1 - a for a in self.a_Drop_p]
        q_bsa_sup = [1 - a for a in self.a_Sup_p]
        c_peg_drop = [fpeg * q**1.63 * qx**1.5 for qx, fpeg in zip(self.qx_Drop_p, self.Pol_Drop_p)]
        c_peg_sup = [fpeg * q**1.63 * qx**1.5 for qx, fpeg in zip(self.qx_Sup_p, self.Pol_Sup_p)]
        axes[2].plot(q_bsa_drop, c_peg_drop, label="Droplet alpha", linestyle="-", color=pcolor)
        axes[2].plot(q_bsa_sup, c_peg_sup, label="Supernatant alpha", linestyle="-", color=pcolor)
        axes[2].set_title("Free volume fraction")
        axes[2].set_xlabel(r"$[BSA]_{sup}$ (mM)")
        axes[2].set_ylabel("Free volume fraction")
        return axes

    def to_legacy_list(self) -> list[Any]:
        """Return results in the notebook's original 32-item list order."""
        return [getattr(self, field) for field in RETURN_FIELDS]

    def _build_params(self) -> list[Any]:
        return [
            self.Rpr_in,
            self.Rpol_in,
            self.mbsa,
            self.mpeg,
            self.solvent,
            self.N_steps,
            self.PhiP_min,
            self.PhiP_max,
            self.outguess_c,
            self.outguess_tp,
            self.guess_b,
            self.TC,
            self.phix_bsa,
            self.phix_peg,
        ]

    def _assign_return_values(self, values: list[Any]) -> None:
        for field, value in zip(RETURN_FIELDS, values):
            setattr(self, field, value)
        self.params = values[-1]
        self._sync_aliases()

    def _sync_aliases(self) -> None:
        self.BSA_Drop = self.Pr_Drop_p
        self.PEG_Drop = self.Pol_Drop_p
        self.BSA_Sup = self.Pr_Sup_p
        self.PEG_Sup = self.Pol_Sup_p
        self.a_Drop = self.a_Drop_p
        self.a_Sup = self.a_Sup_p

    @staticmethod
    def _get_axis(ax):
        if ax is not None:
            return ax
        import matplotlib.pyplot as plt

        return plt.gca()

    @staticmethod
    def _maybe_legend(ax, show_legend: bool) -> None:
        if show_legend:
            ax.legend(loc=(1.1, 0), fontsize="x-large")

    @staticmethod
    def _parse_binodal_data(
        data,
        *,
        mbsa=0,
        mpeg=0,
        phix_bsa=None,
        phix_peg=None,
        peg_scale=1,
    ):
        data = np.asarray(data)
        if data.shape[1] < 16:
            raise ValueError("Expected binodal data with at least 16 columns.")

        bsa_scale = 1
        peg_mass_scale = peg_scale
        if mbsa != 0 and mpeg != 0:
            bsa_scale = 1 / mbsa
            peg_mass_scale = peg_scale / mpeg
            if phix_bsa is not None and phix_peg is not None:
                bsa_scale /= phix_bsa
                peg_mass_scale /= phix_peg

        return (
            data[:, 0] * bsa_scale,
            data[:, 1] * bsa_scale,
            data[:, 2] * bsa_scale,
            data[:, 3] * bsa_scale,
            data[:, 4] * bsa_scale,
            data[:, 5] * bsa_scale,
            data[:, 6] * peg_mass_scale,
            data[:, 7] * peg_mass_scale,
            data[:, 8] * peg_mass_scale,
            data[:, 9] * peg_mass_scale,
            data[:, 10] * peg_mass_scale,
            data[:, 11] * peg_mass_scale,
            data[:, 12] * bsa_scale,
            data[:, 13] * bsa_scale,
            data[:, 14] * peg_mass_scale,
            data[:, 15] * peg_mass_scale,
        )

    @staticmethod
    def _plot_data_errorbars(
        ax,
        plabel,
        pcolor,
        BN_BSA_tot,
        BN_BSA_tot_err,
        BN_BSA_sup,
        BN_BSA_sup_err,
        BN_BSA_drop,
        BN_BSA_drop_err,
        BN_PEG_tot,
        BN_PEG_tot_err,
        BN_PEG_sup,
        BN_PEG_sup_err,
        BN_PEG_drop,
        BN_PEG_drop_err,
        CP_BSA,
        CP_BSA_err,
        CP_PEG,
        CP_PEG_err,
        *,
        n_tie=False,
        plot_tot=False,
    ) -> None:
        ax.errorbar(
            BN_BSA_sup,
            BN_PEG_sup,
            xerr=BN_BSA_sup_err,
            yerr=BN_PEG_sup_err,
            markersize=12,
            fmt="o",
            c=pcolor,
            ecolor=pcolor,
            capsize=5,
            capthick=1,
            label=plabel + " binodal",
        )
        ax.errorbar(
            BN_BSA_drop,
            BN_PEG_drop,
            xerr=BN_BSA_drop_err,
            yerr=BN_PEG_drop_err,
            markersize=12,
            fmt="o",
            c=pcolor,
            ecolor=pcolor,
            capsize=5,
            capthick=1,
        )
        ax.errorbar(
            CP_BSA,
            CP_PEG,
            xerr=CP_BSA_err,
            yerr=CP_PEG_err,
            markersize=12,
            fmt="X",
            c=pcolor,
            ecolor=pcolor,
            capsize=5,
            capthick=1,
            label=plabel + " critical point",
        )
        if plot_tot:
            ax.errorbar(
                BN_BSA_tot,
                BN_PEG_tot,
                xerr=BN_BSA_tot_err,
                yerr=BN_PEG_tot_err,
                markersize=12,
                fmt="o",
                c=pcolor,
                ecolor=pcolor,
                capsize=5,
                capthick=1,
                label=plabel + " binodal",
            )
        if n_tie:
            for idx in range(len(BN_BSA_sup)):
                ax.plot(
                    [BN_BSA_sup[idx], BN_BSA_drop[idx]],
                    [BN_PEG_sup[idx], BN_PEG_drop[idx]],
                    "--",
                    color=pcolor,
                )

    def _require_results(self) -> None:
        if self.phi_R is None:
            raise RuntimeError("Run GFVT() first, or create the object with PD.from_legacy_list(...).")

    @staticmethod
    def _reject_fvt_overlay(FVT_color) -> None:
        if FVT_color:
            raise NotImplementedError("FVT overlays require FVT output and are not supported by the PD class.")

    def _plot_masses(self, mbsa, mpeg):
        return (self.mbsa if mbsa == 0 else mbsa, self.mpeg if mpeg == 0 else mpeg)

    def _full_label(self, plabel) -> str:
        return plabel + f"Rbsa = {self.Rpr_in * 1e9:.2f} nm, Rp = {self.Rpol_in * 1e9:.2f} nm"

    @staticmethod
    def _plot_binodal_lines(ax, x_drop, y_drop, x_sup, y_sup, pcolor, full_label) -> None:
        ax.plot(x_drop, y_drop, "-", color=pcolor, ms=2, label=full_label + " binodal")
        ax.plot(x_sup, y_sup, "-", color=pcolor, ms=2)

    @staticmethod
    def _plot_tie_lines(ax, x_drop, y_drop, x_sup, y_sup, pcolor, n_tie) -> None:
        if n_tie <= 0:
            return
        for idx in range(0, len(x_drop), n_tie):
            ax.plot([x_drop[idx], x_sup[idx]], [y_drop[idx], y_sup[idx]], "--", color=pcolor)

    def _plot_critical_lines(self, ax, crit_color, full_label, *, mass=False, mbsa=1, mpeg=1) -> None:
        if not crit_color or self.cp_list is None or len(self.cp_list) == 0:
            return
        cp_list = np.asarray(self.cp_list)
        q_clist, Rpr_clist, Rpol_clist, phix_peg_clist, crit1_clist, crit2_clist, Pr_Crit_clist, Pol_Crit_clist = cp_list.T
        x = Pr_Crit_clist * mbsa if mass else Pr_Crit_clist
        y = Pol_Crit_clist * mpeg if mass else Pol_Crit_clist
        ax.plot(x, y, "--", color=crit_color, ms=2, label=full_label + " critical line")
        ax.scatter(x[-1], y[-1], color=crit_color, marker="o", s=100, label=full_label + " critical line")

        if self.tp_list is None or len(self.tp_list) == 0:
            return
        tp_list = np.asarray(self.tp_list)
        _, _, _, _, tp1, tp2, tp3, tp4, Pr_tp1, Pr_tp2, Pr_tp3, Pol_tp1, Pol_tp2, Pol_tp3 = tp_list.T
        for idx, (tp_x, tp_y) in enumerate(
            (
                (Pr_tp1, Pol_tp1),
                (Pr_tp2, Pol_tp2),
                (Pr_tp3, Pol_tp3),
            ),
            start=1,
        ):
            x = tp_x * mbsa if mass else tp_x
            y = tp_y * mpeg if mass else tp_y
            ax.plot(x, y, "-", color=crit_color, markersize=6, label=full_label + f" triple line {idx}")

    def _plot_phi_critical_lines(self, ax, crit_color, full_label, *, gamma) -> None:
        if not crit_color or self.cp_list is None or len(self.cp_list) == 0:
            return
        cp_list = np.asarray(self.cp_list)
        q_clist, Rpr_clist, Rpol_clist, phix_peg_clist, crit1_clist, crit2_clist, Pr_Crit_clist, Pol_Crit_clist = cp_list.T
        ax.plot(
            [pr / self.phix_bsa for pr in Pr_Crit_clist],
            [q_item ** (-1 / gamma) * pol / phix for pol, phix, q_item in zip(Pol_Crit_clist, phix_peg_clist, q_clist)],
            "--",
            color=crit_color,
            ms=2,
            label=full_label + " critical line",
        )
        ax.scatter(
            Pr_Crit_clist[-1] / self.phix_bsa,
            q_clist[-1] ** (-1 / gamma) * Pol_Crit_clist[-1] / phix_peg_clist[-1],
            color=crit_color,
            marker="o",
            s=100,
            label=full_label + " critical line",
        )

        if self.tp_list is None or len(self.tp_list) == 0:
            return
        tp_list = np.asarray(self.tp_list)
        q_tlist, _, _, phix_peg_tlist, tp1, tp2, tp3, tp4, Pr_tp1, Pr_tp2, Pr_tp3, Pol_tp1, Pol_tp2, Pol_tp3 = tp_list.T
        for idx, (tp_x, tp_y) in enumerate(
            (
                (Pr_tp1, Pol_tp1),
                (Pr_tp2, Pol_tp2),
                (Pr_tp3, Pol_tp3),
            ),
            start=1,
        ):
            ax.plot(
                tp_x / self.phix_bsa,
                [q_item ** (-1 / gamma) * pol / phix for pol, phix, q_item in zip(tp_y, phix_peg_tlist, q_tlist)],
                color=crit_color,
                label=full_label + f" triple line {idx}",
            )

    def _qxf(self, q, Phi_pol):
        if self.solvent == "good":
            dPidPhiP = (1 + 3.73 * Phi_pol**1.31) / q**3
            qx = 0.865 * (q / (1 + 3.95 * Phi_pol**1.54) ** (1 / 2)) ** 0.88
        elif self.solvent == "theta":
            dPidPhiP = (1 + 12.3 * Phi_pol**2) / q**3
            qx = 0.938 * (q / (1 + 6.02 * Phi_pol**2) ** (1 / 2)) ** 0.9
        elif self.solvent == "RG":
            dPidPhiP = (1 + 3.73 * Phi_pol**1.31) / q**3
            dq = q / (1 + 3.95 * Phi_pol**1.54) ** (1 / 2)
            qx = (1 + 3 * dq + 2.73 * dq**2 - 0.0975 * dq**3) ** (1 / 3) - 1
        else:
            dPidPhiP = 1
            qx = q
        return qx, dPidPhiP

    def _Qsf(self, Phi, Phi_pol):
        y = Phi / (1 - Phi)
        qx, dPidPhiP = self._qxf(self.q, Phi_pol)
        a = 3 * qx + 3 * qx**2 + qx**3
        b = 9 / 2 * qx**2 + 3 * qx**3
        c = 3 * qx**3
        return a, b, c, y, dPidPhiP

    def _alpha(self, Phi, Phi_pol):
        a, b, c, y, _ = self._Qsf(Phi, Phi_pol)
        return (1 - Phi) * np.exp(-(a * y + b * y**2 + c * y**3))

    def _analytical_base_curves(self):
        c_bsa, c_PEG = self.outguess_c
        tp_bsa1, tp_bsa2, tp_bsa3, tp_PEG = self.outguess_tp
        y_a_L = np.linspace(c_PEG, tp_PEG, 2000)
        q = self.q

        if self.solvent == "good":
            gamma = 0.77
            c2 = 1.62
            qq = q ** (-1 / gamma)
            Y0_a_L = y_a_L * qq
            Pv_a_L = q ** (1 / gamma - 3) * Y0_a_L + c2 * Y0_a_L ** (3 * gamma)
            q_a_G = 0.865 * (q**-2 + 3.95 * Y0_a_L ** (2 * gamma)) ** -0.44
        else:
            gamma = 1
            c2 = 4.10
            qq = q ** (-1 / gamma)
            Y0_a_L = y_a_L * qq
            Pv_a_L = q ** (1 / gamma - 3) * Y0_a_L + c2 * Y0_a_L ** (3 * gamma)
            q_a_G = 0.938 * (q**-2 + 6.02 * Y0_a_L ** (2 * gamma)) ** -0.45

        bsa_a_L = c_bsa + (tp_bsa2 - c_bsa) * ((Pv_a_L - Pv_a_L[0]) / (Pv_a_L[-1] - Pv_a_L[0])) ** 0.4
        bsa_a_G = c_bsa + (tp_bsa1 - c_bsa) * ((q_a_G - q_a_G[0]) / (q_a_G[-1] - q_a_G[0])) ** 0.25
        return gamma, c2, qq, y_a_L, bsa_a_L, bsa_a_G
