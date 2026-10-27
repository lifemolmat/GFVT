"""Phase equilibria for the 2011 Chapter 6 spherocylinder/depletant theory.

The old isotropic approximation remains in :mod:`gfvt.rods`.  Here both
orientation and composition equilibrate.  Stability is within the isotropic
and nematic family; smectic and crystalline free energies are not included.
"""

from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.optimize import brentq, least_squares
from tqdm import tqdm

from .rod_thermo import RodEOS, NoOrderedPhase, RodConvergenceError, _scalar


@dataclass
class RodBranch:
    """One continuous coexistence branch, with phases ordered by density.

    ``phi_R`` is dimensionless reservoir overlap concentration.  ``phi``,
    ``alpha``, ``S`` and ``rho`` have two phase columns. ``mu`` is in kT,
    and ``pressure`` is in Pa.  The remaining legacy diagnostics are in
    ``values``.  Unstable points are retained, with ``stable=False``.
    """

    kind: str
    phase_labels: tuple
    phi_R: np.ndarray
    phi: np.ndarray
    alpha: np.ndarray
    S: np.ndarray
    rho: np.ndarray
    mu: np.ndarray
    pressure: np.ndarray
    stable: np.ndarray
    values: dict = field(default_factory=dict)
    stability_margin: np.ndarray = field(default_factory=lambda: np.empty(0))


def _hull(points):
    """Indices of the lower convex envelope of sorted (density, energy)."""
    hull = []
    for i, (x, f, *_) in enumerate(points):
        while len(hull) >= 2:
            a, b = hull[-2:]
            xa, fa = points[a][:2]
            xb, fb = points[b][:2]
            if (fb-fa)*(x-xb) < (f-fb)*(xb-xa):
                break
            hull.pop()
        hull.append(i)
    return hull


class _Equilibria:
    def __init__(self, eos, maximum, guess):
        self.eos, self.maximum, self.guess = eos, maximum, guess
        self.proxy = eos if eos.odf == "gaussian" else RodEOS(
            eos.D, eos.L, eos.Rpol, eos.dep_model, "gaussian", n_poly=eos.n_poly,
        )
        self.grid = np.unique(np.r_[
            np.geomspace(1e-12, 0.07, 82), np.linspace(0.004, 0.8, 151),
            np.linspace(0.8, 0.97, 16),
        ])
        self.scans = {}
        self.previous = {key: [] for key in ("II", "IN", "NN")}
        self.critical = {}

    def scan(self, reservoir):
        """Gaussian candidates; numerical-ODF roots are refined afterwards."""
        reservoir = float(reservoir)
        if reservoir in self.scans:
            return self.scans[reservoir]
        model = self.proxy.at(reservoir)
        states = {phase: [] for phase in ("I", "N")}
        candidates = {kind: [] for kind in ("II", "IN", "NN")}
        for phi in self.grid:
            for phase in states:
                try:
                    state = model.evaluate(float(phi), phase)
                except NoOrderedPhase:
                    continue
                states[phase].append((float(phi), state))
        for phase in states:
            points = [(p, s.omega, phase) for p, s in states[phase]]
            ids = _hull(points)
            for left, right in zip(ids[:-1], ids[1:]):
                if right-left > 1:
                    candidates[phase+phase].append((points[left][0], points[right][0]))
        # At each density retain the lower orientation free energy, then take
        # its convex envelope. Cross-phase chords are I--N candidate tangents.
        best = {}
        for phase, records in states.items():
            for p, s in records:
                if p not in best or s.omega < best[p][1]:
                    best[p] = (p, s.omega, phase)
        points = sorted(best.values())
        ids = _hull(points)
        for left, right in zip(ids[:-1], ids[1:]):
            a, b = points[left], points[right]
            if a[2] != b[2] and b[0]-a[0] > 1e-7:
                candidates["IN"].append((a[0], b[0]))
        result = states, candidates
        self.scans[reservoir] = result
        # Complete figure runs may create many short-lived continuation scans.
        if len(self.scans) > 256:
            self.scans.pop(next(iter(self.scans)))
        return result

    def critical_i(self):
        """An II critical point is conditional on isotropic orientation."""
        high = max(1.0, 4*self.maximum)
        bounds = np.log([1e-7, 1e-9]), np.log([0.9, high])

        def residual(z):
            phi, r = np.exp(z)
            y = phi/(1-phi)
            norm = [1+2*self.eos.A*y+3*self.eos.B*y*y,
                    2*self.eos.A+6*self.eos.B*y]
            return self.eos.at(r).isotropic_slopes(phi)/norm

        seeds = [self.guess, (.05, .4), (.15, .1), (.02, 1), (.25, .8)]
        # Large nominal polymer sizes can put the II critical reservoir far
        # above one. At the old, exclusively dilute-reservoir seeds, polymer
        # insertion may underflow and leave the optimizer with no useful
        # reservoir derivative. Try range-scaled seeds as a fallback, while
        # retaining the original fast path and respecting the same bounds.
        if self.maximum > 0:
            seeds += [(phi, reservoir)
                      for reservoir in (self.maximum, .25*self.maximum,
                                        min(.9*high, 3*self.maximum))
                      for phi in (.04, .01, .15)]
        for seed in seeds:
            if np.shape(seed) != (2,) or np.any(np.asarray(seed) <= 0):
                continue
            sol = least_squares(residual, np.clip(np.log(seed), bounds[0]+1e-8, bounds[1]-1e-8),
                                bounds=bounds, max_nfev=120, ftol=1e-12,
                                xtol=1e-12, gtol=1e-12)
            if sol.success and np.max(np.abs(sol.fun)) < 2e-8 and not np.any(sol.active_mask):
                phi, r = np.exp(sol.x)
                if r <= self.maximum*(1+1e-9):
                    state = self.eos.at(r).evaluate(phi, "I")
                    self.critical["II"] = dict(phi=float(phi), phi_R=float(r), S=0.,
                                                alpha=state.alpha, stable=False)
                return

    def critical_n(self, seed, reservoir):
        """Find a critical point of the orientation-relaxed nematic EOS."""
        bounds = np.log([1e-5, 1e-9]), np.log([.95, max(self.maximum, 1e-8)*1.001])

        def residual(z):
            phi, r = np.exp(z)
            h = min(.002*phi, .1*(1-phi))
            nodes = np.arange(-3., 4.)
            try:
                p = [self.eos.at(r).evaluate(phi+h*v, "N").pressure for v in nodes]
            except (NoOrderedPhase, RodConvergenceError):
                return np.array([1e3, 1e3])
            coefficients = np.polynomial.polynomial.polyfit(nodes, p, 6)
            normalization = max(1., abs(p[3])/phi)
            return [coefficients[1]/h/normalization,
                    2*coefficients[2]/h**2/(normalization/phi)]

        sol = least_squares(residual, np.clip(np.log([np.mean(seed), reservoir*.9]),
                                             bounds[0]+1e-7, bounds[1]-1e-7),
                            bounds=bounds, diff_step=2e-4, max_nfev=90,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if sol.success and np.max(np.abs(sol.fun)) < 2e-5 and not np.any(sol.active_mask):
            phi, r = np.exp(sol.x)
            state = self.eos.at(r).evaluate(phi, "N")
            self.critical["NN"] = dict(phi=float(phi), phi_R=float(r), S=state.S,
                                        alpha=state.alpha, stable=False)

    def solve_pair(self, reservoir, kind, seed):
        phases = tuple(kind)
        low, high = sorted(map(float, seed))
        if low <= 0 or high >= 1 or high-low < 1e-9:
            return None
        midpoint = (low+high)/2
        if kind == "IN":
            # The numerical ODF can shift BOTH phases below the Gaussian
            # candidate midpoint; a midpoint constraint would omit bare I--N.
            bounds = np.log([1e-250, 1e-9]), np.log([1-1e-7, 1-1e-7])
        else:
            bounds = np.log([1e-250, midpoint+1e-10]), np.log([midpoint-1e-10, 1-1e-7])
        model = self.eos.at(reservoir)
        if kind == "II":
            critical=self.critical.get("II")
            center=critical["phi"] if critical is not None else midpoint
            slope=lambda phi:model.isotropic_slopes(phi)[0]
            if slope(center) < 0:
                spin_low=brentq(slope,1e-14,center,xtol=1e-15)
                spin_high=brentq(slope,center,1-1e-7,xtol=1e-15)
                bounds=np.log([1e-250,spin_high]),np.log([spin_low,1-1e-7])
                middle=(spin_low+spin_high)/2
                half=np.sqrt(3)*(spin_high-spin_low)/2
                low=max(middle-half,spin_low/2)
                high=min(middle+half,(1+spin_high)/2)

        def residual(z):
            a, b = np.exp(z)
            if a >= b:
                return np.array([1e3, 1e3])
            try:
                left, right = model.evaluate(a, phases[0]), model.evaluate(b, phases[1])
            except NoOrderedPhase:
                return np.array([1e3, 1e3])
            return np.array([left.mu-right.mu, left.pressure-right.pressure])/(b-a)

        sol = least_squares(residual, np.clip(np.log([low, high]), bounds[0]+1e-9, bounds[1]-1e-9),
                            bounds=bounds, max_nfev=90, x_scale="jac",
                            ftol=3e-12, xtol=3e-12, gtol=3e-12,
                            diff_step=2e-5 if self.eos.odf == "numerical" else None)
        if not sol.success or np.max(np.abs(sol.fun)) > 2e-7:
            return None
        a, b = np.exp(sol.x)
        if b-a < 1e-6 or np.any(sol.active_mask):
            return None
        left, right = model.evaluate(a, phases[0]), model.evaluate(b, phases[1])
        if max(abs(left.mu-right.mu), abs(left.pressure-right.pressure)) > 1e-8:
            return None
        return left, right

    def pairs(self, reservoir, kinds=("II", "IN", "NN")):
        scan_states, candidates = self.scan(reservoir)
        result = {}
        for kind in kinds:
            seeds = list(candidates[kind]) + list(self.previous[kind])
            cp = self.critical.get(kind)
            if cp is not None and reservoir > cp["phi_R"]:
                offset = min(.8, np.sqrt(reservoir/cp["phi_R"]-1))
                seeds.append((cp["phi"]*max(.03, 1-offset),
                              min(.97, cp["phi"]*(1+offset))))
            pairs = []
            for seed in seeds:
                pair = self.solve_pair(reservoir, kind, seed)
                if pair is not None and not any(
                        np.max(np.abs(np.log(np.array([s.phi for s in pair]) /
                                             [s.phi for s in old]))) < 1e-5 for old in pairs):
                    pairs.append(pair)
            pairs.sort(key=lambda pair: pair[0].phi)
            if candidates[kind] and not pairs:
                strong=kind=="IN"
                if kind in ("II","NN"):
                    records=scan_states[kind[0]]
                    for low,high in candidates[kind]:
                        left=next((i for i,(p,_) in enumerate(records) if p==low),0)
                        right=next((i for i,(p,_) in enumerate(records) if p==high),0)
                        if left>0 and right<len(records)-1 and right-left>=3:
                            fa,fb=records[left][1].omega,records[right][1].omega
                            deficit=max(s.omega-(fa+(fb-fa)*(p-low)/(high-low))
                                        for p,s in records[left+1:right])
                            strong |= deficit>5e-11*max(1.,abs(fa),abs(fb))
                if strong:
                    raise RodConvergenceError(
                        f"Could not refine an interior {kind} coexistence tangent "
                        f"at phi_R={reservoir:.12g}; no result was published.")
            result[kind] = pairs
            self.previous[kind] = [(p[0].phi, p[1].phi) for p in pairs]
        return result

    def margin(self, reservoir, pair):
        """Minimum separation between all phase free energies and a tangent."""
        states, _ = self.scan(reservoir)
        mu = (pair[0].mu+pair[1].mu)/2
        pressure = (pair[0].pressure+pair[1].pressure)/2
        model = self.eos.at(reservoir)
        minimum = min(s.omega-mu*s.phi+pressure for s in pair)
        minimum = min(minimum, pressure-np.sum(model.weights))
        # The proxy brackets all extrema; use the requested ODF to refine
        # each local grand-potential minimum, not a mesh-only hull test.
        for phase, records in states.items():
            for (a, sa), (b, sb) in zip(records[:-1], records[1:]):
                if sa.mu <= mu <= sb.mu:
                    try:
                        fa = model.evaluate(a, phase).mu-mu
                        fb = model.evaluate(b, phase).mu-mu
                        if fa*fb > 0:
                            # Numerical and Gaussian phase concentrations differ;
                            # widen the local bracket without changing the theory.
                            for _ in range(16):
                                if fa > 0:
                                    candidate=max(1e-12,a*.9)
                                    try:
                                        next_value=model.evaluate(candidate,phase).mu-mu
                                    except NoOrderedPhase:
                                        # Approach the ordered spinodal from its
                                        # valid side; never skip a minimum merely
                                        # because a broad Gaussian bracket crossed it.
                                        invalid,valid=candidate,a
                                        for _ in range(12):
                                            trial=(invalid+valid)/2
                                            try:
                                                next_value=model.evaluate(trial,phase).mu-mu
                                                valid=trial
                                                # We need a chemical-potential
                                                # bracket, not the spinodal
                                                # itself. Stop once bracketed;
                                                # further boundary refinement
                                                # needlessly enters marginal,
                                                # slowly converging ODF states.
                                                if next_value <= 0:
                                                    break
                                            except NoOrderedPhase:
                                                invalid=trial
                                        candidate=valid
                                        next_value=model.evaluate(candidate,phase).mu-mu
                                    if candidate==a:
                                        break
                                    a,fa=candidate,next_value
                                if fb < 0:
                                    b=min(.999,b*1.1)
                                    fb=model.evaluate(b,phase).mu-mu
                                if fa <= 0 <= fb:
                                    break
                        if fa <= 0 <= fb:
                            phi = brentq(lambda v: model.evaluate(v, phase).mu-mu,
                                         a, b, xtol=2e-12, rtol=1e-11)
                            s = model.evaluate(phi, phase)
                            minimum = min(minimum, s.omega-mu*phi+pressure)
                    except NoOrderedPhase:
                        continue
        return float(minimum)

    def triple(self, kind, reservoir, phases_phi):
        labels = ("I", "I", "N") if kind == "IIN" else ("I", "N", "N")
        bounds = np.log([1e-250]*3+[1e-9]), np.log([1-1e-7]*3+[max(self.maximum, 1e-8)*1.001])

        def residual(z):
            a, b, c, r = np.exp(z)
            if not a < b < c:
                return np.ones(4)*1e3
            try:
                states = [self.eos.at(r).evaluate(phi, phase)
                          for phi, phase in zip((a,b,c), labels)]
            except NoOrderedPhase:
                return np.ones(4)*1e3
            return [
                (states[0].mu-states[1].mu)/(b-a),
                (states[0].pressure-states[1].pressure)/(b-a),
                (states[1].mu-states[2].mu)/(c-b),
                (states[1].pressure-states[2].pressure)/(c-b),
            ]

        sol = least_squares(residual, np.clip(np.log([*phases_phi, reservoir]),
                                             bounds[0]+1e-9, bounds[1]-1e-9),
                            bounds=bounds, max_nfev=100, x_scale="jac",
                            ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if sol.success and np.max(np.abs(sol.fun)) < 1e-7:
            values = np.exp(sol.x)
            if min(np.diff(values[:3])) > 1e-5 and 0 <= values[3] <= self.maximum:
                states = [self.eos.at(values[3]).evaluate(phi, phase)
                          for phi, phase in zip(values[:3], labels)]
                if max(np.ptp([state.mu for state in states]),
                       np.ptp([state.pressure for state in states])) > 1e-8:
                    return None
                return dict(kind=kind, phi_R=float(values[3]), phi=values[:3],
                            S=np.array([s.S for s in states]),
                            alpha=np.array([s.alpha for s in states]),
                            mu=np.array([s.mu for s in states]),
                            pressure=np.array([s.pressure for s in states]))
        return None


def _discover_nematic_critical(engine):
    """Find the onset of an NN loop independently of output sampling."""
    previous_without_nn = 0.
    maximum = engine.maximum
    for r in np.geomspace(max(1e-7, maximum*1e-5), maximum, 16):
        _, candidates = engine.scan(r)
        if candidates["NN"]:
            left, right = previous_without_nn, r
            near = candidates["NN"]
            for _ in range(12):
                middle = (left + right)/2
                _, trial = engine.scan(middle)
                if trial["NN"]:
                    right, near = middle, trial["NN"]
                else:
                    left = middle
            for seed in near:
                engine.critical_n(seed, right)
                if "NN" in engine.critical:
                    return
        else:
            previous_without_nn = r


def _critical_reservoir_grid(engine, base):
    """Include enough postcritical samples to resolve triple equilibria."""
    grid = list(base)
    maximum = engine.maximum
    for cp in engine.critical.values():
        grid.append(cp["phi_R"])
        if maximum > cp["phi_R"]:
            offset = min((maximum-cp["phi_R"])/100, cp["phi_R"]*.001)
            span = min(maximum-cp["phi_R"], cp["phi_R"])
            grid += list(cp["phi_R"] + np.geomspace(offset, span, 17))
    return np.unique(grid)


def _triple_landmarks(engine, samples, kinds):
    """Refine actual stability switches of conditional II/NN tangents."""
    triples = []
    for kind, triple_kind, third_phase in (("II", "IIN", "N"),
                                           ("NN", "INN", "I")):
        if kind not in kinds:
            continue
        previous = None
        for sample in samples:
            pairs = sample["pairs"].get(kind, [])
            if not pairs:
                continue
            pair = pairs[0]
            margin = sample["margins"][kind][0]
            stable = margin >= -2e-7
            if previous is not None and stable != previous[2]:
                reservoir = (sample["phi_R"] + previous[0])/2
                scan_states, _ = engine.scan(reservoir)
                mu = (pair[0].mu + pair[1].mu)/2
                choices = scan_states[third_phase]
                if choices:
                    third = min(choices, key=lambda row: row[1].omega-mu*row[0])[0]
                    guess = sorted([pair[0].phi, pair[1].phi, third])
                    triple = engine.triple(triple_kind, reservoir, guess)
                    if triple is None:
                        old_pair = previous[1]
                        old_mu = (old_pair[0].mu + old_pair[1].mu)/2
                        third = min(choices, key=lambda row: row[1].omega-old_mu*row[0])[0]
                        triple = engine.triple(
                            triple_kind, reservoir,
                            sorted([old_pair[0].phi, old_pair[1].phi, third]))
                    if triple is None:
                        raise RodConvergenceError(
                            f"A {kind} stability change was found between "
                            f"phi_R={previous[0]:.12g} and {sample['phi_R']:.12g}, "
                            f"but its {triple_kind} triple equilibrium did not converge.")
                    if not any(abs(t["phi_R"]-triple["phi_R"]) < 1e-6
                               for t in triples):
                        triples.append(triple)
            previous = (sample["phi_R"], pair, stable)
    return triples


def _in_only_landmarks(engine):
    """Locate meetings of IN families without publishing other branches.

    Conditional II/NN tangents are private diagnostics here.  Their optional
    critical points only supply an adaptive reservoir grid; a missing
    same-phase critical point never prevents an ordinary IN calculation.
    Gaussian proxy scans avoid numerical ODF work where no relevant loop
    exists, as in the Figure 6.6 geometry.
    """
    if engine.maximum == 0 or engine.eos.L == 0:
        return []
    auxiliary = _Equilibria(engine.eos, engine.maximum, engine.guess)
    auxiliary.critical_i()
    _discover_nematic_critical(auxiliary)
    kinds = tuple(auxiliary.critical)
    if not kinds:
        return []
    grid = _critical_reservoir_grid(auxiliary, [0., engine.maximum])
    samples = []
    for reservoir in grid:
        pairs = auxiliary.pairs(reservoir, kinds)
        samples.append({
            "phi_R": reservoir,
            "pairs": pairs,
            "margins": {
                kind: [auxiliary.margin(reservoir, pair) for pair in pairs[kind]]
                for kind in kinds},
        })
    return _triple_landmarks(auxiliary, samples, kinds)


def _make_branch(kind, records, phix, temperature):
    records.sort(key=lambda row: row[0], reverse=True)
    r = np.array([row[0] for row in records])
    states = [row[1] for row in records]
    pressure_scale = phix[0]*8.314*temperature
    extract = lambda attr: np.array([[getattr(s, attr) for s in pair] for pair in states])
    phi, alpha = extract("phi"), extract("alpha")
    branch = RodBranch(kind, tuple(kind), r, phi, alpha, extract("S"), extract("rho"),
                       extract("mu"), extract("pressure")*pressure_scale,
                       np.array([row[2] >= -2e-7 for row in records]),
                       stability_margin=np.array([row[2] for row in records]))
    for side, column in (("Sup", 0), ("Drop", 1)):
        values = {
            f"Pr_{side}_p": phi[:,column]*phix[0],
            f"Pol_{side}_p": alpha[:,column]*r*phix[1],
            f"a_{side}_p": alpha[:,column],
        }
        for attr, name, factor in (
                ("qx", f"qx_{side}_p", 1), ("dpi", f"dPidPhiP_{side}_p", 1),
                ("pressure", f"Pi_{side}", pressure_scale),
                ("pressure0", f"P0_{side}", pressure_scale),
                ("mu", f"Mu_{side}", 1), ("mu0", f"Mu0_{side}", 1),
                ("g", f"g_{side}", 1), ("h", f"h_{side}", 1)):
            values[name] = extract(attr)[:,column]*factor
        values[f"Pp_{side}"] = values[f"Pi_{side}"]-values[f"P0_{side}"]
        values[f"MuR_{side}"] = values[f"Mu_{side}"]-values[f"Mu0_{side}"]
        branch.values.update(values)
    branch.values["phi_R"] = r
    return branch


def _group_records(kind, records, phix, temperature):
    """Separate simultaneous tangent families before plotting/serialization."""
    pooled={}
    for r,pairs in records:
        pool=pooled.setdefault(r,[])
        for pair,margin in pairs:
            if not any(np.max(np.abs(np.log(np.array([s.phi for s in pair])/
                                            [s.phi for s in previous]))) < 1e-7
                       for previous,_ in pool):
                pool.append((pair,margin))
    groups=[];last_seen=[]
    for index,(r,pairs) in enumerate(sorted(pooled.items())):
        available={j for j,seen in enumerate(last_seen) if seen==index-1}
        unmatched=set(range(len(pairs)))
        costs=[]
        for i,(pair,_) in enumerate(pairs):
            for j in available:
                cost=np.linalg.norm(np.log(np.array([s.phi for s in pair])/
                                            [s.phi for s in groups[j][-1][1]]))
                costs.append((cost,i,j))
        # Global nearest matches prevent a newly appearing dilute branch from
        # stealing the existing concentrated branch at a triple point.
        for _,i,j in sorted(costs):
            if i in unmatched and j in available:
                pair,margin=pairs[i]
                groups[j].append((r,pair,margin));last_seen[j]=index
                unmatched.remove(i);available.remove(j)
        for i in sorted(unmatched):
            pair,margin=pairs[i]
            groups.append([(r,pair,margin)]);last_seen.append(index)
    return [_make_branch(kind, group, phix, temperature) for group in groups if len(group)]


def calculate_rods(pd, PhiP_max=1, *, L_rod=None, L=None, coexistence="II",
                   dep_model="int", odf="gaussian", progress=None, verbose=None,
                   n_orient=128, n_azimuth=256, progress_callback=None):
    """Calculate all results locally and publish to PD only after success."""
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")

    def report(stage, completed=None, total=None, **details):
        nonlocal progress_callback
        if progress_callback is not None:
            try:
                progress_callback(dict(stage=stage, completed=completed,
                                       total=total, **details))
            except Exception as error:
                # A presentation observer must not invalidate a physical result.
                progress_callback = None
                warnings.warn(f"Rod progress callback disabled: {error}",
                              RuntimeWarning, stacklevel=2)

    maximum = _scalar(PhiP_max, "PhiP_max")
    if L_rod is not None and L is not None and L_rod != L:
        raise ValueError("L and L_rod specify conflicting cylindrical lengths.")
    length = _scalar(L if L_rod is None and L is not None else (0 if L_rod is None else L_rod), "L_rod")
    if not isinstance(coexistence, str) or coexistence.upper() not in ("II", "IN", "NN", "ALL"):
        raise ValueError("coexistence must be 'II', 'IN', 'NN', or 'all'.")
    selected = ("II", "IN", "NN") if coexistence.upper() == "ALL" else (coexistence.upper(),)
    if dep_model == "int" and (pd.solvent != "good" or getattr(pd,"nu",None) is not None):
        raise ValueError("dep_model='int' uses the published good-solvent closure: solvent='good', nu=None.")
    if getattr(pd, "kuhn_crossover", None) is not None:
        raise ValueError("The book rod models do not support a Kuhn crossover.")
    if pd.N_steps < 2:
        raise ValueError("N_steps must be at least 2.")
    eos = RodEOS(2*pd.Rpr_in, length, pd.Rpol_in, dep_model, odf,
                 n_orient=n_orient,n_azimuth=n_azimuth)
    engine = _Equilibria(eos, maximum, pd.guess_c)
    report("critical_points")
    if "II" in selected:
        engine.critical_i()
    # Find an NN spinodal interval before tracing; the logarithmic scan also
    # resolves the small reservoir fractions for short-ranged PHS depletants.
    if "NN" in selected and length > 0 and maximum > 0:
        _discover_nematic_critical(engine)
    in_landmarks = _in_only_landmarks(engine) if selected == ("IN",) else []
    grid = _critical_reservoir_grid(
        engine, [*np.linspace(0, maximum, pd.N_steps),
                 *(point["phi_R"] for point in in_landmarks)])
    records = {key: [] for key in selected}
    samples = []
    use_progress = pd.progress if progress is None else progress
    report("tracing", 0, len(grid))
    for index, r in enumerate(tqdm(grid, desc="Rod phase equilibria", disable=not use_progress), 1):
        result = engine.pairs(r, selected)
        sample = {"phi_R":r,"pairs":result,"margins":{}}
        for kind in selected:
            rows = []
            for pair in result[kind]:
                margin = engine.margin(r,pair)
                rows.append((pair,margin))
            records[kind].append((r,rows))
            sample["margins"][kind] = [row[1] for row in rows]
        samples.append(sample)
        report("tracing", index, len(grid), phi_R=float(r))

    report("finalizing")
    triples = _triple_landmarks(engine, samples, records)
    triples.extend(in_landmarks)
    # Landmarks are explicit points of every meeting branch, rather than
    # approximate intersections of independently sampled output polylines.
    for triple in triples:
        r=triple["phi_R"]; labels=tuple(triple["kind"])
        states=[eos.at(r).evaluate(phi,phase) for phi,phase in zip(triple["phi"],labels)]
        for i,j in ((0,1),(0,2),(1,2)):
            kind=labels[i]+labels[j]
            if kind in records:
                records[kind].append((r,[((states[i],states[j]),0.)]))
    for kind,cp in engine.critical.items():
        state=eos.at(cp["phi_R"]).evaluate(cp["phi"],kind[0])
        cp_margin=engine.margin(cp["phi_R"],(state,state))
        cp["stable"]=cp_margin >= -2e-7
        if kind in records:
            records[kind].append((cp["phi_R"],[((state,state),cp_margin)]))
    phix=(1/(eos.volume*6.022e23),1/(eos.polymer_volume*6.022e23))
    temperature=273.15+pd.TC
    branches={kind:_group_records(kind,sorted(records.get(kind,[]),key=lambda row:row[0]),
                                  phix,temperature) for kind in ("II","IN","NN")}
    for triple in triples:
        triple["pressure"]*=phix[0]*8.314*temperature
    # Atomic state update: until this point a failed calculation changes no PD.
    pd._clear_scaled_rod_metadata()
    for name in pd.RETURN_FIELDS:
        if name != "params":setattr(pd,name,None)
    pd.geometry="spherocylinders";pd.rod_theory="book2011"
    pd.D=eos.D;pd.L=pd.L_rod=length;pd.sigma=2*pd.Rpol_in
    pd.aspect_ratio=eos.gamma;pd.L_over_D=length/eos.D;pd.q=eos.q
    pd.dep_model=dep_model;pd.odf=odf;pd.depletion_model=dep_model
    pd.vbsa=eos.volume*6.022e23;pd.vpeg=eos.polymer_volume*6.022e23
    pd.phix_bsa,pd.phix_peg=phix
    pd.PhiP_min=0.;pd.PhiP_max=maximum
    pd.rod_branches=branches;pd.rod_critical_points=engine.critical;pd.rod_triple_points=triples
    pd.rod_requested_coexistence="all" if len(selected)>1 else selected[0]
    pd.rod_branch_status={kind:("calculated" if branches[kind]
                               else "no coexistence in the requested reservoir range")
                          for kind in selected}
    pd.rod_orientation_grid=(n_orient,n_azimuth);pd.s=0
    present=[b for group in branches.values() for b in group]
    pd._rod_legacy_ambiguous=len(present)!=1 or len(selected)>1
    pd.cp=pd.tp=pd.outguess_c=pd.outguess_tp=None;pd.cp_list=[];pd.tp_list=[]
    if not pd._rod_legacy_ambiguous:
        branch=present[0]
        for name,value in branch.values.items():setattr(pd,name,value)
        cp=engine.critical.get(branch.kind)
        if cp is not None:
            pd.cp=[cp["phi"]*phix[0],cp["alpha"]*cp["phi_R"]*phix[1]]
            pd.outguess_c=[cp["phi"],cp["phi_R"]]
    pd.params=pd._build_params();pd._sync_aliases()
    if pd.verbose if verbose is None else verbose:
        counts=", ".join(f"{kind}: {sum(len(b.phi_R) for b in branches[kind])} points" for kind in selected)
        print(f"Book rods, {dep_model}, {odf}; L/D={length/eos.D:g}, q={eos.q:g}; {counts}")
    report("complete", len(grid), len(grid))
    return pd
