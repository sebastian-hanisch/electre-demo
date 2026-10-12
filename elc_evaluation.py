"""Auswertung: Fall der App, Experimente über feste Matrizen (SWEEP_SEEDS) und Verdikte. Nur numpy und scipy (über elc_outranking), kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import elc_constants as C
import elc_outranking as ox
import elc_score as sc
from entscheidungsmatrix_generator import SplitMix64, make_matrix

INF = math.inf


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def veto_label(v: float) -> str:
    return "ohne" if math.isinf(v) else de(v, 1)


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    cstar: float = C.DEFAULT_CSTAR
    veto: float = C.DEFAULT_VETO
    decider: str = C.DEFAULT_DECIDER
    delta: float = C.DEFAULT_DELTA
    seed: int = C.DEFAULT_SEED


@dataclass
class Case:
    """Eine Matrix mit simuliertem Entscheider: vorzeichenbehaftete Rohwerte nach Verschmälerung, wahre Gewichte, Nutzen, zulässige Alternativen (Untergrenze)."""
    matrix: object
    goodness: np.ndarray
    S: np.ndarray
    w: np.ndarray
    utility: np.ndarray
    acceptable: np.ndarray


def make_case(n: int, m: int, seed: int, delta: float = 1.0, decider: str = "guete_linear") -> Case:
    scale, form = C.DECIDER_PARTS[decider]
    x = make_matrix(n, m, 0.0, SplitMix64(seed))
    g, _, S = sc.build_case(x.goodness, x.directions, m, delta)
    theta = C.THETA if decider == "untergrenze" else None
    return Case(x, g, S, x.weights, sc.decider_utility(scale, form, g, S, x.weights, m, theta), sc.acceptable(g, C.THETA))


@dataclass
class Result:
    score: np.ndarray          # höher ist besser
    top1: int
    hit: bool
    tau: float
    loss: float
    tie_at_top: bool


def _result(case: Case, score: np.ndarray, best: int) -> Result:
    t = sc.top1(score)
    return Result(score, t, t == best, sc.kendall_tau_b(score, case.utility), sc.utility_loss(case.utility, t), int((score == score.max()).sum()) > 1)


@dataclass
class Analysis:
    settings: Settings
    case: Case
    N: np.ndarray
    conc: np.ndarray           # Konkordanz c[a, b]
    veto: np.ndarray           # Veto-Maske V[a, b]
    S: np.ndarray              # Überordnung
    P: np.ndarray
    I: np.ndarray
    R: np.ndarray
    cycles: list               # Komponenten mit mindestens zwei Alternativen
    kernel: set
    front: np.ndarray
    best: int
    results: dict              # "summe", "copeland"
    summe_in_kernel: bool
    best_in_kernel: bool
    kernel_in_front: bool
    unacceptable_in_kernel: bool


def analyse(settings: Settings) -> Analysis:
    s = settings
    case = make_case(s.n, s.m, s.seed, s.delta, s.decider)
    N = sc.minmax(case.S)
    conc = ox.concordance(N, case.w)
    veto = ox.veto_mask(N, s.veto)
    S = ox.outranking(N, case.w, s.cstar, s.veto)
    P, I, R = ox.relation_classes(S)
    K = ox.kernel(S)
    front = sc.front_mask(case.goodness)
    best = sc.top1(case.utility)
    results = {"summe": _result(case, sc.weighted_score(N, case.w), best), "copeland": _result(case, ox.copeland(S).astype(float), best)}
    return Analysis(s, case, N, conc, veto, S, P, I, R, ox.cycles(S), K, front, best, results, results["summe"].top1 in K, best in K,
                    all(front[i] for i in K), any(not case.acceptable[i] for i in K))


# ----------------------------------------------------------------------------- Experiment 1: Struktur der Relation

def experiment_structure(ms=C.EXP_M, cstars=C.EXP_CSTARS, vetoes=C.EXP_VETOES, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Je (m, c*, Veto) über 200 feste Matrizen: Anteile der ungeordneten Paare (strikt, beidseitig, unvergleichbar), Zyklen in der Relation und unter den strikten
    Präferenzen, Kerngröße, Frontgröße, Kern innerhalb der Front."""
    rows = []
    pairs = n * (n - 1) / 2
    for m in ms:
        cases = [make_case(n, m, s) for s in seeds]
        norm = [sc.minmax(c.S) for c in cases]
        for cstar in cstars:
            for v in vetoes:
                acc = {k: [] for k in ("p", "i", "r", "cycle", "cycle_p", "kernel", "front", "kernel_in_front")}
                for c, N in zip(cases, norm):
                    S = ox.outranking(N, c.w, cstar, v)
                    P, I, R = ox.relation_classes(S)
                    K = ox.kernel(S)
                    front = sc.front_mask(c.goodness)
                    acc["p"].append(P.sum() / pairs)
                    acc["i"].append(I.sum() / 2 / pairs)
                    acc["r"].append(R.sum() / 2 / pairs)
                    acc["cycle"].append(ox.has_cycle(S))
                    acc["cycle_p"].append(ox.has_cycle(P))
                    acc["kernel"].append(len(K))
                    acc["front"].append(int(front.sum()))
                    acc["kernel_in_front"].append(all(front[i] for i in K))
                rows.append({"m": m, "cstar": cstar, "veto": v, **{k: float(np.mean(x)) for k, x in acc.items()}})
    return rows


# ----------------------------------------------------------------------------- Experiment 2: der Kern und der Beste

def experiment_kernel(deciders=C.EXP_KERNEL_DECIDERS, cstars=C.EXP_KERNEL_CSTARS, m=C.EXP_M_MAIN, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for dec in deciders:
        cases = [make_case(n, m, s, 1.0, dec) for s in seeds]
        for cstar in cstars:
            acc = {k: [] for k in ("kernel", "best_in_kernel", "sum_in_kernel", "best_in_front")}
            for c in cases:
                N = sc.minmax(c.S)
                K = ox.kernel(ox.outranking(N, c.w, cstar))
                best = sc.top1(c.utility)
                acc["kernel"].append(len(K))
                acc["best_in_kernel"].append(best in K)
                acc["sum_in_kernel"].append(sc.top1(N @ c.w) in K)
                acc["best_in_front"].append(bool(sc.front_mask(c.goodness)[best]))
            rows.append({"decider": dec, "cstar": cstar, **{k: float(np.mean(x)) for k, x in acc.items()}})
    return rows


# ----------------------------------------------------------------------------- Experiment 3: die Schwellen

THRESHOLD_PAIRS = (((0.6, INF), (0.7, INF)), ((0.7, INF), (0.8, INF)), ((0.6, INF), (0.8, INF)), ((0.7, INF), (0.7, 0.5)), ((0.7, 0.5), (0.7, 0.3)), ((0.6, INF), (0.6, 0.5)))


def experiment_thresholds(m=C.EXP_M_MAIN, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Anteil der Matrizen, in denen sich der Kern bei einem Schwellenwechsel ändert, und mittleres Jaccard (Güte/linear)."""
    cases = [make_case(n, m, s) for s in seeds]
    norm = [sc.minmax(c.S) for c in cases]
    rows = []
    for (c1, v1), (c2, v2) in THRESHOLD_PAIRS:
        diff, jac = [], []
        for c, N in zip(cases, norm):
            k1 = ox.kernel(ox.outranking(N, c.w, c1, v1))
            k2 = ox.kernel(ox.outranking(N, c.w, c2, v2))
            diff.append(k1 != k2)
            jac.append(ox.jaccard(k1, k2))
        rows.append({"c1": c1, "v1": v1, "c2": c2, "v2": v2, "changed": float(np.mean(diff)), "jaccard": float(np.mean(jac))})
    return rows


COPELAND_CONFIGS = ((0.5, INF), (0.6, INF), (0.7, INF), (0.6, 0.5))


def experiment_copeland(deciders=C.EXP_KERNEL_DECIDERS, m=C.EXP_M_MAIN, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Eine Rangfolge aus der Relation ziehen: Platz 1 und tau der Copeland-Zahl gegen die Summe."""
    rows = []
    for dec in deciders:
        stats = {}
        for s in seeds:
            c = make_case(n, m, s, 1.0, dec)
            N = sc.minmax(c.S)
            best = sc.top1(c.utility)
            cands = {"summe": N @ c.w}
            for cstar, v in COPELAND_CONFIGS:
                cands[f"copeland_{cstar}_{veto_label(v)}"] = ox.copeland(ox.outranking(N, c.w, cstar, v)).astype(float)
            for k, score in cands.items():
                r = _result(c, score, best)
                st = stats.setdefault(k, {"hit": [], "tau": [], "tie": []})
                st["hit"].append(r.hit)
                st["tau"].append(r.tau)
                st["tie"].append(r.tie_at_top)
        for k, st in stats.items():
            rows.append({"decider": dec, "method": k, "hit": float(np.mean(st["hit"])), "tau": float(np.mean(st["tau"])), "tie": float(np.mean(st["tie"]))})
    return rows


# ----------------------------------------------------------------------------- Experiment 4: Veto gegen Untergrenze, Genauigkeit

VETO_METHODS = ("summe", "topsis_nicht", "copeland", "copeland_veto_0.5", "copeland_veto_0.3", "summe_nach_filter")


def experiment_veto(m=C.EXP_M_MAIN, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Entscheider mit Untergrenze (kein Kriterium unter THETA): Anteil der Wahlen unter der Untergrenze und Treffer auf den zulässigen Besten; dazu der Kern (c* = 0,6)
    mit und ohne Veto: enthält er Unzulässiges?"""
    keys = ("summe", "copeland", "copeland_veto_0.5", "copeland_veto_0.3", "summe_nach_filter")
    bad = {k: [] for k in keys}
    hit = {k: [] for k in keys}
    kbad = {"kern": [], "kern_veto_0.3": []}
    khit = {"kern": [], "kern_veto_0.3": []}
    unacc, skipped = [], 0
    for s in seeds:
        c = make_case(n, m, s, 1.0, "untergrenze")
        ok = c.acceptable
        unacc.append(1 - ok.mean())
        if not ok.any():
            skipped += 1
            continue
        best = sc.top1(c.utility)
        N = sc.minmax(c.S)
        ws = N @ c.w
        picks = {"summe": int(np.argmax(ws)), "summe_nach_filter": int(np.argmax(np.where(ok, ws, -np.inf)))}
        for key, v in (("copeland", INF), ("copeland_veto_0.5", 0.5), ("copeland_veto_0.3", 0.3)):
            picks[key] = int(np.argmax(ox.copeland(ox.outranking(N, c.w, 0.6, v))))
        for k, p in picks.items():
            bad[k].append(not ok[p])
            hit[k].append(p == best)
        for key, v in (("kern", INF), ("kern_veto_0.3", 0.3)):
            K = ox.kernel(ox.outranking(N, c.w, 0.6, v))
            kbad[key].append(any(not ok[i] for i in K))
            khit[key].append(best in K)
    rows = [{"method": k, "unacceptable": float(np.mean(bad[k])), "hit": float(np.mean(hit[k]))} for k in keys]
    rows += [{"method": k, "unacceptable": float(np.mean(kbad[k])), "hit": float(np.mean(khit[k]))} for k in kbad]
    for r in rows:
        r["unacceptable_share"] = float(np.mean(unacc))
        r["skipped"] = skipped
    return rows


def experiment_accuracy(deciders=("guete_linear", "roh_linear"), cstars=(0.6, 0.7, 0.8), vetoes=(INF, 0.5), m=C.EXP_M_MAIN, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Wenn ELECTRE a strikt über b stellt: Anteil der Paare, die der Entscheider genauso ordnet; gegen die Summe auf denselben Paaren und auf allen Paaren."""
    rows = []
    iu = np.triu_indices(n, 1)
    for dec in deciders:
        cases = [make_case(n, m, s, 1.0, dec) for s in seeds]
        for cstar in cstars:
            for v in vetoes:
                decided, acc_e, acc_s, acc_all = [], [], [], []
                for c in cases:
                    N = sc.minmax(c.S)
                    P = ox.relation_classes(ox.outranking(N, c.w, cstar, v))[0]
                    u, ws = c.utility, N @ c.w
                    truth = u[:, None] > u[None, :]
                    sum_pref = ws[:, None] > ws[None, :]
                    decided.append(P.sum() / (n * (n - 1) / 2))
                    if P.sum():
                        acc_e.append(truth[P].mean())
                        acc_s.append((sum_pref == truth)[P].mean())
                    acc_all.append((sum_pref[iu] == truth[iu]).mean())
                rows.append({"decider": dec, "cstar": cstar, "veto": v, "decided": float(np.mean(decided)), "acc_electre": float(np.mean(acc_e)),
                             "acc_sum_same": float(np.mean(acc_s)), "acc_sum_all": float(np.mean(acc_all))})
    return rows


# ----------------------------------------------------------------------------- Verdikte (live aus den Zeilen)

def _row(rows, **keys):
    return next(r for r in rows if all(r[k] == v for k, v in keys.items()))


def verdict_structure(rows: list[dict]) -> tuple[str, str]:
    m = C.EXP_M_MAIN
    z = {c: _row(rows, m=m, cstar=c, veto=INF)["cycle"] for c in (0.5, 0.6, 0.7)}
    r = _row(rows, m=m, cstar=0.6, veto=INF)
    v = _row(rows, m=m, cstar=0.6, veto=0.3)
    return "warning", (f"Bei {m} Kriterien enthält die Überordnung mit c* = 0,6 in {de(100 * z[0.6], 0)} % der Matrizen einen Zyklus (c* = 0,5: {de(100 * z[0.5], 0)} %, c* = 0,7: "
                       f"{de(100 * z[0.7], 0)} %). Bei c* = 0,6 sind {de(100 * r['r'], 0)} % der Paare unvergleichbar, mit einem Veto von 0,3 sind es {de(100 * v['r'], 0)} %.")


def verdict_kernel(rows: list[dict]) -> tuple[str, str]:
    r = _row(rows, decider="guete_linear", cstar=0.6)
    r8 = _row(rows, decider="guete_linear", cstar=0.8)
    return "info", (f"Bei c* = 0,6 enthält der Kern im Mittel {de(r['kernel'], 1)} von {C.EXP_N} Alternativen; der Beste des Entscheiders (Güte/linear) liegt in {de(100 * r['best_in_kernel'], 0)} % "
                    f"der Fälle darin, Platz 1 der Summe in {de(100 * r['sum_in_kernel'], 0)} %. Bei c* = 0,8 sind es {de(r8['kernel'], 1)} Alternativen, {de(100 * r8['best_in_kernel'], 0)} % und "
                    f"{de(100 * r8['sum_in_kernel'], 0)} %.")


def verdict_thresholds(thr: list[dict], cop: list[dict]) -> tuple[str, str]:
    a = _row(thr, c1=0.6, v1=INF, c2=0.7, v2=INF)
    b = _row(thr, c1=0.6, v1=INF, c2=0.8, v2=INF)
    s = _row(cop, decider="roh_linear", method="summe")["hit"]
    c = _row(cop, decider="roh_linear", method="copeland_0.6_ohne")["hit"]
    return "warning", (f"Wandert die Schwelle von 0,6 auf 0,7, ändert sich der Kern in {de(100 * a['changed'], 0)} % der Matrizen, von 0,6 auf 0,8 in {de(100 * b['changed'], 0)} %. "
                       f"Zieht man aus der Relation die Copeland-Zahl als Rangfolge, trifft sie bei einem Entscheider linear in den Rohwerten Platz 1 in {de(100 * c, 0)} % der Fälle, "
                       f"die Summe in {de(100 * s, 0)} %.")


def verdict_veto(rows: list[dict]) -> tuple[str, str]:
    cop = _row(rows, method="copeland")["unacceptable"]
    c5 = _row(rows, method="copeland_veto_0.5")["unacceptable"]
    c3 = _row(rows, method="copeland_veto_0.3")["unacceptable"]
    k0 = _row(rows, method="kern")["unacceptable"]
    k3 = _row(rows, method="kern_veto_0.3")["unacceptable"]
    return "warning", (f"Bei einer Untergrenze, die {de(100 * rows[0]['unacceptable_share'], 0)} % der Alternativen unzulässig macht, wählt die Copeland-Zahl Unzulässiges in {de(100 * cop, 0)} % der Fälle, "
                       f"mit Veto 0,5 in {de(100 * c5, 0)} %, mit Veto 0,3 in {de(100 * c3, 0)} %. Der Kern enthält trotzdem in {de(100 * k0, 0)} % (mit Veto 0,3: {de(100 * k3, 0)} %) "
                       "mindestens eine unzulässige Alternative: Das Veto macht unvergleichbar, es scheidet nicht aus.")


def verdict_accuracy(rows: list[dict]) -> tuple[str, str]:
    a = _row(rows, decider="guete_linear", cstar=0.6, veto=INF)
    b = _row(rows, decider="roh_linear", cstar=0.6, veto=INF)
    return "warning", (f"Bei c* = 0,6 ohne Veto entscheidet ELECTRE {de(100 * a['decided'], 0)} % der Paare strikt. Der Entscheider ordnet sie in {de(100 * a['acc_electre'], 0)} % genauso "
                       f"(Güte/linear; die Summe auf denselben Paaren in {de(100 * a['acc_sum_same'], 0)} %), in den Rohwerten in {de(100 * b['acc_electre'], 0)} % "
                       f"(Summe: {de(100 * b['acc_sum_same'], 0)} %). Vorsicht heißt hier: weniger entscheiden, nicht genauer entscheiden.")
