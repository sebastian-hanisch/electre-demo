"""Auswertung: Fall der App (Invarianten), Entscheider mit Untergrenze, Experimente im Kleinen, Verdikte."""

import math

import numpy as np
import pytest

import elc_constants as C
import elc_evaluation as ev
import elc_outranking as ox


def test_make_case_is_deterministic_and_the_floor_decider_ranks_unacceptable_last():
    a, b = ev.make_case(20, 5, 7), ev.make_case(20, 5, 7)
    assert (a.S == b.S).all() and (a.w == b.w).all() and a.utility == pytest.approx(a.goodness @ a.w)
    f = ev.make_case(20, 5, 7, 1.0, "untergrenze")
    ok = f.acceptable
    assert ok.any() and (~ok).any()
    assert f.utility[ok].min() > f.utility[~ok].max()
    assert f.utility[ok] == pytest.approx((f.goodness @ f.w)[ok]) and f.utility[~ok] == pytest.approx((f.goodness @ f.w)[~ok] - 1.0)
    assert f.acceptable.tolist() == (f.goodness >= C.THETA).all(axis=1).tolist()


def test_narrowing_changes_only_the_first_column():
    full, narrow = ev.make_case(20, 5, 7, 1.0), ev.make_case(20, 5, 7, 0.05)
    assert (full.goodness[:, 1:] == narrow.goodness[:, 1:]).all()
    assert np.ptp(narrow.goodness[:, 0]) == pytest.approx(0.05 * np.ptp(full.goodness[:, 0]))


def test_analyse_basic_invariants():
    a = ev.analyse(ev.Settings())
    n = a.settings.n
    assert a.N.shape == (n, a.settings.m) and a.S.shape == (n, n) and not a.S.diagonal().any()
    assert a.P.sum() + a.I.sum() / 2 + a.R.sum() / 2 == n * (n - 1) / 2      # jedes ungeordnete Paar liegt in genau einer Klasse
    assert a.kernel == ox.kernel(a.S) and a.best == int(np.argmax(a.case.utility))
    assert a.best_in_kernel == (a.best in a.kernel) and a.summe_in_kernel == (a.results["summe"].top1 in a.kernel)
    assert a.results["copeland"].score.tolist() == ox.copeland(a.S).tolist()
    for r in a.results.values():
        assert r.hit == (r.top1 == a.best) and r.loss >= 0 and -1 <= r.tau <= 1


def test_cycle_preset_instance_and_threshold_effect():
    a = ev.analyse(ev.Settings(**C.PRESETS["Ein Zyklus über drei Kriterien"]))
    assert a.cycles == [[1, 3, 5]] and a.kernel == {1, 3, 5}
    b = ev.analyse(ev.Settings(**dict(C.PRESETS["Die Schwelle verschiebt den Kern"], cstar=0.6)))
    c = ev.analyse(ev.Settings(**C.PRESETS["Die Schwelle verschiebt den Kern"]))
    assert b.kernel != c.kernel and c.best_in_kernel and not b.best_in_kernel


def test_unacceptable_flag_follows_the_kernel_and_the_floor():
    a = ev.analyse(ev.Settings(**C.PRESETS["Veto macht unvergleichbar"]))
    assert a.unacceptable_in_kernel == any(not a.case.acceptable[i] for i in a.kernel)
    b = ev.analyse(ev.Settings(**dict(C.PRESETS["Veto macht unvergleichbar"], veto=math.inf)))
    assert not b.unacceptable_in_kernel and len(b.kernel) < len(a.kernel)


def test_veto_label_and_de():
    assert ev.veto_label(math.inf) == "ohne" and ev.veto_label(0.3) == "0,3" and ev.de(0.1234, 3) == "0,123"


def test_experiment_structure_small_run():
    rows = ev.experiment_structure(ms=(3,), cstars=(0.5, 0.8), vetoes=(math.inf,), seeds=tuple(range(200000, 200012)))
    assert len(rows) == 2
    for r in rows:
        assert r["p"] + r["i"] + r["r"] == pytest.approx(1.0) and 0 <= r["cycle"] <= 1 and r["kernel"] >= 1
    assert next(r for r in rows if r["cstar"] == 0.8)["cycle"] == 0.0


def test_experiment_kernel_and_thresholds_small_run():
    rows = ev.experiment_kernel(deciders=("guete_linear",), cstars=(0.6, 0.8), seeds=tuple(range(200000, 200012)))
    assert len(rows) == 2 and all(r["best_in_front"] == 1.0 for r in rows)
    thr = ev.experiment_thresholds(seeds=tuple(range(200000, 200012)))
    assert len(thr) == len(ev.THRESHOLD_PAIRS) and all(0 <= r["changed"] <= 1 and 0 <= r["jaccard"] <= 1 for r in thr)


def test_experiment_copeland_veto_and_accuracy_small_runs():
    cop = ev.experiment_copeland(deciders=("roh_linear",), seeds=tuple(range(200000, 200010)))
    assert {r["method"] for r in cop} == {"summe", "copeland_0.5_ohne", "copeland_0.6_ohne", "copeland_0.7_ohne", "copeland_0.6_0,5"}
    veto = ev.experiment_veto(seeds=tuple(range(200000, 200015)))
    assert {r["method"] for r in veto} == {"summe", "copeland", "copeland_veto_0.5", "copeland_veto_0.3", "summe_nach_filter", "kern", "kern_veto_0.3"}
    assert next(r for r in veto if r["method"] == "summe_nach_filter")["unacceptable"] == 0.0
    acc = ev.experiment_accuracy(deciders=("guete_linear",), cstars=(0.6,), vetoes=(math.inf,), seeds=tuple(range(200000, 200010)))
    assert len(acc) == 1 and 0 <= acc[0]["acc_electre"] <= 1 and 0 < acc[0]["decided"] <= 1


def test_experiments_are_deterministic():
    seeds = tuple(range(200000, 200006))
    first = ev.experiment_kernel(deciders=("guete_cobb",), cstars=(0.7,), seeds=seeds)
    assert first == ev.experiment_kernel(deciders=("guete_cobb",), cstars=(0.7,), seeds=seeds)


def test_verdicts_read_their_numbers_from_the_rows():
    def srow(m, c, v, cyc, r):
        return {"m": m, "cstar": c, "veto": v, "p": 1 - r, "i": 0.0, "r": r, "cycle": cyc, "cycle_p": cyc, "kernel": 3.0, "front": 14.0, "kernel_in_front": 1.0}

    rows = [srow(5, 0.5, math.inf, 0.685, 0.0), srow(5, 0.6, math.inf, 0.385, 0.204), srow(5, 0.7, math.inf, 0.02, 0.416), srow(5, 0.6, 0.3, 0.005, 0.706)]
    kind, text = ev.verdict_structure(rows)
    assert kind == "warning" and "38 %" in text and "68 %" in text and "2 %" in text and "20 %" in text and "71 %" in text

    krows = [{"decider": "guete_linear", "cstar": c, "kernel": k, "best_in_kernel": b, "sum_in_kernel": s_, "best_in_front": 1.0}
             for c, k, b, s_ in ((0.6, 2.63, 0.775, 0.705), (0.8, 5.29, 0.975, 0.95))]
    kind, text = ev.verdict_kernel(krows)
    assert kind == "info" and "2,6 von 20" in text and "5,3" in text and "98 %" in text

    thr = [{"c1": 0.6, "v1": math.inf, "c2": 0.7, "v2": math.inf, "changed": 0.685, "jaccard": 0.61},
           {"c1": 0.6, "v1": math.inf, "c2": 0.8, "v2": math.inf, "changed": 0.875, "jaccard": 0.44}]
    cop = [{"decider": "roh_linear", "method": "summe", "hit": 0.965}, {"decider": "roh_linear", "method": "copeland_0.6_ohne", "hit": 0.545}]
    kind, text = ev.verdict_thresholds(thr, cop)
    assert kind == "warning" and "68 %" in text and "88 %" in text and "55 %" in text and "96 %" in text

    vrows = [{"method": k, "unacceptable": v, "hit": 0.6, "unacceptable_share": 0.409, "skipped": 0}
             for k, v in (("copeland", 0.19), ("copeland_veto_0.5", 0.03), ("copeland_veto_0.3", 0.01), ("kern", 0.425), ("kern_veto_0.3", 0.70))]
    kind, text = ev.verdict_veto(vrows)
    assert kind == "warning" and "19 %" in text and "3 %" in text and "1 %" in text and "70 %" in text and "41 %" in text and "unvergleichbar" in text

    arows = [{"decider": d, "cstar": 0.6, "veto": math.inf, "decided": 0.796, "acc_electre": e, "acc_sum_same": s_, "acc_sum_all": 0.9}
             for d, e, s_ in (("guete_linear", 0.907, 0.951), ("roh_linear", 0.888, 0.985))]
    kind, text = ev.verdict_accuracy(arows)
    assert kind == "warning" and "80 %" in text and "91 %" in text and "95 %" in text and "89 %" in text and "98 %" in text
