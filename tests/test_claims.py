"""Jede Zahl aus README.md und App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über SWEEP_SEEDS (200 feste Matrizen), n = 20, nie über einen einzelnen Seed. Prozentwerte mit Toleranz 0,5 Punkte, Kerngrößen mit 0,05, Jaccard mit 0,01,
damit Rundung und Bibliotheksversionen (CI installiert ungepinnt) nichts kippen. Der Standardfehler einer Trefferquote ist 1,3 bis 3,5 Punkte: Unterschiede unter etwa 6 Punkten
sind NICHT belastbar und stehen nirgends als Aussage.
"""

import math

import pytest

import elc_constants as C
import elc_evaluation as ev

INF = math.inf


@pytest.fixture(scope="module")
def structure():
    return ev.experiment_structure()


@pytest.fixture(scope="module")
def kernel():
    return ev.experiment_kernel()


@pytest.fixture(scope="module")
def thresholds():
    return ev.experiment_thresholds()


@pytest.fixture(scope="module")
def copeland():
    return ev.experiment_copeland()


@pytest.fixture(scope="module")
def veto():
    return ev.experiment_veto()


@pytest.fixture(scope="module")
def accuracy():
    return ev.experiment_accuracy()


def cell(rows, **keys):
    hits = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    assert len(hits) == 1, (keys, len(hits))
    return hits[0]


def pct(x):
    return 100 * x


# ----------------------------------------------------------------------------- Experiment 1: Struktur

STRUCTURE = {   # (m, c*, Veto): (strikt %, unvergleichbar %, Zyklus %, Kern, Front, Kern in Front %)
    (3, 0.5, INF): (100.0, 0.0, 22.5, 2.72, 7.32, 90.0), (3, 0.6, INF): (84.7, 15.3, 3.5, 2.27, 7.32, 98.0), (3, 0.7, INF): (70.5, 29.5, 0.5, 2.65, 7.32, 100.0),
    (3, 0.8, INF): (56.3, 43.7, 0.0, 3.65, 7.32, 100.0),
    (5, 0.5, INF): (99.9, 0.0, 68.5, 5.85, 14.15, 73.0), (5, 0.5, 0.5): (55.9, 44.1, 45.0, 2.86, 14.15, 99.0), (5, 0.5, 0.3): (33.7, 66.2, 5.0, 4.43, 14.15, 100.0),
    (5, 0.6, INF): (79.6, 20.4, 38.5, 2.63, 14.15, 96.5), (5, 0.6, 0.5): (46.6, 53.4, 8.5, 3.48, 14.15, 100.0), (5, 0.6, 0.3): (29.4, 70.6, 0.5, 5.33, 14.15, 100.0),
    (5, 0.7, INF): (58.4, 41.6, 2.0, 3.25, 14.15, 100.0), (5, 0.8, INF): (39.0, 61.0, 0.0, 5.29, 14.15, 100.0),
    (8, 0.5, INF): (99.9, 0.0, 93.5, 9.59, 18.9, 79.5), (8, 0.6, INF): (71.2, 28.8, 65.0, 3.09, 18.9, 100.0), (8, 0.7, INF): (43.9, 56.1, 1.5, 4.51, 18.9, 100.0),
    (8, 0.8, INF): (23.1, 76.9, 0.0, 7.91, 18.9, 100.0),
}


@pytest.mark.parametrize("key", sorted(STRUCTURE, key=str))
def test_structure_of_the_relation(structure, key):
    m, cstar, v = key
    p, r, cycle, kern, front, kif = STRUCTURE[key]
    row = cell(structure, m=m, cstar=cstar, veto=v)
    assert pct(row["p"]) == pytest.approx(p, abs=0.5) and pct(row["r"]) == pytest.approx(r, abs=0.5) and pct(row["cycle"]) == pytest.approx(cycle, abs=0.5)
    assert row["kernel"] == pytest.approx(kern, abs=0.05) and row["front"] == pytest.approx(front, abs=0.05) and pct(row["kernel_in_front"]) == pytest.approx(kif, abs=0.5)


def test_mutual_outranking_is_practically_absent(structure):
    assert all(pct(r["i"]) <= 0.1 for r in structure)


def test_cycles_fall_with_the_threshold_and_with_the_veto(structure):
    for m in C.EXP_M:
        for v in C.EXP_VETOES:
            cycles = [cell(structure, m=m, cstar=c, veto=v)["cycle"] for c in C.EXP_CSTARS]
            assert cycles == sorted(cycles, reverse=True)
        for c in C.EXP_CSTARS:
            cycles = [cell(structure, m=m, cstar=c, veto=v)["cycle"] for v in C.EXP_VETOES]
            assert cycles == sorted(cycles, reverse=True)


def test_no_cycle_at_the_high_thresholds_in_the_sample(structure):
    """Gemessen (0 von 200), nicht bewiesen: Dreierzyklen sind für c* über 2/3 unmöglich, längere Zyklen bleiben möglich."""
    for m in C.EXP_M:
        assert cell(structure, m=m, cstar=0.8, veto=INF)["cycle"] == 0.0


# ----------------------------------------------------------------------------- Experiment 2: Kern und Bester

KERNEL = {   # (Entscheider, c*): (Kern, Bester im Kern %, Summe im Kern %)
    ("guete_linear", 0.6): (2.63, 77.5, 70.5), ("guete_linear", 0.7): (3.25, 89.0, 84.0), ("guete_linear", 0.8): (5.29, 97.5, 95.0),
    ("guete_cobb", 0.6): (2.63, 70.5, 70.5), ("guete_cobb", 0.7): (3.25, 79.0, 84.0), ("guete_cobb", 0.8): (5.29, 92.0, 95.0),
    ("roh_linear", 0.6): (2.63, 70.5, 70.5), ("roh_linear", 0.7): (3.25, 83.5, 84.0), ("roh_linear", 0.8): (5.29, 94.5, 95.0),
}


@pytest.mark.parametrize("key", sorted(KERNEL))
def test_kernel_and_the_best(kernel, key):
    dec, cstar = key
    k, best, summe = KERNEL[key]
    row = cell(kernel, decider=dec, cstar=cstar)
    assert row["kernel"] == pytest.approx(k, abs=0.05) and pct(row["best_in_kernel"]) == pytest.approx(best, abs=0.5) and pct(row["sum_in_kernel"]) == pytest.approx(summe, abs=0.5)
    assert row["best_in_front"] == 1.0


# ----------------------------------------------------------------------------- Experiment 3: Schwellen

THRESHOLDS = {   # (c1, v1, c2, v2): (anderer Kern %, Jaccard)
    (0.6, INF, 0.7, INF): (68.5, 0.61), (0.7, INF, 0.8, INF): (74.0, 0.63), (0.6, INF, 0.8, INF): (87.5, 0.44),
    (0.7, INF, 0.7, 0.5): (73.5, 0.67), (0.7, 0.5, 0.7, 0.3): (81.0, 0.72), (0.6, INF, 0.6, 0.5): (78.5, 0.57),
}


@pytest.mark.parametrize("key", sorted(THRESHOLDS, key=str))
def test_kernel_changes_with_the_thresholds(thresholds, key):
    c1, v1, c2, v2 = key
    changed, jac = THRESHOLDS[key]
    row = cell(thresholds, c1=c1, v1=v1, c2=c2, v2=v2)
    assert pct(row["changed"]) == pytest.approx(changed, abs=0.5) and row["jaccard"] == pytest.approx(jac, abs=0.01)


COPELAND = {   # (Entscheider, Verfahren): (Platz 1 %, Gleichstand an der Spitze %)
    ("guete_linear", "summe"): (67.0, 0.0), ("guete_linear", "copeland_0.5_ohne"): (54.0, 14.5), ("guete_linear", "copeland_0.6_ohne"): (67.5, 11.5),
    ("guete_linear", "copeland_0.7_ohne"): (72.5, 11.0), ("guete_linear", "copeland_0.6_0,5"): (52.5, 11.5),
    ("guete_cobb", "summe"): (69.0, 0.0), ("guete_cobb", "copeland_0.5_ohne"): (47.0, 14.5), ("guete_cobb", "copeland_0.6_ohne"): (56.0, 11.5),
    ("guete_cobb", "copeland_0.7_ohne"): (58.5, 11.0), ("guete_cobb", "copeland_0.6_0,5"): (60.5, 11.5),
    ("roh_linear", "summe"): (96.5, 0.0), ("roh_linear", "copeland_0.5_ohne"): (50.0, 14.5), ("roh_linear", "copeland_0.6_ohne"): (54.5, 11.5),
    ("roh_linear", "copeland_0.7_ohne"): (56.0, 11.0), ("roh_linear", "copeland_0.6_0,5"): (58.0, 11.5),
}


@pytest.mark.parametrize("key", sorted(COPELAND))
def test_copeland_number_as_a_ranking(copeland, key):
    dec, method = key
    hit, tie = COPELAND[key]
    row = cell(copeland, decider=dec, method=method)
    assert pct(row["hit"]) == pytest.approx(hit, abs=0.5) and pct(row["tie"]) == pytest.approx(tie, abs=0.5)


def test_copeland_number_trails_the_sum_in_the_raw_world_by_a_wide_margin(copeland):
    s = cell(copeland, decider="roh_linear", method="summe")["hit"]
    for m in ("copeland_0.5_ohne", "copeland_0.6_ohne", "copeland_0.7_ohne", "copeland_0.6_0,5"):
        assert pct(s - cell(copeland, decider="roh_linear", method=m)["hit"]) > 35


# ----------------------------------------------------------------------------- Experiment 4: Veto und Genauigkeit

VETO = {"summe": (11.0, 64.5), "copeland": (19.0, 62.0), "copeland_veto_0.5": (3.0, 60.5), "copeland_veto_0.3": (1.0, 53.5), "summe_nach_filter": (0.0, 70.5),
        "kern": (42.5, 73.0), "kern_veto_0.3": (70.0, 94.0)}


@pytest.mark.parametrize("method", sorted(VETO))
def test_veto_against_the_floor(veto, method):
    bad, hit = VETO[method]
    row = cell(veto, method=method)
    assert pct(row["unacceptable"]) == pytest.approx(bad, abs=0.5) and pct(row["hit"]) == pytest.approx(hit, abs=0.5)


def test_unacceptable_alternatives_share_and_no_skipped_matrix(veto):
    assert pct(veto[0]["unacceptable_share"]) == pytest.approx(40.9, abs=0.1) and veto[0]["skipped"] == 0


ACCURACY = {   # (Entscheider, c*, Veto): (entschieden %, ELECTRE %, Summe gleiche Paare %, Summe alle Paare %)
    ("guete_linear", 0.6, INF): (79.6, 90.7, 95.1, 93.5), ("guete_linear", 0.6, 0.5): (46.6, 96.5, 97.4, 93.5), ("guete_linear", 0.7, INF): (58.4, 96.0, 97.2, 93.5),
    ("guete_linear", 0.7, 0.5): (36.3, 98.8, 98.9, 93.5), ("guete_linear", 0.8, INF): (39.0, 99.0, 99.0, 93.5), ("guete_linear", 0.8, 0.5): (26.1, 99.8, 99.7, 93.5),
    ("roh_linear", 0.6, INF): (79.6, 88.8, 98.5, 98.0), ("roh_linear", 0.6, 0.5): (46.6, 95.8, 99.4, 98.0), ("roh_linear", 0.7, INF): (58.4, 94.6, 99.1, 98.0),
    ("roh_linear", 0.7, 0.5): (36.3, 98.4, 99.7, 98.0), ("roh_linear", 0.8, INF): (39.0, 98.4, 99.7, 98.0), ("roh_linear", 0.8, 0.5): (26.1, 99.6, 99.9, 98.0),
}


@pytest.mark.parametrize("key", sorted(ACCURACY, key=str))
def test_accuracy_of_strict_preferences(accuracy, key):
    dec, cstar, v = key
    d, e, s_same, s_all = ACCURACY[key]
    row = cell(accuracy, decider=dec, cstar=cstar, veto=v)
    assert pct(row["decided"]) == pytest.approx(d, abs=0.5) and pct(row["acc_electre"]) == pytest.approx(e, abs=0.5)
    assert pct(row["acc_sum_same"]) == pytest.approx(s_same, abs=0.5) and pct(row["acc_sum_all"]) == pytest.approx(s_all, abs=0.5)


def test_the_sum_is_at_least_as_accurate_on_the_same_pairs_up_to_rounding(accuracy):
    """Die Vorsicht kauft keine Genauigkeit: In keiner Zelle ist ELECTRE auf den entschiedenen Paaren genauer als die Summe um mehr als 0,2 Punkte."""
    for r in accuracy:
        assert pct(r["acc_electre"] - r["acc_sum_same"]) <= 0.2
