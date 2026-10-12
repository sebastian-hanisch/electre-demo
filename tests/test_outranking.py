"""Überordnung: Konkordanz, Veto, Relation, Klassen, Zyklen, Kern, Copeland-Zahl, Jaccard (je Einheit gegen Handwerte der Mini-Instanzen)."""

import math

import numpy as np
import pytest

import elc_outranking as ox


def test_concordance_hand_values(cycle, chain):
    N, w = cycle
    c = ox.concordance(N, w)
    assert c[0, 1] == pytest.approx(2 / 3) and c[1, 2] == pytest.approx(2 / 3) and c[2, 0] == pytest.approx(2 / 3)
    assert c[1, 0] == pytest.approx(1 / 3) and c[2, 1] == pytest.approx(1 / 3) and c[0, 2] == pytest.approx(1 / 3)
    assert np.diag(c) == pytest.approx([1.0, 1.0, 1.0])
    N, w = chain
    c = ox.concordance(N, w)
    assert c[0].tolist() == pytest.approx([1.0, 0.7, 1.0, 1.0]) and c[1].tolist() == pytest.approx([0.3, 1.0, 0.5, 1.0])
    assert c[2].tolist() == pytest.approx([0.0, 0.5, 1.0, 1.0]) and c[3].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_concordance_counts_ties_for_both():
    N = np.array([[0.5, 0.2], [0.5, 0.9]])
    c = ox.concordance(N, np.array([0.4, 0.6]))
    assert c[0, 1] == pytest.approx(0.4) and c[1, 0] == pytest.approx(1.0)


def test_veto_mask_hand_values(chain):
    N, _ = chain
    V = ox.veto_mask(N, 0.3)
    assert V[0, 1] and V[1, 0] and not V[0, 2] and V[2, 0] and V[3, 0] and not V[0, 3]
    assert not ox.veto_mask(N, math.inf).any()
    assert not ox.veto_mask(N, 0.5)[0, 1]                      # 0,4 ist nicht mehr als 0,5


def test_outranking_cycle_instance(cycle):
    N, w = cycle
    S = ox.outranking(N, w, 0.6)
    assert S.astype(int).tolist() == [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    assert not ox.outranking(N, w, 0.7).any()
    assert not ox.outranking(N, w, 0.6, 0.6).any()
    assert ox.outranking(N, w, 0.6, 1.0).astype(int).tolist() == S.astype(int).tolist()      # Veto 1,0: Unterschied 1,0 ist nicht mehr als 1,0


def test_cycle_instance_cycles_and_kernel(cycle):
    N, w = cycle
    S = ox.outranking(N, w, 0.6)
    assert ox.cycles(S) == [[0, 1, 2]] and ox.has_cycle(S)
    assert ox.kernel(S) == {0, 1, 2}
    assert (N @ w).tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert ox.copeland(S).tolist() == [0, 0, 0]


def test_chain_instance_relation_kernel_and_copeland(chain):
    N, w = chain
    S = ox.outranking(N, w, 0.6)
    assert S.astype(int).tolist() == [[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
    assert ox.kernel(S) == {0} and ox.copeland(S).tolist() == [3, 0, 0, -3] and not ox.has_cycle(S)
    S = ox.outranking(N, w, 0.6, 0.3)
    assert S.astype(int).tolist() == [[0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
    assert ox.kernel(S) == {0, 1} and ox.copeland(S).tolist() == [2, 1, 0, -3]
    S = ox.outranking(N, w, 0.8)
    assert ox.kernel(S) == {0, 1} and ox.copeland(S).tolist() == [2, 1, 0, -3]
    assert (N @ w).tolist() == pytest.approx([0.84, 0.64, 0.44, 0.0])


def test_relation_classes(chain):
    N, w = chain
    S = ox.outranking(N, w, 0.6)
    P, I, R = ox.relation_classes(S)
    assert P.sum() == 5 and I.sum() == 0 and R.sum() == 2 * 1          # unvergleichbar: nur das Paar (B, C), beide Richtungen
    assert R[1, 2] and R[2, 1] and not R.diagonal().any()
    S2 = np.array([[0, 1], [1, 0]], dtype=bool)
    P, I, R = ox.relation_classes(S2)
    assert not P.any() and I.sum() == 2 and not R.any()


def test_components_and_cycles_on_a_known_graph():
    S = np.zeros((6, 6), dtype=bool)
    for a, b in ((0, 1), (1, 2), (2, 0), (3, 4), (4, 3), (2, 5)):
        S[a, b] = True
    assert ox.cycles(S) == [[0, 1, 2], [3, 4]]
    k, labels = ox.components(S)
    assert k == 3 and labels[0] == labels[1] == labels[2] and labels[3] == labels[4] and labels[5] not in (labels[0], labels[3])
    assert not ox.has_cycle(np.triu(np.ones((4, 4), dtype=bool), 1))


def test_kernel_on_a_known_graph():
    S = np.zeros((5, 5), dtype=bool)
    for a, b in ((0, 1), (1, 2), (3, 2)):          # 0 -> 1 -> 2 <- 3, 4 isoliert
        S[a, b] = True
    # 0 und 3 sind Quellen, 4 ist isoliert; 1 und 2 werden übertroffen (1 von 0, 2 von 3): Kern {0, 3, 4}
    assert ox.kernel(S) == {0, 3, 4}


def test_kernel_member_that_was_outranked_only_by_a_removed_alternative():
    S = np.zeros((4, 4), dtype=bool)
    for a, b in ((0, 1), (1, 2)):
        S[a, b] = True                              # 0 -> 1 -> 2: Kern {0, 2} (2 wird nur von der entfernten 1 übertroffen)
    assert ox.kernel(S) == {0, 2, 3}


def test_copeland_is_a_signed_count():
    S = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=bool)
    assert ox.copeland(S).tolist() == [2, 0, -2]


def test_jaccard():
    assert ox.jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)
    assert ox.jaccard(set(), set()) == 1.0 and ox.jaccard({1}, {2}) == 0.0
