"""Orakel (DEMO-PLAYBOOK §12): unabhängige Rechenwege für Überordnung, Kern, Zyklen, Dominanz als Sonderfall, Monotonie und Dreierzyklen.

Jede Eigenschaft hat eine Gegenprobe, die zeigt, dass der Test etwas trennen kann (bei c* = 0,5 entstehen Zyklen und der Kern verlässt die Front; bei c* über 2/3 keine Dreierzyklen)."""

import itertools
import math

import numpy as np
import pytest

import elc_outranking as ox
import elc_score as sc


def _random_case(rng, n=8, m=4):
    N = sc.minmax(rng.uniform(0.0, 1.0, size=(n, m)))
    return N, rng.dirichlet(np.ones(m))


def _closure(S):
    """Erreichbarkeit per Warshall (unabhängig von scipy)."""
    R = np.array(S, dtype=bool) | np.eye(len(S), dtype=bool)
    for k in range(len(S)):
        R = R | (R[:, [k]] & R[[k], :])
    return R


# ----------------------------------------------------------------------------- 1. Überordnung gegen die Definition

@pytest.mark.parametrize("cstar", [0.5, 0.6, 0.8])
@pytest.mark.parametrize("veto", [math.inf, 0.5, 0.3])
def test_outranking_equals_the_definition_loop(cstar, veto):
    rng = np.random.default_rng(1)
    for _ in range(15):
        N, w = _random_case(rng)
        n, m = N.shape
        S = ox.outranking(N, w, cstar, veto)
        for a in range(n):
            for b in range(n):
                if a == b:
                    expected = False
                else:
                    conc = sum(w[j] for j in range(m) if N[a, j] >= N[b, j])
                    vetoed = any(N[b, j] - N[a, j] > veto for j in range(m))
                    expected = conc >= cstar - 1e-12 and not vetoed
                assert S[a, b] == expected


# ----------------------------------------------------------------------------- 2. Zyklen und Kern gegen Brute-Force

def _components_by_closure(S):
    R = _closure(S)
    n = len(S)
    comp = [-1] * n
    k = 0
    for i in range(n):
        if comp[i] == -1:
            for j in range(n):
                if R[i, j] and R[j, i]:
                    comp[j] = k
            k += 1
    return k, comp


def test_cycles_equal_the_closure_based_components():
    rng = np.random.default_rng(2)
    for _ in range(60):
        N, w = _random_case(rng, n=9, m=5)
        S = ox.outranking(N, w, 0.5)
        k, comp = _components_by_closure(S)
        groups = [sorted(i for i in range(9) if comp[i] == c) for c in range(k)]
        expected = sorted(g for g in groups if len(g) > 1)
        assert sorted(ox.cycles(S)) == expected


def test_kernel_equals_the_unique_independent_absorbing_set_of_components():
    """Brute-Force: Auf der Kondensation (ein azyklischer Graph) gibt es genau eine unabhängige und absorbierende Menge von Komponenten; ihre Vereinigung ist der Kern."""
    rng = np.random.default_rng(3)
    for _ in range(40):
        N, w = _random_case(rng, n=7, m=4)
        S = ox.outranking(N, w, float(rng.choice([0.5, 0.6, 0.7])), float(rng.choice([math.inf, 0.5, 0.3])))
        k, comp = _components_by_closure(S)
        edge = np.zeros((k, k), dtype=bool)
        for a, b in zip(*np.nonzero(S)):
            if comp[a] != comp[b]:
                edge[comp[a], comp[b]] = True
        sols = []
        for r in range(k + 1):
            for sub in itertools.combinations(range(k), r):
                members = set(sub)
                independent = not any(edge[a, b] for a in members for b in members)
                absorbing = all(any(edge[a, b] for a in members) for b in range(k) if b not in members)
                if independent and absorbing:
                    sols.append(members)
        assert len(sols) == 1
        assert ox.kernel(S) == {i for i in range(7) if comp[i] in sols[0]}


def test_every_non_member_is_outranked_by_a_kernel_member_and_kernel_is_independent_outside_cycles():
    rng = np.random.default_rng(4)
    for _ in range(60):
        N, w = _random_case(rng, n=10, m=5)
        S = ox.outranking(N, w, float(rng.choice([0.5, 0.6, 0.7])))
        K = ox.kernel(S)
        k, comp = _components_by_closure(S)
        for b in range(10):
            if b not in K:
                assert any(S[a, b] for a in K)
        for a in K:
            for b in K:
                if S[a, b]:
                    assert comp[a] == comp[b]


# ----------------------------------------------------------------------------- 3. Dominanz als Sonderfall, Monotonie, Zyklen

def test_concordance_threshold_one_is_weak_dominance():
    rng = np.random.default_rng(5)
    for _ in range(100):
        N, w = _random_case(rng, n=9, m=int(rng.integers(2, 7)))
        S = ox.outranking(N, w, 1.0)
        assert (S == sc.dominance_matrix(N)).all()
        assert not ox.has_cycle(S)


def test_raising_the_threshold_and_tightening_the_veto_only_remove_edges():
    rng = np.random.default_rng(6)
    for _ in range(40):
        N, w = _random_case(rng)
        lo = ox.outranking(N, w, 0.5, math.inf)
        for c2, v2 in ((0.6, math.inf), (0.5, 0.5), (0.7, 0.3)):
            hi = ox.outranking(N, w, c2, v2)
            assert not (hi & ~lo).any()


def test_no_three_cycles_above_two_thirds_but_many_at_one_half():
    rng = np.random.default_rng(7)
    def three_cycles(S):
        n = len(S)
        return any(S[a, b] and S[b, c] and S[c, a] for a in range(n) for b in range(n) for c in range(n) if len({a, b, c}) == 3)
    high = low = 0
    total = 150
    for _ in range(total):
        N, w = _random_case(rng, n=8, m=8)
        high += three_cycles(ox.outranking(N, w, 0.7))
        low += three_cycles(ox.outranking(N, w, 0.5))
    assert high == 0 and low > total / 2


def test_kernel_lies_inside_the_front_without_cycles_and_can_leave_it_with_cycles():
    rng = np.random.default_rng(8)
    inside_acyclic = outside_cyclic = 0
    for _ in range(300):
        N, w = _random_case(rng, n=10, m=5)
        front = sc.front_mask(N)
        for cstar in (0.5, 0.7):
            S = ox.outranking(N, w, cstar)
            K = ox.kernel(S)
            if not ox.has_cycle(S):
                assert all(front[i] for i in K)
                inside_acyclic += 1
            elif not all(front[i] for i in K):
                outside_cyclic += 1
    assert inside_acyclic > 50 and outside_cyclic >= 5


def test_veto_keeps_the_kernel_from_shrinking_to_the_vetoed_pair_only():
    """Ein Veto kann den Kern nur vergrößern oder unverändert lassen, wenn es Kanten entfernt, ohne neue zu schaffen: jede Kern-Alternative bleibt nicht übertroffen."""
    rng = np.random.default_rng(9)
    for _ in range(40):
        N, w = _random_case(rng, n=9, m=5)
        k0 = ox.kernel(ox.outranking(N, w, 0.6))
        k1 = ox.kernel(ox.outranking(N, w, 0.6, 0.3))
        s0 = ox.outranking(N, w, 0.6)
        sources0 = {b for b in range(9) if not s0[:, b].any()}
        assert sources0 <= k1 and sources0 <= k0


def test_copeland_equals_the_definition():
    rng = np.random.default_rng(10)
    N, w = _random_case(rng, n=9, m=5)
    S = ox.outranking(N, w, 0.6)
    assert ox.copeland(S).tolist() == [sum(S[a, b] for b in range(9)) - sum(S[b, a] for b in range(9)) for a in range(9)]
