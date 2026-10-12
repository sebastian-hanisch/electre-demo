"""Score-Helfer: Min-Max, Summe, Nutzen auf beiden Wertskalen und mit Untergrenze, Verschmälerung, Dominanz, Kendalls tau-b."""

import numpy as np
import pytest

import elc_score as sc


def test_minmax_hand_values_and_constant_column():
    S = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    assert sc.minmax(S).tolist() == [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]


def test_weighted_score_and_top1_tie_goes_to_the_first():
    N = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    s = sc.weighted_score(N, np.array([0.5, 0.5]))
    assert s.tolist() == [0.5, 0.5, 0.5] and sc.top1(s) == 0


def test_narrow_criterion_shrinks_around_one_half_and_leaves_the_rest():
    g = np.array([[0.0, 0.3], [1.0, 0.7], [0.5, 0.9]])
    h = sc.narrow_criterion(g, 0.2)
    assert h[:, 0].tolist() == pytest.approx([0.4, 0.6, 0.5]) and (h[:, 1] == g[:, 1]).all()
    assert (sc.narrow_criterion(g, 1.0) == g).all()


def test_acceptable_needs_every_criterion_at_the_floor():
    g = np.array([[0.5, 0.5], [0.05, 0.9], [0.1, 0.1], [0.09, 0.3]])
    assert sc.acceptable(g, 0.1).tolist() == [True, False, True, False]


def test_decider_utility_forms_and_floor():
    g = np.array([[0.25, 0.81], [1.0, 0.04], [0.5, 0.5]])
    w = np.array([0.5, 0.5])
    S = np.array([[20.0, -1000.0], [80.0, -4000.0], [50.0, -2000.0]])
    assert sc.decider_utility("guete", "linear", g, S, w, 2).tolist() == pytest.approx([0.53, 0.52, 0.5])
    assert sc.decider_utility("guete", "konkav", g, S, w, 2)[0] == pytest.approx(0.5 * 0.5 + 0.5 * 0.9)
    assert sc.decider_utility("guete", "cobb", g, S, w, 2)[1] == pytest.approx(0.5 * np.log(1.0) + 0.5 * np.log(0.04))
    u = sc.decider_utility("roh", "linear", g, S, w, 2)
    assert u[0] == pytest.approx(0.5 * 0.2 + 0.5 * (4000 / 4200))
    f = sc.decider_utility("guete", "linear", g, S, w, 2, theta=0.1)         # Alternative 2 hat Güte 0,04 < 0,1: unzulässig, Nutzen - 1
    assert f.tolist() == pytest.approx([0.53, 0.52 - 1.0, 0.5])
    with pytest.raises(ValueError):
        sc.decider_utility("roh", "cobb", g, S, w, 2)
    with pytest.raises(ValueError):
        sc.decider_utility("zehn", "linear", g, S, w, 2)


def test_kendall_tau_b_hand_values():
    assert sc.kendall_tau_b([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert sc.kendall_tau_b([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert sc.kendall_tau_b([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(4 / 6)
    assert sc.kendall_tau_b([1, 1, 1], [1, 2, 3]) == 0.0


def test_utility_loss():
    assert sc.utility_loss(np.array([0.2, 0.9, 0.5]), 2) == pytest.approx(0.4)
    assert sc.utility_loss(np.array([0.2, 0.9, 0.5]), 1) == 0.0


def test_dominance_and_front():
    G = np.array([[1.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    assert sc.front_mask(G).tolist() == [True, False, False, False]
    G2 = np.array([[1.0, 0.0], [0.0, 1.0], [0.4, 0.4], [0.3, 0.3]])
    assert sc.front_mask(G2).tolist() == [True, True, True, False]
    assert sc.front_mask(np.array([[0.5, 0.5], [0.5, 0.5]])).tolist() == [True, True]
