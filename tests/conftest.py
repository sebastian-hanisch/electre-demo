"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def cycle():
    """Zyklus über drei Kriterien: a = (1; 0; 0,5), b = (0,5; 1; 0), c = (0; 0,5; 1), w = 1/3 je Kriterium (Min-Max-normiert).
    c(a, b) = c(b, c) = c(c, a) = 2/3, c(b, a) = c(c, b) = c(a, c) = 1/3. Bei c* = 0,6: a S b, b S c, c S a (ein Zyklus), Kern {a, b, c}; die Summe ist ein Gleichstand 0,5 / 0,5 / 0,5.
    Bei c* = 0,7: keine Relation. Bei c* = 0,6 mit Veto 0,6: jede Kante ist ein Veto (der Unterschied ist 1)."""
    return np.array([[1.0, 0.0, 0.5], [0.5, 1.0, 0.0], [0.0, 0.5, 1.0]]), np.full(3, 1 / 3)


@pytest.fixture
def chain():
    """A = (1; 0,6; 0,8), B = (0,4; 1; 0,7), C = (0,5; 0,5; 0,2), D = (0; 0; 0), w = (0,5; 0,3; 0,2).
    Konkordanz: c(A, B) = 0,7, c(B, A) = 0,3, c(B, C) = 0,5, c(C, B) = 0,5; c* = 0,6, kein Veto: A S B, A S C, A S D, B S D, C S D; Kern {A}; Copeland 3 / 0 / 0 / -3.
    Veto 0,3: B ist in Kriterium 2 um 0,4 besser als A, A S B entfällt; Kern {A, B}; Copeland 2 / 1 / 0 / -3. c* = 0,8: A S B entfällt (0,7 < 0,8), Kern {A, B}.
    Summe 0,84 / 0,64 / 0,44 / 0."""
    return np.array([[1.0, 0.6, 0.8], [0.4, 1.0, 0.7], [0.5, 0.5, 0.2], [0.0, 0.0, 0.0]]), np.array([0.5, 0.3, 0.2])
