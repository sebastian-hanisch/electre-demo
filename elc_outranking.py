"""ELECTRE I (crisp): Konkordanz, Veto, Überordnung, Klassen, Zyklen, Kern der Überordnung, Copeland-Zahl. Nur numpy und scipy (starke Zusammenhangskomponenten).

Alle Funktionen arbeiten auf einer normierten Matrix N (n x m, höher besser, Min-Max) und Gewichten w. a S b ("a übertrifft b") gilt, wenn die Konkordanz
c(a, b) = Summe der Gewichte der Kriterien, in denen a mindestens so gut ist wie b, mindestens c* beträgt und kein Veto vorliegt (b ist in keinem Kriterium um mehr als v besser).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components


def concordance(N: np.ndarray, w: np.ndarray) -> np.ndarray:
    """c[a, b] = Summe der Gewichte der Kriterien, in denen a mindestens so gut ist wie b (Gleichstand zählt für beide)."""
    N = np.asarray(N, dtype=float)
    return ((N[:, None, :] >= N[None, :, :]) * np.asarray(w, dtype=float)).sum(axis=-1)


def veto_mask(N: np.ndarray, veto: float) -> np.ndarray:
    """V[a, b] = True, wenn b in irgendeinem Kriterium um mehr als `veto` besser ist als a (dann darf a b nicht übertreffen)."""
    N = np.asarray(N, dtype=float)
    return ((N[None, :, :] - N[:, None, :]) > veto).any(axis=-1)


def outranking(N: np.ndarray, w: np.ndarray, cstar: float, veto: float = float("inf")) -> np.ndarray:
    """Relation S[a, b] = (a übertrifft b); die Diagonale ist leer. Toleranz 1e-12 gegen Rundung an der Schwelle."""
    S = (concordance(N, w) >= cstar - 1e-12) & ~veto_mask(N, veto)
    np.fill_diagonal(S, False)
    return S


def relation_classes(S: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P, I, R): strikt (a S b, nicht b S a), beidseitig (a S b und b S a), unvergleichbar (weder noch; ohne Diagonale)."""
    S = np.asarray(S, dtype=bool)
    P = S & ~S.T
    I = S & S.T
    R = ~S & ~S.T
    np.fill_diagonal(R, False)
    return P, I, R


def components(S: np.ndarray) -> tuple[int, np.ndarray]:
    """Starke Zusammenhangskomponenten der Relation (Zyklen liegen innerhalb einer Komponente)."""
    k, labels = connected_components(np.asarray(S, dtype=bool), directed=True, connection="strong")
    return int(k), labels


def cycles(S: np.ndarray) -> list[list[int]]:
    """Komponenten mit mindestens zwei Alternativen: die Zyklen der Relation."""
    k, labels = components(S)
    return [sorted(np.flatnonzero(labels == c).tolist()) for c in range(k) if (labels == c).sum() > 1]


def has_cycle(S: np.ndarray) -> bool:
    return bool(cycles(S))


def kernel(S: np.ndarray) -> set[int]:
    """Kern der Überordnung: Iterativ die Quellen-Komponenten der Kondensation der verbleibenden Relation nehmen (Alternativen, die keine Alternative außerhalb ihrer
    Komponente übertrifft), dann alles entfernen, was diese Kern-Mitglieder übertreffen, und wiederholen. Mitglieder eines Zyklus bleiben alle im Kern."""
    S = np.asarray(S, dtype=bool)
    remaining = set(range(len(S)))
    core: set[int] = set()
    while remaining:
        idx = sorted(remaining)
        sub = S[np.ix_(idx, idx)]
        k, labels = components(sub)
        incoming = np.zeros(k, dtype=bool)
        for a, b in zip(*np.nonzero(sub)):
            if labels[a] != labels[b]:
                incoming[labels[b]] = True
        members = {idx[i] for i in range(len(idx)) if not incoming[labels[i]]}
        core |= members
        outranked = {idx[b] for i in range(len(idx)) if idx[i] in members for b in np.nonzero(sub[i])[0]}
        remaining -= members | outranked
    return core


def copeland(S: np.ndarray) -> np.ndarray:
    """Copeland-Zahl: Zahl der übertroffenen minus Zahl der übertreffenden Alternativen."""
    S = np.asarray(S, dtype=bool)
    return S.sum(axis=1) - S.sum(axis=0)


def jaccard(a: set, b: set) -> float:
    """Jaccard-Index zweier Mengen (1, wenn beide leer)."""
    return 1.0 if not (a | b) else len(a & b) / len(a | b)
