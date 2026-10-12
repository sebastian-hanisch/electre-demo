"""Min-Max, gewichtete Summe, Nutzen des Entscheiders auf beiden Wertskalen (auch mit Untergrenze), Verschmälerung, Dominanz. Nur numpy; eigene Kopie der kleinen Helfer aus
gewichtete-summe-demo (kein Import über Repos). Alle Funktionen arbeiten im vorzeichenbehafteten Raum, in dem HÖHER BESSER gilt."""

from __future__ import annotations

import numpy as np

from entscheidungsmatrix_generator import CRITERIA, to_raw, utility


def minmax(S: np.ndarray) -> np.ndarray:
    """Min-Max-Normierung je Kriterium über die Stichprobe; konstante Spalte -> lauter Nullen."""
    S = np.asarray(S, dtype=float)
    rng = S.max(axis=0) - S.min(axis=0)
    return (S - S.min(axis=0)) / np.where(rng == 0, 1.0, rng)


def weighted_score(N: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.asarray(N, dtype=float) @ np.asarray(w, dtype=float)


def top1(score: np.ndarray) -> int:
    """Index des höchsten Scores; bei Gleichstand die kleinste Nummer."""
    return int(np.argmax(score))


def nominal_bounds(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Nenn-Spannweite je Kriterium im vorzeichenbehafteten Raum: ↑ [lo, hi], ↓ [-hi, -lo]."""
    lo = np.array([CRITERIA[k][3] if CRITERIA[k][1] > 0 else -CRITERIA[k][4] for k in range(m)], dtype=float)
    hi = np.array([CRITERIA[k][4] if CRITERIA[k][1] > 0 else -CRITERIA[k][3] for k in range(m)], dtype=float)
    return lo, hi


def narrow_criterion(goodness: np.ndarray, delta: float, column: int = 0) -> np.ndarray:
    """Staucht die Güte eines Kriteriums um 0,5 auf den Anteil `delta` der Streuung: g' = 0,5 + delta (g - 0,5). delta = 1 ändert nichts."""
    g = np.array(goodness, dtype=float, copy=True)
    g[:, column] = 0.5 + delta * (g[:, column] - 0.5)
    return g


def acceptable(goodness: np.ndarray, theta: float) -> np.ndarray:
    """Zulässig sind Alternativen, deren Güte in JEDEM Kriterium mindestens theta beträgt (absolute Mindestanforderung)."""
    return (np.asarray(goodness, dtype=float) >= theta).all(axis=1)


def decider_utility(scale: str, form: str, goodness: np.ndarray, S: np.ndarray, w: np.ndarray, m: int, theta: float | None = None) -> np.ndarray:
    """Nutzen des Entscheiders: in der Güte (linear, konkav, cobb aus dem Generator) oder linear in den Rohwerten bezogen auf die feste Nenn-Spannweite.
    Mit `theta` ist der Nutzen unzulässiger Alternativen um 1 gesenkt: Sie liegen unter allen zulässigen (die Güte-Nutzen liegen in [0, 1])."""
    if scale == "guete":
        u = utility(goodness, w, form)
    elif scale == "roh":
        if form != "linear":
            raise ValueError("in den Rohwerten ist nur die lineare Form definiert")
        lo, hi = nominal_bounds(m)
        u = ((np.asarray(S, dtype=float) - lo) / (hi - lo)) @ np.asarray(w, dtype=float)
    else:
        raise ValueError(f"unbekannte Wertskala: {scale}")
    if theta is not None:
        u = np.where(acceptable(goodness, theta), u, u - 1.0)
    return u


def build_case(goodness: np.ndarray, directions: np.ndarray, m: int, delta: float):
    """Aus Güte und Richtungen: (Güte nach Verschmälerung, Rohwerte, vorzeichenbehaftete Rohwerte)."""
    g = narrow_criterion(goodness, delta)
    raw = to_raw(g, m)
    return g, raw, raw * directions


def utility_loss(utility_values: np.ndarray, chosen: int) -> float:
    return float(np.max(utility_values) - utility_values[chosen])


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """Kendalls tau-b (Gleichstände berücksichtigt); 0, wenn eine Seite keine Unterschiede hat."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = np.sign(a[:, None] - a[None, :])
    sb = np.sign(b[:, None] - b[None, :])
    denom = np.sqrt((sa * sa).sum() * (sb * sb).sum())
    return 0.0 if denom == 0 else float((sa * sb).sum() / denom)


def dominance_matrix(G: np.ndarray) -> np.ndarray:
    """D[i, j] = True, wenn i die Alternative j schwach dominiert (in jedem Kriterium mindestens so gut, in einem besser)."""
    G = np.asarray(G, dtype=float)
    return (G[:, None, :] >= G[None, :, :]).all(axis=2) & (G[:, None, :] > G[None, :, :]).any(axis=2)


def front_mask(G: np.ndarray) -> np.ndarray:
    """Alternativen, die von keiner anderen dominiert werden."""
    return ~dominance_matrix(G).any(axis=0)
