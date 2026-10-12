"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import elc_constants as C
from elc_evaluation import Analysis, de, veto_label

ARROW = {1: "↑", -1: "↓"}
COPELAND_LABELS = {"summe": "Summe (Min-Max)", "copeland_0.5_ohne": "Copeland, c* 0,5", "copeland_0.6_ohne": "Copeland, c* 0,6", "copeland_0.7_ohne": "Copeland, c* 0,7",
                   "copeland_0.6_0,5": "Copeland, c* 0,6, Veto 0,5"}


def alt_names(n: int) -> list[str]:
    return [f"A{i + 1}" for i in range(n)]


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def criterion_label(a: Analysis, k: int) -> str:
    inst = a.case.matrix
    return f"K{k + 1} {ARROW[int(inst.directions[k])]} {inst.units[k]}"


def build_relation_matrix(a: Analysis) -> go.Figure:
    """Zeile a gegenüber Spalte b: 2 = a übertrifft b strikt, 1 = beidseitig, 0 = keine Überordnung. Kern-Mitglieder tragen ein Sternchen."""
    n = a.settings.n
    names = alt_names(n)
    z = np.where(a.P, 2, np.where(a.I, 1, 0))
    labels = [f"{nm}{'*' if i in a.kernel else ''}" for i, nm in enumerate(names)]
    text = [["" if i == j else ("→" if a.P[i, j] else "↔" if a.I[i, j] else "·") for j in range(n)] for i in range(n)]
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, text=text, texttemplate="%{text}", zmin=0, zmax=2, showscale=False,
                               colorscale=[[0, "rgba(150,150,150,0.15)"], [0.5, "#e0b040"], [1, C.COLOR_KERNEL]],
                               hovertemplate="%{y} gegenüber %{x}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=dict(text="Überordnung (Zeile übertrifft Spalte; * = Kern)", font=dict(size=13)))
    return _fixed(fig, 380)


def build_graph(a: Analysis) -> go.Figure:
    """Kreislayout: Pfeile für strikte Überordnung, Doppelstrich für beidseitige; Kern grün, Zyklus-Mitglieder mit Rand, Bester als Stern."""
    n = a.settings.n
    names = alt_names(n)
    ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    x, y = np.cos(ang), np.sin(ang)
    fig = go.Figure()
    in_cycle = {i for cyc in a.cycles for i in cyc}
    for i in range(n):
        for j in range(n):
            if i != j and a.S[i, j] and (a.P[i, j] or i < j):
                fig.add_annotation(x=0.88 * x[j] + 0.12 * x[i], y=0.88 * y[j] + 0.12 * y[i], ax=x[i], ay=y[i], xref="x", yref="y", axref="x", ayref="y",
                                   showarrow=True, arrowhead=3 if a.P[i, j] else 0, arrowsize=1.2, arrowwidth=1.4,
                                   arrowcolor=C.COLOR_WARN if (i in in_cycle and j in in_cycle) else "rgba(80,80,80,0.55)")
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers+text", text=names, textposition="middle center",
                             marker=dict(size=30, color=[C.COLOR_KERNEL if i in a.kernel else C.COLOR_OTHER for i in range(n)],
                                         line=dict(width=[3 if i in in_cycle else 1 for i in range(n)], color=[C.COLOR_WARN if i in in_cycle else "#444" for i in range(n)])),
                             textfont=dict(color="white", size=10), hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter(x=[x[a.best]], y=[y[a.best]], mode="markers", marker=dict(symbol="star", size=14, color=C.COLOR_BEST), name="bester nach Nutzen"))
    fig.update_xaxes(visible=False, range=[-1.25, 1.25])
    fig.update_yaxes(visible=False, range=[-1.25, 1.25], scaleanchor="x")
    fig.update_layout(title=dict(text="Graph (grün: Kern, rot: Zyklus)", font=dict(size=13)))
    return _fixed(fig, 380)


def build_table(a: Analysis) -> pd.DataFrame:
    n = a.settings.n
    names = alt_names(n)
    rank = lambda v: np.argsort(np.argsort(-np.asarray(v), kind="stable"), kind="stable") + 1  # noqa: E731
    df = pd.DataFrame({
        "Alternative": names,
        "übertrifft": [", ".join(names[j] for j in np.flatnonzero(a.S[i])) or "-" for i in range(n)],
        "wird übertroffen von": [", ".join(names[j] for j in np.flatnonzero(a.S[:, i])) or "-" for i in range(n)],
        "Copeland-Zahl": a.results["copeland"].score.astype(int),
        "im Kern": np.where([i in a.kernel for i in range(n)], "ja", "nein"),
        "nichtdominiert": np.where(a.front, "ja", "nein"),
        "Rang Summe": rank(a.results["summe"].score),
        "Rang Entscheider": rank(a.case.utility),
    })
    if a.settings.decider == "untergrenze":
        df["zulässig"] = np.where(a.case.acceptable, "ja", "nein")
    return df


def pair_details(a: Analysis, i: int, j: int) -> pd.DataFrame:
    """Je Kriterium: Werte (Min-Max), ob a mindestens so gut ist wie b, Gewicht; darunter Konkordanz und Veto."""
    m = a.settings.m
    names = alt_names(a.settings.n)
    return pd.DataFrame({
        "Kriterium": [f"K{k + 1}" for k in range(m)], "Gewicht": a.case.w, f"{names[i]} (Min-Max)": a.N[i], f"{names[j]} (Min-Max)": a.N[j],
        f"{names[i]} mindestens so gut": np.where(a.N[i] >= a.N[j], "ja", "nein"), f"{names[j]} besser um mehr als das Veto": np.where(a.N[j] - a.N[i] > a.settings.veto, "ja", "nein"),
    })


def style_pair(df: pd.DataFrame):
    fmt = {c: (lambda v: de(v, 3)) for c in df.columns if c == "Gewicht" or "(Min-Max)" in c}
    return df.style.format(fmt)


def build_method_table(a: Analysis) -> pd.DataFrame:
    names = alt_names(a.settings.n)
    rows = []
    for k, label in (("summe", "Summe (Min-Max, wahre Gewichte)"), ("copeland", "Copeland-Zahl der Überordnung")):
        r = a.results[k]
        rows.append({"Verfahren": label, "Platz 1": names[r.top1] + (" (Gleichstand)" if r.tie_at_top else ""), "Trifft den Besten": "ja" if r.hit else "nein",
                     "τ gegen Nutzen": de(r.tau, 2), "Nutzenverlust": de(r.loss, 3), "im Kern": "ja" if r.top1 in a.kernel else "nein"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Experimente

def build_structure(rows: list[dict], m: int = 5) -> go.Figure:
    """Anteil der Matrizen mit Zyklus je Konkordanzschwelle und Veto."""
    fig = go.Figure()
    for v in C.EXP_VETOES:
        rs = [r for r in rows if r["m"] == m and r["veto"] == v]
        fig.add_trace(go.Bar(x=[f"c* = {de(r['cstar'], 1)}" for r in rs], y=[100 * r["cycle"] for r in rs], name=f"Veto {veto_label(v)}"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Matrizen mit Zyklus in der Überordnung (%)", range=[0, 102])
    return _fixed(fig, 320)


def structure_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Kriterien m": r["m"], "c*": de(r["cstar"], 1), "Veto": veto_label(r["veto"]), "strikt entschieden": f"{de(100 * r['p'], 1)} %",
                          "unvergleichbar": f"{de(100 * r['r'], 1)} %", "Zyklus": f"{de(100 * r['cycle'], 1)} %", "Kern": de(r["kernel"], 1), "Front": de(r["front"], 1),
                          "Kern ⊆ Front": f"{de(100 * r['kernel_in_front'], 1)} %"} for r in rows])


def kernel_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Entscheider": C.DECIDER_SHORT[r["decider"]], "c*": de(r["cstar"], 1), "Kern (von 20)": de(r["kernel"], 1),
                          "Bester im Kern": f"{de(100 * r['best_in_kernel'], 1)} %", "Platz 1 der Summe im Kern": f"{de(100 * r['sum_in_kernel'], 1)} %",
                          "Bester in der Front": f"{de(100 * r['best_in_front'], 1)} %"} for r in rows])


def build_kernel(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    for dec in dict.fromkeys(r["decider"] for r in rows):
        rs = [r for r in rows if r["decider"] == dec]
        fig.add_trace(go.Scatter(x=[de(r["cstar"], 1) for r in rs], y=[100 * r["best_in_kernel"] for r in rs], mode="lines+markers", name=C.DECIDER_SHORT[dec]))
    fig.update_xaxes(title="Konkordanzschwelle c*")
    fig.update_yaxes(title="Bester des Entscheiders liegt im Kern (%)", range=[0, 102])
    return _fixed(fig, 320)


def threshold_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Vergleich": f"c* {de(r['c1'], 1)}, Veto {veto_label(r['v1'])} gegen c* {de(r['c2'], 1)}, Veto {veto_label(r['v2'])}",
                          "anderer Kern": f"{de(100 * r['changed'], 1)} %", "mittleres Jaccard": de(r["jaccard"], 2)} for r in rows])


def copeland_table(rows: list[dict]) -> pd.DataFrame:
    methods = list(dict.fromkeys(r["method"] for r in rows))
    out = []
    for k in methods:
        row = {"Verfahren": COPELAND_LABELS.get(k, k)}
        for r in [x for x in rows if x["method"] == k]:
            row[f"{C.DECIDER_SHORT[r['decider']]}: Platz 1"] = f"{de(100 * r['hit'], 1)} %"
        out.append(row)
    return pd.DataFrame(out)


VETO_LABELS = {"summe": "Summe, Platz 1", "copeland": "Copeland-Zahl (c* 0,6, kein Veto)", "copeland_veto_0.5": "Copeland-Zahl, Veto 0,5", "copeland_veto_0.3": "Copeland-Zahl, Veto 0,3",
               "summe_nach_filter": "Summe nach Filter auf zulässige", "kern": "Kern enthält Unzulässiges (c* 0,6, kein Veto)",
               "kern_veto_0.3": "Kern enthält Unzulässiges (c* 0,6, Veto 0,3)"}


def build_veto(rows: list[dict]) -> go.Figure:
    fig = go.Figure(go.Bar(x=[VETO_LABELS[r["method"]] for r in rows], y=[100 * r["unacceptable"] for r in rows], marker_color=C.COLOR_WARN))
    fig.update_yaxes(title="wählt bzw. enthält Unzulässiges (%)", rangemode="tozero")
    fig.update_xaxes(tickangle=-25)
    return _fixed(fig, 420)


def veto_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Verfahren": VETO_LABELS[r["method"]], "Wahl unzulässig bzw. Kern enthält Unzulässiges": f"{de(100 * r['unacceptable'], 1)} %",
                          "zulässiger Bester getroffen bzw. im Kern": f"{de(100 * r['hit'], 1)} %"} for r in rows])


def accuracy_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Entscheider": C.DECIDER_SHORT[r["decider"]], "c*": de(r["cstar"], 1), "Veto": veto_label(r["veto"]), "entschiedene Paare": f"{de(100 * r['decided'], 1)} %",
                          "Genauigkeit ELECTRE": f"{de(100 * r['acc_electre'], 1)} %", "Summe, gleiche Paare": f"{de(100 * r['acc_sum_same'], 1)} %",
                          "Summe, alle Paare": f"{de(100 * r['acc_sum_all'], 1)} %"} for r in rows])
