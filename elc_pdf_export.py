"""PDF-Export des aktuellen Falls (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import elc_constants as C
from elc_evaluation import Analysis, de, veto_label

_REPLACE = {"↑": "+", "↓": "-", "→": "->", "↔": "<->", "–": "-", "—": "-", "τ": "tau", "σ": "sigma", "δ": "delta", "θ": "theta", "≥": ">=", "≤": "<=", "−": "-",
            "€": "EUR", "·": "*", "⊆": "in", "∞": "inf"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    n = s.n
    names = [f"A{i + 1}" for i in range(n)]
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("ELECTRE: ein schlechter Wert wird nicht ausgeglichen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Alternativen n = {n}, Kriterien m = {s.m}, Konkordanzschwelle c* = {de(s.cstar, 1)}, Veto: {veto_label(s.veto)}, Entscheider: {C.DECIDER_LABELS[s.decider]}, "
                            f"Streuung des ersten Kriteriums delta = {de(s.delta, 2)}, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(f"Kern der Ueberordnung: {', '.join(names[i] for i in sorted(a.kernel))}. Zyklen: "
                            f"{'; '.join('{' + ', '.join(names[i] for i in c) + '}' for c in a.cycles) or 'keine'}. Bester nach Nutzen: {names[a.best]}. "
                            f"Platz 1 der Summe: {names[a.results['summe'].top1]} ({'im Kern' if a.summe_in_kernel else 'nicht im Kern'})."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, _s("Wer uebertrifft wen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for i in range(n):
        over = ", ".join(names[j] for j in range(n) if a.S[i, j]) or "-"
        pdf.cell(0, 5, _s(f"{names[i]}{'*' if i in a.kernel else ''}: uebertrifft {over}; Copeland-Zahl {int(a.results['copeland'].score[i])}; "
                          f"Summe {de(a.results['summe'].score[i], 3)}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Der Entscheider ist ein simuliertes Orakel, kein Mensch; * = Kern. Quelle: electre-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
