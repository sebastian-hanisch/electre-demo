"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import math

N_MIN, N_MAX, DEFAULT_N = 5, 40, 8            # Standard 8: Der Graph im Kreislayout bleibt lesbar (Graph nur bis GRAPH_MAX_N)
M_MIN, M_MAX, DEFAULT_M = 3, 8, 5
SEED_MAX, DEFAULT_SEED = 999999, 1
GRAPH_MAX_N = 12

CSTAR_CHOICES = (0.5, 0.6, 0.7, 0.8, 0.9)     # Konkordanzschwelle c*: Mindestanteil des Gewichts, der für "a übertrifft b" spricht
DEFAULT_CSTAR = 0.6
VETO_CHOICES = (math.inf, 0.7, 0.5, 0.3)      # Veto v: b ist in einem Kriterium um mehr als v (Anteil der Spannweite) besser; inf = kein Veto
DEFAULT_VETO = math.inf

DECIDERS = ("guete_linear", "guete_konkav", "guete_cobb", "roh_linear", "untergrenze")
DECIDER_LABELS = {
    "guete_linear": "Linear in der Güte (Quantil innerhalb des Kriteriums)",
    "guete_konkav": "Konkav in der Güte (Wurzel, gewichtet)",
    "guete_cobb": "Cobb-Douglas in der Güte (Logarithmus, gewichtet)",
    "roh_linear": "Linear in den Rohwerten (feste Nenn-Spannweite)",
    "untergrenze": "Linear in der Güte, mit Untergrenze (kein Kriterium unter 0,1)",
}
DECIDER_SHORT = {"guete_linear": "Güte/linear", "guete_konkav": "Güte/konkav", "guete_cobb": "Güte/Cobb-Douglas", "roh_linear": "Rohwerte/linear",
                 "untergrenze": "Güte/Untergrenze"}
DECIDER_PARTS = {"guete_linear": ("guete", "linear"), "guete_konkav": ("guete", "konkav"), "guete_cobb": ("guete", "cobb"), "roh_linear": ("roh", "linear"),
                 "untergrenze": ("guete", "linear")}
DEFAULT_DECIDER = "guete_linear"
THETA = 0.1                                    # Untergrenze der Güte des Entscheiders mit Untergrenze (Mindestanforderung, absolut)

DELTA_CHOICES = (1.0, 0.5, 0.2, 0.05)         # Anteil der Streuung des ersten Kriteriums (1 = unverändert)
DEFAULT_DELTA = 1.0

# Feste Instanzen für alle Experimente: Mittel über SWEEP_SEEDS, nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(200000, 200200))
EXP_N = 20                                     # Experimente: feste Instanz mit 20 Alternativen, unabhängig vom Regler
EXP_M = (3, 5, 8)
EXP_M_MAIN = 5
EXP_CSTARS = (0.5, 0.6, 0.7, 0.8)
EXP_VETOES = (math.inf, 0.5, 0.3)
EXP_KERNEL_DECIDERS = ("guete_linear", "guete_cobb", "roh_linear")
EXP_KERNEL_CSTARS = (0.6, 0.7, 0.8)

COLOR_KERNEL = "#2F6B65"
COLOR_FRONT = "#4c78a8"
COLOR_OTHER = "rgba(150,150,150,0.55)"
COLOR_WARN = "#c0392b"
COLOR_BEST = "#D68A2E"
COLOR_SUM = "#8A96A6"

PRESETS = {
    "Standardfall (c* = 0,6, kein Veto)": {"n": DEFAULT_N, "m": DEFAULT_M, "cstar": DEFAULT_CSTAR, "veto": DEFAULT_VETO, "decider": DEFAULT_DECIDER,
                                           "delta": 1.0, "seed": DEFAULT_SEED},
    "Ein Zyklus über drei Kriterien": {"n": 6, "m": 3, "cstar": 0.6, "veto": math.inf, "decider": "guete_linear", "delta": 1.0, "seed": 136},
    "Veto macht unvergleichbar": {"n": 8, "m": 5, "cstar": 0.6, "veto": 0.3, "decider": "untergrenze", "delta": 1.0, "seed": 2},
    "Die Schwelle verschiebt den Kern": {"n": 8, "m": 5, "cstar": 0.7, "veto": math.inf, "decider": "guete_linear", "delta": 1.0, "seed": 8},
}
PRESET_HELP = {
    "Standardfall (c* = 0,6, kein Veto)": "8 Alternativen, 5 Kriterien, Konkordanzschwelle 0,6, kein Veto: Der Kern enthält den Besten und die Wahl der Summe, die Relation hat keinen Zyklus.",
    "Ein Zyklus über drei Kriterien": "Drei Alternativen übertreffen einander im Kreis: Jede ist in zwei von drei Kriterien besser als die nächste. Der Kern ist der ganze Zyklus; die Summe ordnet trotzdem.",
    "Veto macht unvergleichbar": "Entscheider mit Untergrenze, Veto 0,3: Eine unzulässige Alternative gerät in den Kern, weil das Veto verhindert, dass sie übertroffen wird. Ohne Veto bleibt der Kern zulässig.",
    "Die Schwelle verschiebt den Kern": "Mit c* = 0,7 liegt der Beste im Kern, mit c* = 0,6 nicht: Die Antwort hängt an der Schwelle.",
}
