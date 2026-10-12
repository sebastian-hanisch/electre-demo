# ELECTRE: ein schlechter Wert wird nicht ausgeglichen – Streamlit-Demo

<!-- Nach dem Deploy hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-electre-demo.streamlit.app/)** -->

Sechstes Stück der Linie **Mehrkriterien-Ranking** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Kind der
[gewichteten Summe](https://github.com/sebastian-hanisch/gewichtete-summe-demo). Die Summe lässt einen sehr guten Wert einen sehr schlechten ausgleichen. ELECTRE vergleicht je
**Paar** von Alternativen: a **übertrifft** b, wenn Kriterien mit zusammen mindestens dem Anteil c\* des Gewichts für a sprechen (**Konkordanz**) und b in keinem Kriterium so weit
vorn liegt, dass es ein **Veto** auslöst. Das Ergebnis ist eine Relation mit einem **Kern der Überordnung**, keine Rangfolge; sie kann **Zyklen** und **Unvergleichbarkeit** enthalten.

Vehikel ist dieselbe **neutrale, simulierte Entscheidungsmatrix** wie in den Vorgängern (Generator `entscheidungsmatrix_generator.py`, wortgleich). Der **Entscheider** kennt seine
Gewichte und seinen Nutzen (linear, konkav oder Cobb-Douglas in der Güte, linear in den Rohwerten, oder linear mit einer Untergrenze); er ist ein Orakel, kein Mensch.

## Warum dieses Problem

**ELECTRE antwortet auf „was wird nicht überboten?“, nicht auf „was ist das Beste?“, und die Antwort hängt an den Schwellen.** Die Kernfrage der Demo: Wie oft ist die Antwort
überhaupt eine Rangfolge, wie stark hängt sie an c\* und am Veto, und kauft die Vorsicht etwas, nämlich Genauigkeit?

## Befunde und Korrekturen gegenüber dem Plan

- **„Ein schlechter Wert wird nicht ausgeglichen“ stimmt nur relativ.** Das Veto verhindert, dass a b übertrifft, wenn b in einem Kriterium weit vorn liegt; es setzt keine absolute
  Untergrenze. Mit einer Untergrenze von 0,1 in jeder Güte wählt die Copeland-Zahl in 19 % der Fälle Unzulässiges, mit Veto 0,5 in 3 %, mit Veto 0,3 in 1 %; der Kern enthält
  trotzdem in 42,5 % (mit Veto 0,3: 70,0 %) eine unzulässige Alternative. Ein Filter vor der Summe ist einfacher und besser (0 %, 70,5 % Treffer gegen 64,5 %).
- **Die Vorsicht kauft keine Genauigkeit.** Auf den entschiedenen Paaren ist die Summe mindestens so genau wie ELECTRE (bei c\* = 0,6 ohne Veto 95,1 gegen 90,7 % bei einem
  Entscheider linear in der Güte, 98,5 gegen 88,8 % in den Rohwerten); ELECTRE entscheidet weniger Paare, nicht bessere.
- **Zyklen sind bei Schwellen nahe 0,5 der Regelfall:** 5 Kriterien, c\* = 0,5 / 0,6 / 0,7: Zyklus in 68,5 / 38,5 / 2,0 % der Matrizen. Zyklen ziehen auch dominierte Alternativen in
  den Kern (Kern ⊆ Front nur in 73,0 / 96,5 / 100 %).
- **Die Rangfolge ist eine Zugabe.** Die Copeland-Zahl der Relation trifft Platz 1 meist schlechter als die Summe (Rohwerte/linear 50,0 bis 58,0 % gegen 96,5 %); in 11 bis 14,5 % der
  Matrizen liegt ein Gleichstand an der Spitze.
- **Die Entscheider sind kompensatorisch** (bis auf einen): ELECTRE ist für Entscheider gebaut, die nicht ausgleichen wollen. „Verliert gegen die Summe“ gilt gegen diese Entscheider.
- **Die 0,0 % Zyklen bei c\* = 0,8 sind gemessen (0 von 200), nicht bewiesen:** Dreierzyklen sind für c\* über 2/3 unmöglich, ein Zyklus der Länge k braucht c\* ≤ (k − 1)/k.
- **Der Kern ist eine Definition dieser Demo** (iterativ über die starken Zusammenhangskomponenten), nicht die des Originals; Mitglieder eines Zyklus bleiben alle im Kern.
- **Die Experimente starten auf Klick** (rund 10 Sekunden), nicht beim Laden.

## Modell

- **Matrix:** n Alternativen (5 bis 40, Standard 8), m Kriterien (3 bis 8), Rohwerte mit Richtung und Einheit; optional ein Kriterium auf den Anteil δ seiner Streuung gestaucht.
- **Normierung und Gewichte:** nur Min-Max; die wahren Gewichte des Entscheiders (die Erhebung steht in den Vorgängern).
- **Konkordanz:** c(a, b) = Summe der Gewichte der Kriterien, in denen a mindestens so gut ist wie b. **Veto:** b ist in einem Kriterium um mehr als v (Anteil der Spannweite) besser.
  **Überordnung:** a S b, wenn c(a, b) ≥ c\* und kein Veto. Klassen: strikt, beidseitig, unvergleichbar.
- **Kern der Überordnung:** iterativ die Quellen-Komponenten der Kondensation nehmen, dann alles entfernen, was sie übertreffen; Zyklen sind starke Zusammenhangskomponenten.
- **Entscheider mit Untergrenze:** Nutzen linear in der Güte; Alternativen mit einer Güte unter 0,1 in irgendeinem Kriterium sind unzulässig und liegen im Nutzen unter allen zulässigen.
- **Fester Rahmen der Experimente:** 200 feste Matrizen (`SWEEP_SEEDS`), n = 20, unabhängig vom Regler n.

Formeln im 📐-Expander der App.

## Methodik

1. **Dieselbe Matrix, derselbe Entscheider, alle Schwellen:** Vergleiche zwischen Schwellen sind gepaart.
2. **Messgrößen sind die der Relation:** Zyklen, Unvergleichbarkeit, Kerngröße, Kern enthält den Besten, Änderung des Kerns bei Schwellenwechsel (Jaccard); keine Trefferquote als Hauptmaß,
   weil ELECTRE keinen einzelnen Platz 1 liefert. Die Copeland-Zahl dient nur dem Vergleich mit der Summe.
3. **Genauigkeit strikter Präferenzen** gegen die Summe auf denselben Paaren (a P b gegen die Ordnung des Entscheiders).
4. **Veto gegen Untergrenze:** Anteil der Wahlen und der Kerne unter der Untergrenze.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: `tests/test_claims.py`. 200 feste Matrizen, n = 20, wahre Gewichte, Min-Max. **Der Standardfehler einer Trefferquote ist 1,3 bis 3,5 Punkte; Unterschiede unter etwa
6 Punkten sind nicht belastbar und stehen nicht als Aussage.**

| Frage | Befund | Test |
|---|---|---|
| Struktur der Relation (5 Kriterien, kein Veto; c\* = 0,5 / 0,6 / 0,7 / 0,8) | strikt entschiedene Paare 99,9 / 79,6 / 58,4 / 39,0 %, unvergleichbar 0,0 / 20,4 / 41,6 / 61,0 %, beidseitig höchstens 0,1 %; **Zyklus 68,5 / 38,5 / 2,0 / 0,0 %**; Kern 5,85 / 2,63 / 3,25 / 5,29 von 20 (Front 14,15). | `test_structure_of_the_relation` |
| Dasselbe bei 3 und 8 Kriterien (c\* = 0,5 / 0,6 / 0,7) | Zyklus 22,5 / 3,5 / 0,5 % (3 Kriterien) und 93,5 / 65,0 / 1,5 % (8 Kriterien). | `test_structure_of_the_relation` |
| Wirkung des Veto (5 Kriterien, c\* = 0,6; kein Veto / 0,5 / 0,3) | Zyklus 38,5 / 8,5 / 0,5 %; unvergleichbar 20,4 / 53,4 / 70,6 %; Kern 2,63 / 3,48 / 5,33 Alternativen. | `test_structure_of_the_relation` |
| Zyklen ziehen Dominiertes in den Kern (Kern ⊆ Front; c\* = 0,5 / 0,6 / 0,7) | 73,0 / 96,5 / 100 % (5 Kriterien); 79,5 / 100 / 100 % (8 Kriterien); bei 3 Kriterien c\* = 0,5: 90,0 %. | `test_structure_of_the_relation` |
| Liegt der Beste im Kern? (5 Kriterien, kein Veto; c\* = 0,6 / 0,7 / 0,8) | Güte/linear 77,5 / 89,0 / 97,5 %; Cobb-Douglas 70,5 / 79,0 / 92,0 %; Rohwerte 70,5 / 83,5 / 94,5 %; **Platz 1 der Summe 70,5 / 84,0 / 95,0 %**; der Beste liegt zu 100 % in der Front. | `test_kernel_and_the_best` |
| Wie stark hängt der Kern an den Schwellen? (5 Kriterien, Güte/linear) | Anderer Kern: c\* 0,6 → 0,7 in 68,5 % (Jaccard 0,61), 0,7 → 0,8 in 74,0 % (0,63), 0,6 → 0,8 in 87,5 % (0,44); Veto ohne → 0,5 bei c\* 0,7 in 73,5 % (0,67), 0,5 → 0,3 in 81,0 % (0,72). | `test_kernel_changes_with_the_thresholds` |
| Stimmen die strikten Präferenzen? (5 Kriterien; c\* = 0,6 / 0,7 / 0,8, kein Veto; Güte/linear) | entschieden 79,6 / 58,4 / 39,0 % der Paare; ELECTRE 90,7 / 96,0 / 99,0 %; Summe auf denselben Paaren 95,1 / 97,2 / 99,0 %; Summe alle Paare 93,5 %. | `test_accuracy_of_strict_preferences` |
| Dasselbe, Rohwerte/linear | ELECTRE 88,8 / 94,6 / 98,4 %; Summe auf denselben Paaren 98,5 / 99,1 / 99,7 %; alle Paare 98,0 %. | `test_accuracy_of_strict_preferences` |
| Veto gegen Untergrenze (θ = 0,1; 40,9 % der Alternativen unzulässig; c\* = 0,6) | Wahl unzulässig: Summe 11,0 %, Copeland 19,0 %, Veto 0,5: 3,0 %, Veto 0,3: 1,0 %, Summe nach Filter 0,0 %; **Kern enthält Unzulässiges 42,5 % (kein Veto) bzw. 70,0 % (Veto 0,3)**. | `test_veto_against_the_floor` |
| Treffer auf den zulässigen Besten (θ = 0,1) | Summe 64,5 %, Copeland 62,0 %, Veto 0,5: 60,5 %, Veto 0,3: 53,5 %, Summe nach Filter 70,5 %; Kern enthält ihn in 73,0 % bzw. 94,0 %. | `test_veto_against_the_floor` |
| Copeland-Zahl als Rangfolge (5 Kriterien; Platz 1) | Güte/linear: c\* 0,5 / 0,6 / 0,7: 54,0 / 67,5 / 72,5 % (Summe 67,0 %); Cobb-Douglas 47,0 / 56,0 / 58,5 % (Summe 69,0 %); **Rohwerte 50,0 / 54,5 / 56,0 % (Summe 96,5 %)**; Gleichstand an der Spitze 14,5 / 11,5 / 11,0 %. | `test_copeland_number_as_a_ranking` |

## Ehrliche Grenzen

- **Die Entscheider sind kompensatorisch** (bis auf den mit Untergrenze). ELECTRE ist für Entscheider gebaut, die ein schlechtes Kriterium nicht ausgleichen wollen; die Genauigkeit
  gegen linearen, konkaven oder Cobb-Douglas-Nutzen sagt nichts über echte Entscheider mit solchen Präferenzen.
- **Nur ELECTRE I, crisp.** Kredibilität, Indifferenz- und Präferenzschwellen und die Destillation (ELECTRE III) sind nicht umgesetzt; die Schwellen sind einheitliche Anteile der
  Spannweite, nicht je Kriterium.
- **Der Kern hängt von dieser Definition ab** und von den anderen Alternativen: Wie er sich beim Hinzufügen einer Alternative ändert, ist hier nicht gemessen.
- **Ein Generator, wahre Gewichte.** n = 20, 3 bis 8 Kriterien; die Gewichte sind bekannt. Quellenangaben nur aus Zusammenfassungen.
- **Kleine Unterschiede sind Rauschen.** Die Reihenfolge von Copeland-Zahl und Summe bei Güte/linear und c\* 0,6 bis 0,7 (67,5 / 72,5 % gegen 67,0 %) liegt innerhalb von zwei
  Standardfehlern.

## Verifikation

- **Kern:** Konkordanz, Veto, Überordnung, Klassen, Zyklen, Kern und Copeland-Zahl gegen Handrechnung auf den Mini-Instanzen (`tests/conftest.py`: Dreierzyklus mit w = 1/3 und
  Konkordanz 2/3; vier Alternativen mit w = (0,5; 0,3; 0,2), Konkordanz 0,7, Veto 0,3, Kern {A} gegen {A, B}).
- **Orakel** (`tests/test_oracle_electre.py`): Überordnung gegen eine Definitionsschleife; Zyklen gegen Komponenten über Erreichbarkeit (Warshall); **Kern gegen die eindeutige
  unabhängige und absorbierende Menge der Kondensation** (Brute-Force über alle Teilmengen); **c\* = 1 ist die schwache Dominanz**; Monotonie in c\* und Veto; **keine Dreierzyklen
  über 2/3, viele bei 0,5**; Kern ⊆ Front ohne Zyklen, Gegenprobe mit Zyklen.
- **Generator** (`tests/test_generator.py`): SplitMix64 gegen unabhängige Python-Fassung, Randverteilung, Spearman-Korrelation der Copula, Richtungen und Bereiche.
- **Aussagen:** jede Zahl dieser README in `tests/test_claims.py`; Presets in `test_presets.py` (jedes zeigt seinen Effekt).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), alle Presets, jeder Entscheider, jede Schwelle, jedes Veto, Permalink, PDF und Experimente
  erst nach Klick, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `entscheidungsmatrix_generator.py` | Generator der Linie (wortgleich in allen Repos der Linie, kein Präfix): SplitMix64, Copula, Rohwerte, Entscheider |
| `elc_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `elc_outranking.py` | Konkordanz, Veto, Überordnung, Klassen, Zyklen, Kern, Copeland-Zahl |
| `elc_score.py` | Min-Max, Summe, Nutzen des Entscheiders (auch mit Untergrenze), Dominanz, Kendalls τ-b |
| `elc_evaluation.py` | Fall der App, vier Experimente, Verdikte |
| `elc_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `elc_visualization.py` | Relationsmatrix, Graph, Tabellen und Plotly-Abbildungen (Achsen gesperrt) |
| `elc_pdf_export.py` | PDF-Export des aktuellen Falls (fpdf2, erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt; `scipy` für die starken Zusammenhangskomponenten) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Mini-Instanzen, Überordnung, Score, Orakel, Generator, Auswertung, Aussagen, Presets, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **ELECTRE III und die Destillation:** Kredibilitätsmatrix und Präferenzschwellen sind ein eigenes Verfahren; die Nettoflüsse als Rangfolge stehen in einem eigenen Stück.
- **Eine Rangfolge als Ergebnis:** Die Copeland-Zahl ist ein Notbehelf, kein Verfahren.
- **Absolute Untergrenzen als Profile:** Zuordnung gegen feste Profile hat ein eigenes Stück.
- **Rangumkehr bei hinzugefügter Alternative:** nicht gemessen; sie hat ein eigenes Stück.
- **Unsichere Gewichte und Schwellen als Bereich:** Experiment 3 zeigt nur die Änderung des Kerns bei Schwellenwechsel; die Verteilung über den Gewichtsraum steht beim SMAA.
- **Mehrere Entscheider:** Der Entscheider ist ein einzelnes, verborgenes Orakel.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas, SciPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
