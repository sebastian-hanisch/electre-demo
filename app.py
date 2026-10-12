"""ELECTRE: ein schlechter Wert wird nicht ausgeglichen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechstes Stück der Linie Mehrkriterien-Ranking (Kind der gewichteten Summe). Dieselbe neutrale Entscheidungsmatrix; ein simulierter Entscheider kennt seine Gewichte und seinen
Nutzen. ELECTRE I vergleicht je Paar (Konkordanz, Veto) und liefert eine Überordnungsrelation mit Kern, keine Rangfolge. Kern: elc_outranking.py, elc_score.py;
Generator: entscheidungsmatrix_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import math

import streamlit as st

import elc_constants as C
from elc_evaluation import (
    Settings,
    analyse,
    de,
    experiment_accuracy,
    experiment_copeland,
    experiment_kernel,
    experiment_structure,
    experiment_thresholds,
    experiment_veto,
    verdict_accuracy,
    verdict_kernel,
    verdict_structure,
    verdict_thresholds,
    verdict_veto,
    veto_label,
)
from elc_pdf_export import build_pdf
from elc_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from elc_visualization import (
    accuracy_table,
    alt_names,
    build_graph,
    build_kernel,
    build_method_table,
    build_relation_matrix,
    build_structure,
    build_table,
    build_veto,
    copeland_table,
    kernel_table,
    pair_details,
    structure_table,
    style_pair,
    threshold_table,
    veto_table,
)

st.set_page_config(page_title="ELECTRE – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Die vier Experimente rechnen …")
def _experiments():
    return {
        "structure": experiment_structure(),
        "kernel": experiment_kernel(),
        "thresholds": experiment_thresholds(),
        "copeland": experiment_copeland(),
        "veto": experiment_veto(),
        "accuracy": experiment_accuracy(),
    }


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🧭 ELECTRE: ein schlechter Wert wird nicht ausgeglichen")
st.markdown(
    """
**ELECTRE antwortet auf „was wird nicht überboten?“, nicht auf „was ist das Beste?“.** Dies ist das sechste Stück der Linie **Mehrkriterien-Ranking**: Die gewichtete Summe lässt einen
sehr guten Wert einen sehr schlechten ausgleichen. ELECTRE vergleicht je **Paar** von Alternativen: a **übertrifft** b, wenn Kriterien mit zusammen mindestens dem Anteil c* des
Gewichts für a sprechen (**Konkordanz**) und b in keinem Kriterium so weit vorn liegt, dass es ein **Veto** auslöst. Das Ergebnis ist eine Relation mit einem **Kern** der nicht
übertroffenen Alternativen, kein Punktwert; sie kann **Zyklen** und **Unvergleichbarkeit** enthalten. Der Entscheider ist simuliert und kennt seinen Nutzen.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert ELECTRE", expanded=True):
    st.markdown(
        """
1. **Konkordanz** c(a, b): die Summe der Gewichte der Kriterien, in denen a mindestens so gut ist wie b (auf Min-Max-normierten Werten).
2. **Veto** v: b ist in irgendeinem Kriterium um mehr als v (Anteil der Spannweite) besser als a; dann darf a b nicht übertreffen, auch bei hoher Konkordanz.
3. **Überordnung** a S b: c(a, b) ≥ c* und kein Veto. Daraus entstehen drei Klassen von Paaren: **strikt** (a S b, nicht b S a), **beidseitig** und **unvergleichbar** (keines).
4. **Kern der Überordnung:** Alternativen, die keine andere außerhalb ihres Zyklus übertrifft, dazu iterativ, was die übrigen nicht übertreffen. Er muss nicht klein sein und
   nicht den Besten enthalten.
5. **Keine Rangfolge:** Die Relation kann im Kreis laufen (a übertrifft b, b übertrifft c, c übertrifft a). Als Notbehelf zeigt die Demo die **Copeland-Zahl** (übertroffene
   minus übertreffende Alternativen).
6. **Gemessen wird** am simulierten Entscheider: liegt der Beste im Kern, stimmen die strikten Präferenzen, was macht das Veto an einer Mindestanforderung? Die Entscheider sind
   bis auf einen kompensatorisch, also nicht die Welt, für die ELECTRE gebaut wurde.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Matrix**")
    n = st.slider("Alternativen n", *bounds("n_slider"), key="n_slider", help=f"Wie viele Alternativen zur Wahl stehen; der Graph wird bis {C.GRAPH_MAX_N} gezeichnet.")
    m = st.slider("Kriterien m", *bounds("m_slider"), key="m_slider", help="Nach wie vielen Kriterien jede Alternative gemessen ist.")
    delta = st.selectbox("Streuung des ersten Kriteriums", C.DELTA_CHOICES, key="delta_select", format_func=lambda d: "unverändert" if d == 1.0 else f"{de(100 * d, 0)} % der Streuung",
                         help="Wie stark sich die Alternativen im ersten Kriterium unterscheiden; der Entscheider selbst ändert sich dadurch nicht.")
    decider = st.selectbox("Entscheider (simuliert)", C.DECIDERS, key="decider_select", format_func=lambda k: C.DECIDER_LABELS[k],
                           help="Wie der verborgene Entscheider bewertet. Nur der Entscheider mit Untergrenze ist nicht kompensatorisch: Kein Kriterium darf unter 0,1 liegen.")
    st.markdown("**Überordnung**")
    cstar = st.selectbox("Konkordanzschwelle c*", C.CSTAR_CHOICES, key="cstar_select", format_func=lambda v: de(v, 1),
                         help="Mindestanteil des Gewichts, der für „a übertrifft b“ sprechen muss. Bei 0,5 ist fast jedes Paar entschieden und die Relation voller Zyklen; bei 0,9 kaum eines.")
    veto = st.selectbox("Veto", C.VETO_CHOICES, key="veto_select", format_func=lambda v: "kein Veto" if math.isinf(v) else f"{de(v, 1)} der Spannweite",
                        help="b ist in einem Kriterium um mehr als diesen Anteil der Spannweite besser: dann darf a b nicht übertreffen. Das Veto ist RELATIV zu den anderen "
                             "Alternativen, keine absolute Untergrenze.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Matrix würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "m_slider": int(m), "cstar_select": float(cstar), "veto_select": float(veto), "decider_select": decider,
                   "delta_select": float(delta), "seed_input": int(seed)})

settings = Settings(int(n), int(m), float(cstar), float(veto), decider, float(delta), int(seed))
a = _analysis(settings)
names = alt_names(settings.n)
pairs = settings.n * (settings.n - 1) / 2
kern_names = ", ".join(names[i] for i in sorted(a.kernel))

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Kern der Überordnung", f"{len(a.kernel)} von {settings.n}", delta=kern_names, delta_color="off",
          help="Alternativen, die keine andere (außerhalb ihres Zyklus) übertrifft, dazu iterativ die nicht übertroffenen der übrigen. Nichtdominierte Alternativen insgesamt: "
               f"{int(a.front.sum())}.")
c2.metric("Zyklen", f"{len(a.cycles)}", delta=("; ".join("{" + ", ".join(names[i] for i in c) + "}" for c in a.cycles) if a.cycles else "keine"), delta_color="off",
          help="Gruppen von Alternativen, die einander im Kreis übertreffen. Alle Mitglieder eines Zyklus bleiben im Kern, auch eine dominierte.")
c3.metric("Strikt entschiedene Paare", f"{de(100 * a.P.sum() / pairs, 0)} %", delta=f"unvergleichbar: {de(100 * a.R.sum() / 2 / pairs, 0)} %", delta_color="off",
          help="Anteil der ungeordneten Paare, bei denen genau eine Alternative die andere übertrifft; der Rest ist beidseitig (selten) oder unvergleichbar.")
c4.metric("Bester im Kern", "ja" if a.best_in_kernel else "nein", delta=f"Platz 1 der Summe: {names[a.results['summe'].top1]} ({'im Kern' if a.summe_in_kernel else 'nicht im Kern'})",
          delta_color="off", help="Der Beste ist die Alternative mit dem höchsten Nutzen des simulierten Entscheiders.")

left, right = st.columns(2)
with left:
    st.plotly_chart(build_relation_matrix(a), width="stretch", key="relation_chart")
with right:
    if settings.n <= C.GRAPH_MAX_N:
        st.plotly_chart(build_graph(a), width="stretch", key="graph_chart")
    else:
        st.info(f"Der Graph wird bis {C.GRAPH_MAX_N} Alternativen gezeichnet; die Matrix links zeigt dieselbe Relation.")

with st.expander("Ein Paar im Einzelnen: Konkordanz und Veto"):
    pa, pb = st.columns(2)
    ia = pa.selectbox("Alternative a", list(range(settings.n)), index=0, format_func=lambda i: names[i], key="pair_a")
    ib = pb.selectbox("Alternative b", list(range(settings.n)), index=min(1, settings.n - 1), format_func=lambda i: names[i], key="pair_b")
    if ia == ib:
        st.info("Bitte zwei verschiedene Alternativen wählen.")
    else:
        st.dataframe(style_pair(pair_details(a, int(ia), int(ib))), hide_index=True, width="stretch")
        st.caption(f"Konkordanz c({names[ia]}, {names[ib]}) = {de(float(a.conc[ia, ib]), 2)}, c({names[ib]}, {names[ia]}) = {de(float(a.conc[ib, ia]), 2)}; Schwelle c* = {de(settings.cstar, 1)}, "
                   f"Veto: {veto_label(settings.veto)}. {names[ia]} übertrifft {names[ib]}: {'ja' if a.S[ia, ib] else 'nein'}"
                   f"{' (Veto)' if a.veto[ia, ib] and a.conc[ia, ib] >= settings.cstar else ''}; {names[ib]} übertrifft {names[ia]}: {'ja' if a.S[ib, ia] else 'nein'}"
                   f"{' (Veto)' if a.veto[ib, ia] and a.conc[ib, ia] >= settings.cstar else ''}.")

st.caption("Matrix: → a übertrifft b strikt, ↔ beidseitig, leer unvergleichbar; * = Kern. Graph: grün = Kern, roter Rand und rote Pfeile = Zyklus, Stern = bester nach Nutzen.")
tab_t, tab_m = st.tabs(["Alle Alternativen", "Summe und Copeland-Zahl"])
with tab_t:
    st.dataframe(build_table(a), hide_index=True, width="stretch")
    if settings.decider == "untergrenze":
        st.caption("Zulässig sind Alternativen mit einer Güte von mindestens 0,1 in jedem Kriterium. Unzulässige haben einen Nutzen unter allen zulässigen.")
with tab_m:
    st.dataframe(build_method_table(a), hide_index=True, width="stretch")
    st.caption("Dieselbe Matrix, derselbe Entscheider. Die Copeland-Zahl ist ein Notbehelf, kein Verfahren von ELECTRE; Gleichstände um Platz 1 sind häufig (siehe Experiment 3).")

if st.button("📄 PDF des aktuellen Falls erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="electre.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ist die Antwort eine Rangfolge, und hängt sie an der Schwelle?")
if a.cycles:
    st.warning(f"Die Überordnung enthält {'einen Zyklus' if len(a.cycles) == 1 else str(len(a.cycles)) + ' Zyklen'} "
               f"({'; '.join('{' + ', '.join(names[i] for i in c) + '}' for c in a.cycles)}): Die Mitglieder übertreffen einander im Kreis, alle bleiben im Kern, und es gibt keine Rangfolge. "
               "Die Summe ordnet trotzdem; sie kompensiert, was ELECTRE nicht ausgleicht.")
elif a.R.sum() > 0:
    st.info(f"Die Überordnung hat keinen Zyklus, aber {de(100 * a.R.sum() / 2 / pairs, 0)} % der Paare sind unvergleichbar: Der Kern ({kern_names}) ist eine Menge, kein Sieger.")
else:
    st.success("Die Überordnung ist vollständig und ohne Zyklus: Alle Paare sind strikt entschieden, der Kern ist ein einzelner Sieger oder die Relation ist eine Ordnung.")
if settings.decider == "untergrenze":
    unacceptable = [names[i] for i in sorted(a.kernel) if not a.case.acceptable[i]]
    if unacceptable:
        st.warning(f"Der Kern enthält unzulässige Alternativen ({', '.join(unacceptable)}): Das Veto vergleicht nur mit den anderen Alternativen, eine absolute Untergrenze setzt es nicht.")
    else:
        st.success("Der Kern enthält nur zulässige Alternativen.")
st.caption("Ändern Sie c* oder das Veto: Der Kern ändert sich oft, ohne dass sich an den Daten etwas geändert hat (Experiment 3).")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die vier Experimente")
st.caption("200 feste Matrizen (SWEEP_SEEDS), n = 20, Min-Max, wahre Gewichte, unabhängig von den Reglern. Die Experimente rechnen zusammen rund 10 Sekunden und starten erst auf Klick.")
if st.button("▶ Experimente rechnen", key="exp_start"):
    st.session_state["exp_rows"] = _experiments()
exp = st.session_state.get("exp_rows")
if exp:
    st.markdown("### Experiment 1: Wie oft entstehen Zyklen?")
    st.caption("Anteil der Matrizen mit Zyklus in der Überordnung, 5 Kriterien; darunter die vollständige Tabelle für 3, 5 und 8 Kriterien.")
    st.plotly_chart(build_structure(exp["structure"], 5), width="stretch", key="structure_chart")
    verdict(verdict_structure(exp["structure"]))
    st.dataframe(structure_table(exp["structure"]), hide_index=True, width="stretch")

    st.markdown("### Experiment 2: Der Kern und der Beste")
    st.caption("5 Kriterien, kein Veto. „Bester im Kern“: Der Kern enthält die Alternative mit dem höchsten Nutzen des Entscheiders.")
    st.plotly_chart(build_kernel(exp["kernel"]), width="stretch", key="kernel_chart")
    verdict(verdict_kernel(exp["kernel"]))
    st.dataframe(kernel_table(exp["kernel"]), hide_index=True, width="stretch")

    st.markdown("### Experiment 3: Die Schwellen")
    st.caption("Wie oft ändert sich der Kern, wenn c* oder das Veto wandert (5 Kriterien, Güte/linear), und wie gut ist die Copeland-Zahl als Rangfolge?")
    st.dataframe(threshold_table(exp["thresholds"]), hide_index=True, width="stretch")
    verdict(verdict_thresholds(exp["thresholds"], exp["copeland"]))
    st.dataframe(copeland_table(exp["copeland"]), hide_index=True, width="stretch")
    st.caption("Platz 1 getroffen. Standardfehler 1,3 bis 3,5 Punkte: Unterschiede unter etwa 6 Punkten sind nicht belastbar.")

    st.markdown("### Experiment 4: Veto ist keine Untergrenze")
    st.caption("Entscheider mit Untergrenze (kein Kriterium unter 0,1 Güte; im Mittel sind 41 % der Alternativen unzulässig), 5 Kriterien, c* = 0,6.")
    st.plotly_chart(build_veto(exp["veto"]), width="stretch", key="veto_chart")
    verdict(verdict_veto(exp["veto"]))
    st.dataframe(veto_table(exp["veto"]), hide_index=True, width="stretch")
    st.markdown("**Wenn ELECTRE a strikt über b stellt:** Anteil der Paare, die der Entscheider genauso ordnet, gegen die Summe auf denselben Paaren (5 Kriterien).")
    st.dataframe(accuracy_table(exp["accuracy"]), hide_index=True, width="stretch")
    verdict(verdict_accuracy(exp["accuracy"]))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Entscheider kompensieren** (linearer, konkaver oder Cobb-Douglas-Nutzen; nur ein Entscheider mit Untergrenze nicht) | ELECTRE ist für Entscheider gebaut, die ein schlechtes Kriterium nicht ausgleichen wollen; gegen kompensatorische Entscheider verliert es an Genauigkeit und Platz 1. Das sagt nichts über echte Entscheider. | – |
| **ELECTRE I, crisp** | Kredibilität, Indifferenz- und Präferenzschwellen und die Destillation (ELECTRE III) sind nicht umgesetzt; die Schwellen sind einheitliche Anteile der Spannweite. | PROMETHEE |
| **Die Gewichte sind bekannt** | In der Demo sind es die wahren Gewichte des Entscheiders; wie man sie erhebt, steht in den Vorgängern. | Gewichte erheben, AHP |
| **Eine Rangfolge ist gewünscht** | Die Relation ordnet nicht vollständig; die Copeland-Zahl ist ein Notbehelf, und der Nettofluss ordnet immer. | PROMETHEE |
| **Das Veto ersetzt die Untergrenze** | Es ist relativ zu den anderen Alternativen: eine schlechte Alternative bleibt im Kern, wenn nichts sie übertreffen darf. Absolute Anforderungen sind Profile. | Zuordnung statt Rangfolge |
| **Die Alternativen sind gegeben** | Der Kern hängt von den anderen Alternativen ab; wie er sich beim Hinzufügen einer Alternative ändert, ist hier nicht gemessen. | Rangumkehr |
| **Eine Schwelle, ein Gewichtsvektor** | Wie sich der Kern über alle verträglichen Schwellen und Gewichte verteilt, zeigt nur Experiment 3 grob. | Unsichere Gewichte |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Normierung.** Min-Max: $n_{ij} = (x_{ij} - \min_k x_{kj}) / (\max_k x_{kj} - \min_k x_{kj})$, höher ist besser; Gewichte $w_j \ge 0$, $\sum_j w_j = 1$.

**Konkordanz.** $c(a, b) = \sum_{j:\, n_{aj} \ge n_{bj}} w_j$; Gleichstände zählen für beide.

**Veto.** $a$ darf $b$ nicht übertreffen, wenn $n_{bj} - n_{aj} > v$ für ein Kriterium $j$.

**Überordnung.** $a\, S\, b \iff c(a, b) \ge c^{*}$ und kein Veto. Klassen: strikt $a\, P\, b \iff a\, S\, b \wedge \neg\, b\, S\, a$; beidseitig $a\, S\, b \wedge b\, S\, a$;
unvergleichbar $\neg\, a\, S\, b \wedge \neg\, b\, S\, a$.

**Zyklen.** Die starken Zusammenhangskomponenten des gerichteten Graphen von $S$; eine Komponente mit mehr als einem Knoten ist ein Zyklus. Ein Dreierzyklus ist für $c^{*} > 2/3$
unmöglich (jedes Kriterium stützt höchstens zwei der drei Kanten); ein Zyklus der Länge $k$ braucht $c^{*} \le (k - 1)/k$.

**Kern.** Wiederhole: nimm die Komponenten der Kondensation ohne eingehende Kante aus den verbleibenden Alternativen in den Kern, entferne sie und alles, was sie übertreffen.
Für einen azyklischen Graphen ist das der eindeutige Kern im Sinn der Graphentheorie (unabhängig und absorbierend).

**Dominanz als Sonderfall.** Für $c^{*} = 1$ und ohne Veto ist $a\, S\, b$ genau die schwache Dominanz.

**Copeland-Zahl.** $\mathrm{Cop}(a) = |\{b : a\, S\, b\}| - |\{b : b\, S\, a\}|$.

**Literatur** (nur aus Zusammenfassungen, nicht im Original geprüft): B. Roy, *Classement et choix en présence de points de vue multiples (la méthode ELECTRE)*, Revue française
d'informatique et de recherche opérationnelle, Série verte, 2(V1), 1968, S. 57–75 (Konkordanz und Diskordanz, Überordnungsrelation); J. Figueira, V. Mousseau, B. Roy, *Electre
Methods*, in: Multiple Criteria Decision Analysis: State of the Art Surveys, Springer, 2005, S. 133–162 (die Familie nach Problemstellungen, Schwellen und Veto).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
