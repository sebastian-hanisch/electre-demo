"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 180
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    import elc_constants as C

    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.slider(key="n_slider").value == preset["n"] and at.slider(key="m_slider").value == preset["m"]
        for key, field in (("cstar_select", "cstar"), ("veto_select", "veto"), ("decider_select", "decider"), ("delta_select", "delta")):
            assert at.selectbox(key=key).value == preset[field], (name, key)
        assert at.number_input(key="seed_input").value == preset["seed"]


def test_every_decider_threshold_and_veto_renders_without_exception():
    import elc_constants as C

    for dec in C.DECIDERS:
        at = _fresh_app()
        at.selectbox(key="decider_select").select(dec).run(timeout=TIMEOUT)
        assert not at.exception, (dec, [str(e) for e in at.exception])
    for cstar in C.CSTAR_CHOICES:
        at = _fresh_app()
        at.selectbox(key="cstar_select").select(cstar).run(timeout=TIMEOUT)
        assert not at.exception, (cstar, [str(e) for e in at.exception])
    for v in C.VETO_CHOICES:
        at = _fresh_app()
        at.selectbox(key="veto_select").select(v).run(timeout=TIMEOUT)
        assert not at.exception, (v, [str(e) for e in at.exception])


def test_extreme_settings_render_without_exception():
    at = _fresh_app()
    at.slider(key="m_slider").set_value(8).run(timeout=TIMEOUT)
    at.slider(key="n_slider").set_value(40).run(timeout=TIMEOUT)         # über GRAPH_MAX_N: nur die Matrix
    at.selectbox(key="cstar_select").select(0.5).run(timeout=TIMEOUT)
    at.selectbox(key="veto_select").select(0.3).run(timeout=TIMEOUT)
    at.selectbox(key="delta_select").select(0.05).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_three_criteria_and_five_alternatives_render_without_exception():
    at = _fresh_app()
    at.slider(key="m_slider").set_value(3).run(timeout=TIMEOUT)
    at.slider(key="n_slider").set_value(5).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_experiments_run_only_after_the_click():
    at = _fresh_app()
    assert "exp_rows" not in at.session_state
    at.button(key="exp_start").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert set(at.session_state["exp_rows"]) == {"structure", "kernel", "thresholds", "copeland", "veto", "accuracy"}


def test_permalink_values_are_clipped_and_unknown_choices_ignored():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "9999"
    at.query_params["m"] = "1"
    at.query_params["cs"] = "0.55"
    at.query_params["dc"] = "orakel"
    at.query_params["vt"] = "inf"
    at.query_params["dl"] = "0.2"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="n_slider").value == 40 and at.slider(key="m_slider").value == 3
    assert at.selectbox(key="cstar_select").value == 0.6 and at.selectbox(key="decider_select").value == "guete_linear"
    assert at.selectbox(key="veto_select").value == float("inf") and at.selectbox(key="delta_select").value == 0.2
