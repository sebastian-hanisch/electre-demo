"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt seinen Effekt."""

import math

import elc_constants as C
import elc_evaluation as ev
import elc_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.M_MIN <= p["m"] <= C.M_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert p["cstar"] in C.CSTAR_CHOICES and p["veto"] in C.VETO_CHOICES and p["decider"] in C.DECIDERS and p["delta"] in C.DELTA_CHOICES, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (c* = 0,6, kein Veto)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_labels_cover_every_choice():
    assert set(C.DECIDER_LABELS) == set(C.DECIDERS) == set(C.DECIDER_SHORT) == set(C.DECIDER_PARTS)
    assert C.DEFAULT_VETO in C.VETO_CHOICES and math.isinf(C.DEFAULT_VETO) and C.DEFAULT_CSTAR in C.CSTAR_CHOICES


def test_sweep_seeds_are_the_documented_fixed_set():
    assert len(C.SWEEP_SEEDS) == 200 and len(set(C.SWEEP_SEEDS)) == 200 and C.SWEEP_SEEDS[0] == 200000


def _analysis(name, **override):
    p = dict(C.PRESETS[name])
    p.update(override)
    return ev.analyse(ev.Settings(**p))


def test_every_preset_shows_the_effect_it_is_named_for():
    """Die Presets sind auf einen Seed abgestimmt, bei dem der Effekt sichtbar ist; ändert sich Generator oder Auswertung, soll das hier auffallen."""
    a = _analysis("Standardfall (c* = 0,6, kein Veto)")
    assert not a.cycles and 2 <= len(a.kernel) <= 4 and a.best_in_kernel and a.summe_in_kernel and a.results["summe"].hit

    a = _analysis("Ein Zyklus über drei Kriterien")
    assert len(a.cycles) == 1 and len(a.cycles[0]) == 3 and a.kernel == set(a.cycles[0]) and a.results["summe"].top1 in a.cycles[0]
    assert not _analysis("Ein Zyklus über drei Kriterien", cstar=0.7).cycles

    a = _analysis("Veto macht unvergleichbar")
    b = _analysis("Veto macht unvergleichbar", veto=math.inf)
    assert a.unacceptable_in_kernel and not b.unacceptable_in_kernel and len(a.kernel) > len(b.kernel) and a.best_in_kernel

    a = _analysis("Die Schwelle verschiebt den Kern")
    b = _analysis("Die Schwelle verschiebt den Kern", cstar=0.6)
    assert a.best_in_kernel and not b.best_in_kernel and a.kernel != b.kernel and not a.cycles
