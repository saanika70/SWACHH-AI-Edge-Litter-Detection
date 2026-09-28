from swachh_ai.simulation import run_scenario, scenario_bystander, scenario_drop, scenario_throw


def kinds(events):
    return [e.kind for e in events]


def test_throw_detected_once():
    events = run_scenario(scenario_throw())
    assert kinds(events) == ["throw"]
    assert 2.0 < events[0].timestamp < 3.5
    assert events[0].label == "plastic_bottle"


def test_drop_detected_once():
    events = run_scenario(scenario_drop())
    assert kinds(events) == ["drop"]


def test_bystander_next_to_existing_litter_raises_nothing():
    assert run_scenario(scenario_bystander()) == []


def test_dustbin_exclusion_zone_suppresses_drop():
    cfg = {"exclusion_zones": [[0.6, 0.8, 1.0, 1.0]]}  # bottle ends at ~(500, 425)
    assert run_scenario(scenario_drop(), cfg) == []


def test_robust_to_noise_seeds():
    for seed in range(5):
        assert kinds(run_scenario(scenario_throw(seed=seed))) == ["throw"]
        assert kinds(run_scenario(scenario_bystander(seed=seed))) == []
