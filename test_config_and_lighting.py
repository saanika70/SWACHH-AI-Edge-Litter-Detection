import textwrap

from swachh_ai.config import load_config
from swachh_ai.hardware import LightingController, MockHardware, Mode


def test_env_expansion_with_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("MQTT_HOST", "broker.test")
    monkeypatch.delenv("MQTT_PORT", raising=False)
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        iot:
          mqtt:
            host: ${MQTT_HOST:-localhost}
            port: ${MQTT_PORT:-1883}
            user: ${NOPE_UNSET:-}
    """))
    cfg = load_config(str(p))
    assert cfg["iot"]["mqtt"] == {"host": "broker.test", "port": "1883", "user": ""}


def test_shipped_config_parses():
    cfg = load_config("config/config.yaml")
    assert cfg["models"]["litter"]["imgsz"] == 640
    assert cfg["hardware"]["pins"]["pir"] == 17


def test_lighting_levels_and_strobe():
    hw = MockHardware()
    light = LightingController(hw, {"fade_per_s": 1e9})  # effectively instant
    assert light.target(Mode.IDLE, dark=False) == 0.0
    assert light.target(Mode.ACTIVE, dark=True) > light.target(Mode.IDLE, dark=True)
    light.update(Mode.WATCH, dark=True)
    assert hw.light_level == 1.0
    light.alert()
    assert ("strobe", 6.0) in hw.events
