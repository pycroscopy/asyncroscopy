import json
import types
import subprocess
import sys

import pytest
import tango

from asyncroscopy.instruments.electron_microscope.auto_script import AutoScriptMicroscope
from asyncroscopy.instruments.electron_microscope.hardware.aperture import ApertureInfo


RESPONSE_FIELDS = {
    "ok",
    "action",
    "mechanism",
    "aperture",
    "insertion_state",
    "autoscript_available",
    "error",
}


@pytest.fixture
def simulated_autoscript_proxy(auto_script_proxy: tango.DeviceProxy) -> tango.DeviceProxy:
    auto_script_proxy.Init()
    return auto_script_proxy


def test_auto_script_microscope_imports_when_autoscript_is_blocked() -> None:
    script = r'''
import builtins

original_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name.startswith("autoscript_tem_microscope_client"):
        raise ImportError("AutoScript deliberately unavailable")
    return original_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
from asyncroscopy.instruments.electron_microscope import auto_script
assert auto_script._AUTOSCRIPT_AVAILABLE is False
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_testing_mode_aperture_list_uses_simulation(
    simulated_autoscript_proxy: tango.DeviceProxy,
) -> None:
    response = json.loads(simulated_autoscript_proxy.aperture_list("{}"))

    assert RESPONSE_FIELDS.issubset(response)
    assert response["ok"] is True
    assert response["action"] == "list"
    assert response["mechanism"] == "all"
    assert response["autoscript_available"] is False
    assert response["error"] is None
    assert response["mechanisms"] == [
        "condenser",
        "objective",
        "selected_area",
        "projector",
    ]


def test_testing_mode_gets_selected_aperture(
    simulated_autoscript_proxy: tango.DeviceProxy,
) -> None:
    request = json.dumps({"mechanism": "condenser"})
    response = json.loads(simulated_autoscript_proxy.aperture_get_selected(request))

    assert response["ok"] is True
    assert response["action"] == "get_selected"
    assert response["mechanism"] == "condenser"
    assert response["aperture"] == {
        "name": "50 um",
        "type": "Circular",
        "diameter_m": pytest.approx(50e-6),
    }
    assert response["insertion_state"] == "Inserted"
    assert response["autoscript_available"] is False


def test_aperture_command_rejects_invalid_json(
    simulated_autoscript_proxy: tango.DeviceProxy,
) -> None:
    response = json.loads(simulated_autoscript_proxy.aperture_get_selected("not-json"))

    assert response["ok"] is False
    assert response["action"] == "get_selected"
    assert response["mechanism"] == ""
    assert response["error"] == "Invalid JSON request."


def test_aperture_command_requires_mechanism(
    simulated_autoscript_proxy: tango.DeviceProxy,
) -> None:
    response = json.loads(simulated_autoscript_proxy.aperture_insert("{}"))

    assert response["ok"] is False
    assert response["error"] == (
        "Aperture request requires a non-empty string 'mechanism'."
    )


def test_aperture_select_requires_name(
    simulated_autoscript_proxy: tango.DeviceProxy,
) -> None:
    request = json.dumps({"mechanism": "objective"})
    response = json.loads(simulated_autoscript_proxy.aperture_select(request))

    assert response["ok"] is False
    assert response["mechanism"] == "objective"
    assert response["error"] == "Aperture request requires a non-empty string 'name'."


def test_testing_mode_reads_enabled_status(
    simulated_autoscript_proxy: tango.DeviceProxy,
) -> None:
    request = json.dumps({"mechanism": "condenser"})
    response = json.loads(simulated_autoscript_proxy.aperture_is_enabled(request))

    assert response["ok"] is True
    assert response["action"] == "is_enabled"
    assert response["mechanism"] == "condenser"
    assert response["enabled"] is True
    assert response["autoscript_available"] is False
    assert response["error"] is None


def test_testing_mode_reads_retractable_status(
    simulated_autoscript_proxy: tango.DeviceProxy,
) -> None:
    request = json.dumps({"mechanism": "objective"})
    response = json.loads(simulated_autoscript_proxy.aperture_is_retractable(request))

    assert response["ok"] is True
    assert response["action"] == "is_retractable"
    assert response["mechanism"] == "objective"
    assert response["retractable"] is True
    assert response["autoscript_available"] is False
    assert response["error"] is None


def test_aperture_status_rejects_unknown_mechanism(
    simulated_autoscript_proxy: tango.DeviceProxy,
) -> None:
    request = json.dumps({"mechanism": "not_a_mechanism"})
    response = json.loads(simulated_autoscript_proxy.aperture_is_enabled(request))

    assert response["ok"] is False
    assert response["action"] == "is_enabled"
    assert response["mechanism"] == "not_a_mechanism"
    assert "Unknown aperture mechanism" in response["error"]


def test_testing_mode_select_returns_selected_aperture(
    simulated_autoscript_proxy: tango.DeviceProxy,
) -> None:
    request = json.dumps({"mechanism": "objective", "name": "100 um"})
    response = json.loads(simulated_autoscript_proxy.aperture_select(request))

    assert response["ok"] is True
    assert response["action"] == "select"
    assert response["mechanism"] == "objective"
    assert response["aperture"] == {
        "name": "100 um",
        "type": "Circular",
        "diameter_m": pytest.approx(100e-6),
    }
    assert response["insertion_state"] == "Inserted"
    assert response["autoscript_available"] is False
    assert response["error"] is None


def test_testing_mode_insert_and_retract_are_explicit(
    simulated_autoscript_proxy: tango.DeviceProxy,
) -> None:
    request = json.dumps({"mechanism": "selected_area"})

    inserted = json.loads(simulated_autoscript_proxy.aperture_insert(request))
    retracted = json.loads(simulated_autoscript_proxy.aperture_retract(request))

    assert inserted["ok"] is True
    assert inserted["action"] == "insert"
    assert inserted["insertion_state"] == "Inserted"
    assert inserted["aperture"] is not None
    assert retracted["ok"] is True
    assert retracted["action"] == "retract"
    assert retracted["insertion_state"] == "Retracted"
    assert retracted["aperture"] is None

class RecordingApertureBackend:
    def __init__(self, *, enabled: bool = True, retractable: bool = True) -> None:
        self.enabled = enabled
        self.retractable = retractable
        self.select_calls = 0
        self.insert_calls = 0
        self.retract_calls = 0
        self.enable_calls = 0
        self.disable_calls = 0
        self.insertion_state = "Retracted"

    def is_enabled(self, mechanism: str) -> bool:
        return self.enabled

    def is_retractable(self, mechanism: str) -> bool:
        return self.retractable

    def _aperture(self, mechanism: str, name: str = "50 um") -> ApertureInfo:
        return ApertureInfo(
            mechanism=mechanism,
            name=name,
            aperture_type="Circular",
            diameter_m=50e-6,
            inserted=True,
            selected=True,
        )

    def enable(self, mechanism: str) -> bool:
        self.enable_calls += 1
        self.enabled = True
        return self.enabled

    def disable(self, mechanism: str) -> bool:
        self.disable_calls += 1
        self.enabled = False
        return self.enabled

    def list_mechanisms(self) -> list[str]:
        return ["C2"]

    def list_apertures(self, mechanism: str) -> list[ApertureInfo]:
        return [self._aperture(mechanism)]

    def select_aperture(self, mechanism: str, name: str) -> ApertureInfo:
        self.select_calls += 1
        return self._aperture(mechanism, name)

    def insert_mechanism(self, mechanism: str) -> str:
        self.insert_calls += 1
        return "Inserted"

    def retract_mechanism(self, mechanism: str) -> str:
        self.retract_calls += 1
        return "Retracted"

    def get_selected_aperture(self, mechanism: str) -> ApertureInfo:
        return self._aperture(mechanism)

    def get_insertion_state(self, mechanism: str) -> str:
        return self.insertion_state


class MissingEnableBackend(RecordingApertureBackend):
    enable = None


class MissingDisableBackend(RecordingApertureBackend):
    disable = None


class MissingEnabledBackend(RecordingApertureBackend):
    is_enabled = None


class MissingRetractableBackend(RecordingApertureBackend):
    is_retractable = None


def make_direct_microscope(adapter) -> AutoScriptMicroscope:
    microscope = AutoScriptMicroscope.__new__(AutoScriptMicroscope)
    microscope._aperture_adapter = adapter
    microscope._aperture_autoscript_available = False
    microscope.error_stream = lambda message: None
    microscope.info_stream = lambda message: None
    microscope._acquisition_active = False
    return microscope



def test_acquisition_command_marks_acquisition_active_until_complete() -> None:
    microscope = make_direct_microscope(RecordingApertureBackend())
    microscope._detector_proxies = {
        "scan": types.SimpleNamespace(
            imsize=16,
            dwell_time=1e-6,
            scan_region=[0.0, 0.0, 1.0, 1.0],
            output_format=".h5",
        )
    }

    def fake_acquire(imsize, dwell_time, detector_list, scan_region, output_format):
        assert microscope._acquisition_active is True
        return "acquired-key"

    microscope._acquire_scanned_image = fake_acquire

    result = microscope.acquire_scanned_image(["haadf"])

    assert result == "acquired-key"
    assert microscope._acquisition_active is False


def test_aperture_mutations_are_blocked_during_active_acquisition() -> None:
    adapter = RecordingApertureBackend(enabled=True, retractable=True)
    microscope = make_direct_microscope(adapter)
    microscope._acquisition_active = True

    responses = [
        json.loads(
            microscope.aperture_select(json.dumps({"mechanism": "C2", "name": "70 um"}))
        ),
        json.loads(microscope.aperture_insert(json.dumps({"mechanism": "C2"}))),
        json.loads(microscope.aperture_retract(json.dumps({"mechanism": "C2"}))),
        json.loads(microscope.aperture_enable(json.dumps({"mechanism": "C2"}))),
        json.loads(microscope.aperture_disable(json.dumps({"mechanism": "C2"}))),
    ]

    assert all(response["ok"] is False for response in responses)
    assert all(
        "aperture mutation is blocked because acquisition is active"
        in response["error"].lower()
        for response in responses
    )
    assert adapter.select_calls == 0
    assert adapter.insert_calls == 0
    assert adapter.retract_calls == 0
    assert adapter.enable_calls == 0
    assert adapter.disable_calls == 0


def test_aperture_mutation_is_allowed_when_acquisition_is_idle() -> None:
    adapter = RecordingApertureBackend(enabled=True, retractable=True)
    microscope = make_direct_microscope(adapter)
    microscope._acquisition_active = False

    response = json.loads(
        microscope.aperture_select(json.dumps({"mechanism": "C2", "name": "70 um"}))
    )

    assert response["ok"] is True
    assert adapter.select_calls == 1


def test_read_only_aperture_commands_are_allowed_during_active_acquisition() -> None:
    adapter = RecordingApertureBackend(enabled=False, retractable=False)
    microscope = make_direct_microscope(adapter)
    microscope._acquisition_active = True

    listed = json.loads(microscope.aperture_list("{}"))
    selected = json.loads(microscope.aperture_get_selected(json.dumps({"mechanism": "C2"})))
    enabled = json.loads(microscope.aperture_is_enabled(json.dumps({"mechanism": "C2"})))
    retractable = json.loads(
        microscope.aperture_is_retractable(json.dumps({"mechanism": "C2"}))
    )

    assert listed["ok"] is True
    assert selected["ok"] is True
    assert enabled["ok"] is True
    assert enabled["enabled"] is False
    assert retractable["ok"] is True
    assert retractable["retractable"] is False

def test_select_is_rejected_when_mechanism_is_disabled() -> None:
    adapter = RecordingApertureBackend(enabled=False)
    microscope = make_direct_microscope(adapter)

    response = json.loads(
        microscope.aperture_select(json.dumps({"mechanism": "C2", "name": "50 um"}))
    )

    assert response["ok"] is False
    assert "disabled" in response["error"]
    assert adapter.select_calls == 0


def test_insert_is_rejected_when_mechanism_is_disabled() -> None:
    adapter = RecordingApertureBackend(enabled=False)
    microscope = make_direct_microscope(adapter)

    response = json.loads(microscope.aperture_insert(json.dumps({"mechanism": "C2"})))

    assert response["ok"] is False
    assert "disabled" in response["error"]
    assert adapter.insert_calls == 0


def test_retract_is_rejected_when_mechanism_is_disabled() -> None:
    adapter = RecordingApertureBackend(enabled=False)
    microscope = make_direct_microscope(adapter)

    response = json.loads(microscope.aperture_retract(json.dumps({"mechanism": "C2"})))

    assert response["ok"] is False
    assert "disabled" in response["error"]
    assert adapter.retract_calls == 0


def test_insert_is_rejected_when_mechanism_is_not_retractable() -> None:
    adapter = RecordingApertureBackend(enabled=True, retractable=False)
    microscope = make_direct_microscope(adapter)

    response = json.loads(microscope.aperture_insert(json.dumps({"mechanism": "C2"})))

    assert response["ok"] is False
    assert "not retractable" in response["error"]
    assert adapter.insert_calls == 0


def test_retract_is_rejected_when_mechanism_is_not_retractable() -> None:
    adapter = RecordingApertureBackend(enabled=True, retractable=False)
    microscope = make_direct_microscope(adapter)

    response = json.loads(microscope.aperture_retract(json.dumps({"mechanism": "C2"})))

    assert response["ok"] is False
    assert "not retractable" in response["error"]
    assert adapter.retract_calls == 0


def test_successful_mutations_still_call_backend() -> None:
    adapter = RecordingApertureBackend(enabled=True, retractable=True)
    microscope = make_direct_microscope(adapter)

    select = json.loads(
        microscope.aperture_select(json.dumps({"mechanism": "C2", "name": "70 um"}))
    )
    insert = json.loads(microscope.aperture_insert(json.dumps({"mechanism": "C2"})))
    retract = json.loads(microscope.aperture_retract(json.dumps({"mechanism": "C2"})))

    assert select["ok"] is True
    assert insert["ok"] is True
    assert retract["ok"] is True
    assert adapter.select_calls == 1
    assert adapter.insert_calls == 1
    assert adapter.retract_calls == 1


def test_read_only_status_commands_still_work_with_false_values() -> None:
    adapter = RecordingApertureBackend(enabled=False, retractable=False)
    microscope = make_direct_microscope(adapter)

    enabled = json.loads(microscope.aperture_is_enabled(json.dumps({"mechanism": "C2"})))
    retractable = json.loads(
        microscope.aperture_is_retractable(json.dumps({"mechanism": "C2"}))
    )

    assert enabled["ok"] is True
    assert enabled["enabled"] is False
    assert retractable["ok"] is True
    assert retractable["retractable"] is False


def test_select_reports_missing_enabled_status_before_backend_call() -> None:
    adapter = MissingEnabledBackend()
    microscope = make_direct_microscope(adapter)

    response = json.loads(
        microscope.aperture_select(json.dumps({"mechanism": "C2", "name": "50 um"}))
    )

    assert response["ok"] is False
    assert "unsupported or missing" in response["error"]
    assert "is_enabled" in response["error"]
    assert adapter.select_calls == 0


def test_insert_reports_missing_retractable_status_before_backend_call() -> None:
    adapter = MissingRetractableBackend()
    microscope = make_direct_microscope(adapter)

    response = json.loads(microscope.aperture_insert(json.dumps({"mechanism": "C2"})))

    assert response["ok"] is False
    assert "unsupported or missing" in response["error"]
    assert "is_retractable" in response["error"]
    assert adapter.insert_calls == 0

def test_enable_calls_backend_enable_and_updates_status() -> None:
    adapter = RecordingApertureBackend(enabled=False)
    microscope = make_direct_microscope(adapter)

    response = json.loads(microscope.aperture_enable(json.dumps({"mechanism": "C2"})))
    status = json.loads(microscope.aperture_is_enabled(json.dumps({"mechanism": "C2"})))

    assert response["ok"] is True
    assert response["enabled"] is True
    assert status["enabled"] is True
    assert adapter.enable_calls == 1


def test_disable_calls_backend_disable_and_updates_status() -> None:
    adapter = RecordingApertureBackend(enabled=True)
    adapter.insertion_state = "Retracted"
    microscope = make_direct_microscope(adapter)

    response = json.loads(microscope.aperture_disable(json.dumps({"mechanism": "C2"})))
    status = json.loads(microscope.aperture_is_enabled(json.dumps({"mechanism": "C2"})))

    assert response["ok"] is True
    assert response["enabled"] is False
    assert status["enabled"] is False
    assert adapter.disable_calls == 1


def test_disable_rejects_inserted_mechanism_before_backend_call() -> None:
    adapter = RecordingApertureBackend(enabled=True)
    adapter.insertion_state = "Inserted"
    microscope = make_direct_microscope(adapter)

    response = json.loads(microscope.aperture_disable(json.dumps({"mechanism": "C2"})))

    assert response["ok"] is False
    assert "not safe to disable" in response["error"]
    assert "Inserted" in response["error"]
    assert adapter.disable_calls == 0


def test_enable_reports_missing_backend_method() -> None:
    adapter = MissingEnableBackend(enabled=False)
    microscope = make_direct_microscope(adapter)

    response = json.loads(microscope.aperture_enable(json.dumps({"mechanism": "C2"})))

    assert response["ok"] is False
    assert "unsupported" in response["error"].lower()
    assert "enable" in response["error"]
    assert adapter.enable_calls == 0


def test_disable_reports_missing_backend_method() -> None:
    adapter = MissingDisableBackend(enabled=True)
    microscope = make_direct_microscope(adapter)

    response = json.loads(microscope.aperture_disable(json.dumps({"mechanism": "C2"})))

    assert response["ok"] is False
    assert "unsupported" in response["error"].lower()
    assert "disable" in response["error"]
    assert adapter.disable_calls == 0


def test_enable_unknown_mechanism_gives_clear_error() -> None:
    class UnknownMechanismBackend(RecordingApertureBackend):
        def enable(self, mechanism: str) -> bool:
            raise ValueError(
                f"Unknown aperture mechanism {mechanism!r}. Available mechanisms: C2."
            )

    adapter = UnknownMechanismBackend()
    microscope = make_direct_microscope(adapter)

    response = json.loads(
        microscope.aperture_enable(json.dumps({"mechanism": "not_a_mechanism"}))
    )

    assert response["ok"] is False
    assert "Unknown aperture mechanism" in response["error"]
