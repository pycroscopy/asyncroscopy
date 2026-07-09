from types import SimpleNamespace

import pytest

from asyncroscopy.instruments.electron_microscope.adapters.autoscript_apertures import (
    AutoScriptApertureAdapter,
)


class FakeAperture:
    def __init__(self, name: str, aperture_type: str, diameter: float | None):
        self.name = name
        self.type = aperture_type
        self.diameter = diameter


class FakeMechanism:
    def __init__(self, *, is_enabled: bool = True, is_retractable: bool = True):
        self.apertures = [
            FakeAperture("30 um", "Circular", 30e-6),
            FakeAperture("50 um", "Circular", 50e-6),
            FakeAperture("Phase plate", "PhasePlate", None),
        ]
        self.aperture = self.apertures[0]
        self.insertion_state = "Inserted"
        self.is_enabled = is_enabled
        self.is_retractable = is_retractable
        self._position = SimpleNamespace(x=1e-6, y=-2e-6)
        self.enable_calls = 0
        self.disable_calls = 0

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        if isinstance(value, tuple):
            self._position = SimpleNamespace(x=value[0], y=value[1])
        else:
            self._position = value

    def enable(self) -> None:
        self.enable_calls += 1
        self.is_enabled = True

    def disable(self) -> None:
        self.disable_calls += 1
        self.is_enabled = False
    def insert(self) -> None:
        self.insertion_state = "Inserted"

    def retract(self) -> None:
        self.insertion_state = "Retracted"
        self.aperture = None


class FakeCollection:
    def __init__(self):
        self.mechanisms = {
            "C2": FakeMechanism(is_enabled=True, is_retractable=True),
            "Objective": FakeMechanism(is_enabled=False, is_retractable=False),
        }

    def get_available(self) -> list[str]:
        return list(self.mechanisms)

    def get_mechanism(self, mechanism: str) -> FakeMechanism:
        return self.mechanisms[mechanism]


def make_adapter() -> tuple[AutoScriptApertureAdapter, FakeCollection]:
    collection = FakeCollection()
    microscope = SimpleNamespace(
        optics=SimpleNamespace(aperture_mechanisms=collection)
    )
    return AutoScriptApertureAdapter(microscope), collection


def test_adapter_lists_mechanisms() -> None:
    adapter, _ = make_adapter()

    assert adapter.list_mechanisms() == ["C2", "Objective"]


def test_adapter_lists_apertures_with_name_type_and_diameter() -> None:
    adapter, _ = make_adapter()

    apertures = adapter.list_apertures("C2")

    assert [item.name for item in apertures] == ["30 um", "50 um", "Phase plate"]
    assert apertures[0].aperture_type == "Circular"
    assert apertures[0].diameter_m == pytest.approx(30e-6)
    assert apertures[0].inserted is True
    assert apertures[0].selected is True
    assert apertures[2].aperture_type == "PhasePlate"
    assert apertures[2].diameter_m is None


def test_adapter_selects_by_aperture_name() -> None:
    adapter, collection = make_adapter()

    selected = adapter.select_aperture("C2", "50 um")

    assert selected.name == "50 um"
    assert selected.aperture_type == "Circular"
    assert selected.diameter_m == pytest.approx(50e-6)
    assert collection.mechanisms["C2"].aperture is collection.mechanisms["C2"].apertures[1]


def test_adapter_gets_selected_aperture() -> None:
    adapter, _ = make_adapter()

    selected = adapter.get_selected_aperture("C2")

    assert selected.name == "30 um"
    assert selected.selected is True


def test_adapter_enable_and_disable_call_documented_methods() -> None:
    adapter, collection = make_adapter()
    mechanism = collection.mechanisms["Objective"]

    assert adapter.enable("Objective") is True
    assert mechanism.enable_calls == 1
    assert mechanism.is_enabled is True

    assert adapter.disable("Objective") is False
    assert mechanism.disable_calls == 1
    assert mechanism.is_enabled is False


def test_adapter_reports_missing_enable_as_unsupported_operation() -> None:
    adapter, collection = make_adapter()
    collection.mechanisms["C2"].enable = None

    with pytest.raises(
        RuntimeError,
        match="Unsupported AutoScript aperture operation 'enable'.*ApertureMechanism.enable",
    ):
        adapter.enable("C2")


def test_adapter_reports_missing_disable_as_unsupported_operation() -> None:
    adapter, collection = make_adapter()
    collection.mechanisms["C2"].disable = None

    with pytest.raises(
        RuntimeError,
        match="Unsupported AutoScript aperture operation 'disable'.*ApertureMechanism.disable",
    ):
        adapter.disable("C2")


def test_adapter_enable_rejects_unknown_mechanism() -> None:
    adapter, _ = make_adapter()

    with pytest.raises(ValueError, match="Unknown AutoScript aperture mechanism 'C3'"):
        adapter.enable("C3")

def test_adapter_reads_insertion_state_and_controls_mechanism() -> None:
    adapter, _ = make_adapter()

    assert adapter.get_insertion_state("C2") == "Inserted"
    assert adapter.retract_mechanism("C2") == "Retracted"
    assert adapter.insert_mechanism("C2") == "Inserted"


def test_adapter_reads_enabled_status() -> None:
    adapter, _ = make_adapter()

    assert adapter.is_enabled("C2") is True
    assert adapter.is_enabled("Objective") is False


def test_adapter_reads_retractable_status() -> None:
    adapter, _ = make_adapter()

    assert adapter.is_retractable("C2") is True
    assert adapter.is_retractable("Objective") is False


def test_adapter_reads_and_sets_documented_position() -> None:
    adapter, _ = make_adapter()

    assert adapter.get_position("C2") == pytest.approx((1e-6, -2e-6))

    assert adapter.set_position("C2", 5e-6, -6e-6) == pytest.approx((5e-6, -6e-6))


def test_adapter_rejects_missing_mechanism() -> None:
    adapter, _ = make_adapter()

    with pytest.raises(ValueError, match="Unknown AutoScript aperture mechanism 'C3'"):
        adapter.list_apertures("C3")


def test_adapter_rejects_unknown_aperture_name() -> None:
    adapter, _ = make_adapter()

    with pytest.raises(ValueError, match="Unknown aperture name '100 um'"):
        adapter.select_aperture("C2", "100 um")


def test_adapter_reports_missing_enabled_as_unsupported_operation() -> None:
    adapter, collection = make_adapter()
    del collection.mechanisms["C2"].is_enabled

    with pytest.raises(
        RuntimeError,
        match="Unsupported AutoScript aperture operation 'is_enabled'.*ApertureMechanism.is_enabled",
    ):
        adapter.is_enabled("C2")


def test_adapter_reports_missing_retractable_as_unsupported_operation() -> None:
    adapter, collection = make_adapter()
    del collection.mechanisms["C2"].is_retractable

    with pytest.raises(
        RuntimeError,
        match="Unsupported AutoScript aperture operation 'is_retractable'.*ApertureMechanism.is_retractable",
    ):
        adapter.is_retractable("C2")


def test_adapter_reports_missing_autoscript_property() -> None:
    microscope = SimpleNamespace(optics=SimpleNamespace())
    adapter = AutoScriptApertureAdapter(microscope)

    with pytest.raises(
        RuntimeError,
        match=r"microscope\.optics\.aperture_mechanisms",
    ):
        adapter.list_mechanisms()
