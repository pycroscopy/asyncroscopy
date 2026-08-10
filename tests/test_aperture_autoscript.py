import math
from types import SimpleNamespace

import pytest

from asyncroscopy.instruments.electron_microscope.hardware.aperture_autoscript import (
    AutoScriptAPERTURE,
)


class FakeAperture:
    def __init__(self, name: str, aperture_type: str, diameter: float | None):
        self.name = name
        self.type = aperture_type
        self.diameter = diameter


class FakeMechanism:
    def __init__(self):
        self.apertures = [
            FakeAperture("30 um", "Circular", 30e-6),
            FakeAperture("50 um", "Circular", 50e-6),
            FakeAperture("Phase plate", "PhasePlate", None),
        ]
        self.aperture = self.apertures[0]
        self.insertion_state = "Inserted"
        self.is_enabled = False
        self.is_retractable = False
        self.position = SimpleNamespace(x=1e-6, y=-2e-6)
        self.calls = []

    def insert(self) -> None:
        self.calls.append("insert")

    def retract(self) -> None:
        self.calls.append("retract")

    def enable(self) -> None:
        self.calls.append("enable")

    def disable(self) -> None:
        self.calls.append("disable")

    def reset_positions(self) -> None:
        self.calls.append("reset_positions")


class FakeApertureMechanisms:
    def __init__(self):
        self.mechanisms = {
            "C2": FakeMechanism(),
            "Objective": FakeMechanism(),
        }

    def get_available(self) -> list[str]:
        return list(self.mechanisms)

    def get_mechanism(self, mechanism: str) -> FakeMechanism:
        return self.mechanisms[mechanism]


def make_aperture() -> tuple[AutoScriptAPERTURE, FakeApertureMechanisms]:
    aperture = AutoScriptAPERTURE.__new__(AutoScriptAPERTURE)
    mechanisms = FakeApertureMechanisms()
    aperture._microscope = SimpleNamespace(
        optics=SimpleNamespace(aperture_mechanisms=mechanisms)
    )
    aperture._mechanism = "C2"
    return aperture, mechanisms


def test_reads_available_mechanisms_and_apertures():
    aperture, _ = make_aperture()

    assert aperture._read_available_mechanisms() == ["C2", "Objective"]
    assert aperture._read_available_apertures() == [
        "30 um",
        "50 um",
        "Phase plate",
    ]


def test_reads_and_writes_selected_aperture():
    aperture, mechanisms = make_aperture()

    assert aperture._read_selected_aperture() == "30 um"
    assert aperture._read_aperture_type() == "Circular"
    assert aperture._read_aperture_diameter() == pytest.approx(30e-6)

    aperture._write_selected_aperture("50 um")

    assert mechanisms.mechanisms["C2"].aperture.name == "50 um"


def test_missing_selected_aperture_returns_empty_values():
    aperture, mechanisms = make_aperture()
    mechanisms.mechanisms["C2"].aperture = None

    assert aperture._read_selected_aperture() == ""
    assert aperture._read_aperture_type() == ""
    assert math.isnan(aperture._read_aperture_diameter())


def test_reads_and_writes_aperture_position():
    aperture, mechanisms = make_aperture()

    assert aperture._read_position() == pytest.approx([1e-6, -2e-6])

    aperture._write_position([5e-6, -6e-6])

    assert mechanisms.mechanisms["C2"].position == pytest.approx((5e-6, -6e-6))


def test_reads_mechanism_state():
    aperture, _ = make_aperture()

    assert aperture._read_insertion_state() == "Inserted"
    assert aperture._read_enabled() is False
    assert aperture._read_retractable() is False


def test_commands_delegate_directly_to_autoscript():
    aperture, mechanisms = make_aperture()
    mechanism = mechanisms.mechanisms["C2"]

    aperture._insert()
    aperture._retract()
    aperture._enable()
    aperture._disable()
    aperture._reset_positions()

    assert mechanism.calls == [
        "insert",
        "retract",
        "enable",
        "disable",
        "reset_positions",
    ]
