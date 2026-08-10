"""
AutoScript-backed APERTURE Tango device.

This device exposes the public APERTURE API and reads or writes the physical
aperture mechanisms through AutoScript.
"""

import math

from tango import DevState
from tango.server import device_property

from asyncroscopy.instruments.electron_microscope.hardware.aperture import APERTURE

# AutoScript imports - only available on the microscope PC.
try:
    from autoscript_tem_microscope_client import TemMicroscopeClient

    _AUTOSCRIPT_AVAILABLE = True
except ImportError:
    _AUTOSCRIPT_AVAILABLE = False


class AutoScriptAPERTURE(APERTURE):
    """AutoScript-backed motorized aperture device."""

    # ------------------------------------------------------------------
    # Device properties - set per-deployment in the Tango DB
    # ------------------------------------------------------------------

    hardware_host = device_property(
        dtype=str,
        default_value="10.46.217.241",
        doc="Hostname or IP of the AutoScript microscope server",
    )

    hardware_port = device_property(
        dtype=int,
        default_value=9095,
        doc="Port of the AutoScript microscope server",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        super().init_device()
        self._microscope = None

        if not _AUTOSCRIPT_AVAILABLE:
            self.warn_stream("AutoScript not available")
            return

        try:
            self._microscope = TemMicroscopeClient()
            self._microscope.connect(self.hardware_host, self.hardware_port)
            mechanisms = self._read_available_mechanisms()
            self._mechanism = mechanisms[0] if mechanisms else ""
            self.info_stream(
                f"Connected AutoScript APERTURE at {self.hardware_host}:{self.hardware_port}"
            )
        except Exception as exc:
            self._microscope = None
            self.set_state(DevState.FAULT)
            self.error_stream(f"AutoScript APERTURE connection failed: {exc}")

    # ------------------------------------------------------------------
    # Attribute read / write
    # ------------------------------------------------------------------

    def _aperture_mechanism(self):
        return self._microscope.optics.aperture_mechanisms.get_mechanism(
            self._mechanism
        )

    def _selected_aperture(self):
        return self._aperture_mechanism().aperture

    def _read_available_mechanisms(self) -> list[str]:
        return list(self._microscope.optics.aperture_mechanisms.get_available())

    def _read_available_apertures(self) -> list[str]:
        return [aperture.name for aperture in self._aperture_mechanism().apertures]

    def _read_selected_aperture(self) -> str:
        aperture = self._selected_aperture()
        return "" if aperture is None else aperture.name

    def _write_selected_aperture(self, value: str) -> None:
        mechanism = self._aperture_mechanism()
        try:
            aperture = next(
                aperture for aperture in mechanism.apertures if aperture.name == value
            )
        except StopIteration as exc:
            raise ValueError(f"Unknown aperture {value!r}") from exc
        mechanism.aperture = aperture

    def _read_aperture_type(self) -> str:
        aperture = self._selected_aperture()
        return "" if aperture is None else aperture.type

    def _read_aperture_diameter(self) -> float:
        aperture = self._selected_aperture()
        if aperture is None or aperture.diameter is None:
            return math.nan
        return float(aperture.diameter)

    def _read_insertion_state(self) -> str:
        return self._aperture_mechanism().insertion_state

    def _read_enabled(self) -> bool:
        return self._aperture_mechanism().is_enabled

    def _read_retractable(self) -> bool:
        return self._aperture_mechanism().is_retractable

    def _read_position(self) -> list[float]:
        position = self._aperture_mechanism().position
        if position is None:
            return []
        return [float(position.x), float(position.y)]

    def _write_position(self, value) -> None:
        x, y = value
        self._aperture_mechanism().position = (float(x), float(y))

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _insert(self) -> None:
        self._aperture_mechanism().insert()

    def _retract(self) -> None:
        self._aperture_mechanism().retract()

    def _enable(self) -> None:
        self._aperture_mechanism().enable()

    def _disable(self) -> None:
        self._aperture_mechanism().disable()

    def _reset_positions(self) -> None:
        self._aperture_mechanism().reset_positions()


AutoscriptAPERTURE = AutoScriptAPERTURE


if __name__ == "__main__":
    AutoScriptAPERTURE.run_server()
