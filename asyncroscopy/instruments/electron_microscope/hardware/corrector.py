"""Backend-neutral aberration-corrector Tango interface."""

from __future__ import annotations

import json
from abc import abstractmethod
from typing import Any

from tango import AttrWriteType, DevState, DevString
from tango.server import Device, attribute, command


class CORRECTOR(Device):
    """Common Tango API implemented by real and simulated correctors.

    The public commands deliberately retain the CEOS-compatible JSON response
    structure used by the existing notebooks. Concrete implementations only
    supply backend operations.
    """

    status_message = attribute(
        label="Status message",
        dtype=str,
        access=AttrWriteType.READ,
        doc="Last backend status string",
    )

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.INIT)
        self._last_status = "Initialising"
        self._simulation_aberrations: dict[str, Any] = {}
        self._connect_backend()

    def read_status_message(self) -> str:
        return self._last_status

    @command(dtype_out=DevString)
    def get_info(self) -> str:
        return self._get_info()

    @command(dtype_in=str, dtype_out=DevString)
    def acquire_tableau(self, args: str) -> str:
        parts = args.strip().split()
        if len(parts) != 2:
            raise ValueError("acquire_tableau expects '<type> <angle>'")
        tab_type, angle = parts
        return self._acquire_tableau(tab_type, float(angle))

    @command(dtype_out=DevString)
    def measure_c1a1(self) -> str:
        return self._measure_c1a1()

    @command(dtype_in=str, dtype_out=DevString)
    def correct_aberration(self, args: str) -> str:
        parts = args.strip().split()
        if len(parts) < 2:
            raise ValueError("correct_aberration expects '<name> <value> [value]' ")
        name = parts[0]
        values = [float(value) for value in parts[1:]]
        return self._correct_aberration(name, values)

    @command
    def reconnect(self) -> None:
        self._connect_backend()

    @command(dtype_in=str)
    def set_aberrations_coeff_sim(self, json_aberrations_string: str) -> None:
        coefficients = json.loads(json_aberrations_string)
        if not isinstance(coefficients, dict):
            raise ValueError("Simulation aberrations must be a JSON object")
        self._set_simulation_aberrations(coefficients)

    @command(dtype_out=str)
    def get_aberrations_coeff_sim(self) -> str:
        return json.dumps(self._get_simulation_aberrations())

    def _set_simulation_aberrations(self, coefficients: dict[str, Any]) -> None:
        self._simulation_aberrations = coefficients

    def _get_simulation_aberrations(self) -> dict[str, Any]:
        return self._simulation_aberrations

    @abstractmethod
    def _connect_backend(self) -> None:
        """Connect or initialize the concrete backend."""

    @abstractmethod
    def _get_info(self) -> str:
        """Return a CEOS-compatible JSON response string."""

    @abstractmethod
    def _acquire_tableau(self, tab_type: str, angle: float) -> str:
        """Acquire aberrations and return a CEOS-compatible JSON response."""

    @abstractmethod
    def _measure_c1a1(self) -> str:
        """Measure first-order aberrations."""

    @abstractmethod
    def _correct_aberration(self, name: str, values: list[float]) -> str:
        """Apply a correction using backend-native semantics."""


if __name__ == "__main__":
    CORRECTOR.run_server()
