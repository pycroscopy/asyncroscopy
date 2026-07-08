"""Simulation-first aperture control device.

This module deliberately has no AutoScript imports and performs no microscope
I/O.  The public Tango interface is intended to remain stable when a hardware
backend is added later.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from tango import AttrWriteType, DevState
from tango.server import Device, attribute, command


@dataclass(frozen=True)
class ApertureInfo:
    """Vendor-neutral description of one aperture."""

    mechanism: str
    name: str
    aperture_type: str
    diameter_m: float | None
    inserted: bool | None
    selected: bool


@dataclass
class _SimulatedMechanism:
    apertures: tuple[tuple[str, str, float | None], ...]
    inserted: bool
    selected_name: str | None
    last_selected_name: str
    enabled: bool
    retractable: bool


class SimulatedApertureBackend:
    """Deterministic in-memory aperture state for a representative Spectra STEM."""

    def __init__(self) -> None:
        self._mechanisms: dict[str, _SimulatedMechanism] = {
            "condenser": self._mechanism((30, 50, 70, 100), "50 um", inserted=True),
            "objective": self._mechanism((20, 40, 70, 100), "40 um", inserted=False),
            "selected_area": self._mechanism((10, 20, 40, 100), "40 um", inserted=False),
            "projector": self._mechanism((50, 100, 200), "100 um", inserted=False),
        }

    @staticmethod
    def _mechanism(
        diameters_um: tuple[int, ...],
        default_name: str,
        *,
        inserted: bool,
    ) -> _SimulatedMechanism:
        apertures = tuple(
            (f"{diameter} um", "Circular", diameter * 1e-6)
            for diameter in diameters_um
        )
        return _SimulatedMechanism(
            apertures=apertures,
            inserted=inserted,
            selected_name=default_name if inserted else None,
            last_selected_name=default_name,
            enabled=True,
            retractable=True,
        )

    @property
    def mechanism_names(self) -> tuple[str, ...]:
        return tuple(self._mechanisms)

    def list_mechanisms(self) -> list[str]:
        """Return simulated mechanism names using the adapter interface."""
        return list(self.mechanism_names)

    def _get(self, mechanism: str) -> _SimulatedMechanism:
        try:
            return self._mechanisms[mechanism]
        except KeyError as exc:
            available = ", ".join(self.mechanism_names)
            raise ValueError(
                f"Unknown aperture mechanism {mechanism!r}. Available mechanisms: {available}."
            ) from exc

    def list_apertures(self, mechanism: str) -> list[ApertureInfo]:
        state = self._get(mechanism)
        return [
            ApertureInfo(
                mechanism=mechanism,
                name=name,
                aperture_type=aperture_type,
                diameter_m=diameter_m,
                inserted=state.inserted,
                selected=name == state.selected_name,
            )
            for name, aperture_type, diameter_m in state.apertures
        ]

    def selected_aperture(self, mechanism: str) -> ApertureInfo | None:
        return next(
            (item for item in self.list_apertures(mechanism) if item.selected),
            None,
        )

    def get_selected_aperture(self, mechanism: str) -> ApertureInfo:
        """Return the selected aperture using the adapter interface."""
        selected = self.selected_aperture(mechanism)
        if selected is None:
            raise RuntimeError(f"No aperture is selected for mechanism {mechanism!r}.")
        return selected

    def is_inserted(self, mechanism: str) -> bool:
        return self._get(mechanism).inserted

    def get_insertion_state(self, mechanism: str) -> str:
        return "Inserted" if self.is_inserted(mechanism) else "Retracted"

    def is_enabled(self, mechanism: str) -> bool:
        return self._get(mechanism).enabled

    def is_retractable(self, mechanism: str) -> bool:
        return self._get(mechanism).retractable

    def enable(self, mechanism: str) -> bool:
        state = self._get(mechanism)
        state.enabled = True
        return state.enabled

    def disable(self, mechanism: str) -> bool:
        state = self._get(mechanism)
        state.enabled = False
        return state.enabled

    def select_aperture(self, mechanism: str, name: str) -> ApertureInfo:
        state = self._get(mechanism)
        available_names = [item[0] for item in state.apertures]
        if name not in available_names:
            available = ", ".join(available_names)
            raise ValueError(
                f"Unknown aperture {name!r} for mechanism {mechanism!r}. "
                f"Available apertures: {available}."
            )

        # AutoScript selection semantics insert the containing mechanism.
        state.inserted = True
        state.selected_name = name
        state.last_selected_name = name
        selected = self.selected_aperture(mechanism)
        assert selected is not None
        return selected

    def insert_mechanism(self, mechanism: str) -> str:
        state = self._get(mechanism)
        state.inserted = True
        state.selected_name = state.last_selected_name
        selected = self.selected_aperture(mechanism)
        assert selected is not None
        return "Inserted"

    def retract_mechanism(self, mechanism: str) -> str:
        state = self._get(mechanism)
        if state.selected_name is not None:
            state.last_selected_name = state.selected_name
        state.inserted = False
        state.selected_name = None
        return "Retracted"


class APERTURE(Device):
    """PyTango aperture device backed only by deterministic simulation state."""

    mechanism_names = attribute(
        label="Aperture Mechanisms",
        dtype=(str,),
        max_dim_x=16,
        access=AttrWriteType.READ,
        doc="Names of simulated aperture mechanisms.",
    )

    selected_aperture_name = attribute(
        label="Selected Aperture Name",
        dtype=str,
        access=AttrWriteType.READ,
        doc="Selected aperture on the mechanism used by the latest successful command.",
    )

    selected_aperture_type = attribute(
        label="Selected Aperture Type",
        dtype=str,
        access=AttrWriteType.READ,
        doc="Type of the selected aperture on the active mechanism.",
    )

    selected_aperture_diameter_m = attribute(
        label="Selected Aperture Diameter",
        dtype=float,
        unit="m",
        access=AttrWriteType.READ,
        doc="Selected circular-aperture diameter in meters, or NaN when unavailable.",
    )

    insertion_state = attribute(
        label="Insertion State",
        dtype=str,
        access=AttrWriteType.READ,
        doc="Inserted or Retracted state of the active simulated mechanism.",
    )

    last_error = attribute(
        label="Last Error",
        dtype=str,
        access=AttrWriteType.READ,
        doc="Most recent command validation error, cleared after a successful command.",
    )

    def init_device(self) -> None:
        Device.init_device(self)
        self._backend = SimulatedApertureBackend()
        self._active_mechanism = self._backend.mechanism_names[0]
        self._last_error = ""
        self.set_state(DevState.ON)
        self.info_stream("APERTURE simulation device initialised")

    @staticmethod
    def _json(payload: dict) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _aperture_payload(aperture: ApertureInfo | None) -> dict | None:
        return asdict(aperture) if aperture is not None else None

    def _response(
        self,
        *,
        status: str,
        mechanism: str,
        selected_aperture: ApertureInfo | None,
        inserted: bool | None,
        error: str | None,
        **extra,
    ) -> str:
        payload = {
            "status": status,
            "mechanism": mechanism,
            "selected_aperture": self._aperture_payload(selected_aperture),
            "inserted": inserted,
            "error": error,
        }
        payload.update(extra)
        return self._json(payload)

    def _success(self, mechanism: str) -> str:
        self._active_mechanism = mechanism
        self._last_error = ""
        return self._response(
            status="ok",
            mechanism=mechanism,
            selected_aperture=self._backend.selected_aperture(mechanism),
            inserted=self._backend.is_inserted(mechanism),
            error=None,
        )

    def _failure(self, mechanism: str, exc: ValueError) -> str:
        self._last_error = str(exc)
        return self._response(
            status="error",
            mechanism=mechanism,
            selected_aperture=None,
            inserted=None,
            error=self._last_error,
        )

    def read_mechanism_names(self) -> list[str]:
        return list(self._backend.mechanism_names)

    def read_selected_aperture_name(self) -> str:
        selected = self._backend.selected_aperture(self._active_mechanism)
        return selected.name if selected is not None else ""

    def read_selected_aperture_type(self) -> str:
        selected = self._backend.selected_aperture(self._active_mechanism)
        return selected.aperture_type if selected is not None else ""

    def read_selected_aperture_diameter_m(self) -> float:
        selected = self._backend.selected_aperture(self._active_mechanism)
        if selected is None or selected.diameter_m is None:
            return math.nan
        return selected.diameter_m

    def read_insertion_state(self) -> str:
        return "Inserted" if self._backend.is_inserted(self._active_mechanism) else "Retracted"

    def read_last_error(self) -> str:
        return self._last_error

    @command(dtype_out=str)
    def list_apertures(self) -> str:
        """Return all simulated mechanisms and apertures as deterministic JSON."""
        self._last_error = ""
        apertures = [
            asdict(aperture)
            for mechanism in self._backend.mechanism_names
            for aperture in self._backend.list_apertures(mechanism)
        ]
        return self._response(
            status="ok",
            mechanism="all",
            selected_aperture=None,
            inserted=None,
            error=None,
            apertures=apertures,
        )

    @command(dtype_in=str, dtype_out=str)
    def get_selected_aperture(self, mechanism: str) -> str:
        """Return the selected aperture for one simulated mechanism."""
        try:
            self._backend.is_inserted(mechanism)
            return self._success(mechanism)
        except ValueError as exc:
            return self._failure(mechanism, exc)

    @command(dtype_in=str, dtype_out=str)
    def select_aperture(self, json_request: str) -> str:
        """Select by JSON request: ``{"mechanism": "...", "name": "..."}``."""
        mechanism = ""
        try:
            try:
                request = json.loads(json_request)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError("Invalid JSON request.") from exc
            if not isinstance(request, dict):
                raise ValueError("Aperture selection request must be a JSON object.")
            requested_mechanism = request.get("mechanism")
            name = request.get("name")
            if not isinstance(requested_mechanism, str) or not requested_mechanism:
                raise ValueError("Selection request requires a non-empty string 'mechanism'.")
            if not isinstance(name, str) or not name:
                raise ValueError("Selection request requires a non-empty string 'name'.")
            mechanism = requested_mechanism
            self._backend.select_aperture(mechanism, name)
            return self._success(mechanism)
        except ValueError as exc:
            return self._failure(mechanism, exc)

    @command(dtype_in=str, dtype_out=str)
    def insert_mechanism(self, mechanism: str) -> str:
        """Insert one simulated mechanism."""
        try:
            self._backend.insert_mechanism(mechanism)
            return self._success(mechanism)
        except ValueError as exc:
            return self._failure(mechanism, exc)

    @command(dtype_in=str, dtype_out=str)
    def retract_mechanism(self, mechanism: str) -> str:
        """Retract one simulated mechanism."""
        try:
            self._backend.retract_mechanism(mechanism)
            return self._success(mechanism)
        except ValueError as exc:
            return self._failure(mechanism, exc)

    @command(dtype_out=str)
    def refresh(self) -> str:
        """Return a deterministic snapshot of all current simulation state."""
        self._last_error = ""
        mechanisms = []
        for mechanism in self._backend.mechanism_names:
            mechanisms.append(
                {
                    "mechanism": mechanism,
                    "selected_aperture": self._aperture_payload(
                        self._backend.selected_aperture(mechanism)
                    ),
                    "inserted": self._backend.is_inserted(mechanism),
                }
            )
        return self._response(
            status="ok",
            mechanism="all",
            selected_aperture=None,
            inserted=None,
            error=None,
            mechanisms=mechanisms,
        )


if __name__ == "__main__":
    APERTURE.run_server()
