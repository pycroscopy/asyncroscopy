"""Isolated adapter for the documented AutoScript motorized-aperture API.

The adapter accepts an already-connected microscope client.  It neither
imports nor constructs ``TemMicroscopeClient``, so importing asyncroscopy and
testing this module never requires an AutoScript installation or connection.
"""

from __future__ import annotations

from typing import Any

from asyncroscopy.instruments.electron_microscope.hardware.aperture import ApertureInfo


class AutoScriptApertureAdapter:
    """Translate AutoScript aperture objects into vendor-neutral models."""

    def __init__(self, microscope: Any) -> None:
        if microscope is None:
            raise ValueError("An existing AutoScript microscope object is required.")
        self._microscope = microscope

    @staticmethod
    def _missing(path: str) -> RuntimeError:
        return RuntimeError(f"Missing AutoScript object/property: {path}")

    @staticmethod
    def _unsupported(operation: str, path: str) -> RuntimeError:
        return RuntimeError(
            f"Unsupported AutoScript aperture operation {operation!r}: "
            f"missing documented object/property {path}."
        )

    @classmethod
    def _require_attribute(cls, owner: Any, name: str, path: str) -> Any:
        try:
            return getattr(owner, name)
        except AttributeError as exc:
            raise cls._missing(path) from exc

    @classmethod
    def _require_method(cls, owner: Any, name: str, path: str):
        method = cls._require_attribute(owner, name, path)
        if not callable(method):
            raise cls._missing(path)
        return method

    def _collection(self) -> Any:
        optics = self._require_attribute(
            self._microscope,
            "optics",
            "microscope.optics",
        )
        return self._require_attribute(
            optics,
            "aperture_mechanisms",
            "microscope.optics.aperture_mechanisms",
        )

    def list_mechanisms(self) -> list[str]:
        """Return mechanism identifiers reported by AutoScript."""
        collection = self._collection()
        get_available = self._require_method(
            collection,
            "get_available",
            "microscope.optics.aperture_mechanisms.get_available",
        )
        return list(get_available())

    def _mechanism(self, mechanism: str) -> Any:
        available = self.list_mechanisms()
        if mechanism not in available:
            names = ", ".join(available)
            raise ValueError(
                f"Unknown AutoScript aperture mechanism {mechanism!r}. "
                f"Available mechanisms: {names}."
            )

        collection = self._collection()
        get_mechanism = self._require_method(
            collection,
            "get_mechanism",
            "microscope.optics.aperture_mechanisms.get_mechanism",
        )
        return get_mechanism(mechanism)

    @classmethod
    def _aperture_info(
        cls,
        mechanism: str,
        aperture: Any,
        *,
        inserted: bool | None,
        selected: bool,
    ) -> ApertureInfo:
        name = cls._require_attribute(aperture, "name", "Aperture.name")
        aperture_type = cls._require_attribute(aperture, "type", "Aperture.type")
        diameter = cls._require_attribute(aperture, "diameter", "Aperture.diameter")
        return ApertureInfo(
            mechanism=mechanism,
            name=name,
            aperture_type=aperture_type,
            diameter_m=float(diameter) if diameter is not None else None,
            inserted=inserted,
            selected=selected,
        )

    @staticmethod
    def _inserted_from_state(insertion_state: str) -> bool | None:
        if insertion_state == "Inserted":
            return True
        if insertion_state == "Retracted":
            return False
        return None

    def list_apertures(self, mechanism: str) -> list[ApertureInfo]:
        """Return all apertures on one discovered mechanism."""
        mechanism_object = self._mechanism(mechanism)
        apertures = self._require_attribute(
            mechanism_object,
            "apertures",
            "ApertureMechanism.apertures",
        )
        selected_aperture = self._require_attribute(
            mechanism_object,
            "aperture",
            "ApertureMechanism.aperture",
        )
        insertion_state = self.get_insertion_state(mechanism)
        inserted = self._inserted_from_state(insertion_state)
        selected_name = (
            self._require_attribute(selected_aperture, "name", "Aperture.name")
            if selected_aperture is not None
            else None
        )
        return [
            self._aperture_info(
                mechanism,
                aperture,
                inserted=inserted,
                selected=self._require_attribute(aperture, "name", "Aperture.name")
                == selected_name,
            )
            for aperture in apertures
        ]

    def get_selected_aperture(self, mechanism: str) -> ApertureInfo:
        """Return the selected aperture or fail clearly when none is selected."""
        mechanism_object = self._mechanism(mechanism)
        aperture = self._require_attribute(
            mechanism_object,
            "aperture",
            "ApertureMechanism.aperture",
        )
        if aperture is None:
            raise RuntimeError(
                f"AutoScript ApertureMechanism.aperture is None for {mechanism!r}."
            )
        inserted = self._inserted_from_state(self.get_insertion_state(mechanism))
        return self._aperture_info(
            mechanism,
            aperture,
            inserted=inserted,
            selected=True,
        )

    def select_aperture(self, mechanism: str, name: str) -> ApertureInfo:
        """Select an AutoScript aperture by its documented unique name."""
        mechanism_object = self._mechanism(mechanism)
        apertures = self._require_attribute(
            mechanism_object,
            "apertures",
            "ApertureMechanism.apertures",
        )
        selected = next(
            (
                aperture
                for aperture in apertures
                if self._require_attribute(aperture, "name", "Aperture.name") == name
            ),
            None,
        )
        if selected is None:
            available = ", ".join(
                self._require_attribute(aperture, "name", "Aperture.name")
                for aperture in apertures
            )
            raise ValueError(
                f"Unknown aperture name {name!r} for mechanism {mechanism!r}. "
                f"Available apertures: {available}."
            )
        try:
            mechanism_object.aperture = selected
        except AttributeError as exc:
            raise self._missing("ApertureMechanism.aperture") from exc
        return self.get_selected_aperture(mechanism)

    def get_insertion_state(self, mechanism: str) -> str:
        """Return the documented AutoScript insertion-state string."""
        mechanism_object = self._mechanism(mechanism)
        return self._require_attribute(
            mechanism_object,
            "insertion_state",
            "ApertureMechanism.insertion_state",
        )

    def is_enabled(self, mechanism: str) -> bool:
        """Return documented `ApertureMechanism.is_enabled` status."""
        mechanism_object = self._mechanism(mechanism)
        try:
            return bool(
                self._require_attribute(
                    mechanism_object,
                    "is_enabled",
                    "ApertureMechanism.is_enabled",
                )
            )
        except RuntimeError as exc:
            raise self._unsupported(
                "is_enabled",
                "ApertureMechanism.is_enabled",
            ) from exc

    def is_retractable(self, mechanism: str) -> bool:
        """Return documented `ApertureMechanism.is_retractable` status."""
        mechanism_object = self._mechanism(mechanism)
        try:
            return bool(
                self._require_attribute(
                    mechanism_object,
                    "is_retractable",
                    "ApertureMechanism.is_retractable",
                )
            )
        except RuntimeError as exc:
            raise self._unsupported(
                "is_retractable",
                "ApertureMechanism.is_retractable",
            ) from exc

    def enable(self, mechanism: str) -> bool:
        """Call documented ``ApertureMechanism.enable()`` and return enabled state."""
        mechanism_object = self._mechanism(mechanism)
        try:
            enable = self._require_method(
                mechanism_object,
                "enable",
                "ApertureMechanism.enable",
            )
        except RuntimeError as exc:
            raise self._unsupported("enable", "ApertureMechanism.enable") from exc
        enable()
        return self.is_enabled(mechanism)

    def disable(self, mechanism: str) -> bool:
        """Call documented ``ApertureMechanism.disable()`` and return enabled state."""
        mechanism_object = self._mechanism(mechanism)
        try:
            disable = self._require_method(
                mechanism_object,
                "disable",
                "ApertureMechanism.disable",
            )
        except RuntimeError as exc:
            raise self._unsupported("disable", "ApertureMechanism.disable") from exc
        disable()
        return self.is_enabled(mechanism)
    def insert_mechanism(self, mechanism: str) -> str:
        """Call ``ApertureMechanism.insert()`` and return its resulting state."""
        mechanism_object = self._mechanism(mechanism)
        insert = self._require_method(
            mechanism_object,
            "insert",
            "ApertureMechanism.insert",
        )
        insert()
        return self.get_insertion_state(mechanism)

    def retract_mechanism(self, mechanism: str) -> str:
        """Call ``ApertureMechanism.retract()`` and return its resulting state."""
        mechanism_object = self._mechanism(mechanism)
        retract = self._require_method(
            mechanism_object,
            "retract",
            "ApertureMechanism.retract",
        )
        retract()
        return self.get_insertion_state(mechanism)

    def get_position(self, mechanism: str) -> tuple[float, float] | None:
        """Return documented selected-aperture position in meters."""
        mechanism_object = self._mechanism(mechanism)
        position = self._require_attribute(
            mechanism_object,
            "position",
            "ApertureMechanism.position",
        )
        if position is None:
            return None
        x = self._require_attribute(position, "x", "Point.x")
        y = self._require_attribute(position, "y", "Point.y")
        return float(x), float(y)

    def set_position(self, mechanism: str, x: float, y: float) -> tuple[float, float] | None:
        """Set documented selected-aperture position with an ``(x, y)`` tuple."""
        mechanism_object = self._mechanism(mechanism)
        try:
            mechanism_object.position = (float(x), float(y))
        except AttributeError as exc:
            raise self._missing("ApertureMechanism.position") from exc
        return self.get_position(mechanism)
