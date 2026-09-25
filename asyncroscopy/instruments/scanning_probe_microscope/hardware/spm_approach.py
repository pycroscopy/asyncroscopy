"""
SPM APPROACH Tango device.

Owns the tip approach/retract sequence. Vendor-specific behaviour is
isolated in the ``_hw_*`` hooks implemented by concrete subclasses
(APPROACH_Jupiter, etc.).
"""

from abc import abstractmethod

import tango
from asyncroscopy.instruments.instrument import CombinedMeta

class SPM_APPROACH(tango.server.Device, metaclass=CombinedMeta):
    """Abstract SPM approach device: tip engage / retract."""

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    approached = tango.server.attribute(
        label="Approached",
        dtype=bool,
        access=tango.AttrWriteType.READ,
        doc="True when the tip is engaged on the surface. Read live from hardware.",
    )


    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        tango.server.Device.init_device(self)
        self.set_state(tango.DevState.ON)
        self.info_stream("SPM APPROACH device initialised")

    # ------------------------------------------------------------------
    # Attribute read
    # ------------------------------------------------------------------

    def read_approached(self) -> bool:
        return self._hw_is_approached()
    
        # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @tango.server.command
    def approach(self) -> None:
        """Engage the tip on the surface. Blocks until engaged."""
        if self.get_state() == tango.DevState.MOVING:
            tango.Except.throw_exception(
                "ApproachInProgress",
                "An approach or retract is already running; call stop first.",
                "approach()",
            )
        self.set_state(tango.DevState.MOVING)
        try:
            self._hw_approach()
        finally:
            self.set_state(tango.DevState.ON)

    @tango.server.command
    def retract(self) -> None:
        """Retract the tip from the surface. Blocks until retracted."""
        if self.get_state() == tango.DevState.MOVING:
            tango.Except.throw_exception(
                "ApproachInProgress",
                "An approach or retract is already running; call stop first.",
                "retract()",
            )
        self.set_state(tango.DevState.MOVING)
        try:
            self._hw_retract()
        finally:
            self.set_state(tango.DevState.ON)

    @tango.server.command
    def stop(self) -> None:
        """Abort a running approach or retract immediately."""
        self._hw_stop()
        self.set_state(tango.DevState.ON)

    # ------------------------------------------------------------------
    # Abstract methods — vendor-specific
    # ------------------------------------------------------------------

    @abstractmethod
    def _hw_approach(self) -> None:
        """Run the approach sequence; return when the tip is engaged."""
        pass

    @abstractmethod
    def _hw_retract(self) -> None:
        """Retract the tip; return when clear of the surface."""
        pass

    @abstractmethod
    def _hw_stop(self) -> None:
        """Abort any running approach/retract motion."""
        pass

    @abstractmethod
    def _hw_is_approached(self) -> bool:
        """Return True if the tip is currently engaged."""
        pass

# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    SPM_APPROACH.run_server()
    