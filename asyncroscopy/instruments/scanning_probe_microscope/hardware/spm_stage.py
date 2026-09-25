"""
SPM STAGE Tango device.

Owns the coarse XY sample stage.

"""

from abc import abstractmethod
import tango # type: ignore
 
from asyncroscopy.instruments.instrument import CombinedMeta

class SPM_STAGE(tango.server.Device, metaclass=CombinedMeta):
    """Abstract SPM coarse stage device: XY position and moves."""

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------
    
    stage_x_m = tango.server.attribute(
        label="Stage X Position",
        dtype=float,
        access=tango.AttrWriteType.READ,
        unit="m",

        doc="Current stage X position in meters. Read live from hardware.",
    )

    stage_y_m = tango.server.attribute(
        label="Stage Y Position",
        dtype=float,
        access=tango.AttrWriteType.READ,
        unit="m",

        doc="Current stage Y position in meters. Read live from hardware.",
    )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        tango.server.Device.init_device(self)
        self.set_state(tango.DevState.ON)
        self.info_stream("SPM STAGE device initialised")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_move(self, dx: float, dy: float) -> None:
        """Run one relative move with MOVING-state bookkeeping."""
        if self.get_state() == tango.DevState.MOVING:
            tango.Except.throw_exception(
                "MoveInProgress",
                "A stage move is already running; call stop first.",
                "move_stage()",
            )
        self.set_state(tango.DevState.MOVING)
        try:
            self._hw_move_stage_relative(dx, dy)
        finally:
            self.set_state(tango.DevState.ON)

    # ------------------------------------------------------------------
    # Attribute read — live from hardware
    # ------------------------------------------------------------------

    def read_stage_x_m(self) -> float:
        return self._hw_read_stage_position()[0]

    def read_stage_y_m(self) -> float:
        return self._hw_read_stage_position()[1]

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @tango.server.command(dtype_in=tango.DevVarDoubleArray)
    def move_stage(self, position) -> None:
        """Move the stage to an absolute position [x, y] in meters. Blocks until done."""
        if len(position) != 2:
            raise ValueError("position must contain exactly two values: [x, y]")
        current = self._hw_read_stage_position()
        self._do_move(float(position[0]) - current[0], float(position[1]) - current[1])

    @tango.server.command(dtype_in=tango.DevVarDoubleArray)
    def move_stage_relative(self, delta) -> None:
        """Move the stage by [dx, dy] in meters. Blocks until done."""
        if len(delta) != 2:
            raise ValueError("delta must contain exactly two values: [dx, dy]")
        self._do_move(float(delta[0]), float(delta[1]))

    @tango.server.command
    def stop(self) -> None:
        """Abort a running stage move immediately."""
        self._hw_stop()
        self.set_state(tango.DevState.ON)

    
    # ------------------------------------------------------------------
    # Abstract methods — vendor-specific
    # ------------------------------------------------------------------

    @abstractmethod
    def _hw_read_stage_position(self) -> list[float]:
        """Return the current stage position [x, y] in meters."""
        pass

    @abstractmethod
    def _hw_move_stage_relative(self, dx: float, dy: float) -> None:
        """Move the stage by (dx, dy) in meters; return when the move is complete."""
        pass

    @abstractmethod
    def _hw_stop(self) -> None:
        """Abort any running stage motion."""
        pass

# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    SPM_STAGE.run_server()

