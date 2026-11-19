"""
SPM FEEDBACK Tango device.

Owns the Z feedback loop: setpoint, gains, and engage/withdraw.

Note: the physical meaning of ``setpoint`` depends on the active SPM
mode (deflection volts in contact, amplitude volts in AC, etc.); this
device passes the value through without interpreting it.
"""

from abc import abstractmethod
import tango # type: ignore

from asyncroscopy.instruments.instrument import CombinedMeta

class SPM_FEEDBACK(tango.server.Device, metaclass=CombinedMeta):
    """Abstract SPM feedback device: Z loop setpoint, gains, engage state."""

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    setpoint = tango.server.attribute(
        label="Setpoint",
        dtype=float,
        access=tango.AttrWriteType.READ_WRITE,
        doc="Z feedback setpoint. Physical meaning depends on the active SPM mode "
            "(deflection in contact, amplitude in AC).",
    )

    i_gain = tango.server.attribute(
        label="Integral Gain",
        dtype=float,
        access=tango.AttrWriteType.READ_WRITE,
        doc="Integral gain of the Z feedback loop.",
    )

    feedback_on_bool = tango.server.attribute(
        label="Feedback On",
        dtype=bool,
        access=tango.AttrWriteType.READ,
        doc="True when the Z feedback loop is engaged. Read live from hardware.",
    )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def init_device(self) -> None:
        tango.server.Device.init_device(self)
        self._refresh_params()
        self.set_state(tango.DevState.ON)
        self.info_stream("SPM FEEDBACK device initialised")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_params(self) -> None:
        """Re-read all feedback parameters from hardware into local fields."""
        params = self._hw_read_feedback_params()
        self._setpoint: float = params["setpoint"]
        self._i_gain: float = params["i_gain"]

    def _write_param(self, name: str, value) -> None:
        """Push one parameter to hardware, then re-read all (hardware may coerce values)."""
        self._hw_write_feedback_param(name, value)
        self._refresh_params()

    # ------------------------------------------------------------------
    # Attribute read / write — writes pushed to hardware
    # ------------------------------------------------------------------

    def read_setpoint(self) -> float:
        return self._setpoint

    def write_setpoint(self, value: float) -> None:
        self._write_param("setpoint", value)

    def read_i_gain(self) -> float:
        return self._i_gain

    def write_i_gain(self, value: float) -> None:
        self._write_param("i_gain", value)

    def read_feedback_on_bool(self) -> bool:
        return self._hw_is_feedback_on()
    
    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @tango.server.command
    def feedback_on(self) -> None:
        """Engage the Z feedback loop."""
        self._hw_feedback_on()

    @tango.server.command
    def feedback_off(self) -> None:
        """Disengage the Z feedback loop."""
        self._hw_feedback_off()

    @tango.server.command
    def refresh_params(self) -> None:
        """Re-read all feedback parameters from hardware, e.g. after changes in the vendor GUI."""
        self._refresh_params()

    # ------------------------------------------------------------------
    # Abstract methods — vendor-specific
    # ------------------------------------------------------------------

    @abstractmethod
    def _hw_read_feedback_params(self) -> dict:
        """Read all feedback parameters from hardware. Keys must match attribute names."""
        pass

    @abstractmethod
    def _hw_write_feedback_param(self, name: str, value) -> None:
        """Push one feedback parameter to hardware."""
        pass

    @abstractmethod
    def _hw_feedback_on(self) -> None:
        """Engage the Z feedback loop."""
        pass

    @abstractmethod
    def _hw_feedback_off(self) -> None:
        """Disengage the Z feedback loop."""
        pass

    @abstractmethod
    def _hw_is_feedback_on(self) -> bool:
        """Return True if the Z feedback loop is currently engaged."""
        pass

# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    SPM_FEEDBACK.run_server()