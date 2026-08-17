"""
Asylum Research Jupiter AFM implementations.

Fills in the ``_hw_*`` hooks of the abstract SPM devices
(SPMMicroscope, SPM_SCAN, SPM_FEEDBACK, SPM_APPROACH, SPM_STAGE)
with calls to the gor Pro control software.

No Tango attributes or commands are (re)defined here — the public
interface lives entirely in the abstract base classes.
"""

from asyncroscopy.instruments.scanning_probe_microscope.scanning_probe_microscope import (
    SPMMicroscope, SPMMode,
)
from asyncroscopy.instruments.scanning_probe_microscope.hardware.spm_scan import SPM_SCAN
from asyncroscopy.instruments.scanning_probe_microscope.hardware.spm_feedback import SPM_FEEDBACK
from asyncroscopy.instruments.scanning_probe_microscope.hardware.spm_approach import SPM_APPROACH
from asyncroscopy.instruments.scanning_probe_microscope.hardware.spm_stage import SPM_STAGE


class JupiterMicroscope(SPMMicroscope):
    """Top-level Jupiter AFM device: vendor connection and instrument-global state."""

    def _connect_hardware(self) -> None:
        """Open the connection to the AR control software; raise on failure."""
        ...

    def _hw_get_spm_mode(self) -> SPMMode:
        """Map the active AR imaging mode to the SPMMode enum."""
        ...

    def _hw_get_meter_values(self) -> dict:
        """Return live photodetector signals as {'sum', 'deflection', 'lateral', 'z'}, in volts."""
        ...


class SCAN_Jupiter(SPM_SCAN):
    """Jupiter scan device: XY piezo frame parameters, scan execution, probe positioning."""

    def _hw_read_scan_params(self) -> dict:
        """Read all scan parameters from AR; keys must match the attribute names
        (x_scan_center_m, y_scan_center_m, scan_size_m, scan_size_px,
        scan_angle_deg, scan_rate_hz)."""
        ...

    def _hw_write_scan_param(self, name: str, value) -> None:
        """Push one scan parameter to AR (name as in _hw_read_scan_params);
        hardware may coerce/couple values — caller re-reads afterwards."""
        ...

    def _hw_acquire_scan(self) -> tuple[list[str], list]:
        """Run one frame with current settings; block until complete.
        Return (channel_names, list of 2D numpy arrays, same length/order)."""
        ...

    def _hw_stop_scan(self) -> None:
        """Abort the running scan immediately."""
        ...

    def _hw_read_probe_position(self) -> list[float]:
        """Return the current probe position [x, y] in meters within the scan frame."""
        ...

    def _hw_move_probe(self, x: float, y: float) -> None:
        """Move the probe to (x, y) in meters; return when the move is complete."""
        ...


class FEEDBACK_Jupiter(SPM_FEEDBACK):
    """Jupiter Z-feedback device: setpoint, gain, engage/disengage."""

    def _hw_read_feedback_params(self) -> dict:
        """Read all feedback parameters from AR; keys must match the attribute
        names (setpoint, i_gain)."""
        ...

    def _hw_write_feedback_param(self, name: str, value) -> None:
        """Push one feedback parameter to AR (name as in _hw_read_feedback_params)."""
        ...

    def _hw_feedback_on(self) -> None:
        """Engage the Z feedback loop."""
        ...

    def _hw_feedback_off(self) -> None:
        """Disengage the Z feedback loop."""
        ...

    def _hw_is_feedback_on(self) -> bool:
        """Return True if the Z feedback loop is currently engaged (read live)."""
        ...


class APPROACH_Jupiter(SPM_APPROACH):
    """Jupiter approach device: tip engage / retract sequence."""

    def _hw_approach(self) -> None:
        """Run the AR approach sequence; block until the tip is engaged."""
        ...

    def _hw_retract(self) -> None:
        """Retract the tip; block until clear of the surface."""
        ...

    def _hw_stop(self) -> None:
        """Abort any running approach/retract motion immediately."""
        ...

    def _hw_is_approached(self) -> bool:
        """Return True if the tip is currently engaged (read live)."""
        ...


class STAGE_Jupiter(SPM_STAGE):
    """Jupiter coarse XY stage device."""

    def _hw_read_stage_position(self) -> list[float]:
        """Return the current stage position [x, y] in meters."""
        ...

    def _hw_move_stage_relative(self, dx: float, dy: float) -> None:
        """Move the stage by (dx, dy) in meters; block until the move completes."""
        ...

    def _hw_stop(self) -> None:
        """Abort any running stage motion immediately."""
        ...
