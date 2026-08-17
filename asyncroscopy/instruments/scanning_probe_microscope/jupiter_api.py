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

import threading

import numpy as np
import tango
import tango.server


try:
    from aespm import read_spm

    _AESPM_AVAILABLE = True
    _AESPM_IMPORT_ERROR = ""
except Exception as exc:
    _AESPM_AVAILABLE = False
    _AESPM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

_IGOR_LOCK = threading.RLock()

def _read(keys: list[str]) -> list[float]:
    """Read AR global variables by name in a single Igor round trip.

    read_spm builds one Igor wave for the whole list, so batching is much cheaper
    than reading keys individually. np.atleast_1d guards the single-key case,
    where np.loadtxt returns a 0-d array.
    """
    if not _AESPM_AVAILABLE:
        tango.Except.throw_exception(
            "AespmNotAvailable",
            "aespm could not be imported, so the Jupiter is unreachable: "
            f"{_AESPM_IMPORT_ERROR}. This device server must run on the Jupiter "
            "control PC with the Asylum Research software installed.",
            "jupiter_api._read()",
        )
    with _IGOR_LOCK:
        values = read_spm(key=list(keys), connection=None)
    return [float(value) for value in np.atleast_1d(values)]

_SCAN_PARAM_KEYS: dict[str, str] = {
    "x_scan_center_m": "XOffset",
    "y_scan_center_m": "YOffset",
    "scan_size_m": "ScanSize",
    "scan_size_px": "ScanLines",
    "scan_angle_deg": "ScanAngle",
    "scan_rate_hz": "ScanRate",
}

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
        names = list(_SCAN_PARAM_KEYS)
        values = _read([_SCAN_PARAM_KEYS[name] for name in names])
        params = dict(zip(names, values))

        # A wrong key name can leave GV() failing inside Igor while readout.txt
        # keeps its previous contents, so bad names surface as NaN or as stale
        # values rather than as an error. NaN is the case we can actually catch.
        invalid = [name for name, value in params.items() if not np.isfinite(value)]
        if invalid:
            tango.Except.throw_exception(
                "ScanParameterUnreadable",
                "AR returned no finite value for: "
                + ", ".join(f"{name} (GV '{_SCAN_PARAM_KEYS[name]}')" for name in invalid),
                "_hw_read_scan_params()",
            )

        params["scan_size_px"] = int(round(params["scan_size_px"]))
        return params

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

# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------
# run_servers.py starts one process per device with
# `python -m ...jupiter_api <key>_instance`, and Device.run_server() uses the
# class name as the Tango server name, so the class is selected here from the
# instance name the process was launched with.
_DEVICE_CLASSES = {
    "instrument": JupiterMicroscope,
    "scan": SCAN_Jupiter,
    "feedback": FEEDBACK_Jupiter,
    "approach": APPROACH_Jupiter,
    "stage": STAGE_Jupiter,
}

if __name__ == "__main__":
    import sys

    instance = sys.argv[1] if len(sys.argv) > 1 else ""
    key = instance.rsplit("_instance", 1)[0]
    device_class = _DEVICE_CLASSES.get(key)
    if device_class is None:
        raise SystemExit(
            f"Cannot pick a device class from instance name {instance!r}. "
            f"Expected one of: {', '.join(f'{k}_instance' for k in _DEVICE_CLASSES)}"
        )
    device_class.run_server()