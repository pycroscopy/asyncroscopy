"""
Asylum Research Jupiter AFM implementations.

Fills in the ``_hw_*`` hooks of the abstract SPM devices
(SPMMicroscope, SPM_SCAN, SPM_FEEDBACK, SPM_APPROACH, SPM_STAGE)
with calls to the gor Pro control software.

Only one command is added here, calibrate_probe_frame,
because the scan-frame to scanner offset is a Jupiter-specific measurement.
"""

from asyncroscopy.instruments.scanning_probe_microscope.scanning_probe_microscope import (
    SPMMicroscope, SPMMode,
)
from asyncroscopy.instruments.scanning_probe_microscope.hardware.spm_scan import SPM_SCAN
from asyncroscopy.instruments.scanning_probe_microscope.hardware.spm_feedback import SPM_FEEDBACK
from asyncroscopy.instruments.scanning_probe_microscope.hardware.spm_approach import SPM_APPROACH
from asyncroscopy.instruments.scanning_probe_microscope.hardware.spm_stage import SPM_STAGE

import math
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import tango
import tango.server

# AR variable names for the scan parameters, keyed by attribute name.
_SCAN_PARAM_KEYS: dict[str, str] = {
    "x_scan_center_m": "XOffset",
    "y_scan_center_m": "YOffset",
    "scan_size_m": "ScanSize",
    "scan_size_px": "ScanLines",
    "scan_angle_deg": "ScanAngle",
    "scan_rate_hz": "ScanRate",
}

_SCAN_WRITE_COMMANDS: dict[str, str] = {
    "x_scan_center_m": 'PV("XOffset", {value})',
    "y_scan_center_m": 'PV("YOffset", {value})',
    "scan_size_m": 'ARExecuteControl("ScanSizeSetVar_0","MasterPanel",{value},"")',
    "scan_size_px": 'ARExecuteControl("PointsLinesSetVar_0","MasterPanel",{value},"")',
    "scan_angle_deg": 'PV("ScanAngle", {value})',
    "scan_rate_hz": 'ARExecuteControl("ScanRateSetVar_0","MasterPanel",{value},"")',
}

# Park the tip at the centre of the current scan frame. Clearing the force-spot
# list first makes "go there" fall back to the frame centre.
_CLEAR_FORCE_COMMAND = 'ARExecuteControl("ClearForce_1","MasterPanel",0,"")'
_GO_TO_CENTER_COMMAND = 'ARExecuteControl("GoForce_1","MasterPanel",0,"")'
_START_SCAN_COMMAND = 'ARExecuteControl("DownScan_0","MasterPanel",0,"")'
_STOP_SCAN_COMMAND = 'ARExecuteControl("StopScan_0","MasterPanel",0,"")'


# Scan centre, LVDT sensitivities (meters per volt) and the frame rotation.
# All plain AR globals, so one round trip covers them.
_PROBE_KEYS = [
    "PIDSLoop.0.Setpoint", "PIDSLoop.1.Setpoint",  # closed-loop X/Y, volts
    "XLVDTSens", "YLVDTSens",                      # meters per volt
    "XOffset", "YOffset",                          # scan centre, meters
    "ScanAngle",                                   # degrees
]


try:
    from aespm import read_spm, write_spm
    from aespm.experiment import _read_out_buffer as _READ_OUT_BUFFER

    _AESPM_AVAILABLE = True
    _AESPM_IMPORT_ERROR = ""
except Exception as exc:
    _AESPM_AVAILABLE = False
    _AESPM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    _READ_OUT_BUFFER = ""

_IGOR_LOCK = threading.RLock()
_READ_OUT_BUFFER_IGOR = _READ_OUT_BUFFER.replace("\\", "\\\\")

# write_spm sleeps this long after handing the command to Igor, which gives the
# AR panels time to update before we read the value back.
_WRITE_SETTLE_S = 0.35
_MOVE_SETTLE_S = 1.
# How often to look in the save folder while a scan is running. Only touches the
# filesystem, never Igor, so it is cheap.
_POLL_INTERVAL_S = 1.0
# The AR software runs Igor Pro; the executable name has varied between
# versions, so we accept any of these.
_IGOR_PROCESS_NAMES = ("Igor.exe", "Igor64.exe", "IgorPro.exe")

#----------------------------------------------------------------------
#-------------------Auxilary methods--------------------------------
#----------------------------------------------------------------------

def _require_aespm(origin: str) -> None:
    """Raise a clear error if aespm could not be imported."""
    if not _AESPM_AVAILABLE:
        tango.Except.throw_exception(
            "AespmNotAvailable",
            "aespm could not be imported, so the Jupiter is unreachable: "
            f"{_AESPM_IMPORT_ERROR}. This device server must run on the Jupiter "
            "control PC with the Asylum Research software installed.",
            origin,
        )

def _require_igor(origin: str) -> None:
    """Raise a clear error if the Asylum Research software is not running.

    aespm talks to Igor by writing a command file and running SendToIgor.bat,
    which ends in Popen(...).wait(). If Igor is already up, the command is handed
    to the live instance and returns at once. If Igor is NOT up, that batch file
    starts the application and the wait never returns, so the device server hangs
    forever with no error message. Checking the process list first turns that
    hang into something readable.

    If the check itself cannot run we let the call through: a broken check must
    never stand in the way of a working instrument.
    """
    try:
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return
    if result.returncode != 0:
        return

    running = result.stdout.lower()
    if any(name.lower() in running for name in _IGOR_PROCESS_NAMES):
        return

    tango.Except.throw_exception(
        "IgorNotRunning",
        "Igor Pro (the Asylum Research software) does not appear to be running. "
        "Start it before using this device: aespm would otherwise block forever "
        "waiting for Igor to launch. Looked for "
        + ", ".join(_IGOR_PROCESS_NAMES)
        + " in the process list.",
        origin,
    )

def _read(keys: list[str]) -> list[float]:
    """Read AR global variables by name in a single Igor round trip.

    read_spm builds one Igor wave for the whole list, so batching is much cheaper
    than reading keys individually. np.atleast_1d guards the single-key case,
    where np.loadtxt returns a 0-d array.
    """

    _require_aespm("jupiter_api._read()")
    _require_igor("jupiter_api._read()")

    with _IGOR_LOCK:
        values = read_spm(key=list(keys), connection=None)
    values = [float(value) for value in np.atleast_1d(values)]

    unreadable = [key for key, value in zip(keys, values) if not math.isfinite(value)]
    if unreadable:
        tango.Except.throw_exception(
            "IgorValueUnreadable",
            "AR returned no finite value for: " + ", ".join(unreadable),
            "jupiter_api._read()",
        )
    return values

def _write(commands: str, settle_s: float = _WRITE_SETTLE_S) -> None:
    """Send one or more Igor commands (newline separated) to the AR software.

    aespm has no error channel: write_spm returns None whether Igor ran the
    command or rejected it. Callers must therefore read the value back and check
    it themselves - see SCAN_Jupiter._write_param.
    """
    _require_aespm("jupiter_api._write()")
    _require_igor("jupiter_api._write()")

    with _IGOR_LOCK:
        write_spm(commands=commands, connection=None, wait=settle_s)

def _first_text_wave_value(raw: str) -> str:
    """Pull the first string out of an Igor text-wave file (what Save/T writes).

    The file looks like:  IGOR / WAVES/T ReadOutText / BEGIN / "the value" / END
    so the value we want is simply the first quoted line.
    """
    for line in raw.splitlines():
        line = line.strip()
        if len(line) >= 2 and line.startswith('"') and line.endswith('"'):
            return line[1:-1]
    return ""


def _igor_path_to_windows(path: str) -> str:
    """Convert an Igor path to a Windows one.

    Igor separates folders with colons and adds a trailing one, so
    'C:Users:Asylum User:Data:' means 'C:\\Users\\Asylum User\\Data'.
    Stripping stray slashes first also copes with versions that already
    return a normal Windows path.
    """
    parts = [part.strip("\\/") for part in path.strip().split(":")]
    parts = [part for part in parts if part]
    if len(parts) < 2:
        return ""
    return parts[0] + ":\\" + "\\".join(parts[1:])


def _read_save_folder() -> Path:
    """Ask Igor where the AR software is currently saving images.

    'SaveImage' is the named path AR keeps for its save folder, and PathInfo puts
    it into the Igor variable S_path. We write that into a one-element text wave
    and save it to the same buffer file _read() uses for numbers.

    The buffer is cleared first for the same reason _read() checks for NaN: when
    an Igor command fails the old file contents stay put, so a stale answer would
    otherwise look like a fresh one.
    """
    _require_aespm("jupiter_api._read_save_folder()")
    _require_igor("jupiter_api._read_save_folder()")

    commands = (
        'PathInfo $"SaveImage"\n'
        "Make/T/O/N=1 ReadOutText\n"
        "ReadOutText[0] = S_path\n"
        f'Save/T/O ReadOutText as "{_READ_OUT_BUFFER_IGOR}"\n'
    )

    buffer = Path(_READ_OUT_BUFFER)
    with _IGOR_LOCK:
        buffer.write_text("", encoding="utf-8")
        write_spm(commands=commands, connection=None, wait=_MOVE_SETTLE_S)
        raw = buffer.read_text(encoding="utf-8", errors="replace")

    folder = _igor_path_to_windows(_first_text_wave_value(raw))
    if not folder:
        tango.Except.throw_exception(
            "SaveFolderUnknown",
            "Igor did not report a save folder. The named path 'SaveImage' is "
            "empty or undefined, which usually means image saving has never been "
            "switched on in the AR GUI.",
            "_read_save_folder()",
        )
    return Path(folder)

def _num(value: float) -> str:
    """Format a number for an Igor command, e.g. '1e-05' or '0.4992'."""
    return f"{float(value):.10g}"


def _is_same_value(requested, current) -> bool:
    """True if the requested value is numerically what we already have."""
    return math.isclose(float(requested), float(current), rel_tol=1e-9, abs_tol=0.0)



#----------------------------------------------------------------------
#------------------Jupiter Class--------------------------------
#----------------------------------------------------------------------

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

    def init_device(self) -> None:
        # Set before the base class runs, because it calls into hardware hooks.
        # Init() re-runs this, so Init() is also how you forget a stale
        # calibration after an LVDT sensitivity change or an AR restart.
        self._center_write_pending = False
        self._scanner_offset: tuple[float, float] | None = None
        super().init_device()

    #----------------------------------------------------------------------
    #-------------------Scan aquiring methods--------------------------------
    #----------------------------------------------------------------------
    def _scan_timeout_s(self) -> float:
        """How long to allow for one frame before giving up.

        A frame takes roughly lines / rate seconds. Double that and add a minute,
        so a slow start or a trace-and-retrace pass cannot trip the timeout.
        """
        rate, lines = self._scan_rate_hz, self._scan_size_px
        if not (math.isfinite(rate) and rate > 0 and lines > 0):
            return 600.0
        return 2.0 * lines / rate + 60.0

    def _wait_for_new_ibw(self, folder: Path, before: set[str], timeout_s: float) -> Path:
        """Wait for the AR software to write a new .ibw into the save folder.

        'before' is the set of filenames that were already there, so anything
        else that turns up is our frame. If more than one appears, take the
        newest by modification time.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            new = {path.name for path in folder.glob("*.ibw")} - before
            if new:
                return max((folder / name for name in new), key=lambda p: p.stat().st_mtime)
            time.sleep(_POLL_INTERVAL_S)

        tango.Except.throw_exception(
            "ScanTimedOut",
            f"No new .ibw appeared in {str(folder)!r} within {timeout_s:.0f} s. "
            "Check that image saving is switched on in the AR GUI.",
            "_hw_acquire_scan()",
        )

    # auxilary
    def _write_param(self, name: str, value) -> None:
        """Write one scan parameter and confirm the instrument actually took it.

        aespm cannot report a rejected command, so a wrong control name looks
        exactly like a successful write. The base class already re-reads every
        parameter after the write, so comparing before with after costs nothing:
        if we asked for a different value and nothing moved at all, Igor ignored
        us. A value the hardware merely rounds (257 -> 256 pixels) does move, so
        legitimate coercion does not trigger this.
        """
        before = getattr(self, f"_{name}")
        super()._write_param(name, value)
        after = getattr(self, f"_{name}")

        if after == before and not _is_same_value(value, before):
            tango.Except.throw_exception(
                "ScanParameterWriteIgnored",
                f"Asked to set {name} to {value}, but it is still {after}. "
                "The AR software probably rejected the command "
                f"'{_SCAN_WRITE_COMMANDS[name]}'.",
                "_write_param()",
            )

    def _hw_read_scan_params(self) -> dict:
        """Read all scan parameters from AR; keys match the attribute names."""
        names = list(_SCAN_PARAM_KEYS)
        values = _read([_SCAN_PARAM_KEYS[name] for name in names])
        params = dict(zip(names, values))
        params["scan_size_px"] = int(round(params["scan_size_px"]))
        return params

    def _hw_write_scan_param(self, name: str, value) -> None:
        """Push one scan parameter to AR. The caller re-reads and checks it."""
        command = _SCAN_WRITE_COMMANDS.get(name)
        if command is None:
            tango.Except.throw_exception(
                "UnknownScanParameter",
                f"No Igor command is defined for '{name}'.",
                "_hw_write_scan_param()",
            )

        # Pixels go into an integer control; everything else is a float.
        if name == "scan_size_px":
            text = str(int(round(float(value))))
        else:
            text = _num(value)

        if name in ("x_scan_center_m", "y_scan_center_m"):
            # AR takes the new offset immediately, but the scanner only adopts it
            # on the next scan, so "go there" would still park at the old centre.
            self._center_write_pending = True

        _write(command.format(value=text))

    def _hw_acquire_scan(self) -> str:
        """Run one frame with current settings; return the file AR wrote.

        The save folder is read from Igor every time rather than configured, so
        it always matches what the AR GUI is actually doing.
        """
        folder = _read_save_folder()
        before = {path.name for path in folder.glob("*.ibw")}

        _write(_START_SCAN_COMMAND)
        path = self._wait_for_new_ibw(folder, before, self._scan_timeout_s())

        # Running a scan is what makes a pending scan centre real, so the
        # calibration guard can be cleared now.
        self._center_write_pending = False
        self.info_stream(f"Scan saved as {path}")
        return str(path)

    def _hw_stop_scan(self) -> None:
        """Abort the running scan immediately."""
        _write(_STOP_SCAN_COMMAND)

    # ------------------------------------------------------------------
    # Probe positioning
    #
    # The probe is positioned by the closed-loop X/Y setpoints, which are
    # voltages in the scanner's own frame:
    #
    #     x_scanner = setpoint_volts * XLVDTSens              [meters]
    #
    # That frame is shifted from the scan frame by a constant the instrument
    # does not report:
    #
    #     x_scanner = x_scan + scanner_offset_x
    # ------------------------------------------------------------------
    def _read_probe(self) -> tuple[float, float, float, float, float, float]:
        """One round trip: setpoints (V), sensitivities (m/V), scan centre (m).

        Refuses a rotated frame - the closed loops drive the scanner axes, not
        the rotated scan frame.
        """
        vx, vy, sx, sy, xo, yo, angle = _read(_PROBE_KEYS)
        if sx == 0.0 or sy == 0.0:
            tango.Except.throw_exception(
                "SensitivityUnavailable",
                f"AR reported an LVDT sensitivity of zero (x={sx}, y={sy}).",
                "_read_probe()",
            )
        if abs(angle) > 1e-6:
            tango.Except.throw_exception(
                "ScanFrameRotated",
                f"ScanAngle is {angle:g} deg; probe moves require ScanAngle = 0.",
                "_read_probe()",
            )
        return vx, vy, sx, sy, xo, yo

    def _measure_scanner_offset(
        self, sx: float, sy: float, xo: float, yo: float
    ) -> tuple[float, float]:
        """Park the tip at the scan centre and measure the frame offset.

        This physically moves the tip. The centre is the one point whose
        scan-frame coordinate we already know, so it is the only place the
        offset can be measured.
        """

        if self._center_write_pending:
            tango.Except.throw_exception(
                "ScanCenterNotApplied",
                "The scan centre was changed and no scan has run since, so the "
                "scanner is still using the old centre and the measured offset "
                "would be wrong by the difference. Run a scan first, or calibrate "
                "before moving the frame. Init() clears this.",
                "_measure_scanner_offset()",
            )

        _write(_CLEAR_FORCE_COMMAND)
        _write(_GO_TO_CENTER_COMMAND, settle_s=_MOVE_SETTLE_S)
        time.sleep(_MOVE_SETTLE_S)  # let the tip arrive before reading

        vx, vy = _read(_PROBE_KEYS)[:2]
        offset = (vx * sx - xo, vy * sy - yo)
        self.info_stream(f"Scanner offset measured: {offset[0]:e}, {offset[1]:e} m")
        return offset

    @tango.server.command(dtype_out=tango.DevVarDoubleArray)
    def calibrate_probe_frame(self) -> list[float]:
        """Measure the scan-frame to scanner offset. Parks the tip at the centre.

        Returns [scanner_offset_x_m, scanner_offset_y_m].
        """
        _, _, sx, sy, xo, yo = self._read_probe()
        self._scanner_offset = self._measure_scanner_offset(sx, sy, xo, yo)
        return list(self._scanner_offset)

    def _hw_read_probe_position(self) -> list[float]:
        """Return the probe position [x, y] in scan-frame meters."""
        if self._scanner_offset is None:
            tango.Except.throw_exception(
                "ProbeFrameNotMeasured",
                "The scan-frame to scanner offset is not known yet. Call "
                "calibrate_probe_frame(), or move the probe once - the first move "
                "measures it by parking the tip at the frame centre.",
                "_hw_read_probe_position()",
            )
        vx, vy, sx, sy, _, _ = self._read_probe()
        offset_x, offset_y = self._scanner_offset
        return [vx * sx - offset_x, vy * sy - offset_y]

    def _hw_move_probe(self, x: float, y: float) -> None:
        """Move the probe to (x, y) in scan-frame meters.

        The first move parks at the frame centre to measure the offset, then
        continues to the target. Needs the X/Y PID loops engaged.
        """
        _, _, sx, sy, xo, yo = self._read_probe()
        if self._scanner_offset is None:
            self._scanner_offset = self._measure_scanner_offset(sx, sy, xo, yo)
        offset_x, offset_y = self._scanner_offset

        _write(
            f'td_WriteValue("PIDSLoop.0.Setpoint",{_num((x + offset_x) / sx)})\n'
            f'td_WriteValue("PIDSLoop.1.Setpoint",{_num((y + offset_y) / sy)})\n',
            settle_s=_MOVE_SETTLE_S,
        )


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