"""
SPM SCAN Tango device.

Owns the scan frame parameters AND scan execution. Vendor-specific
behaviour is isolated in the ``_hw_*`` hooks implemented by concrete
subclasses (SCAN_Jupiter, etc.).

Writes are pushed to hardware and read back.

``acquire_scan`` returns a DATA/Tiled unique id.
"""

from abc import abstractmethod

import tango #type: ignore

from asyncroscopy.data.data_writer import save_acquisition
from asyncroscopy.instruments.instrument import CombinedMeta

class SPM_SCAN(tango.server.Device, metaclass=CombinedMeta):
    """Abstract SPM scan device: frame settings + scan execution."""

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    data_device_address = tango.server.device_property(
        dtype=str,
        default_value="",
        doc="Optional Tango device address for the DATA device, "
            "e.g. 'asyncroscopy/data/default'.",
    )

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    x_scan_center_m = tango.server.attribute(
        label="Scan Center X",
        dtype=float,
        access=tango.AttrWriteType.READ_WRITE,
        unit="m",
        format="%e",
        doc="X center of the scan frame in meters.",
    )

    y_scan_center_m = tango.server.attribute(
        label="Scan Center Y",
        dtype=float,
        access=tango.AttrWriteType.READ_WRITE,
        unit="m",
        format="%e",
        doc="Y center of the scan frame in meters.",
    )

    scan_size_m = tango.server.attribute(
        label="Scan Size (m)",
        dtype=float,
        access=tango.AttrWriteType.READ_WRITE,
        unit="m",
        format="%e",
        doc="Side length of the scan frame in meters.",
    )

    scan_size_px = tango.server.attribute(
        label="Scan Size (px)",
        dtype=int,
        access=tango.AttrWriteType.READ_WRITE,
        unit="px",
        doc="Number of pixels per side of the scan frame.",
    )

    scan_angle_deg = tango.server.attribute(
        label="Scan Angle",
        dtype=float,
        access=tango.AttrWriteType.READ_WRITE,
        unit="deg",
        format="%6.2f",
        doc="Rotation angle of the scan frame in degrees.",
    )

    scan_rate_hz = tango.server.attribute(
        label="Scan Rate",
        dtype=float,
        access=tango.AttrWriteType.READ_WRITE,
        unit="Hz",
        format="%6.2f",
        doc="Line rate in Hz.",
    )

    probe_x_m = tango.server.attribute(
        label="Probe X Position", 
        dtype=float,
        access=tango.AttrWriteType.READ_WRITE,
        unit="m",
        format="%e",
        doc="Current probe X position in meters.",
    )

    probe_y_m = tango.server.attribute(
        label="Probe Y Position",
        dtype=float,
        access=tango.AttrWriteType.READ_WRITE,
        unit="m",
        format="%e",
        doc="Current probe Y position in meters.",
    )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        tango.server.Device.init_device(self)
        self._data_proxy = (
            tango.DeviceProxy(self.data_device_address)
            if self.data_device_address else None
        )
        self._refresh_params()
        self.set_state(tango.DevState.ON)
        self.info_stream("SPM SCAN device initialised")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_params(self) -> None:
        """Re-read all scan parameters from hardware into local fields."""
        params = self._hw_read_scan_params()
        self._x_scan_center_m: float = params["x_scan_center_m"]
        self._y_scan_center_m: float = params["y_scan_center_m"]
        self._scan_size_m: float = params["scan_size_m"]
        self._scan_size_px: int = params["scan_size_px"]
        self._scan_angle_deg: float = params["scan_angle_deg"]
        self._scan_rate_hz: float = params["scan_rate_hz"]

    def _write_param(self, name: str, value) -> None:
        """Push one parameter to hardware, then re-read all (hardware may coerce/couple values)."""
        self._hw_write_scan_param(name, value)
        self._refresh_params()

    def _scan_metadata(self) -> dict:
        """Current scan parameters, saved as dataset attributes."""
        return {
            "x_scan_center_m": self._x_scan_center_m,
            "y_scan_center_m": self._y_scan_center_m,
            "scan_size_m": self._scan_size_m,
            "scan_size_px": self._scan_size_px,
            "scan_angle_deg": self._scan_angle_deg,
            "scan_rate_hz": self._scan_rate_hz,
        }
    
    def _move_probe(self, x: float, y: float) -> None:
        """Run one probe move with state bookkeeping; refuses during a scan."""
        state = self.get_state()
        if state == tango.DevState.RUNNING:
            tango.Except.throw_exception(
                "ScanInProgress",
                "Cannot move probe while a scan is running; call stop_scan first.",
                "move_probe()",
            )
        if state == tango.DevState.MOVING:
            tango.Except.throw_exception(
                "MoveInProgress",
                "A probe move is already running.",
                "move_probe()",
            )
        self.set_state(tango.DevState.MOVING)
        try:
            self._hw_move_probe(x, y)
        finally:
            self.set_state(tango.DevState.ON)

    # ------------------------------------------------------------------
    # Attribute read / write, writes pushed to hardware
    # ------------------------------------------------------------------

    def read_x_scan_center_m(self) -> float:
        return self._x_scan_center_m

    def write_x_scan_center_m(self, value: float) -> None:
        self._write_param("x_scan_center_m", value)

    def read_y_scan_center_m(self) -> float:
        return self._y_scan_center_m

    def write_y_scan_center_m(self, value: float) -> None:
        self._write_param("y_scan_center_m", value)

    def read_scan_size_m(self) -> float:
        return self._scan_size_m

    def write_scan_size_m(self, value: float) -> None:
        self._write_param("scan_size_m", value)

    def read_scan_size_px(self) -> int:
        return self._scan_size_px

    def write_scan_size_px(self, value: int) -> None:
        self._write_param("scan_size_px", value)

    def read_scan_angle_deg(self) -> float:
        return self._scan_angle_deg

    def write_scan_angle_deg(self, value: float) -> None:
        self._write_param("scan_angle_deg", value)

    def read_scan_rate_hz(self) -> float:
        return self._scan_rate_hz

    def write_scan_rate_hz(self, value: float) -> None:
        self._write_param("scan_rate_hz", value)

    def read_probe_x_m(self) -> float:
        return self._hw_read_probe_position()[0]

    def write_probe_x_m(self, value: float) -> None:
        y = self._hw_read_probe_position()[1]
        self._move_probe(value, y)

    def read_probe_y_m(self) -> float:
        return self._hw_read_probe_position()[1]

    def write_probe_y_m(self, value: float) -> None:
        x = self._hw_read_probe_position()[0]
        self._move_probe(x, value)

    

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @tango.server.command(dtype_out=str)
    def acquire_scan(self) -> str:
        """Acquire one frame with current settings; returns a DATA/Tiled key."""
        if self.get_state() == tango.DevState.RUNNING:
            tango.Except.throw_exception(
                "ScanInProgress",
                "A scan is already running",
                "acquire_scan()",
            )
        self.set_state(tango.DevState.RUNNING)
        try:
            channel_names, data = self._hw_acquire_scan()
        finally:
            self.set_state(tango.DevState.ON)
        return save_acquisition(
            self, self._data_proxy, "spm_scan", channel_names, data,
            dataset_attrs=[self._scan_metadata()] * len(channel_names),
        )
    
    @tango.server.command
    def stop_scan(self) -> None:
        """Stop a running scan."""
        self._hw_stop_scan()
        self.set_state(tango.DevState.ON)

    @tango.server.command
    def refresh_params(self) -> None:
        """Re-read all scan parameters from hardware."""
        self._refresh_params()

    @tango.server.command(dtype_in=tango.DevVarDoubleArray)
    def move_probe(self, position) -> None:
        """Move the probe to an absolute position [x, y] in meters."""
        if len(position) != 2:
            raise ValueError("position must contain exactly two values: [x, y]")
        self._move_probe(float(position[0]), float(position[1]))

    # ------------------------------------------------------------------
    # Abstract methods — vendor-specific
    # ------------------------------------------------------------------

    @abstractmethod
    def _hw_read_scan_params(self) -> dict:
        """Read all scan parameters from hardware. Keys must match attribute names."""
        pass

    @abstractmethod
    def _hw_write_scan_param(self, name: str, value) -> None:
        """Push one scan parameter to hardware."""
        pass

    @abstractmethod
    def _hw_acquire_scan(self) -> tuple[list[str], list]:
        """Run one scan; return (channel_names, list of 2D numpy arrays)."""
        pass

    @abstractmethod
    def _hw_stop_scan(self) -> None:
        """Abort the running scan."""
        pass

    @abstractmethod
    def _hw_read_probe_position(self) -> list[float]:
        """Return the current probe position [x, y] in meters."""
        pass

    @abstractmethod
    def _hw_move_probe(self, x: float, y: float) -> None:
        """Move the probe to (x, y) in meters; return when done."""
        pass

# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    SPM_SCAN.run_server()