"""
SCAN hardware settings.
This device holds scan acquisition settings.
It does NOT talk to AutoScript directly — the STEMMicroscope device
reads these attributes via DeviceProxy before acquiring.
"""
from tango import AttrWriteType, DevState
from tango.server import Device, attribute


class SCAN(Device):
    """SCAN detector settings device."""

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    dwell_time = attribute(
        label="Dwell Time",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="s",
        format="%e",
        min_value=1e-9,
        max_value=10,
        doc="Per-pixel dwell time in seconds (e.g. 1e-6 = 1 µs)",
    )

    imsize = attribute(
        label="Image Size",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        unit="px",
        doc="Acquisition width in pixels (should match an AutoScript ImageSize preset)",
    )

    scan_region = attribute(
        label="Scan Region",
        dtype=(float,),
        max_dim_x=4,
        access=AttrWriteType.READ_WRITE,
        unit="fractional",
        min_value=0.0,
        max_value=1.0,
        doc="Relative scan rectangle [left, top, width, height] in the range [0, 1]",
    )

    output_format = attribute(
        label="Output Format",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        doc="Output format for acquired images: '.h5' (default, one file with sub-groups per detector) or '.tiff' (one Velox-compatible TIFF per detector, metadata embedded by AutoScript)",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.ON)
        self._dwell_time: float = 1e-6
        self._imsize: int = 512
        self._scan_region: list[float] = [0.0, 0.0, 1.0, 1.0]
        self._output_format: str = ".h5"
        self.info_stream("SCAN device initialised")

    # ------------------------------------------------------------------
    # Attribute read / write
    # ------------------------------------------------------------------

    def read_dwell_time(self) -> float:
        return self._dwell_time

    def write_dwell_time(self, value: float) -> None:
        self._dwell_time = value

    def read_imsize(self) -> int:
        return self._imsize

    def write_imsize(self, value: int) -> None:
        self._imsize = value

    def read_scan_region(self) -> list[float]:
        return self._scan_region

    def write_scan_region(self, value) -> None:
        self._scan_region = self._validate_scan_region(value)

    def read_output_format(self) -> str:
        return self._output_format

    def write_output_format(self, value: str) -> None:
        self._output_format = value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_scan_region(value) -> list[float]:
        region = [float(item) for item in value]
        if len(region) != 4:
            raise ValueError(
                "scan_region must contain exactly four values: [left, top, width, height]"
            )

        left, top, width, height = region
        if left < 0.0 or top < 0.0:
            raise ValueError("scan_region left and top must be >= 0")
        if width <= 0.0 or height <= 0.0:
            raise ValueError("scan_region width and height must be > 0")
        if left + width > 1.0 or top + height > 1.0:
            raise ValueError("scan_region must fit within the relative [0, 1] scan area")
        return region

# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    SCAN.run_server()
