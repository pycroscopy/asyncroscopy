"""
HAADF (High-Angle Annular Dark-Field) detector Tango device.

This device holds acquisition settings for the HAADF detector.
It does NOT talk to AutoScript directly — the STEMMicroscope device
reads these attributes via DeviceProxy before acquiring.
"""

from tango import AttrWriteType, DevState
from tango.server import Device, attribute


class CAMERA(Device):
    """CAMERA detector settings device."""

    # ------------------------------------------------------------------
    # Device properties — set per-deployment in the Tango DB
    # ------------------------------------------------------------------

    # (no hardware connection properties needed — CAMERA is settings-only)

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    exposure_time = attribute(
        label="Exposure Time",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="s",
        format="%e",
        min_value=1e-7,
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

    readout_area = attribute(
        label="Readout Area",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        doc="Camera readout area preset (e.g. 'Full', 'Half', 'Quarter' — should match an AutoScript ReadoutArea preset)",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.ON)

        # Sensible defaults — operators override via Tango DB or client writes
        self._exposure_time: float = 1e-3   # 1 ms
        self._imsize: int = 1024
        self._readout_area: str = "Full"

        self.info_stream("CAMERA device initialised")

    # ------------------------------------------------------------------
    # Attribute read / write
    # ------------------------------------------------------------------

    def read_exposure_time(self) -> float:
        return self._exposure_time

    def write_exposure_time(self, value: float) -> None:
        self._exposure_time = value

    def read_imsize(self) -> int:
        return self._imsize

    def write_imsize(self, value: int) -> None:
        self._imsize = value

    def read_readout_area(self) -> str:
        return self._readout_area

    def write_readout_area(self, value: str) -> None:
        self._readout_area = value


# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    CAMERA.run_server()