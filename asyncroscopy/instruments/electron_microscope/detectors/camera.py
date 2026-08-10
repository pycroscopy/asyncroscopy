"""Camera acquisition settings Tango device.

This device does not talk to AutoScript directly. The microscope device reads
these attributes through a DeviceProxy when acquiring a camera image.
"""

from tango import AttrWriteType, DevState
from tango.server import Device, attribute


class CAMERA(Device):
    """CAMERA detector settings device."""

    _CAMERA_DETECTORS = {
        "Flucam",
        "BM-Ceta",
        "EF-Ceta",
        "BM-Falcon",
        "EF-Falcon",
        "BM-Empad",
        "SH-Empad",
        "EF-CCD",
        "EF-Empad",
    }
    _READOUT_AREAS = {"Full", "Half", "Quarter"}
    _OUTPUT_FORMATS = {".h5", ".tiff"}

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
        doc="Camera exposure time in seconds.",
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
        doc="Camera readout area preset: 'Full', 'Half', or 'Quarter'.",
    )

    camera_detector = attribute(
        label="Camera Detector",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        doc="AutoScript camera detector name, e.g. 'BM-Ceta' or 'Flucam'.",
    )

    frame_combining = attribute(
        label="Frame Combining",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        min_value=1,
        doc="Number of sub-frames combined by the camera (Ceta-specific).",
    )

    electron_counting = attribute(
        label="Electron Counting",
        dtype=bool,
        access=AttrWriteType.READ_WRITE,
        doc="Produce an electron-counted image on supported counting detectors.",
    )

    output_format = attribute(
        label="Output Format",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        doc="Saved image format: '.h5' or '.tiff'.",
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
        self._camera_detector: str = "BM-Ceta"
        self._frame_combining: int = 1
        self._electron_counting: bool = True
        self._output_format: str = ".h5"

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
        if value not in self._READOUT_AREAS:
            raise ValueError(
                f"Unsupported readout_area {value!r}; expected one of "
                f"{sorted(self._READOUT_AREAS)}"
            )
        self._readout_area = value

    def read_camera_detector(self) -> str:
        return self._camera_detector

    def write_camera_detector(self, value: str) -> None:
        if value not in self._CAMERA_DETECTORS:
            raise ValueError(
                f"Unsupported camera_detector {value!r}; expected one of "
                f"{sorted(self._CAMERA_DETECTORS)}"
            )
        self._camera_detector = value

    def read_frame_combining(self) -> int:
        return self._frame_combining

    def write_frame_combining(self, value: int) -> None:
        if value < 1:
            raise ValueError("frame_combining must be at least 1")
        self._frame_combining = value

    def read_electron_counting(self) -> bool:
        return self._electron_counting

    def write_electron_counting(self, value: bool) -> None:
        self._electron_counting = value

    def read_output_format(self) -> str:
        return self._output_format

    def write_output_format(self, value: str) -> None:
        if value not in self._OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported output_format {value!r}; expected '.h5' or '.tiff'"
            )
        self._output_format = value


# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    CAMERA.run_server()
