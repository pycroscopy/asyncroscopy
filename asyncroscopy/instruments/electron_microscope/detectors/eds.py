"""
EDS (Energy disperive X-ray spectroscopy) detector Tango device.

This device holds acquisition settings for the EDS detector.
It does NOT talk to AutoScript directly — the STEMMicroscope device
reads these attributes via DeviceProxy before acquiring.
"""

from tango import AttrWriteType, DevState
from tango.server import Device, attribute


class EDS(Device):
    """EDS detector settings device."""

    # ------------------------------------------------------------------
    # Device properties — set per-deployment in the Tango DB
    # ------------------------------------------------------------------

    # (no hardware connection properties needed — EDS is settings-only)

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    exposure_time = attribute(
        label="Dwell Time",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="s",
        format="%e",
        min_value=1e-6,
        max_value=1e2,
        doc="Exposure time in seconds (e.g. 1e-6 = 1 µs)",
    )


    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.ON)

        # Sensible defaults — operators override via Tango DB or client writes
        self._exposure_time: float = 1   # 1 s

        self.info_stream("EDS device initialised")

    # ------------------------------------------------------------------
    # Attribute read / write
    # ------------------------------------------------------------------

    def read_exposure_time(self) -> float:
        return self._exposure_time

    def write_exposure_time(self, value: float) -> None:
        self._exposure_time = value


# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    EDS.run_server()