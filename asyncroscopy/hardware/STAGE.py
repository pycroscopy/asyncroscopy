"""
STAGE Tango device.

This device holds params for the scan.
It does NOT talk to AutoScript directly — the Microscope device
reads these attributes via DeviceProxy before acquiring.
"""

import tango

from tango import AttrWriteType, DevState
from tango.server import Device, attribute


class STAGE(Device):
    """Stage/Sample settings device."""

    # ------------------------------------------------------------------
    # Device properties — set per-deployment in the Tango DB
    # ------------------------------------------------------------------

    # (no hardware connection properties needed — STAGE is settings-only)

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

   
    beta_tilt_enabled = attribute(
        label="Beta Tilt Enabled",
        dtype=bool,
        access=AttrWriteType.READ_WRITE,
        doc="Whether the holder supports beta tilt)",
    )

    position = attribute(
        label="Stage Position",
        dtype=(float,),
        max_dim_x=5,
        access=AttrWriteType.READ_WRITE,
        doc="Stage position as [x, y, z, alpha, beta]",
    )
    

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.ON)
        self._position = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._microscope = None
        self.info_stream("STAGE device initialised")

    # Set up microscope proxy
    def _init_microscope_proxy(self):
        """Connect to the microscope on first use."""
        self._microscope = tango.DeviceProxy("asyncroscopy/microscope/default")


    # ------------------------------------------------------------------
    # Attribute read / write
    # ------------------------------------------------------------------

    def read_beta_tilt_enabled(self) -> bool:
        return self._beta_tilt_enabled

    def write_beta_tilt_enabled(self, value: bool) -> None:
        self._beta_tilt_enabled = value

    def read_position(self):
        return tuple(self._position)

    def write_position(self, value):
        try:
            result = self._microscope.set_stage_position(value)
        except:
            self._init_microscope_proxy()
            result = self._microscope.set_stage_position(value)
            self._position = list(result)
            

# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    STAGE.run_server()
