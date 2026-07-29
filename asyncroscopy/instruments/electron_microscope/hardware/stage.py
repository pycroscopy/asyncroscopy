"""
STAGE Tango device.

This device holds params for the scan.
It does NOT talk to AutoScript directly — the STEMMicroscope device
reads these attributes via DeviceProxy before acquiring.
"""

import tango

from tango import AttrWriteType, DevState
from tango.server import Device, attribute
from abc import abstractmethod

class STAGE(Device):
    """Stage/sample settings device.

    Public stage vectors are always [x, y, z, alpha, beta], with x/y/z in
    meters and alpha/beta in degrees.
    """

    # ------------------------------------------------------------------
    # Device properties — set per-deployment in the Tango DB
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    beta_tilt_enabled = attribute(
        label="Beta tilt enabled",
        dtype=bool,
        access=AttrWriteType.READ,
        doc="Whether the beta tilt is enabled on this stage",
    )

    position = attribute(
        label="Position",
        dtype=(float,),
        max_dim_x=5,
        access=AttrWriteType.READ_WRITE,
        unit="m, m, m, deg, deg",
        doc="Stage position [x, y, z, alpha, beta], with tilts in degrees",
    )

    position = attribute(
        label="Stage Position",
        dtype=(float,),
        max_dim_x=5,
        access=AttrWriteType.READ_WRITE,
        unit="m",
        # min_value= TODO: set these - AS-example -  specimen.stage.get_axis_limits
        # max_value= TODO: set these - AS-example -  specimen.stage.get_axis_limits
        doc="Stage X position in meters",
    )

    y = attribute(
        label="Position",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="m",
        # min_value= TODO: set these
        # max_value= TODO: set these
        doc="Stage Y position in meters",
    )

    z = attribute(
        label="Position",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="m",
        # min_value= TODO: set these
        # max_value= TODO: set these
        doc="Stage Z position in meters",
    )

    alpha = attribute(
        label="Alpha tilt",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="degrees",
        min_value = -35,
        max_value = 35,
        doc="Stage alpha tilt in degrees",
    )

    beta = attribute(
        label="Beta tilt",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="degrees",
        min_value = -15,
        max_value = 15,
        doc="Stage beta tilt in degrees",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.ON)

        # Start with zeros, TODO: get real numbers during initialization
        self._beta_tilt_enabled: bool = False
        self._x: float = 0.0
        self._y: float = 0.0
        self._z: float = 0.0
        self._alpha: float = 0.0
        self._beta: float = 0.0


        self.info_stream("STAGE device initialised")

    # Set up microscope proxy
    def _init_microscope_proxy(self):
        """Connect to the microscope on first use."""
        self._microscope = tango.DeviceProxy("asyncroscopy/microscope/default")


    # ------------------------------------------------------------------
    # Attribute read / write
    # ------------------------------------------------------------------

    def read_position(self) -> list[float]:
        return self._read_position()

    def write_position(self, value) -> None:
        self._write_position(value)

    def read_beta_tilt_enabled(self) -> bool:
        return self._read_beta_tilt_enabled()

    @abstractmethod
    def _read_position(self):
        pass

    @abstractmethod
    def _write_position(self, value):
        pass

    @abstractmethod
    def _read_beta_tilt_enabled(self):
        pass

    def read_x(self) -> float:
        return self._x

    def write_x(self, value: float) -> None:
        self._x = value

    def read_y(self) -> float:
        return self._y

    def write_y(self, value: float) -> None:
        self._y = value

    def read_z(self) -> float:
        return self._z

    def write_z(self, value: float) -> None:
        self._z = value

    def read_alpha(self) -> float:
        return self._alpha

    def write_alpha(self, value: float) -> None:
        self._alpha = value

    def read_beta(self) -> float:
        return self._beta

    def write_beta(self, value: float) -> None:
        self._beta = value

# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    STAGE.run_server()
