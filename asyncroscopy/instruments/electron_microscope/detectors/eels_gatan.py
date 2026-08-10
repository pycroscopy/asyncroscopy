"""
Gatan-backed EELS Tango device.

The base class owns the public Tango API; this adapter owns the Pyro
connection and all communication with the Gatan DigitalMicrograph server.
"""

import numpy as np
from Pyro5.api import Proxy  # communication with Gatan server
from tango import DevState
from tango.server import device_property

from asyncroscopy.data.data_writer import save_acquisition
from asyncroscopy.instruments.electron_microscope.detectors.eels import EELSBase


class EELS(EELSBase):
    """Gatan DigitalMicrograph-backed EELS detector."""

    # ------------------------------------------------------------------
    # Device properties — set per-deployment in the Tango DB
    # ------------------------------------------------------------------

    hardware_host = device_property(
        dtype=str,
        default_value="10.46.217.242",
        doc="Hostname or IP of the Gatan server",
    )

    hardware_port = device_property(
        dtype=int,
        default_value=9091,
        doc="Port of the Gatan Pyro server",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        super().init_device()
        self._eels_uri = (
            f"PYRO:eels_server@{self.hardware_host}:{self.hardware_port}"
        )
        self._eels_proxy_factory = Proxy

        try:
            if not self._call_eels("check_server"):
                raise ConnectionError("EELS server did not respond")
            self.info_stream(
                f"Connected to Gatan EELS at {self.hardware_host}:{self.hardware_port}"
            )
        except Exception as exc:
            self.set_state(DevState.FAULT)
            self.error_stream(f"Gatan EELS connection failed: {exc}")

    def _call_eels(self, method_name: str, *args):
        """Call EELS through a proxy owned by the current Tango worker thread.

        Pyro proxies are thread-affine. Tango can initialize a device and execute
        its attributes or commands on different worker threads, so retaining one
        proxy on the device eventually raises Pyro's "calling thread is not the
        owner" error.
        """
        proxy = self._eels_proxy_factory(self._eels_uri)
        try:
            return getattr(proxy, method_name)(*args)
        finally:
            proxy._pyroRelease()

    # ------------------------------------------------------------------
    # Gatan hardware adapter
    # ------------------------------------------------------------------

    def _initialize_eels(self) -> bool:
        result = self._call_eels("initialize_eels")
        return result is not False

    def _write_offset(self, offset: float) -> None:
        self._call_eels("set_eels_offset", float(offset))

    def _acquire_eels_spectrum(self) -> str:
        spectrum, start_energy, dispersion = self._call_eels(
            "get_eels_spectrum",
            self.read_exposure_time(),
            self.read_number_of_frames(),
        )
        self._offset = float(start_energy)
        metadata = {
            "offset": float(start_energy),
            "start_energy": float(start_energy),
            "dispersion": float(dispersion),
            "energy_units": "eV",
            "dispersion_units": "eV/channel",
            "exposure_time": self.read_exposure_time(),
            "number_of_frames": self.read_number_of_frames(),
        }
        return save_acquisition(
            self,
            self._get_data_proxy(),
            "spectrum",
            "eels",
            np.asarray(spectrum),
            dataset_name="spectrum",
            dataset_attrs=metadata,
        )

    def _get_available_dispersions(self) -> str:
        return self._call_eels("get_available_dispersions")

    def _get_eels_dispersion(self) -> list[float]:
        dispersion_index, dispersion = self._call_eels("get_eels_dispersion")
        return [float(dispersion_index), float(dispersion)]

    def _set_eels_dispersion(self, dispersion_index: int) -> None:
        self._call_eels("set_eels_dispersion", int(dispersion_index))

    def _get_eels_aperture(self) -> str:
        return self._call_eels("get_eels_aperture")

    def _set_eels_aperture(self, aperture_index: int) -> None:
        self._call_eels("set_eels_aperture", int(aperture_index))

    def _get_available_apertures(self) -> str:
        return self._call_eels("get_available_apertures")


if __name__ == "__main__":
    EELS.run_server()
