"""
EELS (electron energy-loss spectroscopy) detector Tango device.

This module defines the hardware-independent Tango API. Hardware adapters
inherit :class:`EELSBase` and implement the protected methods below.
"""

from abc import abstractmethod

import tango
from tango import AttrWriteType, DevState, DevVarFloatArray
from tango.server import Device, attribute, command, device_property


class EELSBase(Device):
    """Hardware-independent Tango API for an EELS detector."""

    # ------------------------------------------------------------------
    # Device properties — set per-deployment in the Tango DB
    # ------------------------------------------------------------------

    data_device_address = device_property(
        dtype=str,
        default_value="",
        doc="Optional Tango DATA device used to save and register spectra with Tiled, "
        "e.g. 'asyncroscopy/data/default'.",
    )

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    exposure_time = attribute(
        label="Exposure Time",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="s",
        format="%e",
        min_value=1e-6,
        max_value=5,
        doc="Exposure time in seconds (e.g. 1e-3 = 1 ms)",
    )

    number_of_frames = attribute(
        label="Number of Frames",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        min_value=1,
        max_value=5,
        doc="Number of frames to sum for one spectrum",
    )

    offset = attribute(
        label="Energy Offset",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="eV",
        format="%e",
        doc="Configured EELS spectrum energy offset in electronvolts",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.ON)

        self._exposure_time = 1e-4
        self._number_of_frames = 1
        self._offset = float("nan")
        self._data_proxy = None
        self._connect_data_proxy()

        self.info_stream("EELS device initialised")

    def _connect_data_proxy(self) -> None:
        if not self.data_device_address:
            self.info_stream(
                "No DATA device configured; spectra will be saved locally"
            )
            return

        try:
            self._data_proxy = tango.DeviceProxy(self.data_device_address)
            self._data_proxy.set_timeout_millis(120_000)
            self.info_stream(f"Connected to DATA device at {self.data_device_address}")
        except tango.DevFailed as exc:
            self._data_proxy = None
            self.warn_stream(
                f"Could not connect to DATA device at {self.data_device_address}; "
                f"spectra will be saved locally: {exc}"
            )

    def _get_data_proxy(self):
        """Return the DATA proxy, retrying a startup-time connection failure."""
        if self._data_proxy is None and self.data_device_address:
            self._connect_data_proxy()
        return self._data_proxy

    # ------------------------------------------------------------------
    # Attribute read / write
    # ------------------------------------------------------------------

    def read_exposure_time(self) -> float:
        return self._exposure_time

    def write_exposure_time(self, value: float) -> None:
        self._exposure_time = float(value)

    def read_number_of_frames(self) -> int:
        return self._number_of_frames

    def write_number_of_frames(self, value: int) -> None:
        self._number_of_frames = int(value)

    def read_offset(self) -> float:
        return self._offset

    def write_offset(self, value: float) -> None:
        offset = float(value)
        self._write_offset(offset)
        self._offset = offset

    # ------------------------------------------------------------------
    # Public Tango commands
    # ------------------------------------------------------------------

    @command(dtype_out=bool)
    def initialize_eels(self) -> bool:
        """Initialize spectroscopy mode and verify the EELS hardware."""
        return self._initialize_eels()

    @command(dtype_out=str)
    def acquire_eels_spectrum(self) -> str:
        """Acquire a spectrum and return its DATA/Tiled key or local path."""
        return self._acquire_eels_spectrum()

    @command(dtype_out=str)
    def get_available_dispersions(self) -> str:
        """Return the available dispersions and their indices."""
        return self._get_available_dispersions()

    @command(dtype_out=DevVarFloatArray)
    def get_eels_dispersion(self) -> list[float]:
        """Return the current dispersion as ``[index, eV_per_channel]``."""
        return self._get_eels_dispersion()

    @command(dtype_in=int)
    def set_eels_dispersion(self, dispersion_index: int) -> None:
        """Select a dispersion by index."""
        self._set_eels_dispersion(dispersion_index)

    @command(dtype_out=str)
    def get_eels_aperture(self) -> str:
        """Return the current EELS entrance aperture."""
        return self._get_eels_aperture()

    @command(dtype_in=int)
    def set_eels_aperture(self, aperture_index: int) -> None:
        """Select an EELS entrance aperture by index."""
        self._set_eels_aperture(aperture_index)

    @command(dtype_out=str)
    def get_available_apertures(self) -> str:
        """Return the available EELS entrance apertures and their indices."""
        return self._get_available_apertures()

    # ------------------------------------------------------------------
    # Hardware adapter interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _initialize_eels(self) -> bool:
        pass

    @abstractmethod
    def _write_offset(self, offset: float) -> None:
        pass

    @abstractmethod
    def _acquire_eels_spectrum(self) -> str:
        pass

    @abstractmethod
    def _get_available_dispersions(self) -> str:
        pass

    @abstractmethod
    def _get_eels_dispersion(self) -> list[float]:
        pass

    @abstractmethod
    def _set_eels_dispersion(self, dispersion_index: int) -> None:
        pass

    @abstractmethod
    def _get_eels_aperture(self) -> str:
        pass

    @abstractmethod
    def _set_eels_aperture(self, aperture_index: int) -> None:
        pass

    @abstractmethod
    def _get_available_apertures(self) -> str:
        pass


if __name__ == "__main__":
    EELSBase.run_server()
