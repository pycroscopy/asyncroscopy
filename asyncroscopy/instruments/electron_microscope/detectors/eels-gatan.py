"""
EEeLS (Electron Energy=Loss Spectroscopy) detector Tango device.

Base class for EELS spectrometer of Gatan.
It does NOT talk directly to Gatan but to an servertthat runs under DigitalMicrograph
inherents eelsbas class parent class 
reads these attributes via DeviceProxy before acquiring.
"""

from tango import AttrWriteType, DevState
from tango.server import Device, attribute, command, DevVarFloatArray, device_property
import tango.server

from asyncroscopy.data.data_writer import DEFAULT_ACQUISITION_DIR, save_acquisition

from .eels import EELSBase
import Pyro5

class EEELServer(EELSBase):
    """EELS base class for tango device."""
    # ------------------------------------------------------------------
    # Device properties — configure in Tango DB per deployment
    # ------------------------------------------------------------------
    eels_host = device_property(
        dtype=str,
        default_value="10.46.217.242",
        doc="Hostname or IP of the Gatan server",
    )
    eels_port = device_property(
        dtype=int,
        default_value=9091,
        doc="Port of the AutoScript microscope server",
    )
    hardware_timeout_seconds = device_property(
        dtype=int,
        default_value=120,
        doc="Hardware connection timeout in seconds.",
    )
    acquisition_save_directory = device_property(
        dtype=str,
        default_value=DEFAULT_ACQUISITION_DIR,
        doc="Directory where AutoScript acquisitions are saved before the Tiled server serves them.",
    )
    acquisition_file_format = device_property(
        dtype=str,
        default_value="h5",
        doc="Acquisition file format. HDF5 stores acquisition data and parsed metadata attributes.",
    )
    data_device_address = device_property(
        dtype=str,
        default_value="",
        doc="Optional Tango device address for the DATA device, e.g. 'asyncroscopy/data/default'.",
    )

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------
    # not finishded
    manufacturer = attribute(
        label="Gatan ImageFilter",
        dtype=bool,
        access=AttrWriteType.READ,
        doc="This EELS server uses asyncroscopies EELS server for control and acquisition of spectra",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _connect(self):
        self._connect_hardware()
        self._connect_detector_proxies()
        self.set_state(DevState.ON)

    def _connect_hardware(self) -> None:
        """Establish AutoScript connection from MPC -> hardware."""
        uri = "PYRO:eels_server@{eels_host}:{eels_port}" 
        self.eels_proxy = Pyro5.api.Proxy(uri)if not _AUTOSCRIPT_AVAILABLE or self.testing_mode_bool:
        if not self.eels_proxy.check_server():
            self.warn_stream("EELS Server not available")
            return

        def _connect_detector_proxies(self) -> None:
            """Build DeviceProxy objectsonly for data detector device."""
            try:
                proxy = tango.DeviceProxy(self.data_device_address)
                proxy.set_timeout_millis(12_000)
                self._detector_proxies["data"] = proxy
                self.info_stream(f"Connected to detector proxy: data @ {self.data_device_address}")
            except tango.DevFailed as e:
                    self.error_stream(f"Failed to connect to data proxy at {address}: {e}")
            

    # ------------------------------------------------------------------
    # Public commands
    # ------------------------------------------------------------------

    def _initialize_eels(self) -> None:
        """ Initialize EELS mode and make sure eels server responds"""
        self.eels_proxy.initialize_eels()

   
    def _set_eels_offset(self, offset):
        """ Set the eels energy offset in eV"""
        return self.eels_proxy.set_eels_offset(offset)

   
    def _get_eels_spectrum(self):
        """ Get eels spectrum filename as key for tile server"""
        spectrum, offset, dispersion = self.eels_proxy.get_eels_spectrum(self.exposure_time, self.number)
        energy_scale =np.arange(len(spectrum))*dispersion+offset
        spectrum = np.array(spectrum)

        
    @command(dtype_out=str)
    def get_available_dispersions(self):
        """Get all available dispersions in eV/channel and their index""" 
        return self._get_available_dispersions()

    @command(dtype_out=DevVarFloatArray)
    def get_eels_dispersion(self):
        """Get current dispersion in eV/channel"""
        return self._get_eels_dispersion()

    @command(dtype_in=DevVarFloatArray)                
    def set_eels_dispersion(self, dispersion_index):
        """Get current dispersion in eV/channel"""
        return self._set_eels_dispersion(dispersion_index)
                    
    @command(dtype_out=str)
    def get_eels_aperture(self):
        """Get current EELS entrance aperature as str and index"""
        return self._get_eels_aperture()

    @command(dtype_out=int)
    def set_eels_aperture(self, aperture_index):
        """Set EELS entrance aperature by its index"""
        return self._set_eels_aperture(self, aperture_index)

    @command(dtype_out=str)
    def get_available_apertures(self):
        """Get all available EELS entrance aperatures and their indices"""
        return self._get_available_apertures()

    # ------------------------------------------------------------------
    # Attribute read / write
    # ------------------------------------------------------------------

    def read_exposure_time(self) -> float:
        return self._exposure_time

    def write_exposure_time(self, value: float) -> None:
        self._exposure_time = value

    # ------------------------------------------------------------------
    # Private Functions
    # ------------------------------------------------------------------

    def _set_eels_offset(self, offset):
        pass
    
    def _get_eels_spectrum(self):
        pass    

    def _get_available_dispersions(self):    
        pass

    def _get_eels_dispersion(self):
        pass

    def _set_eels_dispersion(self, dispersion_index):
        pass

    def _get_eels_aperture(self):
        pass

    def _set_eels_aperture(self, aperture_index):
        pass

    def _get_available_apertures(self):
        pass
