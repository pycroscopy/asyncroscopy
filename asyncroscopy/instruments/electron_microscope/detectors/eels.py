"""
EEeLS (Electron Energy=Loss Spectroscopy) detector Tango device.

Base class for EELS spectrometer.
It does NOT talk to Gatan — the eels_gatan parent class 
reads these attributes via DeviceProxy before acquiring.
"""

from tango import AttrWriteType, DevState
from tango.server import Device, attribute, command, DevVarFloatArray
import tango.server

class EELSBase(tango.server.Device):
    """EELS base class for tango device."""
    
    
    exposure_time = attribute(
            label="Dwell Time",
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
                dtype=float,
                access=AttrWriteType.READ_WRITE,
                unit="s",
                format="%e",
                min_value=1e-6,
                max_value=5,
                doc="Number of Frames to be summed over for spectrum e.g.: 1 or 10",
            ) 
    
    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.ON)

        # Sensible defaults — operators override via Tango DB or client writes
        self._exposure_time: float = 1e-4   # 1 s
        if not self._initialize_eels():
            raise ValueError("Could not reach eels_server")
        
        self.info_stream("EELS device initialised")
        
    # ------------------------------------------------------------------
    # Public commands
    # ------------------------------------------------------------------
    
    @command(dtype_out=int)  
    def _initialize_eels(self) -> None:
        """ Initialize EELS mode and make sure eels server responds"""
        return False

    @command(dtype_in=DevVarFloatArray)
    def set_eels_offset(self, offset):
        """ Set the eels energy offset in eV"""
        return self._set_eels_offset(offset)

    @command(dtype_out=str)
    def get_eels_spectrum(self):
        """ Get eels spectrum filename as key for tile server"""
        return self._get_eels_spectrum()

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
