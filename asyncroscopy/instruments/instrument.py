import json
from typing import Optional


from abc import abstractmethod, ABCMeta

import tango

class CombinedMeta(tango.server.DeviceMeta, ABCMeta):
    """Combines Tango DeviceMeta and ABCMeta to allow abstract methods in Devices."""
    pass

class Instrument(tango.server.Device, metaclass=CombinedMeta):

    # ------------------------------------------------------------------
    # Instrument level Device properties — configure in Tango DB per deployment
    # ------------------------------------------------------------------
    data_device_address = tango.server.device_property(
        dtype=str,
        default_value="",
        doc="Optional Tango device address for the DATA device, e.g. 'asyncroscopy/data/default'.",
    )

    testing_mode_bool = tango.server.device_property(
        dtype=bool, 
        default_value=False,
        doc="When True - used for running tests, passed in conftest.py")
    
    # ------------------------------------------------------------------
    # Instrument Attributes
    # ------------------------------------------------------------------

    instrument_type = tango.server.attribute(
        label="Instrument Type",
        dtype=str,
        access=tango.AttrWriteType.READ,
        doc="Instrument modality, for example 'STEM', 'SPM', 'TEM', or 'OPTIC'.",
    )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def init_device(self) -> None:
        tango.server.Device.init_device(self)
        self.set_state(tango.DevState.INIT)

        self._init_device_attributes()
        self._connect()

    # ------------------------------------------------------------------
    # Instrument methods
    # ------------------------------------------------------------------
    @abstractmethod
    def read_instrument_type(self) -> str:
        pass

    @abstractmethod
    def _init_device_attributes(self) -> None:
        """
        Initialize device-specific attributes.

        Define attributes that are specific to a particular instrument type (STEMMicroscope, SPMMicroscope, etc.).
        """
        pass

    @abstractmethod
    def _connect(self):
        pass

    @abstractmethod
    def _disconnect(self):
        pass


    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @tango.server.command
    def Connect(self) -> None:
        """
        Explicitly (re)connect to microscope hardware. Useful after a fault.
        """
        self._connect()

    @tango.server.command
    def Disconnect(self) -> None:
        """Disconnect from microscope hardware gracefully."""
        self.set_state(tango.server.DevState.OFF)
        self._disconnect()
