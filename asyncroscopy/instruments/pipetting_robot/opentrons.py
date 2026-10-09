'''
Opentrons Pipetting Robot Tango Device
'''

import tango
import numpy as np

from asyncroscopy.instruments.instrument import Instrument


class Opentrons(Instrument):
    """Opentrons pipetting robot device."""
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
        doc="Opentrons",
    )

    location_x = tango.server.attribute(
        label="Robot X Position",
        dtype=float,
        access=tango.AttrWriteType.READ,
        unit = "",
        doc="Current tip grabber position in the X direction in xunits",
    )

    location_y = tango.server.attribute(
        label="Robot Y Position",
        dtype=float,
        access=tango.AttrWriteType.READ,
        unit = "",
        doc="Current tip grabber position in the Y direction in xunits",
    )

    location_z = tango.server.attribute(
        label="Robot Z Position",
        dtype=float,
        access=tango.AttrWriteType.READ,
        unit = "",
        doc="Current tip grabber position in the Z direction in xunits",
    )

    has_tip = tango.server.attribute(
        label="Robot Z Position",
        dtype=bool,
        access=tango.AttrWriteType.READ,
        doc="Indicates if the robot has picked up a pipette tip or if the pipette holder is empty",
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

    @abstractmethod
    def _pick_up_tip(self):
        #pick up tip if robot does not already have tip
        self.pick_up_tip(location, presses, increment, prep_after)
        pass

    @abstractmethod
    def _return_tip(self):
        #try to drop tip if robot is holding a tip
        self.return_tip(home_after)
        pass

    @abstractmethod
    def _transfer(self):
        #try to drop tip if robot is holding a tip
        self.transfer(volume, source, dest)
        pass


    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @tango.server.command
    def Connect(self) -> None:
        """
        Explicitly (re)connect to Opentrons hardware. Useful after a fault.
        """
        self._connect()

    @tango.server.command
    def Disconnect(self) -> None:
        """Disconnect from Opentrons hardware gracefully."""
        self.set_state(tango.server.DevState.OFF)
        self._disconnect()


    @tango.server.command
    def pick_up_tip(self) -> None:
        """
        
        """
        self._pick_up_tip()
        self.has_tip()




class TemperatureModule(Instrument):
    """Temperature Module located within Opentrons pipetting robot device."""
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
        doc="Opentrons",
    )

    temperature = tango.server.attribute(
        label="Robot X Position",
        dtype=float,
        access=tango.AttrWriteType.READ,
        unit = "degrees C",
        doc="Current tip grabber position in the X direction in xunits",
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
        Explicitly (re)connect to Opentrons hardware. Useful after a fault.
        """
        self._connect()

    @tango.server.command
    def Disconnect(self) -> None:
        """Disconnect from Opentrons hardware gracefully."""
        self.set_state(tango.server.DevState.OFF)
        self._disconnect()
