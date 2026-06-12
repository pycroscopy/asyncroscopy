"""
Microscope Tango device.

Detector settings are read from the corresponding detector DeviceProxy
so that each detector device is the single source of truth for its own params.

Return convention for image commands
-------------------------------------
Image commands return a string supplied by the concrete microscope
implementation, typically a DATA/Tiled unique id.
"""

import json
from typing import Optional


from abc import abstractmethod, ABCMeta

import tango
from tango import AttrWriteType, DevEncoded, DevState, DevVarFloatArray, DevFloat, DevVarStringArray
from tango.server import Device, DeviceMeta, attribute, command, device_property

class CombinedMeta(DeviceMeta, ABCMeta):
    """Combines Tango DeviceMeta and ABCMeta to allow abstract methods in Devices."""
    pass

class Microscope(Device, metaclass=CombinedMeta):
    """
    Top-level TEM microscope device.
    Detector-specific settings (dwell time, resolution) are stored in
    dedicated detector devices and read via DeviceProxy at acquisition time.
    """

    # ------------------------------------------------------------------
    # Device properties — configure in Tango DB per deployment
    # ------------------------------------------------------------------

    scan_device_address = device_property(
        dtype=str,
        doc="Tango device address for the SCAN settings device. "
            "DB mode: 'test/detector/scan' "
            "No-DB mode: 'tango://127.0.0.1:8888/test/nodb/scan#dbase=no'",
    )
    
    corrector_device_address = device_property(
        dtype=str,
        doc="Tango device address for the aberration corrector settings device. "
            "DB mode: 'test/hardware/corrector' "
            "No-DB mode: 'tango://127.0.0.1:8888/test/nodb/corrector#dbase=no'",
    )

    eds_device_address = device_property(
        dtype=str,
        doc="Tango device address for the EDS settings device. "
            "DB mode: 'asyncroscopy/eds/default' "
            "No-DB mode: 'tango://127.0.0.1:8887/asyncroscopy/haadf/default#dbase=no'",
    )

    stage_device_address = device_property(
        dtype=str,
        doc="Tango device address for the STAGE settings device. "
            "DB mode: 'asyncroscopy/stage/default' "
            "No-DB mode: 'tango://127.0.0.1:8888/asyncroscopy/stage/default#dbase=no'",
    )

    camera_device_address = device_property(
        dtype=str,
        doc="Tango device address for the CAMERA settings . "
            "DB mode: 'asyncroscopy/camera/default' "
            "No-DB mode: 'tango://127.0.0.1:8888/asyncroscopy/camera/default#dbase=no'",
    )

    flucam_device_address = device_property(
        dtype=str,
        default_value="",
        doc="Tango device address for the FLUCAM settings device. "
            "DB mode: 'asyncroscopy/flucam/default' "
            "No-DB mode: 'tango://127.0.0.1:8888/asyncroscopy/flucam/default#dbase=no'",
    )
    testing_mode_bool = device_property(dtype=bool, 
                                        default_value=False,
                                        doc="When True - used for running tests, passed in conftest.py")

    # Add further detector device_property entries here as detectors are added
    # eels_device_address  = device_property(dtype=str, default_value="asyncroscopy/eels/default")

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    stem_mode = attribute(
        label="STEM Mode",
        dtype=bool,
        access=AttrWriteType.READ,
        doc="True when the microscope is in STEM mode",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.INIT)

        self._microscope: Optional[object] = None  # TemMicroscopeClient instance
        self._stem_mode: bool = False

        # Dict mapping detector name string → DeviceProxy
        # Populated in _connect_detector_proxies
        self._detector_proxies: dict[str, tango.DeviceProxy] = {}

        self._connect()

    @abstractmethod
    def _connect(self):
        pass
    
    @abstractmethod
    def _connect_hardware(self) -> None:
        pass

    @abstractmethod
    def _connect_detector_proxies(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Attribute read methods
    # ------------------------------------------------------------------

    def read_stem_mode(self) -> bool:
        # TODO: query self._microscope.optics.mode when AutoScript available
        return self._stem_mode

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @command
    def Connect(self) -> None:
        """Explicitly (re)connect to microscope hardware. Useful after a fault.
        Also, sets the timeout fofr Tango device for 2 minutes (for larger things)
        """
        self._connect()

    @command
    def Disconnect(self) -> None:
        """Disconnect from microscope hardware gracefully."""
        # TODO: self._microscope.disconnect() when AutoScript available
        self._microscope = None
        self.set_state(DevState.OFF)
        self.info_stream("Disconnected from microscope hardware")

    @command(dtype_in=str, dtype_out=str)
    def acquire_spectrum(self, detector_name: str) -> str:
        """Acquire a single spectrum and return its DATA/Tiled unique id."""
        detector_name = detector_name.lower().strip()
        proxy = self._detector_proxies.get(detector_name)
        return self._acquire_spectrum(detector_name, proxy.exposure_time)

    @command(dtype_in=DevVarStringArray, dtype_out=str)
    def acquire_scanned_image(self, detector_list: list[str] = ["haadf"]) -> str:
        """Acquire an image with scanning detectors and return a key pointing to that data. You can get the data with the get_image_from_key command"""
        scan = self._detector_proxies.get("scan")
        return self._acquire_scanned_image(scan.imsize, scan.dwell_time, detector_list, list(scan.scan_region))

    @command(dtype_out=str)
    def acquire_scanned_data_advanced(self) -> str:
        """Trigger an advanced 4D scanned data acquisition with the Ceta camera."""
        scan = self._detector_proxies.get("scan")
        return self._acquire_scanned_data_advanced(scan.imsize, scan.dwell_time, "BM-Ceta", list(scan.scan_region))

    @command(dtype_out=str)
    def acquire_camera_image(self) -> str:
        """Acquire a camera image using settings from the camera device."""
        camera = self._detector_proxies.get("camera")
        return self._acquire_camera_image(camera.imsize, camera.exposure_time, "BM-Ceta", camera.readout_area)

    @command(dtype_out=str)
    def acquire_flucam_image(self) -> str:
        """Acquire a Flucam image using settings from the flucam device."""
        flucam = self._detector_proxies.get("flucam")
        return self._acquire_camera_image(flucam.imsize, flucam.exposure_time, "Flucam", flucam.readout_area)

    @command(dtype_in=int, dtype_out=DevEncoded)
    def get_image_data_cached(self, index: int) -> tuple[str, bytes]:
        """Retrieve cached image by index."""
        if not hasattr(self, '_cached_images'):
            tango.Except.throw_exception("NoCache", "Call acquire_scanned_image() first", "get_image_data()")
        if index >= len(self._cached_images):
            tango.Except.throw_exception("InvalidIndex", f"Index {index} out of range", "get_image_data()")
        
        cached_image = self._cached_images[index]
        img_data = cached_image.data if hasattr(cached_image, 'data') else cached_image
        
        meta = {"shape": list(img_data.shape), "dtype": str(img_data.dtype)}
        return json.dumps(meta), img_data.tobytes()

    @command(dtype_in=DevVarFloatArray, dtype_out=None)
    def place_beam(self, position) -> None:
        """
        sets resting beam position, [0:1]
        """
        self._place_beam(position)

    @command(dtype_in=DevVarFloatArray, dtype_out=None)
    def place_beam_list(self, positions) -> None:
        """
        Place beam at multiple positions sequentially.
        Extension of place_beam command 
        Why not call  place_beam in loop of client side -> It fails
        """
        if len(positions) % 2 != 0:
            raise ValueError("Input must contain pairs of (x, y) values.")

        for i in range(0, len(positions), 2):
            x = float(positions[i])
            y = float(positions[i + 1])

            self._place_beam([x, y])

    @command(dtype_in=str)
    def set_column_valves(self, state: str) -> None:
        """Open or close the column valves"""
        self._set_column_valves(state)

    @command()
    def blank_beam(self) -> None:
        """blank beam"""
        self._blank_beam()

    @command()
    def unblank_beam(self) -> None:
        """
        unblank beam
        """
        self._unblank_beam()

    @command(dtype_in=DevFloat)
    def set_defocus(self, defocus):
        """
        set the defocus in meters
        """
        self._set_defocus(defocus)

    @command(dtype_out=DevFloat)
    def get_defocus(self):
        """
        read the defocus in meters
        """
        return self._get_defocus()

    @command(dtype_in=DevFloat)
    def set_fov(self, fov):
        """
        set the field of view for the next acquisition
        """
        self._set_fov(fov)

    @command(dtype_out=DevFloat)
    def get_fov(self):
        """
        read the field of view for the next acquisition
        """
        return self._get_fov()
    
    @command(dtype_in=DevFloat)
    def set_screen_current(self, current):
        """
        set the screen current in pA
        """
        self._set_screen_current(current)

    @command(dtype_out=DevFloat)
    def get_screen_current(self):
        """
        get the screen current in pA
        """
        return self._get_screen_current()
    
    @command(dtype_in=DevVarFloatArray, dtype_out=DevVarFloatArray)
    def set_stage_position(self, position) -> list:
        """Set stage position [x, y, z, alpha, beta]."""
        return self._set_stage_position(position)

    @command(dtype_out=DevVarFloatArray)
    def get_stage(self):
        """
        Get the current stage position as a list of floats [x, y, z, alpha, beta].

        Returns
        -------
        DevVarFloatArray = [x, y, z, alpha, beta]

        """
        position = self._get_stage()

        return position

    @command(dtype_in=DevVarFloatArray)
    def move_stage(self, position):
        """
        Move the the stage
        to an absolute position  [x, y, z, alpha, beta]

        Parameters
        position: an absolute reference frame move position (not relative)

        """
        self._move_stage(position)

    @command()
    def auto_focus(self):
        """
        Run the microscope's autofocus routine.
        """
        self._auto_focus()

    @command(dtype_in=DevVarFloatArray)
    def set_image_shift(self, shift):
        """
        Set the image shift to the specified values [x_shift, y_shift].

        Parameters
        ----------
        shift: list of two floats [x_shift, y_shift] specifying the desired image shift in meters.
        """
        self._set_image_shift(shift)
    # ------------------------------------------------------------------
    # Internal acquisition helpers
    # ------------------------------------------------------------------
    @abstractmethod
    def _acquire_scanned_image(
        self,
        imsize: int,
        dwell_time: float,
        detector_list: list[str] = ["haadf"],
        scan_region: list[float] = [0.0, 0.0, 1.0, 1.0],
    ) -> str:
        """Vendor-specific scanned image acquisition implementation."""
        pass

    def _acquire_camera_image(self, imsize: int, exposure_time: float, detector: str, readout_area: str) -> str:
        """Vendor-specific camera acquisition implementation."""
        tango.Except.throw_exception(
            "UnsupportedCommand",
            "This microscope does not support camera image acquisition.",
            "_acquire_camera_image()",
        )

    def _acquire_scanned_data_advanced(
        self,
        imsize: int,
        dwell_time: float,
        detector: str,
        scan_region: list[float],
    ) -> str:
        """Vendor-specific advanced 4D scanned data acquisition trigger."""
        tango.Except.throw_exception(
            "UnsupportedCommand",
            "This microscope does not support advanced scanned data acquisition.",
            "_acquire_scanned_data_advanced()",
        )

    def _place_beam(self, position):
        # define in the inherit class
        pass

    def _blank_beam(self):
        # define in the inherit class
        pass

    def _unblank_beam(self):
        # define in the inherit class
        pass

    def _set_defocus(self, defocus):
        # define in the inherit class
        pass

    def _get_defocus(self):
        # define in the inherit class
        pass

    @abstractmethod
    def _set_screen_current(self, current):
        # define in the inherit class
        pass

    @abstractmethod
    def _get_screen_current(self):
        pass

    @abstractmethod
    def _move_stage(self, position):
        # define in the inherit class
        pass

    @abstractmethod
    def _get_stage(self):
        pass

    @abstractmethod
    def _set_fov(self, fov):
        pass

    @abstractmethod
    def _get_fov(self):
        pass

    @abstractmethod
    def _auto_focus(self):
        pass

    @abstractmethod
    def _set_image_shift(self, shift):
        pass
# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    Microscope.run_server()
