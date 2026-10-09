"""
STEMMicroscope Tango device.

Owns the AutoScript connection and all acquisition commands.
Detector settings are read from the corresponding detector DeviceProxy
so that each detector device is the single source of truth for its own params.

AutoScript is an optional dependency; this module imports cleanly without it
and falls back to simulated acquisition. To enable real hardware:

    pip install asyncroscopy[autoscript]

Return convention for real image commands
-----------------------------------------
Real AutoScript image commands save the adorned object on disk and return the
DATA/Tiled unique id for that saved acquisition.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import tango
from tango import AttrWriteType, DevState
from tango.server import attribute, command, device_property

from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope
from asyncroscopy.data.data_writer import DEFAULT_ACQUISITION_DIR, save_acquisition

# AutoScript imports — only available on the microscope PC.
# Wrapped in try/except so the device can still be imported and tested
# on a development machine without AutoScript installed.
try:
    import autoscript_tem_microscope_client
    from autoscript_tem_microscope_client import TemMicroscopeClient
    from autoscript_tem_microscope_client.enumerations import EdsDetectorType
    from autoscript_tem_microscope_client.enumerations import RegionCoordinateSystem, ExposureTimeType
    from autoscript_tem_microscope_client.structures import Region, Rectangle
    from autoscript_tem_microscope_client.structures import StemAcquisitionSettings, EdsAcquisitionSettings, RunOptiStemSettings, CameraAcquisitionSettings

    _AUTOSCRIPT_AVAILABLE = True
except ImportError:
    _AUTOSCRIPT_AVAILABLE = False

AUTOSCRIPT_STEM_OUTPUT_DIRECTORY = Path(r"Y:\AutoScript TEM")


class AutoScriptMicroscope(ElectronMicroscope):
    """
    Manages the AutoScript connection and exposes acquisition commands.
    Detector-specific settings (dwell time, resolution) are stored in
    dedicated detector devices and read via DeviceProxy at acquisition time.
    """

    # ------------------------------------------------------------------
    # Device properties — configure in Tango DB per deployment
    # ------------------------------------------------------------------
    hardware_host = device_property(
        dtype=str,
        default_value="10.46.217.241",
        doc="Hostname or IP of the AutoScript microscope server",
    )
    hardware_port = device_property(
        dtype=int,
        default_value=9095,
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
        label="Thermofisher",
        dtype=bool,
        access=AttrWriteType.READ,
        doc="This microscope uses AutoScript for control and acquisition",
    )

    fov = attribute(
        label="Field of View",
        dtype=float,
        access=AttrWriteType.READ,
        unit = "m",
        # min_value= TODO: set these
        # max_value= TODO: set these
        doc="Current field of view in micrometers",
    )

    defocus = attribute(
        label="Defocus",
        dtype=float,
        access=AttrWriteType.READ,
        unit = "m",
        # min_value= TODO: set these
        # max_value= TODO: set these
        doc="Current defocus in micrometers",
    )

    camera_length = attribute(
        label="Camera Length",
        dtype=float,
        access=AttrWriteType.READ,
        unit = "m",
        # min_value= TODO: set these
        # max_value= TODO: set these
        doc="Current camera length in meters",
    )

    beam_state = attribute(
        label="Beam State",
        dtype=bool,
        access=AttrWriteType.READ,
        doc="Current beam state, either 'blanked' or 'unblanked'",
    )

    acceleration_voltage = attribute(
        label="Acceleration Voltage",
        dtype=float,
        access=AttrWriteType.READ,
        unit = "V",
        # min_value= TODO: set these
        # max_value= TODO: set these
        doc="Current acceleration voltage in volts",
    )


    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _connect(self):
        self._connect_hardware()
        self._connect_detector_proxies()
        self.set_state(DevState.ON)
        self.screen_current_calibration = None

    def _connect_hardware(self) -> None:
        """Establish AutoScript connection from MPC -> hardware."""
        if not _AUTOSCRIPT_AVAILABLE or self.testing_mode_bool:
            self.warn_stream("AutoScript not available")
            return
        try:
            self._microscope = TemMicroscopeClient()
            self._microscope.connect(self.hardware_host, self.hardware_port)
            self.info_stream(f"Connected to AutoScript at {self.hardware_host}:{self.hardware_port}")
            self.is_autoscript = True
        except Exception as e:
            self.error_stream(f"AutoScript connection failed: {e}")
            self.set_state(DevState.FAULT)
            self._microscope = None
            self.is_autoscript = False

    def _connect_detector_proxies(self) -> None:
        """Build DeviceProxy objects for each configured detector device."""
        # Extend this dict as more detectors are added
        # later, we want to do this automatically, not with a dictionary.
        addresses: dict[str, str] = {
            "eds": self.eds_device_address,
            "stage": self.stage_device_address,
            "scan": self.scan_device_address,
            "camera": self.camera_device_address,
            "data": self.data_device_address,
        }
        for name, address in addresses.items():
            if not address:  # <-- minimal fix
                self.info_stream(f"Skipping {name}: no address configured")
                continue
            try:
                proxy = tango.DeviceProxy(address)
                proxy.set_timeout_millis(3_600_000 if name == "data" else 12_000)
                self._detector_proxies[name] = proxy
                self.info_stream(f"Connected to detector proxy: {name} @ {address}")
            except tango.DevFailed as e:
                self.error_stream(f"Failed to connect to {name} proxy at {address}: {e}")

    # ------------------------------------------------------------------
    # Attribute read methods
    # ------------------------------------------------------------------

    def read_manufacturer(self) -> bool:
        # TODO: query self._microscope.optics.mode when AutoScript available
        return self._manufacturer

    def read_fov(self) -> float:
        """Field of view in meters (STEM mode only)."""
        if self._microscope is None:
            return float("nan")
        return self._get_fov()

    def read_defocus(self) -> float:
        """Defocus in meters."""
        if self._microscope is None:
            return float("nan")
        return self._get_defocus()

    def read_acceleration_voltage(self) -> float:
        """Accelerating voltage in volts."""
        if self._microscope is None:
            return float("nan")
        return self._microscope.source.acceleration_voltage.value

    def read_camera_length(self) -> float:
        """Camera length in meters (only meaningful in DIFFRACTION mode)."""
        if self._microscope is None:
            return float("nan")
        return self._microscope.optics.camera_length.value.calibrated

    def read_beam_state(self) -> bool:
        """Beam blanked state: True when blanked, False when unblanked."""
        if self._microscope is None:
            return False
        return self._microscope.optics.blanker.is_beam_blanked

    # ------------------------------------------------------------------
    # Commands pertaining to setting children attributes, e.g. stage position, scan parameters, EDS settings, etc. --> iuser accesses it in a jupyter notebook using the device proxy
    # ------------------------------------------------------------------

    @command
    def register_stage(self):
        """Read the live stage position from hardware and publish it onto the
        STAGE child device.

        Call this from the notebook before reading the STAGE device so its
        x/y/z/alpha attributes reflect the current hardware position.
        """
        self._get_stage()


    # ------------------------------------------------------------------
    # Internal acquisition helpers
    # ------------------------------------------------------------------
    def _acquire_scanned_image(
        self,
        imsize: int,
        dwell_time: float,
        detector_list: list[str] = ["haadf"],
        scan_region: list[float] = [0.0, 0.0, 1.0, 1.0],
        output_format: str = ".h5",
    ) -> str:
        """
        Call AutoScript scanned image acquisition, save it, and return its DATA/Tiled key.
        """
        detector_list = [d.upper() for d in detector_list]
        settings = StemAcquisitionSettings(dwell_time=dwell_time, detector_types=detector_list, size=imsize, region=Region(RegionCoordinateSystem.RELATIVE, Rectangle(*scan_region)))
        adorned = self._microscope.acquisition.acquire_stem_images_advanced(settings)
        if not isinstance(adorned, list):
            adorned = [adorned]
        data_server = self._detector_proxies.get("data")
        return save_acquisition(self, data_server, "stem_image", detector_list, adorned, output_format=output_format)


    def _acquire_camera_image(
        self,
        imsize: int,
        exposure_time: float,
        detector: str,
        readout_area: str,
        frame_combining: int = 1,
        output_format: str = ".h5",
    ) -> str:
        """
        Call advanced AutoScript camera acquisition, save the adorned image,
        and return its DATA/Tiled key.
        """
        camera_detector = {"flucam": "Flucam", "bm-ceta": "BM-Ceta"}.get(detector.lower(), detector)
        fixed_readout_area = {
            "full": "Full",
            "half": "Half",
            "quarter": "Quarter",
        }.get(readout_area.lower(), readout_area)
        settings = CameraAcquisitionSettings(
            camera_detector=camera_detector,
            size=imsize,
            exposure_time=exposure_time,
            fixed_readout_area=fixed_readout_area,
            frame_combining=frame_combining,
        )
        adorned = self._microscope.acquisition.acquire_camera_image_advanced(settings)
        data_server = self._detector_proxies.get("data")
        return save_acquisition(
            self,
            data_server,
            "camera_image",
            str(detector),
            adorned,
            output_format=output_format,
        )


    # the following file is COMPLETELY CORRECT, except the AutoScript acquire_stem_data_advanced is currently trash.
    # it only records the origin corner of the camera, which is covered by the HAADF and ureachable at short camera lengths.
    # waiting on an email back with a fix. -Austin Houston, Aug 11th 2026
    '''
    def _acquire_scanned_data_advanced(self, imsize: int, dwell_time: float, detector: str, scan_region: list[float]) -> str:
        """Acquire AutoScript 4D-STEM data and return its registered HDF5 key."""
        data_server = self._detector_proxies.get("data")
        if not AUTOSCRIPT_STEM_OUTPUT_DIRECTORY.is_dir():
            raise FileNotFoundError(f"AutoScript STEM output directory is not readable: {AUTOSCRIPT_STEM_OUTPUT_DIRECTORY}")

        files_before = set(AUTOSCRIPT_STEM_OUTPUT_DIRECTORY.glob("*.mrc"))
        region = Region(RegionCoordinateSystem.RELATIVE, Rectangle(*scan_region))
        settings = StemDataSettings(dwell_time=dwell_time, detector_types=[CameraType.BM_CETA], size=imsize, region=region)
        self._microscope.acquisition.acquire_stem_data_advanced(settings)
        for _ in range(11):
            new_files = set(AUTOSCRIPT_STEM_OUTPUT_DIRECTORY.glob("*.mrc")) - files_before
            if new_files:
                break
            time.sleep(0.5)

        if len(new_files) != 1:
            raise RuntimeError(f"Expected one new AutoScript MRC file, found {len(new_files)}")

        source_path = new_files.pop()
        request = {
            "source_path": str(source_path),
            "detector": detector,
            "scan_shape": [int(imsize), int(imsize)],
            "dwell_time": float(dwell_time),
            "scan_region": list(scan_region),
        }
        return data_server.copy_and_register_remote_file(json.dumps(request))
        '''

    def _acquire_scanned_data_advanced(self, imsize: int, dwell_time: float, detector: str, scan_region: list[float]) -> str:
        """Acquire 4D-STEM data by placing the probe and recording one Ceta image per scan point."""
        scan_size = int(imsize)
        left, top, width, height = [float(value) for value in scan_region]
        x_positions = left + (np.arange(scan_size) + 0.5) * width / scan_size
        y_positions = top + (np.arange(scan_size) + 0.5) * height / scan_size
        camera_detector = {"flucam": "Flucam", "bm-ceta": "BM-Ceta"}.get(detector.lower(), detector)
        settings = CameraAcquisitionSettings(camera_detector=camera_detector, size=256, exposure_time=dwell_time, fixed_readout_area="Half")
        starting_beam_position = self._microscope.optics.paused_scan_beam_position
        frames = []
        try:
            for y_position in y_positions:
                for x_position in x_positions:
                    self._microscope.optics.paused_scan_beam_position = [float(x_position), float(y_position)]
                    image = self._microscope.acquisition.acquire_camera_image_advanced(settings)
                    frames.append(np.array(image.data, copy=True))
        finally:
            self._microscope.optics.paused_scan_beam_position = starting_beam_position

        stem_data = np.stack(frames).reshape(scan_size, scan_size, 256, 256)
        attrs = {"scan_shape": [scan_size, scan_size], "diffraction_shape": [256, 256], "dwell_time": float(dwell_time), "scan_region": [left, top, width, height], "readout_area": "Half"}
        data_server = self._detector_proxies.get("data")
        return save_acquisition(self, data_server, "stem_data", camera_detector, stem_data, dataset_name="stem_data", dataset_attrs=attrs)


    def _acquire_spectrum(self, detector_name: str, exposure_time: float) -> str:
        settings = EdsAcquisitionSettings()
        settings.eds_detector = EdsDetectorType.SUPER_X
        settings.dispersion = 5
        settings.shaping_time = 3e-6
        settings.exposure_time = exposure_time
        settings.exposure_time_type = ExposureTimeType.LIVE_TIME
        spectrum = self._microscope.analysis.eds.acquire_spectrum(settings)
        data_server = self._detector_proxies.get("data")
        return save_acquisition(self, data_server, "spectrum", detector_name, spectrum, dataset_name="spectrum")

    def _place_beam(self, position) -> None:
        """
        sets resting beam position, [0:1]
        """
        if self._microscope is not None:
            x = float(position[0])
            y = float(position[1])
            self._microscope.optics.paused_scan_beam_position = [x, y]

    def _set_fov(self, fov) -> None:
        """set field of view in meters"""
        self._microscope.optics.scan_field_of_view = fov

    def _get_fov(self) -> float:
        """get field of view in meters"""
        return self._microscope.optics.scan_field_of_view

    def _set_column_valves(self, state: str) -> None:
        """Set column valves state."""
        if self._microscope is not None:
            if state == "open":
                self._microscope.vacuum.column_valves.open()
            elif state == "close":
                self._microscope.vacuum.column_valves.close()
            else:
                print(f"Invalid valve state '{state}'. Use 'open' or 'close'.")

    def _blank_beam(self) -> None:
        """blank beam"""
        if self._microscope is not None:
            self._microscope.optics.blanker.blank()

    def _unblank_beam(self) -> None:
        """
        unblank beam
        """
        self._microscope.optics.blanker.unblank()

    def _set_defocus(self, defocus) -> None:
        """Set defocus in meters."""
        if self._microscope is not None:
            self._microscope.optics.defocus = float(defocus)

    def _get_defocus(self) -> float:
        """Get defocus in meters."""
        return float(self._microscope.optics.defocus)
    
    def _set_screen(self, position: str)->None:
        if position.lower() in ['in', 'insert', 'inserted']:
            if self._microscope.detectors.screen.position == 'Retracted':
                self._microscope.detectors.screen.insert()
        elif position.lower() in ['out', 'retract', 'retracted']:
             if self._microscope.detectors.screen.position == 'Inserted':
                self._microscope.detectors.screen.retract()


    def _calibrate_screen_current(self) -> None:
        """ calibrate screen current with monchromator focus"""
        original_gun_lens = self._microscope.optics.monochromator.focus
        gun_lens_series = np.linspace(10, 150, 15)

        # series of measurements
        current_series = []
        for val in gun_lens_series:
            self._microscope.optics.monochromator.focus = val
            time.sleep(1)
            screen_current = self._microscope.detectors.screen.measure_current()
            current_series.append(screen_current)
        current_series = np.array(current_series) * 1e12
        self._microscope.optics.monochromator.focus = original_gun_lens

        # fit a polynomial and save:
        coeffs = np.polyfit(gun_lens_series, current_series, 11)
        poly_func = np.poly1d(coeffs)
        self.screen_current_calibration = poly_func

    def _set_screen_current(self, current) -> None:
        """set screen current in pA"""
        if self.screen_current_calibration is not None:
            poly_func = self.screen_current_calibration
            adjusted_poly = poly_func - current
            x_candidates = adjusted_poly.r
            x_real = x_candidates[np.isreal(x_candidates)].real
            x_real = np.max(x_real)  # choose the largest real root as the gun lens value
            self._microscope.optics.monochromator.focus = float(x_real)
        else:
            self.warn_stream("Screen current calibration not available. running calibration (should take 15 seconds).")
            self._calibrate_screen_current()

            poly_func = self.screen_current_calibration
            adjusted_poly = poly_func - current
            x_candidates = adjusted_poly.r
            x_real = x_candidates[np.isreal(x_candidates)].real
            x_real = np.max(x_real)  # choose the largest real root as the gun lens value
            self._microscope.optics.monochromator.focus = float(x_real)

    def _get_screen_current(self) -> float:
        """get screen current in pA"""
        screen_current = self._microscope.detectors.screen.measure_current() * 1e12
        return screen_current

    def _get_stage(self):
        """Get the current stage position as a list of floats [x, y, z, alpha, beta]."""
        # set proxy attributes with current stage position
        stage = self._detector_proxies["stage"]

        # TODO: add beta value check
        position = self._microscope.specimen.stage.position
        position = np.array(position)

        stage.x = float(position[1])
        stage.y = float(position[0])
        stage.z = float(position[2])
        stage.alpha = float(math.degrees(position[3]))

        if position[4] is not None:
            return position
        else:
            return position[:4]
        
    
    def _get_parameters(self):
        status = {'system': self._microscope.service.system.name,
                'vacuum': self._microscope.vacuum.state,
                'column_valves': self._microscope.vacuum.column_valves.state,
                'is_accelerator_on': self._microscope.optics.is_accelerator_on,
                'acceleration_voltage': self._microscope.optics.acceleration_voltage.value,
                'optical_mode': self._microscope.optics.optical_mode,
                'illumination_mode': self._microscope.optics.illumination_mode,
                'objective_lens_mode': self._microscope.optics.objective_lens_mode,
                'projector_mode': self._microscope.optics.projector_mode,
                #'convergence_angle': self._microscope.optics.convergence_angle,
                'spot_size': self._microscope.optics.spot_size_index,
                'beam_stopper': self._microscope.optics.beam_stopper.insertion_state,
                'beam_blanker': self._microscope.optics.blanker.is_beam_blanked,
                'is_eftem_on': self._microscope.optics.is_eftem_on,
                }
        if self._microscope.optics.optical_mode == 'Stem':
            status['scan_rotation'] = self._microscope.optics.scan_rotation
            status['scan_field_of_view'] = self._microscope.optics.scan_field_of_view
        for mechanism_type in self._microscope.optics.aperture_mechanisms.get_available():
            mechanism = self._microscope.optics.aperture_mechanisms.get_mechanism(mechanism_type)   
            if not mechanism.is_enabled:
                status[mechanism_type] = 'Disabled'
            elif mechanism.insertion_state == 'Retracted':
                status[mechanism_type] = 'Retracted'
            else:
                status[mechanism_type] = mechanism.aperture.name  
        for deflector in self._microscope.optics.deflectors.get_available_deflectors():  
            defl = self._microscope.optics.deflectors.get_deflector_value(deflector) 
            status[deflector] = [defl.x, defl.y]              

        return json.dumps(status)

    def _get_stage(self):
        """Get the current stage position as [x, y, z, alpha, beta], with tilts in degrees."""
        # set proxy attributes with current stage position
        stage = self._detector_proxies["stage"]
        return stage.position

    def _move_stage(self, position) -> None:
        """Move stage to [x, y, z, alpha, beta], with x/y/z in meters and tilts in degrees."""
        stage_proxy = self._detector_proxies.get("stage")
        stage_proxy.position = position  # expects [x, y, z, alpha, beta] in meters and degrees

    def _auto_focus(self):
        """Perform autofocus routine C1A1"""
        if self._microscope.optics.optical_mode == 'Stem':
            settings = RunOptiStemSettings(method="C1A1")
            self._microscope.auto_functions.run_opti_stem(settings)
        else:
            settings = autoscript_tem_microscope_client.structuresRunObjectiveAutoStigmatorSettings(camera_detector="Flucam")
            self.microscope.auto_functions.run_objective_auto_stigmator(settings)

    def _set_image_shift(self, shift):
        """Apply image shift in meters."""
        x_shift = float(shift[0])
        y_shift = float(shift[1])
        try:
            if self._microscope.optics.optical_mode == 'Stem':
                self._microscope.optics.deflectors.beam_shift = (x_shift, y_shift)
            else:
                self._microscope.optics.deflectors.image_shift = (x_shift, y_shift)
        except Exception as e:
            self.error_stream(f"Failed to set image shift: {e}")

    def _set_diffraction_shift(self, shift):
        """Apply image shift in meters."""
        x_shift = float(shift[0])
        y_shift = float(shift[1])
        try:
            if self._microscope.optics.projector_mode == 'Diffraction':
                self._microscope.optics.deflectors.image_shift = (x_shift, y_shift)
        except Exception as e:
            self.error_stream(f"Failed to set diffraction shift: {e}")
    
    def _get_diffraction_shift(self):
        if self._microscope.optics.optical_mode == 'Diffraction':
            position = self._microscope.optics.deflectors.image_shift
            return np.array([position.x, position.y])
        else:
            return np.array([0, 0])
    
    
    def _get_image_shift(self):
        if self._microscope.optics.optical_mode == 'Stem':
            position = self._microscope.optics.deflectors.beam_shift
        else:
            position = self._microscope.optics.deflectors.image_shift
        return np.array([position.x, position.y])
    
    def _get_beam_tilt(self):
        tilt = self._microscope.optics.deflectors.beam_tilt
        return np.array([tilt .x, tilt.y])

    def _set_beam_tilt(self, tilt):
        """Apply beam tilt in radians."""
        x_tilt = float(tilt[0])
        y_tilt = float(tilt[1])
        try:
            self._microscope.optics.deflectors.beam_tilt = (x_tilt, y_tilt)
        except Exception as e:
            self.error_stream(f"Failed to set beam tilt: {e}")
    
# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    AutoScriptMicroscope.run_server()
