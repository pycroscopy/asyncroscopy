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
from typing import Any

import numpy as np
import tango
from tango import AttrWriteType, DevState
from tango.server import attribute, command, device_property

from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope
from asyncroscopy.instruments.electron_microscope.adapters.autoscript_apertures import (
    AutoScriptApertureAdapter,
)
from asyncroscopy.instruments.electron_microscope.hardware.aperture import (
    ApertureInfo,
    SimulatedApertureBackend,
)
from asyncroscopy.data.data_writer import DEFAULT_ACQUISITION_DIR, save_acquisition

# AutoScript imports — only available on the microscope PC.
# Wrapped in try/except so the device can still be imported and tested
# on a development machine without AutoScript installed.
try:
    from autoscript_tem_microscope_client import TemMicroscopeClient
    from autoscript_tem_microscope_client.enumerations import EdsDetectorType
    from autoscript_tem_microscope_client.enumerations import CameraType, RegionCoordinateSystem, ExposureTimeType
    from autoscript_tem_microscope_client.structures import Region, Rectangle
    from autoscript_tem_microscope_client.structures import StemAcquisitionSettings, EdsAcquisitionSettings, RunOptiStemSettings, CameraAcquisitionSettings, StemDataSettings

    _AUTOSCRIPT_AVAILABLE = True
except ImportError:
    _AUTOSCRIPT_AVAILABLE = False


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
        self._aperture_logging_ready = True
        self._connect_hardware()
        self._configure_aperture_adapter()
        self._connect_detector_proxies()
        self.set_state(DevState.ON)
        self.screen_current_calibration = None

    def _connect_hardware(self) -> None:
        """Establish AutoScript connection from MPC -> hardware."""
        if not _AUTOSCRIPT_AVAILABLE or self.testing_mode_bool:
            self.warn_stream("AutoScript not available")
            self._microscope = None
            self.is_autoscript = False
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

    def _configure_aperture_adapter(self) -> None:
        """Select the real adapter only for a successful AutoScript connection."""
        if self._microscope is not None and getattr(self, "is_autoscript", False):
            self._aperture_adapter = AutoScriptApertureAdapter(self._microscope)
            self._aperture_autoscript_available = True
            self.info_stream("AutoScript aperture adapter initialised")
        else:
            self._aperture_adapter = SimulatedApertureBackend()
            self._aperture_autoscript_available = False
            self.info_stream("Simulated aperture backend initialised")

    def _connect_detector_proxies(self) -> None:
        """Build DeviceProxy objects for each configured detector device."""
        # Extend this dict as more detectors are added
        # later, we want to do this automatically, not with a dictionary.
        addresses: dict[str, str] = {
            "eds": self.eds_device_address,
            "stage": self.stage_device_address,
            "scan": self.scan_device_address,
            "camera": self.camera_device_address,
            "flucam": self.flucam_device_address,
            "data": self.data_device_address,
        }
        for name, address in addresses.items():
            if not address:  # <-- minimal fix
                self.info_stream(f"Skipping {name}: no address configured")
                continue
            try:
                proxy = tango.DeviceProxy(address)
                proxy.set_timeout_millis(12_000)
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

    @staticmethod
    def _parse_aperture_request(
        json_request: str,
        *,
        require_mechanism: bool,
        require_name: bool = False,
    ) -> dict:
        try:
            request = json.loads(json_request or "{}")
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("Invalid JSON request.") from exc
        if not isinstance(request, dict):
            raise ValueError("Aperture request must be a JSON object.")

        mechanism = request.get("mechanism")
        if require_mechanism and (not isinstance(mechanism, str) or not mechanism):
            raise ValueError("Aperture request requires a non-empty string 'mechanism'.")
        if mechanism is not None and not isinstance(mechanism, str):
            raise ValueError("Aperture request field 'mechanism' must be a string.")

        name = request.get("name")
        if require_name and (not isinstance(name, str) or not name):
            raise ValueError("Aperture request requires a non-empty string 'name'.")
        if name is not None and not isinstance(name, str):
            raise ValueError("Aperture request field 'name' must be a string.")
        return request

    @staticmethod
    def _aperture_response_data(aperture: ApertureInfo | None) -> dict | None:
        if aperture is None:
            return None
        return {
            "name": aperture.name,
            "type": aperture.aperture_type,
            "diameter_m": aperture.diameter_m,
        }

    def _aperture_response(
        self,
        *,
        ok: bool,
        action: str,
        mechanism: str,
        aperture: ApertureInfo | None = None,
        insertion_state: str | None = None,
        error: str | None = None,
        **extra,
    ) -> str:
        payload = {
            "ok": ok,
            "action": action,
            "mechanism": mechanism,
            "aperture": self._aperture_response_data(aperture),
            "insertion_state": insertion_state,
            "autoscript_available": self._aperture_autoscript_available,
            "error": error,
        }
        payload.update(extra)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def _aperture_error(self, action: str, mechanism: str, exc: Exception) -> str:
        message = str(exc)
        self.error_stream(
            f"Aperture action failed: action={action}, mechanism={mechanism!r}, error={message}"
        )
        return self._aperture_response(
            ok=False,
            action=action,
            mechanism=mechanism,
            error=message,
        )

    def _require_aperture_enabled(self, mechanism: str, action: str) -> None:
        try:
            enabled = self._aperture_adapter.is_enabled(mechanism)
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "Aperture status attribute unsupported or missing: "
                "ApertureMechanism.is_enabled."
            ) from exc
        if not enabled:
            raise RuntimeError(
                f"Aperture mechanism {mechanism!r} is disabled; "
                f"refusing to {action}."
            )

    def _require_aperture_retractable(self, mechanism: str, action: str) -> None:
        try:
            retractable = self._aperture_adapter.is_retractable(mechanism)
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "Aperture status attribute unsupported or missing: "
                "ApertureMechanism.is_retractable."
            ) from exc
        if not retractable:
            raise RuntimeError(
                f"Aperture mechanism {mechanism!r} is not retractable; "
                f"refusing to {action}."
            )

    def _assert_no_live_acquisition_for_aperture_mutation(self) -> None:
        if getattr(self, "_acquisition_active", False):
            raise RuntimeError(
                "Aperture mutation is blocked because acquisition is active."
            )

    def _precheck_aperture_mutation(
        self,
        mechanism: str,
        action: str,
        *,
        require_retractable: bool,
    ) -> None:
        self._assert_no_live_acquisition_for_aperture_mutation()
        self._require_aperture_enabled(mechanism, action)
        if require_retractable:
            self._require_aperture_retractable(mechanism, action)


    def _precheck_aperture_disable(self, mechanism: str) -> None:
        try:
            insertion_state = self._aperture_adapter.get_insertion_state(mechanism)
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "Aperture insertion status unsupported or missing: "
                "ApertureMechanism.insertion_state."
            ) from exc
        if insertion_state != "Retracted":
            raise RuntimeError(
                f"Aperture mechanism {mechanism!r} is not safe to disable while "
                f"insertion_state is {insertion_state!r}; retract it first."
            )


    def _warn_aperture_metadata(self, message: str) -> None:
        """Log metadata warnings without breaking lightweight unit instances."""
        if not getattr(self, "_aperture_logging_ready", False):
            return
        try:
            self.warn_stream(message)
        except Exception:
            pass

    def _get_aperture_metadata(self) -> dict[str, Any]:
        """Return a non-fatal snapshot of current aperture state."""
        using_autoscript = bool(
            getattr(self, "_aperture_autoscript_available", False)
        )
        metadata: dict[str, Any] = {
            "source": "autoscript" if using_autoscript else "simulation",
            "reason": (
                None
                if using_autoscript
                else "AutoScript unavailable or testing mode; using simulated aperture state."
            ),
            "mechanisms": [],
            "selected_apertures": {},
            "insertion_states": {},
            "errors": {},
        }

        adapter = getattr(self, "_aperture_adapter", None)
        if adapter is None:
            message = "Aperture adapter is not initialised."
            metadata["source"] = "unavailable"
            metadata["reason"] = message
            metadata["errors"]["adapter"] = message
            self._warn_aperture_metadata(message)
            return metadata

        try:
            mechanisms = list(adapter.list_mechanisms())
            metadata["mechanisms"] = mechanisms
        except Exception as exc:
            message = f"Could not list aperture mechanisms: {exc}"
            metadata["errors"]["adapter"] = message
            self._warn_aperture_metadata(message)
            return metadata

        for mechanism in mechanisms:
            mechanism_errors: list[str] = []
            selected = None
            try:
                apertures = adapter.list_apertures(mechanism)
                selected = next(
                    (aperture for aperture in apertures if aperture.selected),
                    None,
                )
            except Exception as exc:
                mechanism_errors.append(f"selected aperture: {exc}")

            metadata["selected_apertures"][mechanism] = (
                self._aperture_response_data(selected)
            )

            try:
                metadata["insertion_states"][mechanism] = (
                    adapter.get_insertion_state(mechanism)
                )
            except Exception as exc:
                metadata["insertion_states"][mechanism] = None
                mechanism_errors.append(f"insertion_state: {exc}")

            if mechanism_errors:
                message = "; ".join(mechanism_errors)
                metadata["errors"][mechanism] = message
                self._warn_aperture_metadata(
                    f"Could not fully read aperture metadata for {mechanism!r}: {message}"
                )

        return metadata

    @command(dtype_in=str, dtype_out=str)
    def aperture_list(self, json_request: str = "{}") -> str:
        """List available apertures without changing microscope state."""
        action = "list"
        mechanism = ""
        try:
            request = self._parse_aperture_request(
                json_request,
                require_mechanism=False,
            )
            mechanism = request.get("mechanism", "")
            mechanism_names = (
                [mechanism]
                if mechanism
                else self._aperture_adapter.list_mechanisms()
            )
            apertures = [
                self._aperture_response_data(aperture)
                | {"mechanism": aperture.mechanism, "selected": aperture.selected}
                for mechanism_name in mechanism_names
                for aperture in self._aperture_adapter.list_apertures(mechanism_name)
            ]
            insertion_state = (
                self._aperture_adapter.get_insertion_state(mechanism)
                if mechanism
                else None
            )
            return self._aperture_response(
                ok=True,
                action=action,
                mechanism=mechanism or "all",
                insertion_state=insertion_state,
                mechanisms=self._aperture_adapter.list_mechanisms(),
                apertures=apertures,
            )
        except Exception as exc:
            return self._aperture_error(action, mechanism, exc)

    @command(dtype_in=str, dtype_out=str)
    def aperture_get_selected(self, json_request: str) -> str:
        """Return the selected aperture for one explicit mechanism."""
        action = "get_selected"
        mechanism = ""
        try:
            request = self._parse_aperture_request(json_request, require_mechanism=True)
            mechanism = request["mechanism"]
            aperture = self._aperture_adapter.get_selected_aperture(mechanism)
            return self._aperture_response(
                ok=True,
                action=action,
                mechanism=mechanism,
                aperture=aperture,
                insertion_state=self._aperture_adapter.get_insertion_state(mechanism),
            )
        except Exception as exc:
            return self._aperture_error(action, mechanism, exc)

    @command(dtype_in=str, dtype_out=str)
    def aperture_is_enabled(self, json_request: str) -> str:
        """Return whether one explicit aperture mechanism is enabled."""
        action = "is_enabled"
        mechanism = ""
        try:
            request = self._parse_aperture_request(json_request, require_mechanism=True)
            mechanism = request["mechanism"]
            return self._aperture_response(
                ok=True,
                action=action,
                mechanism=mechanism,
                enabled=self._aperture_adapter.is_enabled(mechanism),
            )
        except Exception as exc:
            return self._aperture_error(action, mechanism, exc)

    @command(dtype_in=str, dtype_out=str)
    def aperture_is_retractable(self, json_request: str) -> str:
        """Return whether one explicit aperture mechanism is retractable."""
        action = "is_retractable"
        mechanism = ""
        try:
            request = self._parse_aperture_request(json_request, require_mechanism=True)
            mechanism = request["mechanism"]
            return self._aperture_response(
                ok=True,
                action=action,
                mechanism=mechanism,
                retractable=self._aperture_adapter.is_retractable(mechanism),
            )
        except Exception as exc:
            return self._aperture_error(action, mechanism, exc)

    @command(dtype_in=str, dtype_out=str)
    def aperture_enable(self, json_request: str) -> str:
        """Enable one explicit aperture mechanism when AutoScript supports it."""
        action = "enable"
        mechanism = ""
        try:
            request = self._parse_aperture_request(json_request, require_mechanism=True)
            mechanism = request["mechanism"]
            self._assert_no_live_acquisition_for_aperture_mutation()
            try:
                enabled = self._aperture_adapter.enable(mechanism)
            except (AttributeError, TypeError) as exc:
                raise RuntimeError(
                    "Unsupported AutoScript aperture operation 'enable': "
                    "missing documented method ApertureMechanism.enable."
                ) from exc
            return self._aperture_response(
                ok=True,
                action=action,
                mechanism=mechanism,
                enabled=enabled,
            )
        except Exception as exc:
            return self._aperture_error(action, mechanism, exc)

    @command(dtype_in=str, dtype_out=str)
    def aperture_disable(self, json_request: str) -> str:
        """Disable one retracted aperture mechanism when AutoScript supports it."""
        action = "disable"
        mechanism = ""
        try:
            request = self._parse_aperture_request(json_request, require_mechanism=True)
            mechanism = request["mechanism"]
            self._assert_no_live_acquisition_for_aperture_mutation()
            self._precheck_aperture_disable(mechanism)
            try:
                enabled = self._aperture_adapter.disable(mechanism)
            except (AttributeError, TypeError) as exc:
                raise RuntimeError(
                    "Unsupported AutoScript aperture operation 'disable': "
                    "missing documented method ApertureMechanism.disable."
                ) from exc
            return self._aperture_response(
                ok=True,
                action=action,
                mechanism=mechanism,
                enabled=enabled,
            )
        except Exception as exc:
            return self._aperture_error(action, mechanism, exc)


    @command(dtype_in=str, dtype_out=str)
    def aperture_select(self, json_request: str) -> str:
        """Select an aperture only after an explicit JSON command."""
        action = "select"
        mechanism = ""
        try:
            request = self._parse_aperture_request(
                json_request,
                require_mechanism=True,
                require_name=False,
            )
            mechanism = request["mechanism"]
            name = request.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Aperture request requires a non-empty string 'name'.")
            self.info_stream(
                f"Aperture change requested: action={action}, mechanism={mechanism!r}, name={name!r}"
            )
            self._precheck_aperture_mutation(
                mechanism,
                "select aperture",
                require_retractable=False,
            )
            aperture = self._aperture_adapter.select_aperture(mechanism, name)
            return self._aperture_response(
                ok=True,
                action=action,
                mechanism=mechanism,
                aperture=aperture,
                insertion_state=self._aperture_adapter.get_insertion_state(mechanism),
            )
        except Exception as exc:
            return self._aperture_error(action, mechanism, exc)

    @command(dtype_in=str, dtype_out=str)
    def aperture_insert(self, json_request: str) -> str:
        """Insert one mechanism only after an explicit JSON command."""
        action = "insert"
        mechanism = ""
        try:
            request = self._parse_aperture_request(json_request, require_mechanism=True)
            mechanism = request["mechanism"]
            self.info_stream(
                f"Aperture change requested: action={action}, mechanism={mechanism!r}"
            )
            self._precheck_aperture_mutation(
                mechanism,
                "insert aperture mechanism",
                require_retractable=True,
            )
            insertion_state = self._aperture_adapter.insert_mechanism(mechanism)
            aperture = self._aperture_adapter.get_selected_aperture(mechanism)
            return self._aperture_response(
                ok=True,
                action=action,
                mechanism=mechanism,
                aperture=aperture,
                insertion_state=insertion_state,
            )
        except Exception as exc:
            return self._aperture_error(action, mechanism, exc)

    @command(dtype_in=str, dtype_out=str)
    def aperture_retract(self, json_request: str) -> str:
        """Retract one mechanism only after an explicit JSON command."""
        action = "retract"
        mechanism = ""
        try:
            request = self._parse_aperture_request(json_request, require_mechanism=True)
            mechanism = request["mechanism"]
            self.info_stream(
                f"Aperture change requested: action={action}, mechanism={mechanism!r}"
            )
            self._precheck_aperture_mutation(
                mechanism,
                "retract aperture mechanism",
                require_retractable=True,
            )
            insertion_state = self._aperture_adapter.retract_mechanism(mechanism)
            return self._aperture_response(
                ok=True,
                action=action,
                mechanism=mechanism,
                insertion_state=insertion_state,
            )
        except Exception as exc:
            return self._aperture_error(action, mechanism, exc)


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
        return save_acquisition(
            self,
            data_server,
            "stem_image",
            detector_list,
            adorned,
            file_attrs={"apertures": self._get_aperture_metadata()},
            output_format=output_format,
        )


    def _acquire_camera_image(self, imsize: int, exposure_time: float, detector: str, readout_area: str) -> str:
        """
        Call AutoScript acquisition, save the adorned image, and return its DATA/Tiled key.
        this is the advanced version
        """
        settings = CameraAcquisitionSettings(camera_detector=detector, size=imsize, exposure_time=exposure_time, fixed_readout_area=readout_area, frame_combining=1)
        adorned = self._microscope.acquisition.acquire_camera_image_advanced(settings)
        data_server = self._detector_proxies.get("data")
        return save_acquisition(
            self,
            data_server,
            "camera_image",
            str(detector),
            adorned,
            file_attrs={"apertures": self._get_aperture_metadata()},
        )

    def _acquire_scanned_data_advanced(self, imsize: int, dwell_time: float, detector: str, scan_region: list[float]) -> str:
        """
        Trigger AutoScript advanced scanned data acquisition with a camera detector.

        AutoScript offloads the 4D scanned data storage for Ceta acquisitions, so
        this command returns an acknowledgement and the settings used rather
        than a local saved file path.
        """
        camera_detector = CameraType.BM_CETA if detector == "BM-Ceta" else detector
        settings = StemDataSettings(dwell_time=dwell_time, detector_types=[camera_detector], size=imsize, region=Region(RegionCoordinateSystem.RELATIVE, Rectangle(*scan_region)))
        adorned = self._microscope.acquisition.acquire_stem_data_advanced(settings)
        data_server = self._detector_proxies.get("data")
        return save_acquisition(
            self,
            data_server,
            "stem_data",
            str(detector),
            adorned,
            dataset_name="stem_data",
            file_attrs={"apertures": self._get_aperture_metadata()},
        )

    # test: not sure this is how we want to save
    def _acquire_spectrum(self, detector_name: str, exposure_time: float) -> str:
        settings = EdsAcquisitionSettings()
        settings.eds_detector = EdsDetectorType.SUPER_X
        settings.dispersion = 5
        settings.shaping_time = 3e-6
        settings.exposure_time = exposure_time
        settings.exposure_time_type = ExposureTimeType.LIVE_TIME
        spectrum = self._microscope.analysis.eds.acquire_spectrum(settings)
        data_server = self._detector_proxies.get("data")
        return save_acquisition(
            self,
            data_server,
            "spectrum",
            detector_name,
            spectrum,
            dataset_name="spectrum",
            file_attrs={"apertures": self._get_aperture_metadata()},
        )

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
        
    
    def _get_status(self):
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

    def _move_stage(self, position) -> None:
        """Move stage to specified position [x, y, z, alpha, beta]."""
        # TODO: add beta value check

        x = float(position[0])
        y = float(position[1])
        z = float(position[2])
        alpha = float(math.radians(position[3]))

        if len(position) > 4 and position[4] is not None:
            beta = float(math.radians(position[4]))
        else:
            beta = None

        self._microscope.specimen.stage.absolute_move((x, y, z, alpha, beta))
        # self._get_stage()  # link the proxy with real state

    def _auto_focus(self):
        """Perform autofocus routine C1A1"""
        settings = RunOptiStemSettings(method="C1A1")  # method=OptiStemMethod.C1_A1, dwell_time=2e-06, cutoff_in_pixels=5)
        self._microscope.auto_functions.run_opti_stem(settings)

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
