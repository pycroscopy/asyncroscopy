"""Digital twin for FIB-lamella tilt and diffraction workflows."""

from __future__ import annotations

import json

import numpy as np
import tango
from ase import Atoms
from ase.build import bulk
from tango.server import device_property

from asyncroscopy.data.data_writer import (
    acquisition_filename,
    save_acquisition,
    save_acquisition_hdf5,
)
from asyncroscopy.instruments.electron_microscope.digital_twin import (
    DEFAULT_ACQUISITION_DIR,
    DigitalTwin,
)


class DigitalTwinTilt(DigitalTwin):
    """Silicon FIB-lamella twin with an imperfect, displaced tilt axis."""

    acquisition_save_directory = device_property(
        dtype=str,
        default_value=DEFAULT_ACQUISITION_DIR,
        doc="Directory used for simulated acquisitions.",
    )
    sample_seed = device_property(
        dtype=int,
        default_value=731,
        doc="Seed for sample texture and repeatable instrument randomness.",
    )
    lamella_width_nm = device_property(dtype=float, default_value=620.0)
    lamella_height_nm = device_property(dtype=float, default_value=360.0)
    lamella_thickness_nm = device_property(dtype=float, default_value=80.0)
    gridbar_width_nm = device_property(dtype=float, default_value=220.0)
    layer_period_nm = device_property(dtype=float, default_value=45.0)
    overview_fov_nm = device_property(dtype=float, default_value=1100.0)
    diffraction_image_size = device_property(
        dtype=int,
        default_value=32,
        doc="Reciprocal-space size for simulated 4D-STEM datasets.",
    )
    diffraction_max_angle_mrad = device_property(dtype=float, default_value=60.0)
    convergence_angle_mrad = device_property(
        dtype=float,
        default_value=8.0,
        doc="Probe convergence semi-angle in mrad.",
    )
    silicon_lattice_parameter_angstrom = device_property(
        dtype=float,
        default_value=5.431,
        doc="Unstrained silicon lattice parameter.",
    )
    lattice_parameter_gradient_x_percent = device_property(
        dtype=float,
        default_value=2.0,
        doc="End-to-end lattice-parameter change across the lamella width.",
    )
    lattice_parameter_gradient_y_percent = device_property(
        dtype=float,
        default_value=0.5,
        doc="End-to-end lattice-parameter change across the lamella height.",
    )
    crystal_rotation_deg = device_property(
        dtype=float,
        default_value=1.5,
        doc="Mean in-plane silicon crystal rotation.",
    )
    crystal_rotation_gradient_x_deg = device_property(
        dtype=float,
        default_value=4.0,
        doc="End-to-end in-plane rotation change across the lamella width.",
    )
    crystal_rotation_gradient_y_deg = device_property(
        dtype=float,
        default_value=-2.0,
        doc="End-to-end in-plane rotation change across the lamella height.",
    )
    crystal_tilt_x_mrad = device_property(dtype=float, default_value=3.0)
    crystal_tilt_y_mrad = device_property(dtype=float, default_value=-2.0)
    crystal_tilt_gradient_x_mrad = device_property(
        dtype=float,
        default_value=10.0,
        doc="End-to-end x-tilt change across the lamella width.",
    )
    crystal_tilt_gradient_y_mrad = device_property(
        dtype=float,
        default_value=-7.0,
        doc="End-to-end y-tilt change across the lamella height.",
    )
    electron_energy_ev = device_property(dtype=float, default_value=200_000.0)
    potential_sampling_angstrom = device_property(
        dtype=float,
        default_value=0.08,
        doc="Real-space sampling used to construct each abTEM potential.",
    )
    potential_slice_thickness_angstrom = device_property(
        dtype=float,
        default_value=1.0,
        doc="Multislice potential slice thickness.",
    )
    multislice_supercell_xy = device_property(
        dtype=int,
        default_value=4,
        doc="Silicon unit-cell repetitions in x and y for every probe position.",
    )
    multislice_supercell_z = device_property(
        dtype=int,
        default_value=8,
        doc="Silicon unit-cell repetitions along the beam for every probe position.",
    )
    multislice_vacuum_angstrom = device_property(dtype=float, default_value=2.0)
    frozen_phonon_sigma_angstrom = device_property(
        dtype=float,
        default_value=0.03,
        doc="ASE rattle standard deviation before each multislice calculation.",
    )
    rotation_center_x_nm = device_property(dtype=float, default_value=-55.0)
    rotation_center_y_nm = device_property(dtype=float, default_value=35.0)
    rotation_center_z_nm = device_property(dtype=float, default_value=125.0)
    randomness_scale = device_property(
        dtype=float,
        default_value=1.0,
        doc="Master multiplier for every random instrument imperfection.",
    )
    stage_translation_noise_std_nm = device_property(dtype=float, default_value=0.35)
    tilt_angle_noise_std_deg = device_property(dtype=float, default_value=0.06)
    rotation_center_jitter_std_nm = device_property(dtype=float, default_value=1.5)
    tilt_wobble_std_nm = device_property(dtype=float, default_value=1.0)
    tilt_drift_std_nm_per_degree = device_property(dtype=float, default_value=0.08)
    beam_tilt_noise_std_mrad = device_property(dtype=float, default_value=0.03)
    beam_tilt_image_shift_nm_per_mrad = device_property(dtype=float, default_value=0.8)
    diffraction_center_jitter_std_px = device_property(dtype=float, default_value=0.35)
    detector_noise_fraction = device_property(dtype=float, default_value=0.012)
    focus_drift_nm_per_degree = device_property(dtype=float, default_value=1.4)
    autofocus_residual_std_nm = device_property(dtype=float, default_value=2.0)
    haadf_poisson_counts = device_property(dtype=float, default_value=180.0)

    def init_device(self) -> None:
        super().init_device()
        self._manufacturer = "UTKTeam Silicon Tilt Twin"
        self._initialize_tilt_state()

    def _initialize_tilt_state(self) -> None:
        self._fov = float(self.overview_fov_nm) * 1e-9
        self._beam_pos_x = 0.5
        self._beam_pos_y = 0.5
        self._stage_command = np.zeros(5, dtype=np.float64)
        self._stage_position = np.zeros(5, dtype=np.float64)
        self._image_shift = np.zeros(2, dtype=np.float64)
        self._beam_tilt_command = np.zeros(2, dtype=np.float64)
        self._beam_tilt = np.zeros(2, dtype=np.float64)
        self._diffraction_shift = np.zeros(2, dtype=np.float64)
        self._rotation_center_jitter_nm = np.zeros(3, dtype=np.float64)
        self._tilt_wobble_nm = np.zeros(3, dtype=np.float64)
        self._tilt_drift_nm = np.zeros(2, dtype=np.float64)
        self._move_index = 0
        self._acquisition_index = 0
        self._screen_position = "out"
        self._screen_current_pa = 50.0
        self._defocus = 0.0

    def _connect_detector_proxies(self) -> None:
        addresses = {
            "camera": self.camera_device_address,
            "scan": self.scan_device_address,
            "stage": self.stage_device_address,
            "data": self.data_device_address,
        }
        for name, address in addresses.items():
            if not address:
                continue
            try:
                proxy = tango.DeviceProxy(address)
                proxy.set_timeout_millis(12_000)
                self._detector_proxies[name] = proxy
            except tango.DevFailed as exc:
                self.error_stream(f"Failed to connect to {name} proxy at {address}: {exc}")

    def _generate_sample(self, seed: int) -> None:
        self._fov = float(self.overview_fov_nm) * 1e-9
        width = float(self.lamella_width_nm)
        height = float(self.lamella_height_nm)
        thickness = float(self.lamella_thickness_nm)
        grid_width = float(self.gridbar_width_nm)
        self._world_bounds_ang = {
            "x_min": (-0.5 * width - grid_width) * 10.0,
            "x_max": 0.5 * width * 10.0,
            "y_min": -0.7 * height * 10.0,
            "y_max": 0.7 * height * 10.0,
            "z_min": -0.5 * thickness * 10.0,
            "z_max": 0.5 * thickness * 10.0,
        }
        self._sample_atoms_base = bulk(
            "Si",
            "diamond",
            a=float(self.silicon_lattice_parameter_angstrom),
            cubic=True,
        )
        self._sample_atoms_view = self._sample_atoms_base.copy()
        self._particle_records_base = []
        self._particle_records_view = []
        self._all_sample_elements = ["Si"]
        self._cached_pose_key = None

    def _rng(self, event: str, index: int | None = None) -> np.random.Generator:
        event_code = sum((position + 1) * ord(character) for position, character in enumerate(event))
        sequence = self._acquisition_index if index is None else int(index)
        seed = (int(self.sample_seed) * 1_000_003 + event_code * 9_973 + sequence) % (2**32)
        return np.random.default_rng(seed)

    def _lattice_fields(self, local_x_nm, local_y_nm):
        x_fraction = np.clip(
            np.asarray(local_x_nm) / float(self.lamella_width_nm),
            -0.5,
            0.5,
        )
        y_fraction = np.clip(
            np.asarray(local_y_nm) / float(self.lamella_height_nm),
            -0.5,
            0.5,
        )
        strain = (
            float(self.lattice_parameter_gradient_x_percent) * x_fraction
            + float(self.lattice_parameter_gradient_y_percent) * y_fraction
        ) / 100.0
        lattice_parameter = float(self.silicon_lattice_parameter_angstrom) * (1.0 + strain)
        rotation_deg = (
            float(self.crystal_rotation_deg)
            + float(self.crystal_rotation_gradient_x_deg) * x_fraction
            + float(self.crystal_rotation_gradient_y_deg) * y_fraction
        )
        tilt_x_mrad = float(self.crystal_tilt_x_mrad) + float(
            self.crystal_tilt_gradient_x_mrad
        ) * x_fraction
        tilt_y_mrad = float(self.crystal_tilt_y_mrad) + float(
            self.crystal_tilt_gradient_y_mrad
        ) * y_fraction
        return lattice_parameter, rotation_deg, tilt_x_mrad, tilt_y_mrad

    def _apply_stage_target(self, position) -> None:
        target = np.asarray(position, dtype=np.float64)
        if target.shape != (5,):
            raise ValueError("Stage position must be [x, y, z, alpha, beta]")

        previous_alpha = float(self._stage_command[3])
        previous_beta = float(self._stage_command[4])
        tilt_changed = not np.allclose(target[3:5], self._stage_command[3:5], atol=1e-12)
        self._stage_command = target.copy()
        self._move_index += 1
        rng = self._rng("stage", self._move_index)
        scale = max(0.0, float(self.randomness_scale))

        actual = target.copy()
        actual[:3] += rng.normal(
            0.0,
            float(self.stage_translation_noise_std_nm) * scale * 1e-9,
            size=3,
        )
        if tilt_changed:
            actual[3:5] += rng.normal(
                0.0,
                float(self.tilt_angle_noise_std_deg) * scale,
                size=2,
            )
            self._tilt_wobble_nm = rng.normal(
                0.0,
                float(self.tilt_wobble_std_nm) * scale,
                size=3,
            )
            self._rotation_center_jitter_nm = rng.normal(
                0.0,
                float(self.rotation_center_jitter_std_nm) * scale,
                size=3,
            )
            angle_change = np.hypot(target[3] - previous_alpha, target[4] - previous_beta)
            self._tilt_drift_nm += rng.normal(
                0.0,
                float(self.tilt_drift_std_nm_per_degree) * scale * angle_change,
                size=2,
            )
        self._stage_position = actual
        self._cached_pose_key = None

    def _sync_stage_from_proxy(self) -> None:
        stage = self._detector_proxies.get("stage")
        if stage is None:
            return
        try:
            target = np.asarray(stage.position, dtype=np.float64)
        except (AttributeError, tango.DevFailed):
            try:
                target = np.array(
                    [stage.x, stage.y, stage.z, stage.alpha, stage.beta],
                    dtype=np.float64,
                )
            except tango.DevFailed:
                self.error_stream("Failed to read stage proxy; using internal stage state.")
                return
        if not np.allclose(target, self._stage_command, rtol=0.0, atol=1e-15):
            self._apply_stage_target(target)

    def _rotation_matrix(self) -> np.ndarray:
        alpha, beta = np.radians(self._stage_position[3:5])
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        cos_b, sin_b = np.cos(beta), np.sin(beta)
        rotate_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]]
        )
        rotate_y = np.array(
            [[cos_b, 0.0, sin_b], [0.0, 1.0, 0.0], [-sin_b, 0.0, cos_b]]
        )
        return rotate_y @ rotate_x

    def _sample_map(
        self,
        imsize: int,
        scan_region: list[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._sync_stage_from_proxy()
        size = int(imsize)
        left, top, width, height = scan_region or [0.0, 0.0, 1.0, 1.0]
        columns = left + (np.arange(size) + 0.5) / size * width
        rows = top + (np.arange(size) + 0.5) / size * height
        screen_x = (columns - 0.5) * self._fov * 1e9
        screen_y = (rows - 0.5) * self._fov * 1e9
        xx, yy = np.meshgrid(screen_x, screen_y)

        rotation = self._rotation_matrix()
        center = np.array(
            [
                float(self.rotation_center_x_nm),
                float(self.rotation_center_y_nm),
                float(self.rotation_center_z_nm),
            ]
        )
        center += self._rotation_center_jitter_nm
        stage_nm = self._stage_position[:3] * 1e9
        beam_shift_nm = self._beam_tilt * 1e3 * float(self.beam_tilt_image_shift_nm_per_mrad)
        offset = center - rotation @ center - stage_nm + self._tilt_wobble_nm
        offset[:2] += self._tilt_drift_nm + self._image_shift * 1e9 + beam_shift_nm

        projection = rotation[:2, :2]
        inverse_projection = np.linalg.inv(projection)
        local_x = inverse_projection[0, 0] * (xx - offset[0]) + inverse_projection[0, 1] * (
            yy - offset[1]
        )
        local_y = inverse_projection[1, 0] * (xx - offset[0]) + inverse_projection[1, 1] * (
            yy - offset[1]
        )

        lamella = (
            (np.abs(local_x) <= 0.5 * float(self.lamella_width_nm))
            & (np.abs(local_y) <= 0.5 * float(self.lamella_height_nm))
        )
        gridbar = (
            (local_x >= -0.5 * float(self.lamella_width_nm) - float(self.gridbar_width_nm))
            & (local_x < -0.5 * float(self.lamella_width_nm))
            & (np.abs(local_y) <= 0.68 * float(self.lamella_height_nm))
        )

        layer_phase = np.floor(
            (local_y + 0.5 * float(self.lamella_height_nm)) / float(self.layer_period_nm)
        )
        layers = 0.5 * (1.0 + np.where(layer_phase % 2 == 0, 1.0, -1.0))
        wedge = np.clip(
            0.68 + 0.30 * (local_x / float(self.lamella_width_nm) + 0.5),
            0.55,
            1.0,
        )
        curtains = 0.025 * np.sin(2.0 * np.pi * local_x / 38.0) ** 2
        texture = 0.025 * np.sin(local_x / 21.0 + local_y / 33.0)
        image = np.full((size, size), 0.008, dtype=np.float64)
        image[lamella] = (
            0.28 + 0.11 * layers[lamella] + 0.22 * wedge[lamella] + texture[lamella]
        )
        image[lamella] += curtains[lamella]
        image[gridbar] = 0.97

        projected_thickness = float(self.lamella_thickness_nm) / max(
            abs(float(rotation[2, 2])), 0.25
        )
        image[lamella] *= np.clip(projected_thickness / float(self.lamella_thickness_nm), 1.0, 2.3)
        return np.clip(image, 0.0, 1.0), lamella, gridbar, local_x, local_y

    def _render_stem_image(
        self,
        imsize: int,
        dwell_time: float,
        detector_list: list,
        scan_region: list[float] | None = None,
    ) -> np.ndarray:
        image, _lamella, _gridbar, _local_x, _local_y = self._sample_map(
            imsize,
            scan_region,
        )
        true_focus = self._stage_position[2] + (
            float(self.focus_drift_nm_per_degree)
            * np.sin(np.radians(self._stage_position[3]))
            * 1e-9
        )
        blur_sigma = min(4.0, abs(self._defocus - true_focus) * 1e9 / 18.0)
        if blur_sigma > 0.05:
            fy = np.fft.fftfreq(image.shape[0])[:, None]
            fx = np.fft.fftfreq(image.shape[1])[None, :]
            transfer = np.exp(-2.0 * np.pi**2 * blur_sigma**2 * (fx**2 + fy**2))
            image = np.fft.ifft2(np.fft.fft2(image) * transfer).real

        self._acquisition_index += 1
        rng = self._rng("overview")
        counts = max(
            1.0,
            float(self.haadf_poisson_counts) * max(float(dwell_time), 1e-7) / 1e-6,
        )
        image = rng.poisson(np.clip(image, 0.0, 1.0) * counts) / counts
        image += rng.normal(
            0.0,
            float(self.detector_noise_fraction) * max(0.0, float(self.randomness_scale)),
            size=image.shape,
        )
        return np.clip(image, 0.0, 1.0).astype(np.float32)

    def _beam_material(self) -> tuple[str, float, float]:
        px, py = self.read_beam_pos()
        image, lamella, gridbar, local_x, local_y = self._sample_map(512)
        column = int(np.clip(round(float(px) * 511), 0, 511))
        row = int(np.clip(round(float(py) * 511), 0, 511))
        if gridbar[row, column]:
            material = "gridbar"
        elif lamella[row, column]:
            material = "silicon"
        else:
            material = "vacuum"
        return material, float(local_x[row, column]), float(local_y[row, column])

    def _silicon_atoms(self, local_x_nm: float, local_y_nm: float, event_index: int):
        lattice_parameter, rotation_deg, tilt_x_mrad, tilt_y_mrad = self._lattice_fields(
            local_x_nm,
            local_y_nm,
        )
        lattice_parameter = float(lattice_parameter)
        rotation_deg = float(rotation_deg)
        tilt_x_mrad = float(tilt_x_mrad)
        tilt_y_mrad = float(tilt_y_mrad)
        repetitions = (
            int(self.multislice_supercell_xy),
            int(self.multislice_supercell_xy),
            int(self.multislice_supercell_z),
        )
        atoms = bulk(
            "Si",
            "diamond",
            a=lattice_parameter,
            cubic=True,
        ).repeat(repetitions)
        atoms.rotate(rotation_deg, "z", center="COP")
        atoms.rotate(
            float(self._stage_position[3]) + np.degrees(tilt_x_mrad * 1e-3),
            "x",
            center="COP",
        )
        atoms.rotate(
            float(self._stage_position[4]) + np.degrees(tilt_y_mrad * 1e-3),
            "y",
            center="COP",
        )
        sigma = (
            float(self.frozen_phonon_sigma_angstrom)
            * max(0.0, float(self.randomness_scale))
        )
        if sigma > 0.0:
            seed = int(self._rng("frozen_phonon", event_index).integers(0, 2**31 - 1))
            atoms.rattle(stdev=sigma, seed=seed)
        atoms.center(vacuum=float(self.multislice_vacuum_angstrom))
        atoms.info.update(
            {
                "local_x_nm": float(local_x_nm),
                "local_y_nm": float(local_y_nm),
                "lattice_parameter_angstrom": lattice_parameter,
                "nearest_neighbor_distance_angstrom": lattice_parameter * np.sqrt(3.0) / 4.0,
                "crystal_rotation_deg": rotation_deg,
                "crystal_tilt_x_mrad": tilt_x_mrad,
                "crystal_tilt_y_mrad": tilt_y_mrad,
            }
        )
        return atoms

    def _vacuum_diffraction(self, imsize: int) -> np.ndarray:
        lateral_size = (
            float(self.silicon_lattice_parameter_angstrom)
            * int(self.multislice_supercell_xy)
            + 2.0 * float(self.multislice_vacuum_angstrom)
        )
        atoms = Atoms(
            cell=[lateral_size, lateral_size, 10.0],
            pbc=(False, False, False),
        )
        return self._simulate_diffraction_from_atoms(atoms, imsize)

    def _simulate_diffraction_from_atoms(self, atoms, imsize: int) -> np.ndarray:
        try:
            import abtem
        except ImportError as exc:
            raise RuntimeError(
                "abTEM is required for the tilt twin. "
                "Install it with `uv sync --extra diffraction`."
            ) from exc

        potential = abtem.Potential(
            atoms,
            sampling=float(self.potential_sampling_angstrom),
            slice_thickness=float(self.potential_slice_thickness_angstrom),
        )
        probe = abtem.Probe(
            energy=float(self.electron_energy_ev),
            semiangle_cutoff=float(self.convergence_angle_mrad),
            tilt=tuple(self._beam_tilt * 1e3),
        )
        probe.grid.match(potential)
        exit_wave = probe.multislice(potential)
        pattern = exit_wave.diffraction_patterns(
            max_angle=float(self.diffraction_max_angle_mrad)
        ).compute()
        array = np.asarray(getattr(pattern, "array", pattern), dtype=np.float32)
        array = np.squeeze(array)
        if array.ndim > 2:
            array = array.reshape((-1,) + array.shape[-2:])[0]
        result = self._resize_nearest(array, int(imsize))
        pixel_size_mrad = 2.0 * float(self.diffraction_max_angle_mrad) / int(imsize)
        shift_px = self._diffraction_shift * 1e3 / pixel_size_mrad
        shift_px += self._rng("diffraction_center").normal(
            0.0,
            float(self.diffraction_center_jitter_std_px)
            * max(0.0, float(self.randomness_scale)),
            size=2,
        )
        if np.any(shift_px):
            fy = np.fft.fftfreq(result.shape[0])[:, None]
            fx = np.fft.fftfreq(result.shape[1])[None, :]
            phase = np.exp(-2j * np.pi * (fy * shift_px[1] + fx * shift_px[0]))
            result = np.fft.ifft2(np.fft.fft2(result) * phase).real.astype(np.float32)
        return np.clip(result, 0.0, None)

    @staticmethod
    def _resize_nearest(array: np.ndarray, imsize: int) -> np.ndarray:
        if array.shape == (imsize, imsize):
            result = array.astype(np.float32)
        else:
            yi = np.rint(np.linspace(0, array.shape[0] - 1, imsize)).astype(int)
            xi = np.rint(np.linspace(0, array.shape[1] - 1, imsize)).astype(int)
            result = array[np.ix_(yi, xi)].astype(np.float32)
        result -= float(result.min())
        maximum = float(result.max())
        return result / maximum if maximum > 0.0 else result

    def _acquire_scanned_image(
        self,
        imsize: int,
        dwell_time: float,
        detector_list: list[str] = ["haadf"],
        scan_region: list[float] = [0.0, 0.0, 1.0, 1.0],
        output_format: str = ".h5",
    ) -> str:
        detector_list = [detector.upper() for detector in detector_list]
        images = [
            self._render_stem_image(int(imsize), float(dwell_time), [detector], scan_region)
            for detector in detector_list
        ]
        attrs = {
            "pixel_size_nm": float(self._fov * 1e9 / int(imsize)),
            "fov_nm": float(self._fov * 1e9),
            "stage_command": self._stage_command.tolist(),
            "stage_actual": self._stage_position.tolist(),
            "rotation_center_nm": [
                float(self.rotation_center_x_nm),
                float(self.rotation_center_y_nm),
                float(self.rotation_center_z_nm),
            ],
        }
        return save_acquisition(
            self,
            self._detector_proxies.get("data"),
            "stem_image",
            detector_list,
            images,
            dataset_attrs=attrs,
            output_format=output_format,
        )

    def _acquire_camera_image(
        self,
        imsize: int,
        exposure_time: float,
        detector: str,
        readout_area: str,
        frame_combining: int = 1,
        electron_counting: bool = True,
        output_format: str = ".h5",
    ) -> str:
        self._sync_stage_from_proxy()
        material, local_x, local_y = self._beam_material()
        self._acquisition_index += 1
        atoms = None
        if material == "silicon":
            atoms = self._silicon_atoms(local_x, local_y, self._acquisition_index)
            diffraction = self._simulate_diffraction_from_atoms(atoms, int(imsize))
        elif material == "vacuum":
            diffraction = self._vacuum_diffraction(int(imsize))
        else:
            diffraction = np.zeros((int(imsize), int(imsize)), dtype=np.float32)
        rng = self._rng("camera_detector")
        diffraction += rng.normal(
            0.0,
            float(self.detector_noise_fraction) * max(0.0, float(self.randomness_scale)),
            size=diffraction.shape,
        ).astype(np.float32)
        diffraction = np.clip(diffraction, 0.0, None)
        attrs = {
            "material": material,
            "local_x_nm": local_x,
            "local_y_nm": local_y,
            "pixel_size_mrad": float(2.0 * float(self.diffraction_max_angle_mrad) / int(imsize)),
            "max_angle_mrad": float(self.diffraction_max_angle_mrad),
            "convergence_angle_mrad": float(self.convergence_angle_mrad),
            "stage_command": self._stage_command.tolist(),
            "stage_actual": self._stage_position.tolist(),
            "beam_tilt_mrad": (self._beam_tilt * 1e3).tolist(),
            "image_shift_m": self._image_shift.tolist(),
            "defocus_m": float(self._defocus),
            "exposure_time": float(exposure_time),
            "readout_area": str(readout_area),
            "frame_combining": int(frame_combining),
            "electron_counting": bool(electron_counting),
            "simulation": "ASE silicon + abTEM multislice",
            "electron_energy_ev": float(self.electron_energy_ev),
            "potential_sampling_angstrom": float(self.potential_sampling_angstrom),
            "potential_slice_thickness_angstrom": float(
                self.potential_slice_thickness_angstrom
            ),
        }
        if atoms is not None:
            attrs.update(atoms.info)
        return save_acquisition(
            self,
            self._detector_proxies.get("data"),
            "diffraction",
            str(detector),
            diffraction,
            dataset_name="image",
            dataset_attrs=attrs,
            output_format=output_format,
        )

    def _acquire_scanned_data_advanced(
        self,
        imsize: int,
        dwell_time: float,
        detector: str,
        scan_region: list[float],
    ) -> str:
        self._sync_stage_from_proxy()
        scan_size = int(imsize)
        dp_size = int(self.diffraction_image_size)
        self._acquisition_index += 1
        _overview, lamella, gridbar, local_x, local_y = self._sample_map(
            scan_size,
            scan_region,
        )
        vacuum = self._vacuum_diffraction(dp_size)
        datacube = np.broadcast_to(
            vacuum,
            (scan_size, scan_size, dp_size, dp_size),
        ).copy()
        lattice_parameter = np.full((scan_size, scan_size), np.nan, dtype=np.float32)
        bond_distance = np.full_like(lattice_parameter, np.nan)
        crystal_rotation = np.full_like(lattice_parameter, np.nan)
        crystal_tilt_x = np.full_like(lattice_parameter, np.nan)
        crystal_tilt_y = np.full_like(lattice_parameter, np.nan)

        for point_index, (row, column) in enumerate(np.argwhere(lamella)):
            atoms = self._silicon_atoms(
                float(local_x[row, column]),
                float(local_y[row, column]),
                self._acquisition_index * scan_size**2 + point_index,
            )
            datacube[row, column] = self._simulate_diffraction_from_atoms(atoms, dp_size)
            lattice_parameter[row, column] = atoms.info["lattice_parameter_angstrom"]
            bond_distance[row, column] = atoms.info["nearest_neighbor_distance_angstrom"]
            crystal_rotation[row, column] = atoms.info["crystal_rotation_deg"]
            crystal_tilt_x[row, column] = atoms.info["crystal_tilt_x_mrad"]
            crystal_tilt_y[row, column] = atoms.info["crystal_tilt_y_mrad"]

        datacube[gridbar] = 0.0
        rng = self._rng("4dstem")
        datacube += rng.normal(
            0.0,
            float(self.detector_noise_fraction) * max(0.0, float(self.randomness_scale)),
            size=datacube.shape,
        ).astype(np.float32)
        datacube = np.clip(datacube, 0.0, None).astype(np.float32)
        attrs = {
            "scan_shape": [scan_size, scan_size],
            "diffraction_shape": [dp_size, dp_size],
            "pixel_size_mrad": float(2.0 * float(self.diffraction_max_angle_mrad) / dp_size),
            "convergence_angle_mrad": float(self.convergence_angle_mrad),
            "beam_tilt_mrad": (self._beam_tilt * 1e3).tolist(),
            "stage_actual": self._stage_position.tolist(),
            "simulation": "one ASE silicon supercell and abTEM multislice calculation per scan point",
            "electron_energy_ev": float(self.electron_energy_ev),
            "potential_sampling_angstrom": float(self.potential_sampling_angstrom),
            "potential_slice_thickness_angstrom": float(
                self.potential_slice_thickness_angstrom
            ),
            "multislice_supercell": [
                int(self.multislice_supercell_xy),
                int(self.multislice_supercell_xy),
                int(self.multislice_supercell_z),
            ],
        }
        data_server = self._detector_proxies.get("data")
        path = acquisition_filename(self, "stem_data", str(detector), data_server)
        save_acquisition_hdf5(
            path,
            [
                {"name": "stem_data", "source": datacube, "attrs": attrs},
                {
                    "name": "simulation/lattice_parameter_angstrom",
                    "source": lattice_parameter,
                    "attrs": {"description": "Local ASE cubic cell parameter"},
                },
                {
                    "name": "simulation/bond_distance_angstrom",
                    "source": bond_distance,
                    "attrs": {"description": "Local Si nearest-neighbor distance"},
                },
                {
                    "name": "simulation/crystal_rotation_deg",
                    "source": crystal_rotation,
                    "attrs": {"description": "Local in-plane ASE rotation"},
                },
                {
                    "name": "simulation/crystal_tilt_x_mrad",
                    "source": crystal_tilt_x,
                    "attrs": {"description": "Local ASE x tilt before stage tilt"},
                },
                {
                    "name": "simulation/crystal_tilt_y_mrad",
                    "source": crystal_tilt_y,
                    "attrs": {"description": "Local ASE y tilt before stage tilt"},
                },
            ],
        )
        return data_server.register_path(str(path)) if data_server is not None else str(path)

    def _move_stage(self, position) -> None:
        target = np.asarray(position, dtype=np.float64)
        if target.shape != (5,):
            raise ValueError("Stage position must be [x, y, z, alpha, beta]")
        stage = self._detector_proxies.get("stage")
        if stage is not None:
            try:
                stage.position = target.tolist()
            except (AttributeError, tango.DevFailed):
                stage.x, stage.y, stage.z = target[:3]
                stage.alpha, stage.beta = target[3:]
        self._apply_stage_target(target)

    def _get_stage(self):
        self._sync_stage_from_proxy()
        return self._stage_position

    def _set_image_shift(self, shift) -> None:
        value = np.asarray(shift, dtype=np.float64)
        if value.shape != (2,):
            raise ValueError("Image shift must be [x, y] in meters")
        self._image_shift = value

    def _get_image_shift(self):
        return self._image_shift

    def _set_beam_tilt(self, tilt) -> None:
        command = np.asarray(tilt, dtype=np.float64)
        if command.shape != (2,):
            raise ValueError("Beam tilt must be [x, y] in radians")
        self._beam_tilt_command = command
        self._move_index += 1
        noise = self._rng("beam_tilt", self._move_index).normal(
            0.0,
            float(self.beam_tilt_noise_std_mrad)
            * max(0.0, float(self.randomness_scale))
            * 1e-3,
            size=2,
        )
        self._beam_tilt = command + noise

    def _get_beam_tilt(self):
        return self._beam_tilt

    def _set_diffraction_shift(self, shift) -> None:
        value = np.asarray(shift, dtype=np.float64)
        if value.shape != (2,):
            raise ValueError("Diffraction shift must be [x, y] in radians")
        self._diffraction_shift = value

    def _get_diffraction_shift(self):
        return self._diffraction_shift

    def _auto_focus(self) -> None:
        self._sync_stage_from_proxy()
        true_focus = self._stage_position[2] + (
            float(self.focus_drift_nm_per_degree)
            * np.sin(np.radians(self._stage_position[3]))
            * 1e-9
        )
        self._acquisition_index += 1
        residual = self._rng("autofocus").normal(
            0.0,
            float(self.autofocus_residual_std_nm)
            * max(0.0, float(self.randomness_scale))
            * 1e-9,
        )
        self._defocus = float(true_focus + residual)

    def _set_screen(self, position: str) -> None:
        self._screen_position = str(position)

    def _set_screen_current(self, current) -> None:
        self._screen_current_pa = float(current)

    def _calibrate_screen_current(self) -> None:
        return None

    def _get_screen_current(self):
        return float(self._screen_current_pa)

    def _set_fov(self, fov) -> None:
        self._fov = float(fov)

    def _get_fov(self) -> float:
        return float(self._fov)

    def _get_parameters(self) -> str:
        self._sync_stage_from_proxy()
        return json.dumps(
            {
                "sample": "bulk silicon FIB lamella",
                "stage_command": self._stage_command.tolist(),
                "stage_actual": self._stage_position.tolist(),
                "rotation_center_nm": [
                    float(self.rotation_center_x_nm),
                    float(self.rotation_center_y_nm),
                    float(self.rotation_center_z_nm),
                ],
                "image_shift_m": self._image_shift.tolist(),
                "beam_tilt_mrad": (self._beam_tilt * 1e3).tolist(),
                "convergence_angle_mrad": float(self.convergence_angle_mrad),
                "diffraction_image_size": int(self.diffraction_image_size),
                "diffraction_max_angle_mrad": float(self.diffraction_max_angle_mrad),
                "lamella_width_nm": float(self.lamella_width_nm),
                "lamella_height_nm": float(self.lamella_height_nm),
                "lamella_thickness_nm": float(self.lamella_thickness_nm),
                "silicon_lattice_parameter_angstrom": float(
                    self.silicon_lattice_parameter_angstrom
                ),
                "lattice_parameter_gradient_x_percent": float(
                    self.lattice_parameter_gradient_x_percent
                ),
                "lattice_parameter_gradient_y_percent": float(
                    self.lattice_parameter_gradient_y_percent
                ),
                "crystal_rotation_deg": float(self.crystal_rotation_deg),
                "crystal_rotation_gradient_x_deg": float(
                    self.crystal_rotation_gradient_x_deg
                ),
                "crystal_rotation_gradient_y_deg": float(
                    self.crystal_rotation_gradient_y_deg
                ),
                "crystal_tilt_x_mrad": float(self.crystal_tilt_x_mrad),
                "crystal_tilt_y_mrad": float(self.crystal_tilt_y_mrad),
                "crystal_tilt_gradient_x_mrad": float(
                    self.crystal_tilt_gradient_x_mrad
                ),
                "crystal_tilt_gradient_y_mrad": float(
                    self.crystal_tilt_gradient_y_mrad
                ),
                "electron_energy_ev": float(self.electron_energy_ev),
                "potential_sampling_angstrom": float(self.potential_sampling_angstrom),
                "potential_slice_thickness_angstrom": float(
                    self.potential_slice_thickness_angstrom
                ),
                "multislice_supercell": [
                    int(self.multislice_supercell_xy),
                    int(self.multislice_supercell_xy),
                    int(self.multislice_supercell_z),
                ],
                "frozen_phonon_sigma_angstrom": float(
                    self.frozen_phonon_sigma_angstrom
                ),
                "randomness_scale": float(self.randomness_scale),
            }
        )


if __name__ == "__main__":
    DigitalTwinTilt.run_server()
