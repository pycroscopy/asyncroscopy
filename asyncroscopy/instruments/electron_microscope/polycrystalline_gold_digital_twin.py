"""Atomistic polycrystalline gold digital twin."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pyTEMlib.image_tools as image_tools
import pyTEMlib.probe_tools as probe_tools
from ase.build import bulk
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from tango.server import command, device_property

from asyncroscopy.data.data_writer import save_acquisition
from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin


@dataclass(frozen=True)
class GrainParameters:
    label: int
    occupied: bool
    center_angstrom: tuple[float, float, float]
    zone_axis: tuple[int, int, int]
    zone_offset_deg: float
    orientation_euler_deg: tuple[float, float, float]
    composition: dict[str, float]


class PolycrystallineGoldDigitalTwin(DigitalTwin):
    """Atomistic FCC Au grains rendered through a digital-twin corrector probe."""

    volume_size_xy_nm = device_property(dtype=float, default_value=200.0)
    volume_thickness_nm = device_property(dtype=float, default_value=20.0)
    grain_size_angstrom = device_property(dtype=float, default_value=10.0)
    empty_grain_fraction = device_property(dtype=float, default_value=0.15)
    lattice_constant_angstrom = device_property(dtype=float, default_value=4.08)
    zone_axis_max_index = device_property(dtype=int, default_value=2)
    zone_axis_max_deviation_deg = device_property(dtype=float, default_value=2.0)
    acceleration_voltage_ev = device_property(dtype=float, default_value=200_000.0)
    convergence_angle_mrad = device_property(dtype=float, default_value=30.0)
    beam_current_pa = device_property(dtype=float, default_value=100.0)
    blur_noise_level = device_property(dtype=float, default_value=0.5)

    def init_device(self) -> None:
        super().init_device()
        self._fov = 20e-9
        self._manufacturer = "UTKTeam Atomistic Polycrystalline Gold Digital Twin"

    def _generate_sample(self, seed: int) -> None:
        sample_xy_angstrom = float(self.volume_size_xy_nm) * 10.0
        sample_z_angstrom = float(self.volume_thickness_nm) * 10.0
        grain_size_angstrom = float(self.grain_size_angstrom)
        empty_grain_fraction = float(self.empty_grain_fraction)
        zone_axis_max_index = int(self.zone_axis_max_index)
        zone_axis_max_deviation_deg = float(self.zone_axis_max_deviation_deg)
        if sample_xy_angstrom <= 0 or sample_z_angstrom <= 0 or grain_size_angstrom <= 0:
            raise ValueError("Sample dimensions and grain size must be positive")
        if not 0.0 <= empty_grain_fraction < 1.0:
            raise ValueError("empty_grain_fraction must be in [0, 1)")
        if zone_axis_max_index < 1 or zone_axis_max_deviation_deg < 0.0:
            raise ValueError("Zone-axis index must be positive and deviation must be nonnegative")

        rng = np.random.default_rng(int(seed))
        grain_count = max(1, int(sample_xy_angstrom**2 // (4.0 / 3.0 * np.pi * grain_size_angstrom**2)))
        sample_size = np.array([sample_xy_angstrom, sample_xy_angstrom, sample_z_angstrom])
        self._grain_centers = rng.random((grain_count, 3)) * sample_size
        empty_count = int(grain_count * empty_grain_fraction)
        self._empty_grain_indices = set(int(index) for index in rng.choice(grain_count, empty_count, replace=False))
        zone_axes = [(n, m, ell) for n in range(zone_axis_max_index + 1) for m in range(n, zone_axis_max_index + 1) for ell in range(m, zone_axis_max_index + 1) if (n, m, ell) != (0, 0, 0) and np.gcd.reduce((n, m, ell)) == 1]
        self._grain_zone_axes = [zone_axes[int(rng.integers(len(zone_axes)))] for _index in range(grain_count)]
        self._grain_zone_offsets = []
        self._grain_rotations = []
        self._grain_angles = []
        for zone_axis in self._grain_zone_axes:
            zone_direction = np.asarray(zone_axis, dtype=float)
            zone_direction /= np.linalg.norm(zone_direction)
            while True:
                tilt_xy_deg = rng.normal(0.0, zone_axis_max_deviation_deg / 3.0, 2) if zone_axis_max_deviation_deg > 0.0 else np.zeros(2)
                if np.linalg.norm(tilt_xy_deg) <= zone_axis_max_deviation_deg:
                    break
            alignment, _error = Rotation.align_vectors([[0.0, 0.0, 1.0]], [zone_direction])
            tilt = Rotation.from_rotvec(np.radians([tilt_xy_deg[0], tilt_xy_deg[1], 0.0]))
            in_plane = Rotation.from_rotvec([0.0, 0.0, rng.uniform(0.0, 2.0 * np.pi)])
            rotation = in_plane * tilt * alignment
            self._grain_zone_offsets.append(float(np.linalg.norm(tilt_xy_deg)))
            self._grain_rotations.append(rotation.as_matrix())
            self._grain_angles.append(tuple(float(value) for value in rotation.as_euler("ZYX", degrees=True)))
        self._grain_tree = cKDTree(self._grain_centers)
        self._region_parameters = [GrainParameters(index + 1, index not in self._empty_grain_indices, tuple(float(value) for value in self._grain_centers[index]), self._grain_zone_axes[index], self._grain_zone_offsets[index], self._grain_angles[index], {"Au": 1.0} if index not in self._empty_grain_indices else {}) for index in range(grain_count)]
        self._world_bounds_ang = {"x_min": 0.0, "x_max": sample_xy_angstrom, "y_min": 0.0, "y_max": sample_xy_angstrom, "z_min": 0.0, "z_max": sample_z_angstrom}
        self._all_sample_elements = ["Au"]
        self._particle_records_base = []
        self._particle_records_view = []
        self._last_rendered_atom_count = 0
        self._raw_potential_cache = None
        self._raw_potential_cache_key = None
        self._raw_potential_cache_atom_count = 0
        self._raw_potential_cache_hits = 0
        self._sample_metadata = {"sample_type": "atomistic_polycrystalline_gold_slab", "volume_size_xy_nm": float(self.volume_size_xy_nm), "volume_thickness_nm": float(self.volume_thickness_nm), "grain_size_angstrom": grain_size_angstrom, "lattice_constant_angstrom": float(self.lattice_constant_angstrom), "grain_count": grain_count, "occupied_grain_count": grain_count - empty_count, "empty_grain_count": empty_count, "zone_axis_families": [list(axis) for axis in zone_axes], "zone_axis_max_deviation_deg": zone_axis_max_deviation_deg, "corrector_backend": "DigitalTwinCorrector", "probe_backend": "pyTEMlib.probe_tools.get_probe", "acceleration_voltage_ev": float(self.acceleration_voltage_ev), "convergence_angle_mrad": float(self.convergence_angle_mrad)}

    def _corrector_coefficients(self) -> dict:
        corrector = self._detector_proxies.get("corrector")
        if corrector is None:
            raise RuntimeError("PolycrystallineGoldDigitalTwin requires the corrector digital twin")
        info = json.loads(corrector.get_info()).get("result", {})
        if not info.get("simulation") or info.get("model") != "DigitalTwinCorrector":
            raise RuntimeError("PolycrystallineGoldDigitalTwin requires DigitalTwinCorrector")
        coefficients = json.loads(corrector.get_aberrations_coeff_sim())
        if not coefficients:
            raise RuntimeError("DigitalTwinCorrector returned no aberrations")
        return coefficients

    def _set_defocus(self, defocus) -> None:
        coefficients = self._corrector_coefficients()
        coefficients["C1"] = [float(defocus)]
        self._detector_proxies["corrector"].set_aberrations_coeff_sim(json.dumps(coefficients))
        self._defocus = float(defocus)

    def _get_defocus(self) -> float:
        coefficients = self._corrector_coefficients()
        return float(coefficients.get("C1", [self._defocus])[0])

    def _raw_sample_potential(self, imsize: int) -> np.ndarray:
        self._sync_stage_from_proxy()
        self._imsize = int(imsize)
        fov_angstrom = float(self._fov) * 1e10
        pixel_size_angstrom = fov_angstrom / imsize
        edge_crop = max(12, int(round(0.06 * imsize)))
        padded_size = imsize + 2 * edge_crop
        padded_fov_angstrom = pixel_size_angstrom * padded_size
        cache_key = (int(imsize), round(fov_angstrom, 9), tuple(np.round(self._stage_position, 12)))
        if cache_key == self._raw_potential_cache_key:
            self._raw_potential_cache_hits += 1
            self._last_rendered_atom_count = self._raw_potential_cache_atom_count
            return self._raw_potential_cache
        sample_xy_angstrom = float(self.volume_size_xy_nm) * 10.0
        sample_z_angstrom = float(self.volume_thickness_nm) * 10.0
        center_x = sample_xy_angstrom / 2.0 + self._stage_position[0] * 1e10
        center_y = sample_xy_angstrom / 2.0 + self._stage_position[1] * 1e10
        x_min = center_x - padded_fov_angstrom / 2.0
        x_max = center_x + padded_fov_angstrom / 2.0
        y_min = center_y - padded_fov_angstrom / 2.0
        y_max = center_y + padded_fov_angstrom / 2.0

        grid_x = np.linspace(x_min, x_max, 12)
        grid_y = np.linspace(y_min, y_max, 12)
        grid_z = np.linspace(0.0, sample_z_angstrom, 8)
        query_points = np.stack(np.meshgrid(grid_x, grid_y, grid_z, indexing="ij"), axis=-1).reshape(-1, 3)
        neighbor_count = min(8, len(self._grain_centers))
        neighbor_distances, neighbor_indices = self._grain_tree.query(query_points, k=neighbor_count)
        candidate_indices = np.unique(np.asarray(neighbor_indices).reshape(-1))
        local_radius = max(float(np.max(neighbor_distances)) * 2.0, float(self.grain_size_angstrom) * 2.0)
        gold = bulk("Au", "fcc", a=float(self.lattice_constant_angstrom), cubic=True)
        repeat = int(np.ceil(2.0 * local_radius / float(self.lattice_constant_angstrom))) + 2
        supercell = gold.repeat((repeat, repeat, repeat))
        base_positions = supercell.get_positions()
        base_positions -= base_positions.mean(axis=0)
        all_positions = []

        for grain_index in candidate_indices:
            if int(grain_index) in self._empty_grain_indices:
                continue
            positions = base_positions @ self._grain_rotations[int(grain_index)].T + self._grain_centers[int(grain_index)]
            inside = (positions[:, 0] >= x_min) & (positions[:, 0] < x_max) & (positions[:, 1] >= y_min) & (positions[:, 1] < y_max) & (positions[:, 2] >= 0.0) & (positions[:, 2] < sample_z_angstrom)
            positions = positions[inside]
            if len(positions) == 0:
                continue
            nearest_grains = self._grain_tree.query(positions)[1]
            grain_positions = positions[nearest_grains == grain_index]
            if len(grain_positions):
                all_positions.append(grain_positions)

        atom_positions = np.vstack(all_positions) if all_positions else np.empty((0, 3))
        self._last_rendered_atom_count = len(atom_positions)
        atom_frame = 11
        padding = atom_frame
        potential = np.zeros((padded_size + 2 * padding, padded_size + 2 * padding), dtype=np.float32)
        pixel_coordinates = (atom_positions[:, :2] - np.array([x_min, y_min])) / pixel_size_angstrom
        rounded_coordinates = np.round(pixel_coordinates)
        fractional_offsets = pixel_coordinates - rounded_coordinates
        stamp_coordinates = np.arange(atom_frame) - (atom_frame - 1) / 2.0
        stamp_x, stamp_y = np.meshgrid(stamp_coordinates, stamp_coordinates)
        for rounded, offset in zip(rounded_coordinates.astype(int), fractional_offsets):
            gaussian = np.exp(-((stamp_x + offset[0]) ** 2 + (stamp_y + offset[1]) ** 2) / 2.0)
            gaussian /= gaussian.max()
            start_x = rounded[0] + padding - atom_frame // 2
            start_y = rounded[1] + padding - atom_frame // 2
            potential[start_x:start_x + atom_frame, start_y:start_y + atom_frame] += gaussian * 79.0
        potential = potential[padding:-padding, padding:-padding]
        if potential.max() > 0:
            potential /= potential.max()
        self._raw_potential_cache = potential
        self._raw_potential_cache_key = cache_key
        self._raw_potential_cache_atom_count = self._last_rendered_atom_count
        return potential

    def _render_stem_image(self, imsize: int, dwell_time: float) -> np.ndarray:
        potential = self._raw_sample_potential(imsize)
        self._imsize = int(imsize)
        fov_angstrom = float(self._fov) * 1e10
        pixel_size_angstrom = fov_angstrom / imsize
        edge_crop = max(12, int(round(0.06 * imsize)))
        padded_size = imsize + 2 * edge_crop
        padded_fov_angstrom = pixel_size_angstrom * padded_size
        coefficients = self._corrector_coefficients()
        aberrations = probe_tools.get_target_aberrations("Spectra300", int(self.acceleration_voltage_ev))
        mappings = {"C1": ("C10",), "A1": ("C12a", "C12b"), "B2": ("C21a", "C21b"), "A2": ("C23a", "C23b"), "C3": ("C30",), "S3": ("C32a", "C32b"), "A3": ("C34a", "C34b"), "D4": ("C41a", "C41b"), "B4": ("C43a", "C43b"), "A4": ("C45a", "C45b")}
        for source, destinations in mappings.items():
            values = np.atleast_1d(coefficients[source]).astype(float)
            if len(values) != len(destinations):
                raise ValueError(f"Corrector coefficient {source} has {len(values)} value(s); expected {len(destinations)}")
            for destination, value_m in zip(destinations, values):
                aberrations[destination] = float(value_m * 1e9)
        aberrations["acceleration_voltage"] = float(self.acceleration_voltage_ev)
        aberrations["FOV"] = padded_fov_angstrom / 10.0
        aberrations["convergence_angle"] = float(self.convergence_angle_mrad)
        aberrations["wavelength"] = image_tools.get_wavelength(aberrations["acceleration_voltage"])
        probe, _aperture, _chi = probe_tools.get_probe(aberrations, padded_size, padded_size, verbose=False)
        image = np.fft.ifft2(np.fft.fft2(potential) * np.fft.fft2(np.fft.ifftshift(probe)))
        image = np.absolute(image)[edge_crop:-edge_crop, edge_crop:-edge_crop]

        scan_time = dwell_time * imsize * imsize
        counts = scan_time * (float(self.beam_current_pa) * 1e-12) / 1.602e-19 / 100.0
        pose_seed = abs(hash((int(self.sample_seed), int(imsize), round(fov_angstrom, 6), tuple(np.round(self._stage_position, 12))))) % (2**32)
        rng = np.random.default_rng(pose_seed)
        image -= image.min()
        image /= image.sum() if image.sum() > 0 else 1.0
        noisy = rng.poisson(image * counts).astype(np.float32)
        noisy -= noisy.min()
        noisy /= noisy.max() if noisy.max() > 0 else 1.0
        noise = rng.normal(0.0, 0.1, noisy.shape)
        noise_fft = np.fft.fft2(noise)
        frequencies = np.fft.fftfreq(imsize)
        frequency_filter = np.outer(np.exp(-np.square(frequencies) / (2.0 * 0.5**2)), np.exp(-np.square(frequencies) / (2.0 * 0.5**2)))
        blur_noise = np.fft.ifft2(noise_fft * frequency_filter).real
        blur_noise -= blur_noise.min()
        blur_noise /= blur_noise.max() if blur_noise.max() > 0 else 1.0
        return np.asarray(noisy + blur_noise * float(self.blur_noise_level), dtype=np.float32)

    def _acquire_scanned_image(self, imsize: int, dwell_time: float, detector_list: list[str] = ["haadf"], scan_region: list[float] = [0.0, 0.0, 1.0, 1.0], output_format: str = ".h5") -> str:
        detector_list = [detector.upper() for detector in detector_list]
        images = [self._render_stem_image(int(imsize), float(dwell_time)) for _detector in detector_list]
        metadata = {**self._sample_metadata, "last_rendered_atom_count": self._last_rendered_atom_count, "raw_potential_cache_hits": self._raw_potential_cache_hits, "defocus_m": self._get_defocus(), "corrector_aberrations": self._corrector_coefficients()}
        attrs = [metadata.copy() for _image in images]
        return save_acquisition(self, self._detector_proxies.get("data"), "stem_image", detector_list, images, dataset_attrs=attrs, file_attrs=metadata, output_format=output_format)

    @command(dtype_out=str)
    def get_volume_metadata(self) -> str:
        metadata = {**self._sample_metadata, "last_rendered_atom_count": self._last_rendered_atom_count, "raw_potential_cache_hits": self._raw_potential_cache_hits, "defocus_m": self._get_defocus(), "corrector_aberrations": self._corrector_coefficients()}
        return json.dumps(metadata)

    @command(dtype_out=str)
    def get_region_parameters(self) -> str:
        return json.dumps([asdict(parameter) for parameter in self._region_parameters])


if __name__ == "__main__":
    PolycrystallineGoldDigitalTwin.run_server()
