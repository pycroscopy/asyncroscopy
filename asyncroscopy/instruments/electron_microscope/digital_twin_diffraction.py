"""Digital twin for parked-beam nanoparticle diffraction."""

from __future__ import annotations

import numpy as np
import tango
from ase import Atoms
from ase.build import bulk
from tango.server import device_property

from asyncroscopy.data.data_writer import save_acquisition
from asyncroscopy.instruments.electron_microscope.digital_twin import DEFAULT_ACQUISITION_DIR, DigitalTwin

FOV_NM = 500.0
FOV_M = FOV_NM * 1e-9
DEFAULT_MAP_SIZE = 1024
DEFAULT_OVERVIEW_IMAGE_SIZE = 1024
DIFFRACTION_MAX_ANGLE_MRAD = 80.0
TARGET_COVERAGE = 0.22
VACUUM_LATERAL_SIZE_ANGSTROM = 40.0
ROUND_CENTER_LATTICE_MEAN_ANGSTROM = 4.078
OVAL_CENTER_LATTICE_MEAN_ANGSTROM = 4.678
CENTER_LATTICE_STD_ANGSTROM = 0.015
EDGE_LATTICE_CHANGE_FRACTION = 0.25
AMORPHOUS_EDGE_WIDTH_PX = 5.0


class DigitalTwinDiffraction(DigitalTwin):
    """Digital twin that pairs a HAADF nanoparticle overview with local diffraction."""

    acquisition_save_directory = device_property(
        dtype=str,
        default_value=DEFAULT_ACQUISITION_DIR,
        doc='Directory where simulated acquisitions are saved before the Tiled server serves them.',
    )
    map_size = device_property(
        dtype=int,
        default_value=DEFAULT_MAP_SIZE,
        doc='Internal pixel size for the fixed 500 nm nanoparticle field.',
    )
    diffraction_image_size = device_property(
        dtype=int,
        default_value=128,
        doc='Diffraction image size used when the camera device does not provide one.',
    )
    overview_image_size = device_property(
        dtype=int,
        default_value=DEFAULT_OVERVIEW_IMAGE_SIZE,
        doc='Fixed HAADF overview image size in pixels.',
    )
    convergence_angle_mrad = device_property(
        dtype=float,
        default_value=5.0,
        doc='Probe convergence semi-angle in mrad for abTEM diffraction simulations.',
    )
    haadf_poisson_counts = device_property(
        dtype=float,
        default_value=120.0,
        doc='Mean per-pixel count scale used for Poisson noise in the HAADF overview.',
    )

    def init_device(self) -> None:
        super().init_device()
        self._manufacturer = 'UTKTeam Diffraction Twin'
        self._fov = FOV_M

    def _connect_detector_proxies(self) -> None:
        addresses: dict[str, str] = {
            'camera': self.camera_device_address,
            'scan': self.scan_device_address,
            'stage': self.stage_device_address,
            'data': self.data_device_address,
        }
        for name, address in addresses.items():
            if not address:
                self.info_stream(f'Skipping {name}: no address configured')
                continue
            try:
                proxy = tango.DeviceProxy(address)
                proxy.set_timeout_millis(12_000)
                self._detector_proxies[name] = proxy
                self.info_stream(f'Connected to detector proxy: {name} @ {address}')
            except tango.DevFailed as exc:
                self.error_stream(f'Failed to connect to {name} proxy at {address}: {exc}')

    def _update_view_cache(self, force: bool = False) -> None:
        self._cached_pose_key = tuple(np.round(self._stage_position, 12))

    def _generate_sample(self, seed: int) -> None:
        rng = np.random.default_rng(int(seed))
        size = max(128, int(self.map_size))
        self._fov = FOV_M
        self._world_bounds_ang = {
            'x_min': -FOV_NM * 5.0,
            'x_max': FOV_NM * 5.0,
            'y_min': -FOV_NM * 5.0,
            'y_max': FOV_NM * 5.0,
            'z_min': -50.0,
            'z_max': 50.0,
        }

        particle_id = np.zeros((size, size), dtype=np.int16)
        haadf = np.zeros((size, size), dtype=np.float32)
        rattle = np.zeros((size, size), dtype=np.float32)
        lattice_parameter_map = np.zeros((size, size), dtype=np.float32)
        particle_records = []
        yy, xx = np.mgrid[:size, :size]
        coverage = 0.0
        attempts = 0
        kind_counts = {'round': 0, 'oval': 0}

        while coverage < TARGET_COVERAGE and attempts < 4000:
            attempts += 1
            if kind_counts['round'] == kind_counts['oval']:
                kind = 'round' if rng.random() < 0.5 else 'oval'
            else:
                kind = 'round' if kind_counts['round'] < kind_counts['oval'] else 'oval'
            radius = float(rng.uniform(18.0, 34.0))
            if kind == 'round':
                axis_x = radius * float(rng.uniform(0.97, 1.03))
                axis_y = radius * float(rng.uniform(0.97, 1.03))
                center_lattice_parameter = float(
                    rng.normal(ROUND_CENTER_LATTICE_MEAN_ANGSTROM, CENTER_LATTICE_STD_ANGSTROM)
                )
                edge_lattice_change = -EDGE_LATTICE_CHANGE_FRACTION
            else:
                axis_x = radius * float(rng.uniform(1.25, 1.65))
                axis_y = radius * float(rng.uniform(0.75, 0.95))
                center_lattice_parameter = float(
                    rng.normal(OVAL_CENTER_LATTICE_MEAN_ANGSTROM, CENTER_LATTICE_STD_ANGSTROM)
                )
                edge_lattice_change = EDGE_LATTICE_CHANGE_FRACTION

            margin = int(max(axis_x, axis_y) * 1.2 + 4)
            if margin * 2 >= size:
                continue
            cx = float(rng.uniform(margin, size - margin))
            cy = float(rng.uniform(margin, size - margin))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            x0 = max(0, int(np.floor(cx - margin)))
            x1 = min(size, int(np.ceil(cx + margin + 1)))
            y0 = max(0, int(np.floor(cy - margin)))
            y1 = min(size, int(np.ceil(cy + margin + 1)))
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            dx = xx[y0:y1, x0:x1] - cx
            dy = yy[y0:y1, x0:x1] - cy
            xr = dx * cos_a + dy * sin_a
            yr = -dx * sin_a + dy * cos_a
            theta = np.arctan2(yr / max(axis_y, 1e-6), xr / max(axis_x, 1e-6))
            blob_scale = 0.35 if kind == 'round' else 1.0
            blob = 1.0 + 0.09 * blob_scale * np.sin(3.0 * theta + rng.uniform(0.0, 2.0 * np.pi))
            blob += 0.07 * blob_scale * np.sin(5.0 * theta + rng.uniform(0.0, 2.0 * np.pi))
            normalized_radius = np.sqrt((xr / axis_x) ** 2 + (yr / axis_y) ** 2) / blob
            mask = normalized_radius <= 1.0
            particle_view = particle_id[y0:y1, x0:x1]
            if np.any(mask & (particle_view > 0)):
                continue
            particle_pixels = int(np.count_nonzero(mask))
            if particle_pixels / mask.size < 0.002:
                continue

            record_id = len(particle_records) + 1
            distance_to_edge_px = (1.0 - normalized_radius) * min(axis_x, axis_y)
            edge_rattle = np.clip(1.0 - distance_to_edge_px / AMORPHOUS_EDGE_WIDTH_PX, 0.0, 1.0) ** 0.6
            edge_rattle = (edge_rattle * mask).astype(np.float32)
            local_lattice_parameter = self._local_lattice_parameter(
                center_lattice_parameter,
                edge_lattice_change,
                normalized_radius,
            )
            local_intensity = float(rng.uniform(0.72, 1.0))
            texture = rng.normal(0.0, 0.035, size=mask.shape).astype(np.float32)
            rattle_view = rattle[y0:y1, x0:x1]
            lattice_view = lattice_parameter_map[y0:y1, x0:x1]
            haadf_view = haadf[y0:y1, x0:x1]
            particle_view[mask] = record_id
            rattle_view[mask] = edge_rattle[mask]
            lattice_view[mask] = local_lattice_parameter[mask]
            haadf_view[mask] = local_intensity - 0.18 * edge_rattle[mask] + texture[mask]
            particle_records.append(
                {
                    'id': record_id,
                    'kind': kind,
                    'pixel_count': particle_pixels,
                    'center_nm': [
                        float((cx / (size - 1) - 0.5) * FOV_NM),
                        float((cy / (size - 1) - 0.5) * FOV_NM),
                    ],
                    'axes_nm': [
                        float(axis_x / size * FOV_NM),
                        float(axis_y / size * FOV_NM),
                    ],
                    'angle_rad': angle,
                    'rotation_degrees': float(np.degrees(angle)),
                    'center_lattice_parameter': center_lattice_parameter,
                    'lattice_parameter': center_lattice_parameter,
                    'edge_lattice_change_fraction': edge_lattice_change,
                }
            )
            kind_counts[kind] += 1
            coverage = np.count_nonzero(particle_id) / particle_id.size

        haadf += rng.normal(0.015, 0.01, size=(size, size)).astype(np.float32)
        haadf -= float(haadf.min())
        max_val = float(haadf.max())
        self._nanoparticle_map = np.clip(haadf / max_val if max_val > 0.0 else haadf, 0.0, 1.0)
        self._particle_id_map = particle_id
        self._rattle_map = rattle
        self._lattice_parameter_map = lattice_parameter_map
        self._particle_records_base = particle_records
        self._particle_records_view = particle_records
        self._sample_atoms_base = bulk('Au', 'fcc', a=4.08, cubic=True)
        self._sample_atoms_view = self._sample_atoms_base.copy()
        self._cached_pose_key = None

    def _render_stem_image(self, imsize: int, dwell_time: float, detector_list: list) -> np.ndarray:
        self._sync_stage_from_proxy()
        self._imsize = imsize
        source = self._nanoparticle_map
        source_size = source.shape[0]
        stage_nm = self._stage_position[:2] * 1e9
        axis = (np.arange(int(imsize)) + 0.5) / int(imsize) - 0.5
        x_nm = axis * FOV_NM + stage_nm[0]
        y_nm = axis * FOV_NM + stage_nm[1]
        xi = np.rint((x_nm / FOV_NM + 0.5) * (source_size - 1)).astype(int)
        yi = np.rint((y_nm / FOV_NM + 0.5) * (source_size - 1)).astype(int)
        image = np.zeros((int(imsize), int(imsize)), dtype=np.float32)
        valid_x = (xi >= 0) & (xi < source_size)
        valid_y = (yi >= 0) & (yi < source_size)
        image[np.ix_(valid_y, valid_x)] = source[np.ix_(yi[valid_y], xi[valid_x])]
        seed = int(abs(hash((int(self.sample_seed), tuple(np.round(self._stage_position, 10)), int(imsize)))) % (2**32))
        rng = np.random.default_rng(seed)
        counts = max(1.0, float(self.haadf_poisson_counts))
        image = rng.poisson(np.clip(image, 0.0, 1.0) * counts).astype(np.float32) / counts
        image += rng.normal(0.0, 0.008, size=image.shape).astype(np.float32)
        return np.clip(image, 0.0, 1.0)

    def _beam_particle(self) -> tuple[dict | None, float]:
        self._sync_stage_from_proxy()
        px, py = self.read_beam_pos()
        size = self._particle_id_map.shape[0]
        stage_nm = self._stage_position[:2] * 1e9
        x_nm = (float(px) - 0.5) * FOV_NM + stage_nm[0]
        y_nm = (float(py) - 0.5) * FOV_NM + stage_nm[1]
        ix = int(round((x_nm / FOV_NM + 0.5) * (size - 1)))
        iy = int(round((y_nm / FOV_NM + 0.5) * (size - 1)))
        if ix < 0 or ix >= size or iy < 0 or iy >= size:
            return None, 0.0
        particle_id = int(self._particle_id_map[iy, ix])
        if particle_id <= 0:
            return None, 0.0
        particle = self._particle_records_base[particle_id - 1].copy()
        center_lattice_parameter = float(particle['center_lattice_parameter'])
        local_lattice_parameter = float(self._lattice_parameter_map[iy, ix])
        particle['lattice_parameter'] = local_lattice_parameter
        particle['lattice_strain_fraction'] = local_lattice_parameter / center_lattice_parameter - 1.0
        return particle, float(self._rattle_map[iy, ix])

    def _simulate_abtem_diffraction(self, particle: dict, rattle_value: float, imsize: int) -> np.ndarray:
        atoms = self._particle_atoms(particle, rattle_value)
        return self._simulate_diffraction_from_atoms(atoms, imsize)

    @staticmethod
    def _particle_seed(sample_seed: int, stage_position: np.ndarray, beam_position: list[float]) -> int:
        return int(abs(hash((int(sample_seed), tuple(np.round(stage_position, 10)), tuple(np.round(beam_position, 6))))) % (2**32))

    def _particle_atoms(self, particle: dict, rattle_value: float):
        # A conventional cubic FCC cell puts the beam along the <100> zone-axis family.
        atoms = bulk('Au', 'fcc', a=float(particle['lattice_parameter']), cubic=True).repeat((4, 4, 4))
        atoms.center(vacuum=2.0)
        atoms.rotate(float(particle['rotation_degrees']), 'z', center='COP')
        if rattle_value > 0.0:
            seed = self._particle_seed(int(self.sample_seed), self._stage_position, self.read_beam_pos())
            atoms.rattle(stdev=self._rattle_stdev(rattle_value), seed=seed)
        return atoms

    @staticmethod
    def _rattle_stdev(rattle_value: float) -> float:
        return 0.03 + 0.7 * float(rattle_value)

    @staticmethod
    def _local_lattice_parameter(center_value: float, edge_change: float, normalized_radius):
        radial_fraction = np.clip(normalized_radius, 0.0, 1.0)
        smooth_radial_fraction = radial_fraction**2 * (3.0 - 2.0 * radial_fraction)
        return center_value * (1.0 + edge_change * smooth_radial_fraction)

    def _vacuum_diffraction(self, imsize: int) -> np.ndarray:
        cell = [VACUUM_LATERAL_SIZE_ANGSTROM, VACUUM_LATERAL_SIZE_ANGSTROM, 10.0]
        atoms = Atoms(cell=cell, pbc=(False, False, False))
        return self._simulate_diffraction_from_atoms(atoms, imsize)

    def _simulate_diffraction_from_atoms(self, atoms, imsize: int) -> np.ndarray:
        try:
            import abtem
        except ImportError as exc:
            raise RuntimeError(
                "abTEM is required for diffraction simulation. "
                "Install it with `uv sync --extra diffraction`."
            ) from exc

        potential = abtem.Potential(atoms, sampling=0.08, slice_thickness=1.0)
        probe = abtem.Probe(energy=200e3, semiangle_cutoff=float(self.convergence_angle_mrad))
        probe.grid.match(potential)
        exit_wave = probe.multislice(potential)
        pattern = exit_wave.diffraction_patterns(max_angle=DIFFRACTION_MAX_ANGLE_MRAD).compute()
        array = np.asarray(getattr(pattern, 'array', pattern), dtype=np.float32)
        array = np.squeeze(array)
        if array.ndim > 2:
            array = array.reshape((-1,) + array.shape[-2:])[0]
        return self._resize_nearest(array, imsize)

    @staticmethod
    def _resize_nearest(array: np.ndarray, imsize: int) -> np.ndarray:
        if array.shape == (imsize, imsize):
            result = array.astype(np.float32)
        else:
            yi = np.rint(np.linspace(0, array.shape[0] - 1, imsize)).astype(int)
            xi = np.rint(np.linspace(0, array.shape[1] - 1, imsize)).astype(int)
            result = array[np.ix_(yi, xi)].astype(np.float32)
        result -= float(result.min())
        max_val = float(result.max())
        return result / max_val if max_val > 0.0 else result

    def _acquire_camera_image(self, imsize: int, exposure_time: float, detector: str, readout_area: str) -> str:
        particle, rattle_value = self._beam_particle()
        image_size = int(imsize or self.diffraction_image_size)
        diffraction = self._vacuum_diffraction(image_size) if particle is None else self._simulate_abtem_diffraction(particle, rattle_value, image_size)
        data_server = self._detector_proxies.get('data')
        attrs = {
            'pixel_size_mrad': float(2.0 * DIFFRACTION_MAX_ANGLE_MRAD / image_size),
            'pixel_size_rad': float(2.0 * DIFFRACTION_MAX_ANGLE_MRAD * 1e-3 / image_size),
            'max_angle_mrad': float(DIFFRACTION_MAX_ANGLE_MRAD),
            'rattle_value': float(rattle_value),
            'beam_position_x': float(self._beam_pos_x),
            'beam_position_y': float(self._beam_pos_y),
        }
        if particle is not None:
            attrs.update(
                {
                    'particle_id': int(particle['id']),
                    'particle_kind': str(particle['kind']),
                    'particle_lattice_parameter_angstrom': float(particle['lattice_parameter']),
                    'particle_center_lattice_parameter_angstrom': float(particle['center_lattice_parameter']),
                    'particle_lattice_strain_fraction': float(particle['lattice_strain_fraction']),
                    'particle_lattice_strain_percent': float(100.0 * particle['lattice_strain_fraction']),
                    'particle_rotation_degrees': float(particle['rotation_degrees']),
                    'particle_zone_axis': '<100>',
                }
            )
        return save_acquisition(self, data_server, 'diffraction', str(detector), diffraction, dataset_name='image', dataset_attrs=attrs)

    def _acquire_scanned_image(
        self,
        imsize: int,
        dwell_time: float,
        detector_list: list[str] = ['haadf'],
        scan_region: list[float] = [0.0, 0.0, 1.0, 1.0],
        output_format: str = '.h5',
    ) -> str:
        image_size = int(self.overview_image_size)
        detector_list = [detector.upper() for detector in detector_list]
        data_server = self._detector_proxies.get('data')
        attrs = {
            'pixel_size_nm': float(FOV_NM / image_size),
            'pixel_size_m': float(FOV_M / image_size),
            'fov_nm': float(FOV_NM),
            'fov_m': float(FOV_M),
            'map_size_px': int(self.map_size),
            'sample_pixel_size_nm': float(FOV_NM / int(self.map_size)),
        }
        images = [self._render_stem_image(image_size, float(dwell_time), [detector]) for detector in detector_list]
        return save_acquisition(self, data_server, 'stem_image', detector_list, images, dataset_attrs=attrs, output_format=output_format)

    def _set_fov(self, fov) -> None:
        self.warn_stream('DigitalTwinDiffraction uses a fixed 500 nm field of view.')
        self._fov = FOV_M

    def _get_fov(self) -> float:
        return FOV_M


if __name__ == '__main__':
    DigitalTwinDiffraction.run_server()
