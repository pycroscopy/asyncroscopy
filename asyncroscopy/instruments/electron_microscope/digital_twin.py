"""
Digital twin version of AutoScriptMicroscope for HAADF-EDX.

Useful for testing and development without requiring AutoScript hardware.
"""

import json

import numpy as np
import pyTEMlib.image_tools as it
import pyTEMlib.probe_tools as pt
import tango
from ase import Atoms
from ase.build import bulk
from tango import AttrWriteType, DevState
from tango.server import Device, attribute, device_property

from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope
from asyncroscopy.data.data_writer import save_acquisition

DEFAULT_ACQUISITION_DIR = "outputs/tiled_acquisitions"


class DigitalTwin(ElectronMicroscope):
    """
    Persistent ASE-backed sample simulation with stage-coupled viewport rendering.
    """

    # ------------------------------------------------------------------
    # Device properties — configure in Tango DB per deployment
    # ------------------------------------------------------------------
    
    sample_seed = device_property(
        dtype=int,
        default_value=12345,
        doc="Seed used to generate deterministic sample geometry.",
    )
    sample_particle_count = device_property(
        dtype=int,
        default_value=40,
        doc="Number of particles in the generated sample.",
    )
    sample_size_xy = device_property(
        dtype=float,
        default_value=6e-9,
        doc="Absolute sample XY size in meters (must be > 0).",
    )
    sample_size_z = device_property(
        dtype=float,
        default_value=6e-9,
        doc="Absolute sample thickness (Z) in meters (must be > 0).",
    )
    stage_move_noise_std = device_property(
        dtype=float,
        default_value=0.0,
        doc="Gaussian move noise standard deviation in meters (applied to x,y,z).",
    )
    acquisition_save_directory = device_property(
        dtype=str,
        default_value=DEFAULT_ACQUISITION_DIR,
        doc="Directory where simulated acquisitions are saved before the Tiled server serves them.",
    )
    acquisition_file_format = device_property(
        dtype=str,
        default_value="h5",
        doc="Acquisition file format. HDF5 stores simulated acquisition data and metadata.",
    )
    data_device_address = device_property(
        dtype=str,
        default_value="",
        doc="Optional Tango device address for the DATA device, e.g. 'asyncroscopy/data/default'.",
    )

    manufacturer = attribute(
        label="DigitalTwin",
        dtype=str,
        doc="Simulation backend",
    )

    beam_pos = attribute(
        label="Beam Position",
        dtype=(float,),
        max_dim_x=2,
        access=AttrWriteType.READ_WRITE,
        unit="fractional",
        min_value=0.0,
        max_value=1.0,
        doc="Beam position as [x, y] fractional coordinates, each in range [0.0, 1.0]",
    )

    def init_device(self) -> None:
        """Initialize the Tango device and simulation state variables."""
        Device.init_device(self)
        self.set_state(DevState.INIT)

        self._stem_mode = True
        self._detector_proxies: dict[str, tango.DeviceProxy] = {}
        self._manufacturer = "UTKTeam"
        self._beam_pos_x = 0.5
        self._beam_pos_y = 0.5
        self._defocus = 0.0
        self._imsize = 512
        self._fov = 200e-10  # meters
        self._stage_position = np.zeros(5, dtype=np.float64)  # x, y, z, alpha, beta

        self._sample_atoms_base = Atoms()
        self._sample_atoms_view = Atoms()
        self._particle_records_base: list[dict] = []
        self._particle_records_view: list[dict] = []
        self._world_bounds_ang = {
            "x_min": -1.0,
            "x_max": 1.0,
            "y_min": -1.0,
            "y_max": 1.0,
            "z_min": -1.0,
            "z_max": 1.0,
        }
        self._cached_pose_key: tuple | None = None
        self._all_sample_elements: list[str] = []

        self._connect()

    def _connect(self):
        """Simulate connection by connecting to detector proxies."""
        self._connect_detector_proxies()
        self._generate_sample(seed=int(self.sample_seed))
        self.set_state(DevState.ON)

    def _connect_detector_proxies(self) -> None:
        """Build DeviceProxy objects for each configured detector device."""
        addresses: dict[str, str] = {
            "eds": self.eds_device_address,
            "stage": self.stage_device_address,
            "scan": self.scan_device_address,
            "corrector": self.corrector_device_address,
            "data": self.data_device_address,
        }
        for name, address in addresses.items():
            if not address:
                self.info_stream(f"Skipping {name}: no address configured")
                continue
            try:
                proxy = tango.DeviceProxy(address)
                proxy.set_timeout_millis(12_000)
                self._detector_proxies[name] = proxy
                self.info_stream(f"Connected to detector proxy: {name} @ {address}")
            except tango.DevFailed as e:
                self.error_stream(f"Failed to connect to {name} proxy at {address}: {e}")

    def _sync_stage_from_proxy(self) -> None:
        """Fetch the current stage position from the stage device proxy."""
        stage = self._detector_proxies.get("stage")
        if stage is None:
            return
        try:
            self._stage_position = np.array(
                [stage.x, stage.y, stage.z, stage.alpha, stage.beta],
                dtype=np.float64,
            )
            self._update_view_cache(force=False)
        except tango.DevFailed:
            self.error_stream("Failed to read stage proxy position; using internal stage state.")

    def _update_view_cache(self, force: bool = False) -> None:
        """Update the viewed sample positions by applying current stage rotations and translations."""
        pose_key = tuple(np.round(self._stage_position, 12))
        if not force and pose_key == self._cached_pose_key:
            return

        alpha_deg, beta_deg = self._stage_position[3], self._stage_position[4]
        stage_xyz_ang = self._stage_position[:3] * 1e10

        self._sample_atoms_view = self._sample_atoms_base.copy()
        if len(self._sample_atoms_view) > 0:
            if alpha_deg != 0.0:
                self._sample_atoms_view.rotate(alpha_deg, 'x', center=(0, 0, 0))
            if beta_deg != 0.0:
                self._sample_atoms_view.rotate(beta_deg, 'y', center=(0, 0, 0))
            self._sample_atoms_view.translate(-stage_xyz_ang)

        self._particle_records_view = []
        for rec in self._particle_records_base:
            dummy = Atoms("H", positions=[rec["center"]])
            if alpha_deg != 0.0:
                dummy.rotate(alpha_deg, 'x', center=(0, 0, 0))
            if beta_deg != 0.0:
                dummy.rotate(beta_deg, 'y', center=(0, 0, 0))
            dummy.translate(-stage_xyz_ang)
            
            self._particle_records_view.append(
                {
                    "center": dummy.positions[0],
                    "radius": rec["radius"],
                    "btype": rec["btype"],
                    "composition": rec["composition"],
                }
            )
        self._cached_pose_key = pose_key

    @staticmethod
    def _sub_pix_gaussian(size: int = 11, sigma: float = 0.8, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
        """Generate a 2D Gaussian kernel with sub-pixel shifts for atomic rendering."""
        coords = np.arange(size) - (size - 1) / 2.0
        xx, yy = np.meshgrid(coords, coords)
        g = np.exp(-(((xx + dx) ** 2 + (yy + dy) ** 2) / (2 * sigma**2)))
        m = np.max(g)
        return g / m if m > 0 else g
    # ------------------------------------------------------------------
    # Digital twin rendering helpers.
    # ------------------------------------------------------------------
    def _create_pseudo_potential(
        self,
        xtal: Atoms,
        pixel_size: float,
        sigma: float,
        bounds: tuple[float, float, float, float],
        atom_frame: int = 11,
    ) -> np.ndarray:
        """Project 3D atomic coordinates into a 2D density map representing pseudo-potential."""
        x_min, x_max, y_min, y_max = bounds
        pixels_x = int(np.round((x_max - x_min) / pixel_size))
        pixels_y = int(np.round((y_max - y_min) / pixel_size))
        potential_map = np.zeros((pixels_x, pixels_y), dtype=np.float32)
        padding = atom_frame
        padded = np.pad(potential_map, padding, mode="constant", constant_values=0.0)

        if len(xtal) == 0:
            return potential_map

        atomic_numbers = np.asarray(xtal.get_atomic_numbers(), dtype=np.float32)
        positions = xtal.get_positions()[:, :2]
        mask = (
            (positions[:, 0] >= x_min)
            & (positions[:, 0] < x_max)
            & (positions[:, 1] >= y_min)
            & (positions[:, 1] < y_max)
        )
        positions = positions[mask]
        atomic_numbers = atomic_numbers[mask]

        half = atom_frame // 2
        for pos, atomic_number in zip(positions, atomic_numbers):
            x_rel = (pos[0] - x_min) / pixel_size
            y_rel = (pos[1] - y_min) / pixel_size
            x_round = int(np.round(x_rel))
            y_round = int(np.round(y_rel))
            dx = x_rel - x_round
            dy = y_rel - y_round

            atom_patch = self._sub_pix_gaussian(size=atom_frame, sigma=sigma, dx=dx, dy=dy) * atomic_number
            x0 = x_round + padding - half
            y0 = y_round + padding - half
            x1 = x0 + atom_frame
            y1 = y0 + atom_frame
            if x0 < 0 or y0 < 0 or x1 > padded.shape[0] or y1 > padded.shape[1]:
                continue
            padded[x0:x1, y0:y1] += atom_patch

        potential = padded[padding:-padding, padding:-padding]
        max_val = float(np.max(potential)) if potential.size else 0.0
        if max_val > 0:
            potential = potential / max_val
        return potential.astype(np.float32)

    @staticmethod
    def _poisson_noise(image: np.ndarray, counts: float, rng: np.random.Generator) -> np.ndarray:
        """Apply normalized Poisson noise based on simulated electron counts."""
        image = image - image.min()
        total = float(image.sum())
        if total <= 0 or counts <= 0:
            return np.zeros_like(image, dtype=np.float32)
        image = image / total
        noisy = rng.poisson(image * counts).astype(np.float32)
        noisy -= noisy.min()
        m = float(noisy.max())
        if m > 0:
            noisy /= m
        return noisy

    @staticmethod
    def _lowfreq_noise(
        image: np.ndarray,
        noise_level: float,
        freq_scale: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate low-frequency spatial noise to simulate background drift or detector artifacts."""
        size_x, size_y = image.shape
        noise = rng.normal(0, noise_level, (size_x, size_y))
        noise_fft = np.fft.fft2(noise)
        x_freqs = np.fft.fftfreq(size_x)
        y_freqs = np.fft.fftfreq(size_y)
        freq_filter = np.outer(
            np.exp(-np.square(x_freqs) / (2 * freq_scale**2)),
            np.exp(-np.square(y_freqs) / (2 * freq_scale**2)),
        )
        filtered_noise = np.fft.ifft2(noise_fft * freq_filter).real
        filtered_noise -= filtered_noise.min()
        m = float(filtered_noise.max())
        if m > 0:
            filtered_noise /= m
        return filtered_noise.astype(np.float32)

    def _generate_sample(self, seed: int) -> None:
        """Procedurally generate the underlying base sample composed of bulk nanoparticles."""
        rng = np.random.default_rng(int(seed))
        sample_xy = float(self.sample_size_xy) * 1e10
        sample_z = float(self.sample_size_z) * 1e10
        if sample_xy <= 0.0 or sample_z <= 0.0:
            raise ValueError(
                "sample_size_xy and sample_size_z must both be > 0."
            )

        particle_radius = 16.0
        radius_std = 2.0
        aspect_ratio = 0.4
        min_separation = 3.0
        n_particles = max(1, int(self.sample_particle_count))
        max_attempts = 500

        bulk_types = {
            "Au": bulk("Au", "fcc", a=4.08),
            "Pt": bulk("Pt", "fcc", a=3.92),
            "Fe": bulk("Fe", "bcc", a=2.87),
        }
        bulk_names = list(bulk_types.keys())
        desired_angles = [(0, 0, 0), (60, 0, 0), (45, 45, 45)]

        placed_centers: list[tuple[float, float, float]] = []
        placed_particles: list[tuple[str, np.ndarray, float, tuple[float, float, float]]] = []
        particle_records: list[dict] = []

        x_min, x_max = -sample_xy * 0.5, sample_xy * 0.5
        y_min, y_max = -sample_xy * 0.5, sample_xy * 0.5
        z_mid = 0.0
        for _ in range(max_attempts * n_particles):
            if len(placed_particles) >= n_particles:
                break
            radius = float(np.clip(rng.normal(particle_radius, radius_std), 3.0, None))
            margin = radius + 2.0
            cx = float(rng.uniform(x_min + margin, x_max - margin))
            cy = float(rng.uniform(y_min + margin, y_max - margin))
            cz = z_mid

            too_close = False
            for px, py, pr in placed_centers:
                if np.hypot(cx - px, cy - py) < (radius + pr + min_separation):
                    too_close = True
                    break
            if too_close:
                continue

            placed_centers.append((cx, cy, radius))
            btype = str(rng.choice(bulk_names))
            idx = len(placed_particles)
            angles = desired_angles[idx] if idx < len(desired_angles) else tuple(rng.uniform(0, 360, size=3))
            center = np.array([cx, cy, cz], dtype=np.float64)
            placed_particles.append((btype, center, radius, angles))

            counts_dict: dict[str, int] = {}
            for symbol in bulk_types[btype].get_chemical_symbols():
                counts_dict[symbol] = counts_dict.get(symbol, 0) + 1
            total = sum(counts_dict.values())
            composition = {symbol: count / total for symbol, count in counts_dict.items()}
            particle_records.append(
                {
                    "center": center,
                    "radius": radius,
                    "btype": btype,
                    "composition": composition,
                }
            )

        all_positions: list[np.ndarray] = []
        all_symbols: list[str] = []
        for btype, center, radius, angles in placed_particles:
            this_bulk = bulk_types[btype]
            a_lat = this_bulk.cell.lengths()[0]
            z_radius = radius * aspect_ratio

            rep = int(radius * 2 / a_lat) + 3
            supercell = this_bulk.repeat((rep, rep, rep))
            
            # Center the supercell at origin
            supercell.center(about=(0.0, 0.0, 0.0))
            
            # Apply rotations (Z, Y, X by corresponding euler angles)
            if angles[2] != 0.0:
                supercell.rotate(angles[2], 'x', center=(0, 0, 0))
            if angles[1] != 0.0:
                supercell.rotate(angles[1], 'y', center=(0, 0, 0))
            if angles[0] != 0.0:
                supercell.rotate(angles[0], 'z', center=(0, 0, 0))
                
            positions = supercell.get_positions()
            
            r_scaled = np.sqrt(
                (positions[:, 0] / radius) ** 2
                + (positions[:, 1] / radius) ** 2
                + (positions[:, 2] / z_radius) ** 2
            )
            mask = r_scaled <= 1.0
            selected_positions = positions[mask] + center
            selected_symbols = [s for s, m in zip(supercell.get_chemical_symbols(), mask) if m]
            if len(selected_positions) == 0:
                continue
            all_positions.append(selected_positions)
            all_symbols.extend(selected_symbols)

        if all_positions:
            stacked_positions = np.vstack(all_positions)
            self._sample_atoms_base = Atoms(symbols=all_symbols, positions=stacked_positions)
        else:
            self._sample_atoms_base = Atoms()

        self._particle_records_base = particle_records
        self._all_sample_elements = sorted({el for rec in particle_records for el in rec["composition"]})
        self._world_bounds_ang = {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": -sample_z * 0.5,
            "z_max": sample_z * 0.5,
        }
        self._cached_pose_key = None
        self._update_view_cache(force=True)

    # ------------------------------------------------------------------
    # Attribute read methods
    # ------------------------------------------------------------------
    def read_manufacturer(self) -> str:
        """Read method for the manufacturer attribute."""
        return self._manufacturer

    def read_beam_pos(self):
        """Read method for the beam position attribute."""
        return [self._beam_pos_x, self._beam_pos_y]

    def write_beam_pos(self, value):
        """Write method for the beam position attribute."""
        x, y = value[0], value[1]
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"beam_pos values must be in [0.0, 1.0], got x={x}, y={y}")
        self._beam_pos_x = float(x)
        self._beam_pos_y = float(y)

    def _viewport_metadata(self) -> dict:
        fov_ang = self._fov * 1e10
        stage_xyz_ang = self._stage_position[:3] * 1e10
        return {
            "stage_position": [float(v) for v in self._stage_position],
            "beam_position": [float(self._beam_pos_x), float(self._beam_pos_y)],
            "fov_m": float(self._fov),
            "fov_angstrom": float(fov_ang),
            "imsize": int(self._imsize),
            "sample_seed": int(self.sample_seed),
            "sample_size_xy": float(self.sample_size_xy),
            "sample_size_z": float(self.sample_size_z),
            "viewport_world_angstrom": {
                "x_min": float(stage_xyz_ang[0] - fov_ang * 0.5),
                "x_max": float(stage_xyz_ang[0] + fov_ang * 0.5),
                "y_min": float(stage_xyz_ang[1] - fov_ang * 0.5),
                "y_max": float(stage_xyz_ang[1] + fov_ang * 0.5),
                "z_center": float(stage_xyz_ang[2]),
            },
            "world_bounds_angstrom": self._world_bounds_ang,
            "particle_count": len(self._particle_records_base),
        }

    def _render_stem_image(self, imsize: int, dwell_time: float, detector_list: list) -> np.ndarray:
        """Simulate STEM image acquisition using convolutions of the pseudo-potential and electron probe."""
        self._sync_stage_from_proxy()
        self._imsize = imsize
        self._update_view_cache(force=False)

        size = imsize
        fov_ang = self._fov * 1e10
        edge_crop = 20
        beam_current = 1000.0  # pA
        blur_noise_level = 0.1
        pixel_size = fov_ang / size

        frame_half = fov_ang * 0.5
        edge_ang = edge_crop * pixel_size
        frame = (-frame_half - edge_ang, frame_half + edge_ang, -frame_half - edge_ang, frame_half + edge_ang)
        potential = self._create_pseudo_potential(
            self._sample_atoms_view,
            pixel_size=pixel_size,
            sigma=1.0,
            bounds=frame,
            atom_frame=11,
        )

        ab = pt.get_target_aberrations("Spectra300", 200000)
        ab["acceleration_voltage"] = 200e3
        ab["FOV"] = fov_ang / 10.0  # nm
        ab["convergence_angle"] = 30
        ab["wavelength"] = it.get_wavelength(ab["acceleration_voltage"])

        probe, _a_k, _chi = pt.get_probe(ab, size + 2 * edge_crop, size + 2 * edge_crop, verbose=False)
        psf_shifted = np.fft.ifftshift(probe)
        image = np.fft.ifft2(np.fft.fft2(potential) * np.fft.fft2(psf_shifted))
        image = np.abs(image)
        image = image[edge_crop:-edge_crop, edge_crop:-edge_crop]

        scan_time = dwell_time * size * size
        counts = scan_time * (beam_current * 1e-12) / (1.602e-19)
        pose_seed = int(abs(hash((tuple(np.round(self._stage_position, 10)), round(self._fov, 14), size))) % (2**32))
        rng = np.random.default_rng(pose_seed + int(self.sample_seed))
        noisy_image = self._poisson_noise(image, counts=counts, rng=rng)
        noisy_image += self._lowfreq_noise(noisy_image, noise_level=0.1, freq_scale=0.1, rng=rng) * blur_noise_level
        return np.clip(noisy_image, 0.0, 1.0).astype(np.float32)

    def _acquire_scanned_image(
        self,
        imsize: int,
        dwell_time: float,
        detector_list: list[str] = ["haadf"],
        scan_region: list[float] = [0.0, 0.0, 1.0, 1.0],
        output_format: str = ".h5",
    ) -> str:
        """Simulate STEM acquisition, save the data with metadata, and return its DATA/Tiled key."""
        detector_list = [detector.upper() for detector in detector_list]
        data_server = self._detector_proxies.get("data")
        images = []
        for detector in detector_list:
            image = self._render_stem_image(int(imsize), float(dwell_time), [detector])
            images.append(image)
        return save_acquisition(self, data_server, "stem_image", detector_list, images, output_format=output_format)

    def _simulate_spectrum(self, detector_name: str, exposure_time: float) -> dict[str, float]:
        """Simulate EDS spectrum acquisition at the current beam position weighted by surrounding particles."""
        self._sync_stage_from_proxy()
        self._update_view_cache(force=False)

        px, py = self.read_beam_pos()
        fov_ang = self._fov * 1e10
        beam_x = (px - 0.5) * fov_ang
        beam_y = (py - 0.5) * fov_ang

        weighted: dict[str, float] = {}
        weight_sum = 0.0
        for rec in self._particle_records_view:
            cx, cy = rec["center"][:2]
            radius = rec["radius"]
            dist = float(np.hypot(beam_x - cx, beam_y - cy))
            if dist <= radius:
                w = max(1e-6, 1.0 - (dist / radius))
                weight_sum += w
                for element, frac in rec["composition"].items():
                    weighted[element] = weighted.get(element, 0.0) + w * frac

        spectrum_seed = int(
            abs(
                hash(
                    (
                        tuple(np.round(self._stage_position, 10)),
                        round(px, 6),
                        round(py, 6),
                        round(exposure_time, 6),
                        int(self.sample_seed),
                    )
                )
            )
            % (2**32)
        )
        rng = np.random.default_rng(spectrum_seed)

        if weight_sum <= 0.0:
            return {
                element: float(np.abs(rng.normal(0.0, 0.02)))
                for element in (self._all_sample_elements or ["Au", "Pt", "Fe"])
            }

        normalized = {el: val / weight_sum for el, val in weighted.items()}
        noisy = {}
        for element, value in normalized.items():
            noisy[element] = float(max(0.0, value + rng.normal(0.0, 0.01)))
        total = sum(noisy.values())
        if total <= 0.0:
            return noisy
        return {el: val / total for el, val in noisy.items()}

    def _acquire_spectrum(self, detector_name: str, exposure_time: float) -> str:
        """Simulate spectrum acquisition, save HDF5 data, and return its DATA/Tiled key."""
        spectrum = self._simulate_spectrum(detector_name, exposure_time)
        data_server = self._detector_proxies.get("data")
        spectrum_array = np.array(list(spectrum.values()), dtype=np.float64)
        return save_acquisition(self, data_server, "spectrum", detector_name, spectrum_array, dataset_name="spectrum")

    def _place_beam(self, position) -> None:
        """Place the electron beam at the specified [x, y] coordinates."""
        x, y = position
        self.write_beam_pos([x, y])

    def _set_fov(self, fov) -> None:
        """Set the field of view in meters."""
        self._fov = float(fov)

    def _set_defocus(self, defocus) -> None:
        """Set defocus in meters."""
        self._defocus = float(defocus)

    def _get_defocus(self) -> float:
        """Get defocus in meters."""
        return self._defocus

    def _get_stage(self):
        """Return the current 5-axis stage position."""
        self._sync_stage_from_proxy()
        return self._stage_position

    def _move_stage(self, position):
        """Move the stage to the requested 5-axis vector with optional simulated precision noise."""
        if len(position) != 5:
            raise ValueError("Stage position must have 5 values: [x, y, z, alpha, beta]")
        target = np.array(position, dtype=np.float64)

        std = float(self.stage_move_noise_std)
        if std > 0:
            noise = np.zeros(5, dtype=np.float64)
            noise[:3] = np.random.normal(0.0, std, size=3)
            target = target + noise

        stage = self._detector_proxies.get("stage")
        if stage is not None:
            stage.x = float(target[0])
            stage.y = float(target[1])
            stage.z = float(target[2])
            stage.alpha = float(target[3])
            stage.beta = float(target[4])

        self._stage_position = target
        self._update_view_cache(force=False)

    def get_viewport_metadata(self) -> str:
        """Return JSON-formatted metadata regarding the current simulation viewport and environment state."""
        self._sync_stage_from_proxy()
        fov_ang = self._fov * 1e10
        stage_xyz_ang = self._stage_position[:3] * 1e10
        viewport = {
            "x_min": float(stage_xyz_ang[0] - fov_ang * 0.5),
            "x_max": float(stage_xyz_ang[0] + fov_ang * 0.5),
            "y_min": float(stage_xyz_ang[1] - fov_ang * 0.5),
            "y_max": float(stage_xyz_ang[1] + fov_ang * 0.5),
            "z_center": float(stage_xyz_ang[2]),
        }
        metadata = {
            "stage_position": [float(v) for v in self._stage_position],
            "fov_m": float(self._fov),
            "fov_angstrom": float(fov_ang),
            "imsize": int(self._imsize),
            "sample_seed": int(self.sample_seed),
            "sample_size_xy": float(self.sample_size_xy),
            "sample_size_z": float(self.sample_size_z),
            "sample_size_xy_generated_angstrom": float(self._world_bounds_ang["x_max"] - self._world_bounds_ang["x_min"]),
            "sample_size_z_generated_angstrom": float(self._world_bounds_ang["z_max"] - self._world_bounds_ang["z_min"]),
            "viewport_world_angstrom": viewport,
            "world_bounds_angstrom": self._world_bounds_ang,
            "particle_count": len(self._particle_records_base),
        }
        return json.dumps(metadata)

if __name__ == "__main__":
    DigitalTwin.run_server()
