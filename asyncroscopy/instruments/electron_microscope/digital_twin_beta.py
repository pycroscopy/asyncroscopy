"""
Digital twin version of AutoScriptMicroscope for HAADF-EDX.

Useful for testing and development without requiring AutoScript hardware.
"""


import json

import numpy as np
import pyTEMlib.probe_tools as pt
import pyTEMlib.image_tools as it

import tango
from tango import AttrWriteType, DevState
from tango.server import Device, attribute

from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope

class DigitalTwinBeta(ElectronMicroscope):
    """
    Detector-specific settings (dwell time, resolution) are stored in
    dedicated detector devices and read via DeviceProxy at acquisition time.
    """

    # ------------------------------------------------------------------
    # Device properties — configure in Tango DB per deployment
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------
    # not finishded
    manufacturer = attribute(
        label="DigitalTwinBeta",
        dtype=str,
        doc="Simulation backend",
    )

    beam_pos = attribute(
        label="Beam Position",
        dtype=(float,),        # 1D array of floats
        max_dim_x=2,           # exactly 2 elements: [x, y]
        access=AttrWriteType.READ_WRITE,
        unit="fractional",
        min_value=0.0,
        max_value=1.0,
        doc="Beam position as [x, y] fractional coordinates, each in range [0.0, 1.0]",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.INIT)
        
        # Internal state
        self._stem_mode = True
        self._detector_proxies = {}
        self._manufacturer = "UTKTeam"
        self._beam_pos_x = 0.5
        self._beam_pos_y = 0.5
        self._defocus = 0.0
        self._particle_records = []
        self._imsize = 512
        self._fov = 200e-10  # meters, i.e. 200 angstroms
        self._stage_position = np.random.rand(3) * 1e-6  # random initial stage position in meters
        
        self._connect()
        
    def _connect(self):
        """Simulate connection by connecting to detector proxies."""
        self._connect_detector_proxies()
        self.set_state(DevState.ON)


    def _connect_detector_proxies(self) -> None:
        """Build DeviceProxy objects for each configured detector device."""
        # Extend this dict as more detectors are added
        # later, we want to do this automatically, not with a dictionary.
        addresses: dict[str, str] = {
            "AdvancedAcquistion": self.advanced_acquisition_device_address,
            "eds":  self.eds_device_address,
            "stage": self.stage_device_address,
            "scan": self.scan_device_address,
        }
        print(addresses)
        for name, address in addresses.items():
            if not address:   # <-- minimal fix
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


    def read_beam_pos(self):
        """Return beam position as [x, y] fractional coordinates."""
        return [self._beam_pos_x, self._beam_pos_y]

    # --- Write Method ---

    def write_beam_pos(self, value):
        """Set beam position from [x, y] fractional coordinates."""
        x, y = value[0], value[1]

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(
                f"beam_pos values must be in [0.0, 1.0], got x={x}, y={y}"
            )

        self._beam_pos_x = x
        self._beam_pos_y = y


    # ------------------------------------------------------------------
    # Internal acquisition helpers
    # ------------------------------------------------------------------
    def _acquire_scanned_image(
        self,
        imsize: int,
        dwell_time: float,
        detector_list: list[str] = ["haadf"],
        scan_region: list[float] = [0.0, 0.0, 1.0, 1.0],
    ) -> np.ndarray:
        """
        Acquire a simulated STEM image using the pre-cooked sample state.
        Requires _cook_sample_recipe() to have been called immediately before this.
        """
        size = imsize
        self._imsize = imsize
        fov      = self._fov * 1e10          # Å
        edge_crop = 20
        beam_current     = 1000              # pA
        blur_noise_level = 0.1
        pixel_size       = fov / size

        # ── Probe aberrations (unchanged) ─────────────────────────────────────────
        ab = pt.get_target_aberrations("Spectra300", 200000)
        ab['acceleration_voltage'] = 200e3
        ab['FOV']                  = fov / 10   # nm
        ab['convergence_angle']    = 30         # mrad
        ab['wavelength']           = it.get_wavelength(ab['acceleration_voltage'])

        # ── Helper functions (unchanged from original) ────────────────────────────
        def sub_pix_gaussian(size=10, sigma=0.2, dx=0.0, dy=0.0):
            coords = np.arange(size) - (size - 1) / 2.0
            x, y   = np.meshgrid(coords, coords)
            g      = np.exp(-(((x + dx)**2 + (y + dy)**2) / (2 * sigma**2)))
            return g / g.max()

        def create_pseudo_potential(xtal, pixel_size, sigma, bounds, atom_frame=11):
            x_min, x_max = bounds[0], bounds[1]
            y_min, y_max = bounds[2], bounds[3]
            pixels_x = int((x_max - x_min) / pixel_size)
            pixels_y = int((y_max - y_min) / pixel_size)
            potential_map = np.zeros((pixels_x, pixels_y))
            padding       = atom_frame
            potential_map = np.pad(potential_map, padding, mode='constant', constant_values=0.0)

            atomic_numbers = xtal.get_atomic_numbers()
            positions      = xtal.get_positions()[:, :2]

            mask = (
                (positions[:, 0] >= x_min) & (positions[:, 0] < x_max) &
                (positions[:, 1] >= y_min) & (positions[:, 1] < y_max)
            )
            positions      = positions[mask]
            atomic_numbers = atomic_numbers[mask]

            for pos, Z in zip(positions, atomic_numbers):
                x, y   = np.round(pos / pixel_size)
                dx, dy = pos - np.round(pos)
                atom   = sub_pix_gaussian(size=atom_frame, sigma=sigma, dx=dx, dy=dy) * Z
                potential_map[
                    int(x + padding + dx - padding//2 - 1):int(x + padding + dx + padding//2),
                    int(y + padding + dy - padding//2 - 1):int(y + padding + dy + padding//2)
                ] += atom

            potential_map = potential_map[padding:-padding, padding:-padding]
            return potential_map / np.max(potential_map)

        def poisson_noise(image, counts=1e9):
            image = image - image.min()
            image = image / image.sum()
            noisy = np.random.poisson(image * counts).astype(float)
            noisy = noisy - noisy.min()
            return noisy / noisy.max()

        def lowfreq_noise(image, noise_level=0.1, freq_scale=0.1):
            size_x, size_y = image.shape
            noise      = np.random.normal(0, noise_level, (size_x, size_y))
            noise_fft  = np.fft.fft2(noise)
            x_freqs    = np.fft.fftfreq(size_x)
            y_freqs    = np.fft.fftfreq(size_y)
            freq_filter = np.outer(
                np.exp(-np.square(x_freqs) / (2 * freq_scale**2)),
                np.exp(-np.square(y_freqs) / (2 * freq_scale**2)),
            )
            noisy = np.fft.ifft2(noise_fft * freq_filter).real
            noisy = noisy - noisy.min()
            return noisy / noisy.max()

        # ── Pull cooked state ─────────────────────────────────────────────────────
        # _cook_sample_recipe must have been called before this method.
        xtal             = self._cooked_atoms
        proj             = self._cooked_projection   # dict from _cook_sample_recipe
        projected_label  = proj['projected_label']   # (nx_world, ny_world) uint8
        particle_lookup  = self._particle_lookup     # {label -> metadata}

        # ── Rebuild _particle_records from cooked projection ──────────────────────
        # This keeps the contract that _acquire_scanned_image always refreshes
        # _particle_records to match whatever the current image will show,
        # accounting for stage shift/tilt applied in _cook_sample_recipe.
        #
        # We crop the projected label map to the image FOV (with edge padding
        # removed), then find the bounding-box centre of each visible label.
        # Particles that have scrolled off-screen are omitted automatically.

        world_nx, world_ny = projected_label.shape

        # The image FOV sits at the centre of the world map.
        # edge_crop pixels of padding are added on each side for convolution,
        # so the "live" image region in world-map coordinates is:
        img_origin_x = (world_nx - size) // 2
        img_origin_y = (world_ny - size) // 2

        # Crop the projection to the padded image region (size + 2*edge_crop)
        pad = edge_crop
        x0  = max(img_origin_x - pad, 0)
        y0  = max(img_origin_y - pad, 0)
        x1  = min(img_origin_x + size + pad, world_nx)
        y1  = min(img_origin_y + size + pad, world_ny)
        label_crop = projected_label[x0:x1, y0:y1]     # (size+2*pad, size+2*pad)

        particle_records = []
        for label, meta in particle_lookup.items():
            yx = np.argwhere(label_crop == label)
            if yx.size == 0:
                continue   # particle not visible in this FOV

            # Centre of mass in cropped-label coordinates → image pixel coordinates
            centre_crop = yx.mean(axis=0)           # (row≡x, col≡y) in label_crop
            # Subtract the edge_crop offset to get image pixel coords (0..size-1)
            cx_pix = centre_crop[0] - pad
            cy_pix = centre_crop[1] - pad

            # Radius: use the stored physical radius converted to pixels
            radius_pix = meta['radius_ang'] / pixel_size

            particle_records.append({
                'center':      np.array([cx_pix, cy_pix]),
                'radius':      radius_pix,
                'btype':       meta['btype'],
                'composition': meta['composition'],
            })

        self._particle_records = particle_records
        print(f"_acquire_scanned_image: {len(particle_records)} particles in FOV")

        # ── Pseudo-potential + PSF convolution (unchanged logic) ──────────────────
        edge     = 2 * edge_crop * pixel_size
        frame    = (0, fov + edge, 0, fov + edge)

        # Clip the cooked atoms to the padded frame before making the potential.
        # _cooked_atoms already has world-coordinates; we just pass the frame bounds
        # and let create_pseudo_potential mask by position, same as before.
        potential = create_pseudo_potential(
            xtal, pixel_size, sigma=1, bounds=frame, atom_frame=11
        )

        probe, A_k, chi = pt.get_probe(
            ab, size + 2*edge_crop, size + 2*edge_crop, verbose=True
        )

        psf_shifted = np.fft.ifftshift(probe)
        image = np.fft.ifft2(np.fft.fft2(potential) * np.fft.fft2(psf_shifted))
        image = np.absolute(image)
        image = image[edge_crop:-edge_crop, edge_crop:-edge_crop]

        # ── Noise chain (unchanged) ───────────────────────────────────────────────
        scan_time   = dwell_time * size * size
        counts      = scan_time * (beam_current * 1e-12) / 1.602e-19
        noisy_image = poisson_noise(image, counts=counts)
        blur_noise  = lowfreq_noise(noisy_image, noise_level=0.1, freq_scale=0.1)
        noisy_image += blur_noise * blur_noise_level

        return np.array(noisy_image, dtype=np.float32)

    def _make_sample_recipe(self):
        """
        Build three persistent data structures for the sample:
        1. _particle_lookup  : {label_int -> {'btype', 'composition', 'center_vox', 'radius_vox', 'angles'}}
        2. _particle_label_map : 3-D uint8 array (nx, ny, nz) – 0 = vacuum, label = particle id
        3. _atoms_object     : ASE Atoms with all nanoparticle atoms + Cell set from world size
        """
        from ase import Atoms
        from ase.build import bulk
        import numpy as np

        # ── World geometry ──────────────────────────────────────────────────────────
        fov_ang     = self._fov * 1e10          # Angstroms, lateral
        world_z_ang = fov_ang * 0.5             # thin slab, same as _acquire_scanned_image
        vox_size    = fov_ang / self._imsize    # Angstroms per voxel (isotropic)

        nx = ny = self._imsize
        nz = max(1, int(round(world_z_ang / vox_size)))

        # ── Particle parameters (mirror _acquire_scanned_image) ────────────────────────
        particle_radius  = 16.0
        radius_std       = 2.0
        aspect_ratio     = 0.4
        min_separation   = 3.0
        n_particles      = 40
        max_attempts     = 500
        desired_angles   = [(0, 0, 0), (60, 0, 0), (45, 45, 45)]

        bulk_types = {
            'Au': bulk('Au', 'fcc', a=4.08),
            'Pt': bulk('Pt', 'fcc', a=3.92),
            'Fe': bulk('Fe', 'bcc', a=2.87),
        }
        bulk_names = list(bulk_types.keys())

        def rotation_matrix(alpha, beta, gamma):
            a, b, g = np.radians([alpha, beta, gamma])
            Rz = np.array([[np.cos(a), -np.sin(a), 0],
                        [np.sin(a),  np.cos(a), 0],
                        [0, 0, 1]])
            Ry = np.array([[np.cos(b), 0,  np.sin(b)],
                        [0, 1, 0],
                        [-np.sin(b), 0, np.cos(b)]])
            Rx = np.array([[1, 0, 0],
                        [0, np.cos(g), -np.sin(g)],
                        [0, np.sin(g),  np.cos(g)]])
            return Rz @ Ry @ Rx

        # ── 1. Place particle centres (same exclusion logic as _acquire_scanned_image) ─
        placed_centers   = []
        placed_particles = []
        particle_lookup  = {}   # label (1-based int) -> metadata dict

        for _ in range(max_attempts * n_particles):
            if len(placed_particles) >= n_particles:
                break

            radius = np.clip(np.random.normal(particle_radius, radius_std), 3.0, None)
            margin = radius + 2.0
            sample_fov = (fov_ang * 1.5, fov_ang * 1.5, world_z_ang)

            cx = np.random.uniform(margin, sample_fov[0] - margin)
            cy = np.random.uniform(margin, sample_fov[1] - margin)
            cz = sample_fov[2] * 0.5

            too_close = any(
                np.sqrt((cx - px)**2 + (cy - py)**2) < radius + pr + min_separation
                for px, py, pr in placed_centers
            )
            if too_close:
                continue

            placed_centers.append((cx, cy, radius))
            btype  = np.random.choice(bulk_names)
            i      = len(placed_particles)
            angles = desired_angles[i] if i < len(desired_angles) else tuple(np.random.rand(3) * 360)
            placed_particles.append((btype, np.array([cx, cy, cz]), radius, angles))

            symbols_in_bulk = bulk_types[btype].get_chemical_symbols()
            counts_dict = {}
            for s in symbols_in_bulk:
                counts_dict[s] = counts_dict.get(s, 0) + 1
            total = sum(counts_dict.values())
            composition = {s: c / total for s, c in counts_dict.items()}

            label = i + 1   # 1-based so 0 stays "vacuum"
            particle_lookup[label] = {
                'btype':       btype,
                'composition': composition,
                'center_ang':  np.array([cx, cy, cz]),
                'center_vox':  np.array([cx / vox_size, cy / vox_size, cz / vox_size]),
                'radius_ang':  radius,
                'radius_vox':  radius / vox_size,
                'angles':      angles,
            }

        print(f"_make_sample_recipe: placed {len(placed_particles)} particles")

        # ── 2. Build 3-D label map ─────────────────────────────────────────────────
        # Each voxel gets the integer label of whichever particle owns it (0 = none).
        # We use an ellipsoidal test identical to _acquire_scanned_image's r_scaled mask.
        label_map = np.zeros((nx, ny, nz), dtype=np.uint8)

        # Pre-build voxel coordinate arrays once (Angstrom positions of voxel centres)
        xv = (np.arange(nx) + 0.5) * vox_size
        yv = (np.arange(ny) + 0.5) * vox_size
        zv = (np.arange(nz) + 0.5) * vox_size
        XX, YY, ZZ = np.meshgrid(xv, yv, zv, indexing='ij')  # (nx, ny, nz)

        for label, (btype, center, radius, angles) in enumerate(placed_particles, start=1):
            z_radius = radius * aspect_ratio
            R = rotation_matrix(*angles)
            # Rotate the offset of every voxel from the particle centre
            # Broadcasting: offset shape (nx, ny, nz, 3) -> after R: same
            dX = XX - center[0]
            dY = YY - center[1]
            dZ = ZZ - center[2]
            # Apply inverse rotation to map world coords into particle frame
            Rinv = R.T
            dXr = Rinv[0, 0]*dX + Rinv[0, 1]*dY + Rinv[0, 2]*dZ
            dYr = Rinv[1, 0]*dX + Rinv[1, 1]*dY + Rinv[1, 2]*dZ
            dZr = Rinv[2, 0]*dX + Rinv[2, 1]*dY + Rinv[2, 2]*dZ

            inside = (dXr/radius)**2 + (dYr/radius)**2 + (dZr/z_radius)**2 <= 1.0
            # Only overwrite vacuum voxels (first-come wins; avoids blend artefacts)
            label_map[inside & (label_map == 0)] = label

        # ── 3. Build ASE Atoms object ──────────────────────────────────────────────
        all_positions = []
        all_symbols   = []

        for (btype, center, radius, angles) in placed_particles:
            this_bulk = bulk_types[btype]
            a_lat     = this_bulk.cell.lengths()[0]
            z_radius  = radius * aspect_ratio
            rep       = int(radius * 2 / a_lat) + 3
            supercell = this_bulk.repeat((rep, rep, rep))

            R         = rotation_matrix(*angles)
            positions = supercell.get_positions().copy()
            positions -= positions.mean(axis=0)
            positions  = positions @ R.T

            r_scaled = np.sqrt(
                (positions[:, 0] / radius)**2 +
                (positions[:, 1] / radius)**2 +
                (positions[:, 2] / z_radius)**2
            )
            mask = r_scaled <= 1.0
            positions = positions[mask] + center
            symbols   = [s for s, m in zip(supercell.get_chemical_symbols(), mask) if m]

            all_positions.append(positions)
            all_symbols.extend(symbols)

        all_positions = np.vstack(all_positions)
        cell = [fov_ang * 1.5, fov_ang * 1.5, world_z_ang]
        atoms_object = Atoms(
            symbols=all_symbols,
            positions=all_positions,
            cell=cell,
            pbc=False,
        )

        # ── Persist ────────────────────────────────────────────────────────────────
        self._vox_size          = vox_size
        self._world_shape       = (nx, ny, nz)
        self._particle_lookup   = particle_lookup
        self._particle_label_map = label_map
        self._atoms_object      = atoms_object


    def _cook_sample_recipe(self):
        """
        Apply the current stage position (x, y, z translation + alpha/beta tilt)
        to both the atom positions and the label map, then project both to 2-D.

        After this call, self._cooked_atoms and self._cooked_projection are ready
        for consumption by _acquire_scanned_image.

        Projection strategy
        -------------------
        The electron beam travels along the optical axis (z after tilt).  After
        applying the tilt rotation we simply sum the label map along axis=2 to get
        a 2-D occupancy map and sum the pseudo-potential contribution of every atom
        along the same axis.  This is the standard "projected potential"
        approximation used throughout the rest of the simulator.

        Coordinate conventions
        ----------------------
        * Stage (x, y, z) are in *metres* (from the proxy), converted to Å here.
        * Stage (alpha, beta) are in *degrees*.  alpha tilts around the y-axis
        (in-plane), beta tilts around the x-axis.
        * All rotations are applied around the centre of the world volume.
        * The map is wrapped (np.roll) so the sample never "runs out" when shifted.
        """
        import numpy as np

        stage    = self._detector_proxies.get("stage")
        x_m, y_m, z_m  = stage.x, stage.y, stage.z      # metres
        alpha_deg, beta_deg = stage.alpha, stage.beta     # degrees

        # Convert translation to Angstroms
        x_ang = x_m * 1e10
        y_ang = y_m * 1e10
        z_ang = z_m * 1e10

        nx, ny, nz  = self._world_shape
        vox         = self._vox_size          # Å/voxel
        label_map   = self._particle_label_map   # (nx, ny, nz) uint8
        atoms       = self._atoms_object         # ASE Atoms

        # ── Rotation matrix for tilt (alpha around Y, beta around X) ──────────────
        def _Ry(deg):
            t = np.radians(deg)
            return np.array([[ np.cos(t), 0, np.sin(t)],
                            [ 0,         1, 0        ],
                            [-np.sin(t), 0, np.cos(t)]])

        def _Rx(deg):
            t = np.radians(deg)
            return np.array([[1, 0,        0       ],
                            [0, np.cos(t),-np.sin(t)],
                            [0, np.sin(t), np.cos(t)]])

        R_stage = _Rx(beta_deg) @ _Ry(alpha_deg)

        # ── 1. Cook the atoms ──────────────────────────────────────────────────────
        positions  = atoms.get_positions().copy()           # (N, 3) in Å
        world_centre_ang = np.array([
            nx * vox / 2.0,
            ny * vox / 2.0,
            nz * vox / 2.0,
        ])

        # Translate stage shift (move world relative to beam)
        positions[:, 0] -= x_ang
        positions[:, 1] -= y_ang
        positions[:, 2] -= z_ang

        # Rotate around world centre
        positions -= world_centre_ang
        positions  = positions @ R_stage.T
        positions += world_centre_ang

        # Wrap atom x/y into the FOV periodically so atoms never "run out"
        fov_x = nx * vox
        fov_y = ny * vox
        positions[:, 0] = np.mod(positions[:, 0], fov_x)
        positions[:, 1] = np.mod(positions[:, 1], fov_y)

        cooked_atoms = atoms.copy()
        cooked_atoms.set_positions(positions)
        self._cooked_atoms = cooked_atoms

        # ── 2. Cook the label map ─────────────────────────────────────────────────
        # Translate via np.roll (integer-pixel wrap, cheap and artefact-free)
        shift_x_pix = int(round(x_ang / vox))
        shift_y_pix = int(round(y_ang / vox))
        rolled = np.roll(label_map, -shift_x_pix, axis=0)
        rolled = np.roll(rolled,    -shift_y_pix, axis=1)

        # Rotate the map by remapping voxel coordinates through R_stage
        # We build the rotated map by inverse-mapping: for each output voxel,
        # find where it came from in the un-rotated volume.
        xv = (np.arange(nx) - nx / 2.0)
        yv = (np.arange(ny) - ny / 2.0)
        zv = (np.arange(nz) - nz / 2.0)
        XX, YY, ZZ = np.meshgrid(xv, yv, zv, indexing='ij')   # (nx, ny, nz)

        # Inverse-rotate output coords back to input coords
        Rinv = R_stage.T    # orthogonal matrix: R^{-1} = R^T
        src_x = Rinv[0,0]*XX + Rinv[0,1]*YY + Rinv[0,2]*ZZ + nx / 2.0
        src_y = Rinv[1,0]*XX + Rinv[1,1]*YY + Rinv[1,2]*ZZ + ny / 2.0
        src_z = Rinv[2,0]*XX + Rinv[2,1]*YY + Rinv[2,2]*ZZ + nz / 2.0

        # Nearest-neighbour lookup (label maps are integer, NN is correct)
        src_xi = np.clip(np.round(src_x).astype(int), 0, nx - 1)
        src_yi = np.clip(np.round(src_y).astype(int), 0, ny - 1)
        src_zi = np.clip(np.round(src_z).astype(int), 0, nz - 1)

        rotated_map = rolled[src_xi, src_yi, src_zi]   # (nx, ny, nz) uint8

        # ── 3. Project to 2-D ─────────────────────────────────────────────────────
        # The beam travels along z (after tilt has been absorbed into the map).
        # Projection = "is any particle present in this column?"
        # For the label map we store the label of the *first* (shallowest) particle
        # hit, which is useful for EDS/EELS spectral routing later.

        # Occupancy map: True wherever any particle voxel exists
        occupancy_3d = rotated_map > 0          # (nx, ny, nz) bool

        # 2-D projected label: label of the particle in the first occupied z-slice
        # (shallow-most = smallest z index = beam enters first)
        projected_label = np.zeros((nx, ny), dtype=np.uint8)
        for iz in range(nz):
            slice_label = rotated_map[:, :, iz]
            unfilled = projected_label == 0
            projected_label[unfilled] = slice_label[unfilled]

        # Projected thickness (number of occupied voxels per column, in Angstroms)
        projected_thickness_ang = occupancy_3d.sum(axis=2).astype(np.float32) * vox

        self._cooked_projection = {
            'label_map_3d':       rotated_map,          # (nx, ny, nz) – tilted+shifted
            'projected_label':    projected_label,       # (nx, ny) – which particle per pixel
            'projected_thickness': projected_thickness_ang,  # (nx, ny) – Å of material
        }

    def _acquire_spectrum(self, detector_name: str, exposure_time: float):
        px, py = self.read_beam_pos()   # fractional [0, 1]
        px_pix = px * self._imsize
        py_pix = py * self._imsize

        for rec in self._particle_records:
            cx, cy = rec['center']   # pixels
            r      = rec['radius']   # pixels
            if (px_pix - cx)**2 + (py_pix - cy)**2 <= r**2:
                raw   = {el: frac for el, frac in rec['composition'].items()}
                total = sum(raw.values())
                return json.dumps({el: v / total + np.random.normal(0.01, 0.1) for el, v in raw.items()})

        all_elements = {el for rec in self._particle_records for el in rec['composition']}
        return json.dumps({el: np.abs(np.random.normal(0, 0.05)) for el in all_elements})


    def _place_beam(self, position) -> None:
        """
        sets resting beam position, [0:1]
        """
        x, y = position
        self.write_beam_pos([x, y])


    def _set_fov(self, fov) -> None:
        """set field of view in meters"""
        # For the digital twin, we can just store this as a property and use it in acquisition simulations.
        self._fov = fov

    def _set_defocus(self, defocus) -> None:
        """Set defocus in meters."""
        self._defocus = float(defocus)

    def _get_defocus(self) -> float:
        """Get defocus in meters."""
        return self._defocus


    def _get_stage(self):
        """Return current stage position as [x, y, z, alpha, beta], with tilts in degrees."""
        return self._stage_position
    
    def _move_stage(self, position):
        """Move stage to [x, y, z, alpha, beta], with x/y/z in meters and tilts in degrees."""
        self.old_pos = self._stage_position

        # shift the particle records/ atoms object positions by this much, negative

        random_shift = np.random.normal(0, 5e-8, size=5) 
        self._stage_position = position + random_shift

# ----------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    DigitalTwinBeta.run_server()
