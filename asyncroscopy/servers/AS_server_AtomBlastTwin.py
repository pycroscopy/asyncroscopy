# AS_server_SimAtomRes.py
# Digital Twin made by Austin Houston
# Enhanced with beam damage simulation capabilities

"""
Enhanced digital twin for simulating atomic resolution images
with electron beam damage effects.
Tracks accumulated dose and simulates atom knockout probabilistically.
"""
import ast
import sys
import time
import numpy as np

from asyncroscopy.clients.notebook_client import NotebookClient
from asyncroscopy.servers.protocols.execution_protocol import ExecutionProtocol
from asyncroscopy.servers.protocols.utils import package_message, unpackage_message

from pathlib import Path
from ase.io import read
from ase import Atoms
from ase.neighborlist import NeighborList

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

try:
    from ase.io import read
except ImportError:
    print("ASE not installed; some functionality may be limited.")
try:
    from asyncroscopy.cloned_repos.pystemsim import data_generator as dg
except ImportError:
    print("pystemsim not installed; some functionality may be limited.")

import pyTEMlib.probe_tools as pt
import pyTEMlib.image_tools as it

from twisted.internet import reactor, protocol
from twisted.internet.defer import Deferred, inlineCallbacks, returnValue

import autoscript_tem_microscope_client as auto_script


# FACTORY — holds shared state (persistent across all connections)
class ASFactory(protocol.Factory):
    def __init__(self):
        # Persistent states for all protocol instances
        self.microscope = None
        self.detectors = {}
        self.status = "Offline"
        
        # Beam damage simulation state
        self.atoms = None  # ASE Atoms object
        self.dose_map = None  # 2D array tracking accumulated dose (e/Å²)
        self.beam_position = None  # (x, y) in angstroms
        self.beam_blanked = True
        self.beam_current = 100.0  # pA
        self.acceleration_voltage = 60e3  # eV
        
        # Simulation parameters
        self.fov = 96.0  # angstroms
        self.pixel_size = 0.106  # angstrom/pixel
        self.grid_shape = None  # Will be set when atoms are loaded
        
        # Damage model parameters (placeholders - can be tuned)
        self.knockout_cross_section = 1e-24  # cm² (knock-on)
        self.ionization_cross_section = 5e-22  # cm² (ionization)
        self.healing_rate = 0.01  # probability per second
        self.damage_threshold = 1e6  # e/Å² before significant damage

    def buildProtocol(self, addr):
        """Create a new protocol instance and attach the factory (shared state)."""
        proto = ASProtocol()
        proto.factory = self
        return proto


# PROTOCOL — handles per-connection command execution
class ASProtocol(ExecutionProtocol):
    def __init__(self):
        super().__init__()
        allowed = []
        for name, value in ExecutionProtocol.__dict__.items():
            if callable(value) and not name.startswith("_"):
                allowed.append(name)
        self.allowed_commands = set(allowed)

    def connect_AS(self, args: dict):
        """Connect to the microscope via AutoScript"""
        host = args.get('host')
        port = args.get('port')
        
        self.log.info(f"[AS] Connecting to microscope at {host}:{port}...")
        self.factory.microscope = 'Debugging'
        self.factory.status = "Ready"
        msg = "Connected to Digital Twin microscope."
        self.sendString(package_message(msg))

    def load_sample(self, args: dict):
        """Load a crystal structure from a CIF file"""
        cif_path = args.get('cif_path', None)
        replicate = args.get('replicate', (30, 20, 1))  # (x, y, z) replication
        
        # Parse replicate if it comes as a string
        if isinstance(replicate, str):
            try:
                replicate = ast.literal_eval(replicate)
            except (ValueError, SyntaxError):
                self.log.warning(f"[AS] Could not parse replicate '{replicate}', using default (30, 20, 1)")
                replicate = (30, 20, 1)
        
        if cif_path is None:
            # Default to WS2
            cif_path = (
                PROJECT_ROOT
                / "cloned_repos"
                / "pystemsim"
                / "WS2_ortho.cif"
            )
        
        self.log.info(f"[AS] Loading sample from {cif_path}")
        xtal = read(cif_path)
        xtal = xtal * replicate
        xtal.set_pbc((True, True, False))
        # Store atoms in factory (persistent state)
        self.factory.atoms = xtal
        
        # Initialize dose map
        ny = int(self.factory.fov / self.factory.pixel_size)
        nx = int(self.factory.fov / self.factory.pixel_size)
        self.factory.grid_shape = (ny, nx)
        self.factory.dose_map = np.zeros((ny, nx), dtype=np.float64)
        
        # Initialize beam position at center
        self.factory.beam_position = (self.factory.fov / 2, self.factory.fov / 2)
        
        msg = f"Loaded sample with {len(xtal)} atoms. Dose map initialized."
        self.sendString(package_message(msg))

    def place_beam(self, args: dict):
        """Move the beam to a specific position (x, y in normalized coordinates 0-1)"""
        x = float(args.get('x'))
        y = float(args.get('y'))
        
        # Validate normalized coordinates
        if 0 <= x <= 1 and 0 <= y <= 1:
            self.factory.beam_position = (x, y)
            # Convert to angstroms for logging
            x_ang = x * self.factory.fov
            y_ang = y * self.factory.fov
            msg = f"Beam positioned at ({x:.3f}, {y:.3f}) normalized = ({x_ang:.2f}, {y_ang:.2f}) Å"
            self.log.info(f"[AS] {msg}")
            self.sendString(package_message(msg))
        else:
            msg = f"Error: Position ({x}, {y}) outside normalized range (0-1)"
            self.log.error(f"[AS] {msg}")
            self.sendString(package_message(msg))

    def blank_beam(self, args=None):
        """Blank the electron beam (stop dose accumulation)"""
        self.factory.beam_blanked = True
        msg = "Beam blanked"
        self.log.info(f"[AS] {msg}")
        self.sendString(package_message(msg))

    def unblank_beam(self, args: dict):
        """Unblank the electron beam and start dose accumulation"""
        duration = float(args.get('duration', 1))  # seconds
        
        self.factory.beam_blanked = False
        msg = f"Beam unblanked"
        self.log.info(f"[AS] {msg}")
        
        if duration > 0:
            # Apply dose for specified duration
            self._apply_beam_dose(duration)
            msg += f" for {duration}s. Damage applied."
        
        self.sendString(package_message(msg))

    def set_beam_current(self, args: dict):
        """Set the beam current in pA"""
        current = float(args.get('current'))
        self.factory.beam_current = current
        msg = f"Beam current set to {current} pA"
        self.log.info(f"[AS] {msg}")
        self.sendString(package_message(msg))

    def set_fov(self, args: dict):
        """Set the field of view in angstroms"""
        fov = float(args.get('fov'))
        old_fov = self.factory.fov
        self.factory.fov = fov
        
        # Update grid shape if atoms are loaded
        if self.factory.atoms is not None:
            ny = int(self.factory.fov / self.factory.pixel_size)
            nx = int(self.factory.fov / self.factory.pixel_size)
            old_shape = self.factory.grid_shape
            self.factory.grid_shape = (ny, nx)
            
            # Resize dose map (interpolate or recreate)
            self.factory.dose_map = np.zeros((ny, nx), dtype=np.float64)
            msg = f"FOV changed from {old_fov} to {fov} Å. Grid: {old_shape} → {self.factory.grid_shape}. Dose map reset."
        else:
            msg = f"FOV set to {fov} Å"
        
        self.log.info(f"[AS] {msg}")
        self.sendString(package_message(msg))

    def _apply_beam_dose(self, duration):
        """Apply electron dose to the dose map using the real probe"""
        if self.factory.atoms is None:
            self.log.warning("[AS] No sample loaded. Cannot apply dose.")
            return

        if self.factory.beam_blanked:
            return

        e_charge = 1.602e-19  # Coulombs
        total_electrons = (
            self.factory.beam_current * 1e-12 * duration
        ) / e_charge

        # Get real probe from microscope model
        tem = NotebookClient.connect(host='localhost', port=9000)
        ab = tem.send_command(destination='Ceos', command='getAberrations', args={})
        ab = ast.literal_eval(ab)
        ab['acceleration_voltage'] = self.factory.acceleration_voltage
        ab['FOV'] = self.factory.fov / 12
        ab['convergence_angle'] = 30  # mrad
        ab['wavelength'] = it.get_wavelength(ab['acceleration_voltage'])
        probe, A_k, chi  = pt.get_probe(ab, self.factory.grid_shape[0], self.factory.grid_shape[1], verbose= True)


        print('probe created')
        # Convert normalized position → pixel indices
        x_norm, y_norm = self.factory.beam_position
        ny, nx = self.factory.grid_shape
        x_pix = int(x_norm * nx)
        y_pix = int(y_norm * ny)
        cx = nx // 2
        cy = ny // 2
        shift_x = x_pix - cx
        shift_y = y_pix - cy

        # this can lead to unphysical results near edge
        beam_profile = np.roll(probe,shift=(shift_y, shift_x),axis=(0, 1))

        pixel_area = self.factory.pixel_size ** 2  # Å²
        dose_increment = beam_profile * total_electrons / pixel_area
        self.factory.dose_map += dose_increment

        self._apply_damage_model()

    def _apply_damage_model(self, dose_map=None):
        """
        Damage model with three independent channels:
        1) Knock-on (ballistic)
        2) Radiolysis / ionization
        3) Neighbor-induced structural instability

        All channels contribute additively to a Poisson hazard.
        """

        if self.factory.atoms is None:
            return

        atoms = self.factory.atoms
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()

        # scanned image passes custon dose map
        # else, use factory dose map
        if dose_map is None:
            dose_map = self.factory.dose_map

        pixel_size = self.factory.pixel_size
        ny, nx = self.factory.grid_shape

        # Cross sections / rates (Å² or rate-like prefactors)
        # Knock-on (200 kV)
        sigma_knockon = {"S": 3e-7,"Se": 1e-7,"Mo": 1e-9,"W": 5e-10}

        # Radiolysis / ionization
        sigma_radiolysis = {"S": 5e-9,"Se": 2e-9,"Mo": 1e-10,"W": 5e-11}

        # Structural instability prefactor
        gamma_instability = {"S": 0.3,"Se": 0.3,"Mo": 0.3,"W": 0.3}

        # Ideal coordination numbers
        ideal_coordination = {"S": 3,"Se": 3,"Mo": 6,"W": 6,}

        # Neighbor list (first coordination shell)
        cutoffs = []
        for sym in symbols:
            if sym in ("S", "Se"):
                cutoffs.append(1.2)
            else:
                cutoffs.append(1.2)

        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)

        atoms_to_remove = []
        for i, (pos, sym) in enumerate(zip(positions, symbols)):
            # Dose lookup
            x, y = pos[:2]
            ix = int(x / pixel_size)
            iy = int(y / pixel_size)

            if not (0 <= ix < nx and 0 <= iy < ny):
                continue

            local_dose = dose_map[iy, ix]
            if local_dose <= 0.0:
                continue

            # Knock-on hazard
            lambda_knock = (sigma_knockon.get(sym, 0.0) * local_dose)

            # Radiolysis hazard
            lambda_rad = (sigma_radiolysis.get(sym, 0.0) * local_dose)

            # Neighbor instability hazard
            neighbors, offsets = nl.get_neighbors(i)
            N = len(neighbors)
            N0 = ideal_coordination.get(sym, N)
            missing = max(N0 - N, 0)
            if missing > 0:
                frac_lost = missing / N0
                coord_amp = np.exp(frac_lost)
                lambda_inst = (gamma_instability.get(sym, 0.0) * coord_amp * local_dose / 1e4) # change here
            else:
                lambda_inst = 0.0
            if N <=1:
                lambda_inst  = 1e6  # Isolated atom, very unstable
            # Total hazard and removal
            lambda_total = lambda_knock + lambda_rad + lambda_inst
            p_remove = 1.0 - np.exp(-lambda_total)

            if np.random.rand() < p_remove:
                atoms_to_remove.append(i)

        # Apply removals
        if atoms_to_remove:
            mask = np.ones(len(atoms), dtype=bool)
            mask[atoms_to_remove] = False
            self.factory.atoms = atoms[mask]

            self.log.info(
                f"[AS] Damage model removed {len(atoms_to_remove)} atoms "
                f"(knock-on + radiolysis + instability)")


    def get_dose_map(self, args=None):
        """Return the current accumulated dose map"""
        if self.factory.dose_map is None:
            msg = "No dose map available. Load a sample first."
            self.sendString(package_message(msg))
        else:
            dose_map = np.array(self.factory.dose_map, dtype=np.float32)
            self.sendString(package_message(dose_map))

    def get_atom_count(self, args=None):
        """Return the current number of atoms in the sample"""
        if self.factory.atoms is None:
            count = 0
        else:
            count = len(self.factory.atoms)
        
        msg = f"Current atom count: {count}"
        self.log.info(f"[AS] {msg}")
        self.sendString(package_message(msg))

    def reset_sample(self, args=None):
        """Reset the sample to original state and clear dose map"""
        # Reload the original sample
        self.load_sample({})
        msg = "Sample and dose map reset to initial state"
        self.log.info(f"[AS] {msg}")
        self.sendString(package_message(msg))

    def get_scanned_image(self, args: dict):
        """Return a scanned image using the indicated detector"""
        scanning_detector = args.get('scanning_detector')
        size = args.get('size')
        dwell_time = args.get('dwell_time')
        size = int(size)
        dwell_time = float(dwell_time)

        if dwell_time * size * size > 600:  # frame time > 10 minutes
            self.log.info(f"[AS] Error: Acquisition too long: {dwell_time*size*size} seconds")
            return None
        else:
            self.log.info(f"[AS] Acquiring image with detector '{scanning_detector}', size={size}, dwell_time={dwell_time}s")
            self.factory.status = "Busy"

            # Get probe
            tem = NotebookClient.connect(host='localhost', port=9000)
            ab = tem.send_command(destination='Ceos', command='getAberrations', args={})
            ab = ast.literal_eval(ab)
            ab['acceleration_voltage'] = self.factory.acceleration_voltage
            fov = self.factory.fov
            ab['FOV'] = fov / 12
            ab['convergence_angle'] = 30  # mrad
            ab['wavelength'] = it.get_wavelength(ab['acceleration_voltage'])

            # Use current atoms state (with damage applied)
            if self.factory.atoms is None:
                self.log.warning("[AS] No sample loaded. Loading default...")
                self.load_sample({})
            
            xtal = self.factory.atoms
            positions = xtal.get_positions()[:, :2]
            pixel_size = self.factory.pixel_size
            frame = (0, fov, 0, fov)
            
            # Generate image from current atom configuration
            potential = dg.create_pseudo_potential(xtal, pixel_size, sigma=1, bounds=frame, atom_frame=11)
            probe = dg.get_probe(ab, potential)
            image = dg.convolve_kernel(potential, probe)
            noisy_image = dg.lowfreq_noise(image, noise_level=0.5, freq_scale=.04)

            scan_time = dwell_time * size * size
            counts = scan_time * (self.factory.beam_current * 1e-12) / (1.602e-19)
            sim_im = dg.poisson_noise(noisy_image, counts=counts)

            # Apply dose during scan
            self.factory.dose_map += dwell_time * (self.factory.beam_current * 1e-12) / (1.602e-19)
            delta_dose = np.zeros_like(self.factory.dose_map) + dwell_time * (self.factory.beam_current * 1e-12) / (1.602e-19)
            self._apply_damage_model(dose_map=delta_dose)

            image = np.array(sim_im, dtype=np.float32)
            self.factory.status = "Ready"
            self.sendString(package_message(image))


    def get_stage(self, args=None):
        """Return current stage position (placeholder)"""
        positions = [np.random.uniform(-10, 10) for _ in range(5)]
        positions = np.array(positions, dtype=np.float32)
        self.sendString(package_message(positions))

    def get_status(self, args=None):
        """Return the server status"""
        beam_status = "blanked" if self.factory.beam_blanked else "unblanked"
        if self.factory.atoms is not None:
            atom_count = len(self.factory.atoms)
            msg = f"Microscope is {self.factory.status}. Beam: {beam_status}. Atoms: {atom_count}"
        else:
            msg = f"Microscope is {self.factory.status}. Beam: {beam_status}. No sample loaded."
        self.sendString(package_message(msg))


if __name__ == "__main__":
    port = 9001
    print(f"[AS] Server running on port {port}...")
    reactor.listenTCP(port, ASFactory())
    reactor.run()
