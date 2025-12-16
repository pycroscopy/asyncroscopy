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

sys.path.insert(0, "/Users/austin/Desktop/Projects/autoscript_tem_microscope_client")
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
        duration = float(args.get('duration', 0))  # seconds
        
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
        """Apply electron dose to the dose map and calculate damage"""
        if self.factory.atoms is None:
            self.log.warning("[AS] No sample loaded. Cannot apply dose.")
            return
        
        if self.factory.beam_blanked:
            return
        
        # Calculate total electrons delivered
        # current (pA) * duration (s) / electron charge (C)
        e_charge = 1.602e-19  # Coulombs
        total_electrons = (self.factory.beam_current * 1e-12 * duration) / e_charge
        
        # Create Gaussian beam profile
        # Convert normalized coordinates to angstroms
        x_norm, y_norm = self.factory.beam_position
        x_beam = x_norm * self.factory.fov
        y_beam = y_norm * self.factory.fov
        
        probe_size = 0.5  # angstroms (FWHM)
        sigma = probe_size / 2.355  # Convert FWHM to sigma
        
        # Create coordinate grids
        y_coords = np.linspace(0, self.factory.fov, self.factory.grid_shape[0])
        x_coords = np.linspace(0, self.factory.fov, self.factory.grid_shape[1])
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Gaussian beam profile
        beam_profile = np.exp(-((X - x_beam)**2 + (Y - y_beam)**2) / (2 * sigma**2))
        beam_profile /= beam_profile.sum()  # Normalize
        
        # Add dose to dose map (electrons per Ų)
        pixel_area = self.factory.pixel_size ** 2
        dose_increment = beam_profile * total_electrons / pixel_area
        self.factory.dose_map += dose_increment
        
        # Apply damage based on accumulated dose
        self._apply_damage_model()
        
        max_dose = self.factory.dose_map.max()
        self.log.info(f"[AS] Applied {total_electrons:.2e} electrons. Max dose: {max_dose:.2e} e/Ų")

    def _apply_damage_model(self):
        """
        Apply probabilistic damage to atoms based on dose map.
        Damage mechanisms:
        1. Knock-on damage (elastic scattering)
        2. Ionization damage (inelastic scattering)
        3. Healing (random atom recovery - simplified)
        """
        if self.factory.atoms is None:
            return
        
        positions = self.factory.atoms.get_positions()[:, :2]  # x, y only
        pixel_size = self.factory.pixel_size
        
        # For each atom, calculate local dose and damage probability
        atoms_to_remove = []
        
        for i, (x, y) in enumerate(positions):
            # Find pixel indices
            ix = int(x / pixel_size)
            iy = int(y / pixel_size)
            
            # Check bounds
            if 0 <= ix < self.factory.grid_shape[1] and 0 <= iy < self.factory.grid_shape[0]:
                local_dose = self.factory.dose_map[iy, ix]
                
                # Calculate damage probability
                # P_knockout = 1 - exp(-sigma * dose)
                # Combine knock-on and ionization
                
                # Convert cross sections from cm² to Ų (1 cm² = 1e16 Ų)
                sigma_ko = self.factory.knockout_cross_section * 1e16
                sigma_ion = self.factory.ionization_cross_section * 1e16
                
                # Total damage probability
                p_damage = 1 - np.exp(-(sigma_ko + sigma_ion) * local_dose)
                
                # Add randomness
                if np.random.rand() < p_damage:
                    atoms_to_remove.append(i)
        
        # Remove damaged atoms
        if atoms_to_remove:
            mask = np.ones(len(self.factory.atoms), dtype=bool)
            mask[atoms_to_remove] = False
            self.factory.atoms = self.factory.atoms[mask]
            self.log.info(f"[AS] Removed {len(atoms_to_remove)} atoms due to beam damage")

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
