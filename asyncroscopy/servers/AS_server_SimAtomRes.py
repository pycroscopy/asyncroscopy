# AS_server_SimAtomRes.py
# Digital Twin made by Austin Houston
# for simulating atomic resolution images

"""
designed to work ith Ceos digital twin
to get real probes and simulate images
mirrors the real thing.
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

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # removes "servers"



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

# sys.path.insert(0, "C:\\AE_future\\autoscript_1_14\\")
sys.path.insert(0, "/Users/austin/Desktop/Projects/autoscript_tem_microscope_client")
import autoscript_tem_microscope_client as auto_script


# FACTORY — holds shared state (persistent across all connections)
class ASFactory(protocol.Factory):
    def __init__(self):
        # persistent states for all protocol instances
        self.microscope = None
        self.detectors = {}
        self.status = "Offline"

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

    # working here
    def get_scanned_image(self, args: dict):
        """Return a scanned image using the indicated detector"""
        scanning_detector = args.get('scanning_detector')
        size = args.get('size')
        dwell_time = args.get('dwell_time')
        size = int(size)
        dwell_time = float(dwell_time)

        if dwell_time * size * size > 600: # frame time > 10 minutes
            self.log.info(f"[AS] Error: Acquisition too long: {dwell_time*size*size} seconds")
            return None
        else:
            self.log.info(f"[AS] Acquiring image with detector '{scanning_detector}', size={size}, dwell_time={dwell_time}s")
            self.factory.status = "Busy"

            # get probe
            # connect to central through the client
            tem = NotebookClient.connect(host='localhost',port=9000)
            ab = tem.send_command(destination = 'Ceos', command = 'getAberrations', args = {})
            ab = ast.literal_eval(ab)
            ab['acceleration_voltage'] = 60e3 # eV
            fov = 96 # angstroms
            ab['FOV'] = fov /12 # Angstroms
            ab['convergence_angle'] = 30 # mrad
            ab['wavelength'] = it.get_wavelength(ab['acceleration_voltage'])

            # make image
            # with pystemsim data generator
            # print("check the cif path ")
            cif_path = (
                PROJECT_ROOT
                / "cloned_repos"
                / "pystemsim"
                / "WS2_ortho.cif"
            )
            print("Reading CIF from:", cif_path)
            xtal = read(cif_path)
            # xtal = read('asyncroscopy/cloned_repos/pystemsim/WS2_ortho.cif')
            xtal = xtal * (30, 20, 1)
            positions = xtal.get_positions()[:, :2]
            pixel_size = 0.106 # angstrom/pixel
            frame = (0,fov,0,fov) # limits of the image in angstroms
            potential = dg.create_pseudo_potential(xtal, pixel_size, sigma=1, bounds=frame, atom_frame=11)
            probe = dg.get_probe(ab, potential)
            image = dg.convolve_kernel(potential, probe)
            noisy_image = dg.lowfreq_noise(image, noise_level=0.5, freq_scale=.04)
            sim_im = dg.poisson_noise(noisy_image, counts=1e7)
            # convert args dict

            # time.sleep(1)
            image = np.array(image, dtype=np.float32)
            # image = (np.random.rand(size, size) * 255).astype(np.uint8)
            self.factory.status = "Ready"
            self.sendString(package_message(image))


    def get_stage(self, args=None):
        """Return current stage position (placeholder)"""
        positions = [np.random.uniform(-10, 10) for _ in range(5)]
        positions = np.array(positions, dtype=np.float32)
        self.sendString(package_message(positions))

    def get_status(self, args=None):
        """Return the server status"""
        msg = f"Microscope is {self.factory.status}"
        self.sendString(package_message(msg))


if __name__ == "__main__":
    port = 9001
    print(f"[AS] Server running on port {port}...")
    reactor.listenTCP(port, ASFactory())
    reactor.run()