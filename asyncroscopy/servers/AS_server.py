# AS_server.py

"""
the real thing.
"""

from twisted.internet import reactor, protocol
import numpy as np
import time
import sys

from asyncroscopy.servers.protocols.execution_protocol import ExecutionProtocol
from asyncroscopy.servers.protocols.utils import package_message, unpackage_message
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
        
        print(f"[AS] Connecting to microscope at {host}:{port}...")
        try:
            self.factory.microscope = auto_script.TemMicroscopeClient()
            self.factory.microscope.connect(host=str(host), port=int(port))
            self.factory.status = "Ready"
            msg = "[AS] Connected to microscope."
        except Exception as e:
            msg = f"[AS] Failed to connect to microscope: {e}"
            self.factory.microscope = None
        self.log.info(msg)
        self.sendString(package_message(msg))


    def calibrate_screen_current(self, args: dict = None):
        """
        calibrates the gun lens values to screen current
        start with screen current at ~100 pA
        screen must be inserted
        """
        mic = self.factory.microscope
        original_gun_lens = mic.optics.monochromator.focus
        gun_lens_series = np.linspace(5, 100, 15)

        # series of measurements
        current_series = []
        for val in gun_lens_series:
            mic.optics.monochromator.focus = val # original_gun_lens + val
            time.sleep(1)
            screen_current = mic.detectors.screen.measure_current()
            current_series.append(screen_current)
        current_series = np.array(current_series) * 1e12
        mic.optics.monochromator.focus = original_gun_lens

        # fit a polynomial:
        coeffs = np.polyfit(gun_lens_series, current_series, 11)
        poly_func = np.poly1d(coeffs)
        x_fit = np.linspace(min(gun_lens_series), max(gun_lens_series), 500)
        y_fit = poly_func(x_fit)

        # save polyfunc
        self.factory.current_calibration = poly_func
        msg = f"[AS] calibrated screen current"
        self.log.info(msg)
        self.sendString(package_message(msg))

    # gerd's code - check
#     def carth2polar(z):
#         return np.linalg.norm(z), np.degrees(np.arctan2(z[1], z[0]))
#     # def carth2polar, correct_3rd_orders, correct_low_orders
#     def aberration_correction(self, args: dict):
#         """Perform aberration correction"""
#         tem = NotebookClient.connect(host='localhost',port=9000)
#         mic = self.factory.microscope
#         mic.optics.scan_field_of_view  = 348*1e-9
#         order = parameter.get('order, 1')
# 
#         print(f"Performing aberration correction of order {order}")
#         if order <3:
#             tableau_result = tem.send_command(destination = 'Ceos', command = 'acquireTableau', args = {'tabType':"Fast", 'angle':1})
#             for key in ['C1', 'A1', 'B2', 'A2']:
#                 amplitude , angle = carth2polar(tableau_result['aberrations'][key])                                         
#                 print(f" {key}: {amplitude*1e9:.2f}nm {angle:.2f}deg")
#             print(f" WD: {np.linalg.norm(tableau_result['aberrations']['WD'])*1e3:.3f}mrad ")
#             tableau_result['corrected'] = correct_low_orders(self.ceos, tableau_result['aberrations'])
#         else:
#             tableau_result = self.ceos.run_tableau(tab_type="Enhanced", angle=40)
# 
#             for key in ['C3', 'S3', 'A3', 'A4', 'D4', 'B4']:   
#                 amplitude , angle = carth2polar(tableau_result['aberrations'][key])                                         
#                 print(f" {key}: {amplitude*1e9:.2f}nm {angle:.2f}deg")
#                 tableau_result['corrected'] = correct_3rd_orders(self.ceos, tableau_result['aberrations'])
#         
#         self.sendString(package_message(out_dict))
        

# ---------------------------


    def set_current(self, args:dict):
        """
        set screen current (via gun lens)
        must have screen current calibrated
        """
        mic = self.factory.microscope
        current = args.get('current')
        current = float(current)
        poly_func = self.factory.current_calibration

        adjusted_poly = poly_func - current
        x_candidates = adjusted_poly.r
        x_real = x_candidates[np.isreal(x_candidates)].real
        if len(x_real) >= 1:
            x_real = np.max(x_real)

        mic.optics.monochromator.focus = float(x_real)

        msg = f"[AS] current set to {current}"
        self.log.info(msg)
        self.sendString(package_message(msg))

    def place_beam(self, args: dict = None):
        """
        sets resting beam position, [0:1]
        """
        mic = self.factory.microscope
        beam_pos = args.get('beam_pos')
        x = beam_pos[0]
        y = beam_pos[1]
        mic.optics.paused_scan_beam_position = (x, y)

        msg = f"Beam moved to {x}, {y}"
        self.log.info(f"[AS] {msg}")
        self.sendString(package_message(msg))

    def blank_beam(self, args: dict):
        """blank beam"""
        self.factory.microscope.optics.blanker.blank()

        msg = f"Beam unblanked for {dwell_time}s"
        self.log.info(f"[AS] {msg}")
        self.sendString(package_message(msg))

    def unblank_beam(self, args: dict = None):
        """
        unblank beam
        optional dwell time, then auto-blank
        """
        dwell_time = args.get('dwell_time')
        
        # unblank here
        msg = "Beam unblanked"
        self.factory.microscope.optics.blanker.unblank()

        if dwell_time: # blank if desired
            time.sleep(dwell_time)
            self.blank_beam()
            msg = f"Beam unblanked for {dwell_time}s"
        
        self.log.info(f"[AS] {msg}")
        self.sendString(package_message(msg))

    def get_scanned_image(self, args: dict):
        """Return a scanned image using the indicated detector"""
        scanning_detector = args.get('scanning_detector')
        size = args.get('size')
        dwell_time = args.get('dwell_time')
        size = int(size)
        dwell_time = float(dwell_time)

        if dwell_time * size * size > 600: # frame time > 10 minutes
            print(f"[AS] Error: Acquisition too long: {dwell_time*size*size} seconds")
            return None
        else:
            self.factory.status = "Busy"
            image = self.factory.microscope.acquisition.acquire_stem_image(
                scanning_detector = 'HAADF', 
                size = size, 
                dwell_time = dwell_time)
            image = np.array(image.data, dtype=np.float32)
            self.factory.status = "Ready"
            self.sendString(package_message(image))

    def get_stage(self, args: dict = None):
        """Return current stage position"""
        positions = self.factory.microscope.specimen.stage.position
        positions = np.array(positions, dtype=np.float32)
        self.sendString(package_message(positions))

    def get_status(self, args: dict = None):
        """Return the server status"""
        msg = f"Microscope is {self.factory.status}"
        self.sendString(package_message(msg))


if __name__ == "__main__":
    port = 9001
    print(f"[AS] Server running on port {port}...")
    reactor.listenTCP(port, ASFactory())
    reactor.run()