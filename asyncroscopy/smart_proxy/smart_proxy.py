""" First server"""

import sys
import os

import numpy as np
import Pyro5.api
from base_proxy import MicroscopeServer, serialize

from CEOSacquisition import CEOSacquisitionTCP


sys.path.insert(0, "C:\\AE_future\\autoscript_1_14\\")
import autoscript_tem_microscope_client as auto_script

os.environ["PYRO_LOGLEVEL"] = "DEBUG"

def default_flu_camera(detector_dict):
    detector_dict['flu_camera'] = {'size': 512,
                                   'exposure': 0.1,
                                   'binning': 1,
                                   'save_to_disc': False,
                                    'insert': 'flu_camera',
                                    'retract': None}
def default_ceta_camera(detector_dict):
    detector_dict['ceta_camera'] = {'data': np.zeros((512, 512), dtype=np.uint16),
                                    'size': 512,
                                    'exposure': 0.1,
                                    'binning': 1,
                                    'insert': 'ceta_camera',
                                    'retract': 'flu_camera'}
def default_scan(detector_dict):
    detector_dict['scan'] = {'data': np.zeros((512, 512), dtype=np.uint16),
                                    'size': 512,
                                    'exposure': 4e-6,
                                    'field_of_view': (1e-6, 1e-6),
                                    'detectors': ['HAADF'],
                                    'camera_length': 91e-3,}
def default_eds(detector_dict):
    detector_dict['super_x'] = {'data': np.zeros(2048, dtype=np.uint16),
                                'size': 2048,
                                'exposure': 0.1,
                                'binning': 2,
                                'energy_window': (0, 20000),
                                'insert': None,
                                'retract': None}
default_haadf_detector = {'detector_type': 'HAADF',
                          'collection_angle_inner': 50e-3,
                          'collection_angle_outer': 200e-3,
                          }

@Pyro5.api.expose
class TEMServer(MicroscopeServer):
    """Class to handle the array server"""
    def __init__(self):
        super().__init__()

        default_flu_camera(self.detectors)
        default_ceta_camera(self.detectors)
        default_scan(self.detectors)
        default_eds(self.detectors)
        self.status_parameters = ['vacuum', 'column_valve', 'stage_position', 'beam_current']

        self.microscope = auto_script.TemMicroscopeClient()
        self.connect_to_autoscript()
        print('initialized')
        self.ceos = CEOSacquisitionTCP() # host="10.46.217.241", port=9092)
    
    def get_detectors(self):
        """Get the list of available detectors"""
        return list(self.detectors.keys())

    def get_instrument(self):
        """Get the current instrument"""
        return self.microscope

    def get_commands(self):
        """Get the list of available commands"""
        return []

    def help_commands(self):
        """Get help for available commands"""
        return "help"

    def get_instrument_status(self, parameters=None):
        """Get the current microscope status"""

        vacuum = self.microscope.vacuum.state
        column_valve = self.microscope.vacuum.column_valves.state
        stage_position = self.microscope.specimen.stage.position
        beam_current = self.microscope.detectors.screen.measure_current()
        out_dict = {'vacuum': vacuum,
                    'column_valve': column_valve,
                    # 'stage_position': stage_position,
                    'beam_current': beam_current*1e9}
        print(out_dict)
        if parameters is not None:
            return {param: out_dict.get(param, None) for param in parameters}
        return out_dict
    
    def activate_device(self, device):
        """Activate the specified device for acquisition"""
        if device in self.detectors:
            print(f"{device.capitalize()} activated")
            return 1
        else:
            print(f"Device {device} not found")
            return 0
        
    def device_settings(self, device, **args):
        """Set the device settings"""
        if device in self.detectors:
            print(f"Setting {device} settings: {args}")
            for key, value in args.items():
                if key in self.detectors[device]:
                    self.detectors[device][key] = value
            return 1
        else:
            print(f"Device {device} not found")
            return 0

    def get_stage(self):
        """Get the current stage position"""
        positions = self.microscope.specimen.stage.position
        if self.microscope.specimen.stage.get_holder_type() == "SingleTilt":
            return [float(positions[0]), float(positions[1]), float(positions[2]), float(positions[3]), 0]
        else:
            return [float(positions[0]), float(positions[1]), float(positions[2]), float(positions[3]), float(positions[4])]

    def set_stage(self, stage_positions, relative=True):
        """Set the stage position in nm and degrees"""
        stage_move = auto_script.structures.StagePosition()
        for index, direction in enumerate(['x', 'y', 'z', 'a', 'b']):
            move = stage_positions.get(direction, None)
            setattr(stage_move, direction, move*1e-69)
        if relative:
            self.microscope.specimen.stage.relative_move_safe(stage_move)
        else:
            self.microscope.specimen.stage.absolute_move_safe(stage_move)
        print(f"Moving stage by {stage_move}")

    def acquire_image(self, device_name, **args):
        """Acquire an image from the specified device"""
        if device_name in self.detectors:
            print(f"Acquiring image from {device_name}")
            camera = None
            if device_name == 'flu_camera':
                camera = auto_script.enumerations.CameraType.FLUCAM
            elif device_name == 'ceta_camera':
                camera = auto_script.enumerations.CameraType.BM_CETA
            device = self.microscope.detectors.get_camera_detector(camera)
            self.activate_device(device_name)
            image = self.microscope.acquisition.acquire_camera_image(
                                                    camera,
                                                    self.detectors[device_name]['size'],
                                                    self.detectors[device_name]['exposure'])
            return serialize(image.data)
        else:
            print(f"Device {device_name} not found")
            return None

    def acquire_image_stack(self, device):
        """Acquire an image from the specified device"""
        if device in self.detectors:
            print(f"Acquiring image from {device}")
            return self.detectors[device]
        else:
            print(f"Device {device} not found")
            return None

    def acquire_spectrum(self, device, **args):
        """Acquire a spectrum from the specified device"""
        if device in self.detectors:
            print(f"Acquiring spectrum from {device}")
            return np.zeros((2048,), dtype=np.float32)
        else:
            print(f"Device {device} not found")
            return None

    def acquire_spectrum_points(self, device, points, **args):
        """Acquire a spectrum stack from the specified device"""
        if device in self.detectors:
            print(f"Acquiring spectrum stack from {device}")
            spectra = []
            for point in points:
                self.set_beam_position(point[0], point[1])
                spectra.append(self.acquire_spectrum(device, **args))
                print(f"  at point {point}")
            
        else:
            print(f"Device {device} not found")
            return None
    
    def set_probe_position(self, x, y):
        """Set the beam position in nm"""
        print(f"Moving beam to ({x}, {y})")
        return 1
    
    def set_microscope_status(self, parameter=None, value=None):
        """Set the current microscope status"""
        if parameter == 'column_valve':
            if value == 'open':
                self.microscope.vacuum.column_valves.open()
            else:
                self.microscope.vacuum.column_valves.close()
            print(f"Setting column valve to {value}")
        elif parameter == 'optics_mode':
            if value == 'TEM':
                self.microscope.optics.optical_mode = auto_script.enumerations.OpticalMode.TEM
            elif value == 'STEM':
                self.microscope.optics.optical_mode = auto_script.enumerations.OpticalMode.STEM
            print(f"Setting optics mode to {value}")
        else:
            print(f"Parameter {parameter} not recognized or cannot be set")

    
    def connect_to_autoscript(self):
        ip = "127.0.0.1"
        # ip = "10.46.217.241"
        self.microscope.connect(ip, port = 9095)

    def aberration_correction(self, order, **args):
        """Perform aberration correction"""
        print(f"Performing aberration correction of order {order}")
        dd = self.ceos.run_tableau()
        # print(dd)
        return 1

    def close(self):
        """Close the server"""
        print("Closing server")
        return 1
    



def main(host = "10.46.217.241", port = 9093):
    """Main function to start the server"""
    daemon = Pyro5.api.Daemon(host=host, port=port)
    uri = daemon.register(TEMServer, objectId="tem.server")
    print("Server is ready. Object uri =", uri)
    daemon.requestLoop()

if __name__ == "__main__":
    main()
