import sys
import os

import numpy as np
import Pyro5.api

os.environ["PYRO_LOGLEVEL"] = "DEBUG"


def serialize(array):
    array_list = array.tolist()
    dtype = str(array.dtype)
    return array_list, array.shape, dtype


class MicroscopeServer(object):
    """Class to handle the array server"""
    def __init__(self):
        print('init')
        self.detectors  = {}

        self.microscope = None
        self.available_parameters = []

    @Pyro5.api.expose
    def get_instrument(self):
        """Get the current instrument"""
        return self.microscope

    @Pyro5.api.expose
    def get_commands(self):
        """Get the list of available commands"""
        return []

    @Pyro5.api.expose
    def help_commands(self):
        """Get help for available commands"""
        return "help"
    
    @Pyro5.api.expose
    def get_instrument_status(self, parameters=None):
        """Get the current microscope status"""

    @Pyro5.api.expose
    def microscope_command(self, comand):
        """Send a command to the microscope"""
        print(f"Sending command to microscope: {comand}")
        return 1

    @Pyro5.api.expose
    def align_microscope(self, device, order, **args):
        """Align the microscope"""
        print(f"Aligning microscope with {device} for {order} order")
        return 1

    @Pyro5.api.expose
    def get_detectors(self):
        """Get the list of available detectors"""
        return list(self.detectors.keys())

    @Pyro5.api.expose
    def activate_device(self, device):
        """Activate the specified device for acquisition"""
        if device in self.detectors:
            print(f"{device.capitalize()} activated")
            return 1
        else:
            print(f"Device {device} not found")
            return 0
        
    @Pyro5.api.expose
    def device_settings(self, device, **args):
        """Set the device settings"""
        if device in self.detectors:
            print(f"Setting {device} settings: {args}")
            return 1
        else:
            print(f"Device {device} not found")
            return 0

    @Pyro5.api.expose
    def get_stage(self):
        """Get the current stage position"""
        positions = [0, 0, 0, 0, 0]
        return positions
    
    @Pyro5.api.expose
    def set_stage(self, stage_positions, relative=True):
        """Set the stage position in nm and degrees"""
        stage_move = auto_script.structures.StagePosition()
        for index, direction in enumerate(['x', 'y', 'z', 'a', 'b']):
            pass
        print(f"Moving stage by {stage_move}")

    @Pyro5.api.expose
    def acquire_image(self, device, **args):
        """Acquire an image from the specified device"""
        if device in self.detectors:
            print(f"Acquiring image from {device}")
            image = np.zeros([512,512], dtype=np.uint16)
            return serialize(image)
            
        else:
            print(f"Device {device} not found")
            return None

    @Pyro5.api.expose
    def acquire_image_stack(self, device):
        """Acquire an image from the specified device"""
        if device in self.detectors:
            print(f"Acquiring image from {device}")
            return self.detectors[device]
        else:
            print(f"Device {device} not found")
            return None

    @Pyro5.api.expose
    def acquire_spectrum(self, device, **args):
        """Acquire a spectrum from the specified device"""
        if device in self.detectors:
            print(f"Acquiring spectrum from {device}")
            return np.zeros((2048,), dtype=np.float32)
        else:
            print(f"Device {device} not found")
            return None

    @Pyro5.api.expose
    def acquire_spectrum_points(self, device, points, **args):
        """Acquire a spectrum stack from the specified device"""
        if device in self.detectors:
            print(f"  at point {point}")
        else:
            print(f"Device {device} not found")
            return None

    @Pyro5.api.expose
    def set_beam_position(self, x, y):
        """Set the beam position in nm"""
        print(f"Moving beam to ({x}, {y})")
        return 1
    
    @Pyro5.api.expose
    def get_microscope_status(self, parameters=None):
        """Get the current microscope status"""
        for parameter in parameters:
            if parameter in available_parameters:
                if parameter == 'vacuum':
                    return "vacuum"
                elif parameter == 'column_valve':
                    return "Open"
        return "Idle"
    
    def send_data(self, data):
        """Serialize a numpy array to make it transferable via Pyro"""
        out_data = data.copy()
        if isinstance(data, dict):
            for key, item in data.items():
                if isinstance(item, np.ndarray):
                    data[key] = serialize(item)
                elif not isinstance(item, (list, int, float, str)):
                    out_data.pop(key)
        elif isinstance(data,  np.ndarray):
            out_data = serialize(data)
        elif not isinstance(data, (list, int, float, str)):
            out_data = None
        return out_data

    @Pyro5.api.expose
    def close(self):
        """Close the server"""
        print("Closing server")
        return 1
