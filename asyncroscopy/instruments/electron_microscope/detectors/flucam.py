"""
Flucam Tango settings device.

This device intentionally mirrors the generic CAMERA settings device. The
STEMMicroscope device reads these attributes via DeviceProxy before acquiring from
AutoScript's "Flucam" camera detector.
"""

from asyncroscopy.instruments.electron_microscope.detectors.camera import CAMERA


class FLUCAM(CAMERA):
    """FLUCAM detector settings device."""


if __name__ == "__main__":
    FLUCAM.run_server()
