
## Adding a new detector

1. Copy `asyncroscopy/detectors/HAADF.py` to `asyncroscopy/detectors/NEWDET.py` and adjust the attributes for that detector's settings.
2. Add a `device_property` in ElectronMicroscope.py:
   ```python
   newdet_device_address = device_property(dtype=str, default_value="asyncroscopy/newdet/default")
   ```
3. Register it in `_connect_detector_proxies()` - see step 4 in  [modify_auto_script_microscope.md](../Microscopy/modify_auto_script_microscope.md)
   ```python
   "newdet": self.newdet_device_address,
   ```
- note : base class `ElectronMicroscope` at `asyncroscopy/instruments/electron_microscope/electron_microscope.py` is not the right place for this:

4. Add acquisition logic:
- see step 3 in [modify_base_electron_microscope](../Microscopy/modify_base_electron_microscope.md) 
- see step 5 in [modify_auto_script_microscope](../Microscopy/modify_auto_script_microscope.md)

5. Add `tests/detectors/test_NEWDET.py` following `test_HAADF.py` as a template.
