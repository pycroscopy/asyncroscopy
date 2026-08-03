# Modifying the base `ElectronMicroscope`

`ElectronMicroscope` (`asyncroscopy/instruments/electron_microscope/electron_microscope.py`) is the **vendor-agnostic** Tango
device. It owns the public `@command` API and the abstract `_helper` methods
each vendor subclass (e.g. `AutoScriptMicroscope`) must fill in.

**The pattern:** a public `@command` validates input and reads settings from the
detector `DeviceProxy` objects, then delegates to a vendor `_helper`. Acquisition
commands return a **Tiled key (a string)**; the client reads the data back from
the Tiled server with that key.

If you're editing this class, you're usually doing one of these:

1. **Adding or modifying an attribute**
   (Expose new device state clients can read, e.g. `stem_mode`.)

2. **Updating an attribute read/write method**
   (Control how a value is validated, stored, or synced with the vendor API.)

3. **Adding or modifying a public command**
   Add a thin `@command` that validates input, reads any settings from a
   detector proxy, then calls a vendor `_helper`. Existing groups:
   - acquisition — `acquire_scanned_image`, `acquire_spectrum`,
     `acquire_camera_image`, `acquire_flucam_image`,
     `acquire_scanned_data_advanced`
   - beam / optics — `place_beam`, `place_beam_list`, `blank_beam`,
     `unblank_beam`, `set_defocus` / `get_defocus`, `set_image_shift`,
     `set_column_valves`
   - stage — `get_stage`, `move_stage`
   - imaging conditions — `set_fov` / `get_fov`, `set_screen_current` /
     `get_screen_current`, `auto_focus`

4. **Adding or changing a vendor `_helper` contract**
   The `_helper` methods are the vendor extension points. Add the public
   `@command` here, and the `_helper` it delegates to.
   - **Required** (declared `@abstractmethod` — every subclass must implement):
     `_connect`, `_connect_hardware`, `_connect_detector_proxies`,
     `_acquire_scanned_image`, `_get_stage`, `_move_stage`,
     `_set_fov` / `_get_fov`, `_set_screen_current` / `_get_screen_current`,
     `_auto_focus`, `_set_image_shift`.
   - **Optional** (default no-op or "unsupported" — override only if the vendor
     supports it): `_acquire_camera_image`, `_acquire_scanned_data_advanced`,
     `_place_beam`, `_blank_beam`, `_unblank_beam`, `_set_defocus` /
     `_get_defocus`.
   - **Note:** `acquire_spectrum` delegates to `_acquire_spectrum`, which is
     *not* declared on the base — it's defined only in the vendor subclass.
     If you add a new vendor, you must provide `_acquire_spectrum` yourself.

   Stage positions use `[x, y, z, alpha, beta]`; x/y/z are meters and alpha/beta
   are degrees in the public Tango API. Vendor helpers should convert only at
   the hardware boundary if the vendor API expects another unit.

5. **Changing the return / transport convention**
   Acquisition commands return a Tiled key string; the actual save happens in
   the vendor helper via `save_acquisition` (`asyncroscopy/data/data_writer.py`)
   and registration via the DATA device (`asyncroscopy/data/data.py`). See
   [data_integration.md](../Tiled_server/data_integration.md). The legacy
   `get_image_data_cached` (returns `DevEncoded`) is the only remaining
   byte-over-Tango path.

6. **Improving robustness**
   (Connection failures, missing proxies, vendor-API errors, simulation
   fallback, or state transitions like `FAULT` / `ON` / `OFF`.)
