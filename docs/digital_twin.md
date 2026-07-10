# DigitalTwin

`DigitalTwin` is the simulated version of the `AutoScriptMicroscope`.  
It provides realistic-enough image and spectrum behavior for development, testing, and demos without requiring AutoScript or hardware.

## How it works

1. On startup, the twin generates a **persistent synthetic sample** (deterministic from seed).
2. Stage pose (`x, y, z, alpha, beta`) defines the current viewport into that sample.
3. `acquire_scanned_image(["haadf"])` calls the inherited microscope command, which delegates to `_acquire_scanned_image()`.
4. `_acquire_scanned_image()` renders the current pose/FoV, writes HDF5 data with metadata attributes, and returns the DATA/Tiled key when a DATA device is configured.
5. `acquire_spectrum("eds")` delegates to `_acquire_spectrum()`, which estimates composition at the current beam position and saves a HDF5 spectrum.

This means moving the stage navigates the sample, and revisiting the same pose can reproduce the same view when stage noise is disabled.

## Available features

- Persistent sample per device session
- Deterministic sample generation via seed
- Stage-coupled navigation in **XY + Z + alpha/beta tilt**
- Beam-position-dependent spectrum simulation
- File-backed acquisition output instead of in-memory image caches
- Configurable stage move noise
- Viewport metadata reporting
- Manual sample regeneration with a new seed

Simulated image and spectrum acquisitions use the same HDF5 writer as the hardware-backed microscope path.

## Diffraction twin

`DigitalTwinDiffraction` is a sibling twin for parked-beam nanoparticle diffraction.
It keeps the HAADF overview workflow, fixes the field of view at 500 nm, and uses `acquire_camera_image()` to save a simulated diffraction pattern at the current beam position.
If the beam is not on a nanoparticle, it waits 3 seconds and saves detector noise.

The diffraction simulation uses `abTEM` with a small Au FCC unit cell and `ase` for the structure/rattle step.
Install that optional environment with:

```bash
uv sync --extra diffraction
```

## Key properties

- `sample_seed`: controls deterministic sample generation
- `sample_particle_count`: controls synthetic particle count
- `sample_extent_scale`: controls sample XY size relative to FoV
- `stage_move_noise_std`: adds Gaussian perturbation to stage moves

## Key commands

- `move_stage([x, y, z, alpha, beta])`
- `get_stage()`
- `set_fov(fov)`
- `set_defocus(defocus)`
- `get_defocus()`
- `acquire_scanned_image(["haadf"])`
- `place_beam([x, y])`
- `acquire_spectrum("eds")`
- `get_viewport_metadata()`
- `regenerate_sample(seed)`
