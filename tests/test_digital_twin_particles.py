import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import tango


def test_particle_twin_starts_with_persistent_voxel_world(particles_twin_proxy):
    assert particles_twin_proxy.state() == tango.DevState.ON
    assert particles_twin_proxy.manufacturer == "UTKTeam Particle Digital Twin"


def test_particle_twin_saves_haadf_with_world_metadata(
    particles_twin_proxy,
    scan_proxy,
    stage_proxy,
):
    scan_proxy.imsize = 48
    scan_proxy.dwell_time = 1e-6
    stage_proxy.position = [0.0, 0.0, 0.0, 0.0, 0.0]

    path = Path(particles_twin_proxy.acquire_scanned_image(["HAADF"]))

    with h5py.File(path, "r") as h5:
        image = h5["image/HAADF"]
        assert image.shape == (48, 48)
        assert image.attrs["sample_type"] == "oriented_particles"
        assert image.attrs["particle_count"] > 0
        assert json.loads(image.attrs["world_shape"]) == [48, 48, 24]


def test_particle_twin_stage_tilt_recooks_projection(
    particles_twin_proxy,
    scan_proxy,
    stage_proxy,
):
    scan_proxy.imsize = 48
    stage_proxy.position = [0.0, 0.0, 0.0, 0.0, 0.0]
    untilted_path = Path(particles_twin_proxy.acquire_scanned_image(["HAADF"]))
    with h5py.File(untilted_path, "r") as h5:
        untilted = h5["image/HAADF"][()]

    stage_proxy.position = [0.0, 0.0, 0.0, 12.0, 0.0]
    tilted_path = Path(particles_twin_proxy.acquire_scanned_image(["HAADF"]))
    with h5py.File(tilted_path, "r") as h5:
        tilted = h5["image/HAADF"][()]

    assert np.mean(np.abs(tilted - untilted)) > 1e-4


def test_particle_twin_fov_round_trip(particles_twin_proxy):
    particles_twin_proxy.set_fov(18e-9)
    assert particles_twin_proxy.get_fov() == pytest.approx(18e-9)
