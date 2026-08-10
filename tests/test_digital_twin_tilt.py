import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from asyncroscopy.instruments.electron_microscope.digital_twin_tilt import (
    DigitalTwinTilt,
)


def twin_for_path(tmp_path: Path, randomness_scale: float = 0.0) -> DigitalTwinTilt:
    twin = DigitalTwinTilt.__new__(DigitalTwinTilt)
    twin._tango_properties = {}
    twin._detector_proxies = {}
    properties = {
        "sample_seed": 731,
        "lamella_width_nm": 620.0,
        "lamella_height_nm": 360.0,
        "lamella_thickness_nm": 80.0,
        "gridbar_width_nm": 220.0,
        "layer_period_nm": 45.0,
        "overview_fov_nm": 1100.0,
        "diffraction_image_size": 16,
        "diffraction_max_angle_mrad": 60.0,
        "convergence_angle_mrad": 8.0,
        "silicon_lattice_parameter_angstrom": 5.431,
        "lattice_parameter_gradient_x_percent": 2.0,
        "lattice_parameter_gradient_y_percent": 0.5,
        "crystal_rotation_deg": 1.5,
        "crystal_rotation_gradient_x_deg": 4.0,
        "crystal_rotation_gradient_y_deg": -2.0,
        "crystal_tilt_x_mrad": 3.0,
        "crystal_tilt_y_mrad": -2.0,
        "crystal_tilt_gradient_x_mrad": 10.0,
        "crystal_tilt_gradient_y_mrad": -7.0,
        "electron_energy_ev": 200_000.0,
        "potential_sampling_angstrom": 0.08,
        "potential_slice_thickness_angstrom": 1.0,
        "multislice_supercell_xy": 4,
        "multislice_supercell_z": 8,
        "multislice_vacuum_angstrom": 2.0,
        "frozen_phonon_sigma_angstrom": 0.03,
        "rotation_center_x_nm": -55.0,
        "rotation_center_y_nm": 35.0,
        "rotation_center_z_nm": 125.0,
        "randomness_scale": randomness_scale,
        "stage_translation_noise_std_nm": 0.35,
        "tilt_angle_noise_std_deg": 0.06,
        "rotation_center_jitter_std_nm": 1.5,
        "tilt_wobble_std_nm": 1.0,
        "tilt_drift_std_nm_per_degree": 0.08,
        "beam_tilt_noise_std_mrad": 0.03,
        "beam_tilt_image_shift_nm_per_mrad": 0.8,
        "diffraction_center_jitter_std_px": 0.35,
        "detector_noise_fraction": 0.012,
        "focus_drift_nm_per_degree": 1.4,
        "autofocus_residual_std_nm": 2.0,
        "haadf_poisson_counts": 180.0,
        "acquisition_save_directory": str(tmp_path),
    }
    for name, value in properties.items():
        setattr(twin, name, value)
    twin._initialize_tilt_state()
    twin._generate_sample(seed=int(twin.sample_seed))
    return twin


def test_sample_is_layered_lamella_attached_to_opaque_gridbar(tmp_path: Path):
    twin = twin_for_path(tmp_path)

    image, lamella, gridbar, _local_x, local_y = twin._sample_map(256)

    assert image.shape == (256, 256)
    assert np.count_nonzero(lamella) > 8_000
    assert np.count_nonzero(gridbar) > 3_000
    assert image[gridbar].mean() > image[lamella].mean()
    layer_means = []
    for layer in range(6):
        lower = -0.5 * twin.lamella_height_nm + layer * twin.layer_period_nm
        selection = lamella & (local_y >= lower) & (local_y < lower + twin.layer_period_nm)
        layer_means.append(float(image[selection].mean()))
    assert np.ptp(layer_means) > 0.07


def test_offset_rotation_center_moves_lamella_and_image_shift_tracks_it(tmp_path: Path):
    twin = twin_for_path(tmp_path)
    size = 256
    _image, untilted, _gridbar, _x, _y = twin._sample_map(size)
    untilted_center = np.argwhere(untilted).mean(axis=0)

    twin._apply_stage_target([0.0, 0.0, 0.0, 30.0, 0.0])
    _image, tilted, _gridbar, _x, _y = twin._sample_map(size)
    tilted_center = np.argwhere(tilted).mean(axis=0)
    row_motion = tilted_center[0] - untilted_center[0]

    assert row_motion > 10.0

    pixel_size_m = twin._fov / size
    twin._set_image_shift([0.0, -row_motion * pixel_size_m])
    _image, tracked, _gridbar, _x, _y = twin._sample_map(size)
    tracked_center = np.argwhere(tracked).mean(axis=0)

    assert tracked_center[0] == pytest.approx(untilted_center[0], abs=1.0)


def test_tilt_randomness_is_repeatable_and_has_master_disable(tmp_path: Path):
    first = twin_for_path(tmp_path / "first", randomness_scale=1.0)
    second = twin_for_path(tmp_path / "second", randomness_scale=1.0)
    ideal = twin_for_path(tmp_path / "ideal", randomness_scale=0.0)
    target = [4e-9, -3e-9, 2e-9, 20.0, 0.0]

    first._apply_stage_target(target)
    second._apply_stage_target(target)
    ideal._apply_stage_target(target)

    np.testing.assert_allclose(first._stage_position, second._stage_position)
    np.testing.assert_allclose(first._rotation_center_jitter_nm, second._rotation_center_jitter_nm)
    assert not np.allclose(first._stage_position, target, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(ideal._stage_position, target, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(ideal._tilt_wobble_nm, 0.0)


def test_local_ase_cells_encode_continuous_lattice_and_orientation_fields(tmp_path: Path):
    twin = twin_for_path(tmp_path)

    left = twin._silicon_atoms(-0.5 * twin.lamella_width_nm, 0.0, 1)
    center = twin._silicon_atoms(0.0, 0.0, 2)
    right = twin._silicon_atoms(0.5 * twin.lamella_width_nm, 0.0, 3)

    assert left.info["lattice_parameter_angstrom"] < center.info[
        "lattice_parameter_angstrom"
    ] < right.info["lattice_parameter_angstrom"]
    assert right.info["lattice_parameter_angstrom"] / left.info[
        "lattice_parameter_angstrom"
    ] == pytest.approx(1.01 / 0.99)
    assert left.info["crystal_rotation_deg"] == pytest.approx(-0.5)
    assert right.info["crystal_rotation_deg"] == pytest.approx(3.5)
    assert left.info["crystal_tilt_x_mrad"] == pytest.approx(-2.0)
    assert right.info["crystal_tilt_x_mrad"] == pytest.approx(8.0)
    assert center.info["nearest_neighbor_distance_angstrom"] == pytest.approx(
        center.info["lattice_parameter_angstrom"] * np.sqrt(3.0) / 4.0
    )
    distances = center.get_all_distances(mic=False)
    nearest_neighbor = distances[distances > 0].min()
    assert nearest_neighbor == pytest.approx(
        center.info["nearest_neighbor_distance_angstrom"]
    )


def test_real_abtem_multislice_returns_diffraction_pattern(tmp_path: Path):
    pytest.importorskip("abtem", exc_type=ImportError)
    twin = twin_for_path(tmp_path)
    twin.multislice_supercell_xy = 2
    twin.multislice_supercell_z = 2
    twin.potential_sampling_angstrom = 0.15
    twin.potential_slice_thickness_angstrom = 2.0
    atoms = twin._silicon_atoms(0.0, 0.0, 1)
    strained_atoms = twin._silicon_atoms(0.5 * twin.lamella_width_nm, 0.0, 2)

    pattern = twin._simulate_diffraction_from_atoms(atoms, 24)
    strained_pattern = twin._simulate_diffraction_from_atoms(strained_atoms, 24)

    assert pattern.shape == (24, 24)
    assert np.isfinite(pattern).all()
    assert pattern.max() == pytest.approx(1.0)
    assert pattern.std() > 0.01
    assert np.mean(np.abs(pattern - strained_pattern)) > 1e-4


def test_move_probe_and_autofocus_update_instrument_state(tmp_path: Path):
    twin = twin_for_path(tmp_path)

    twin._move_stage([5e-9, -6e-9, 7e-9, 25.0, 0.0])
    twin._place_beam([0.6, 0.4])
    twin._auto_focus()

    np.testing.assert_allclose(twin._get_stage(), [5e-9, -6e-9, 7e-9, 25.0, 0.0])
    assert twin.read_beam_pos() == [0.6, 0.4]
    expected_focus = 7e-9 + twin.focus_drift_nm_per_degree * np.sin(np.radians(25.0)) * 1e-9
    assert twin._get_defocus() == pytest.approx(expected_focus)


def test_camera_diffraction_uses_local_silicon_ase_model(
    monkeypatch,
    tmp_path: Path,
):
    twin = twin_for_path(tmp_path)
    twin._place_beam([0.5, 0.5])
    simulated_atoms = []

    def fake_multislice(atoms, imsize):
        simulated_atoms.append(atoms.copy())
        return np.ones((imsize, imsize), dtype=np.float32)

    monkeypatch.setattr(twin, "_simulate_diffraction_from_atoms", fake_multislice)

    saved_path = Path(
        twin._acquire_camera_image(
            64,
            0.2,
            "BM-Ceta",
            "Half",
            frame_combining=2,
            electron_counting=True,
        )
    )

    with h5py.File(saved_path, "r") as h5:
        image = h5["image"]
        assert image.shape == (64, 64)
        assert image.attrs["material"] == "silicon"
        assert image.attrs["convergence_angle_mrad"] == 8.0
        assert image.attrs["pixel_size_mrad"] == pytest.approx(120.0 / 64)
        assert image.attrs["frame_combining"] == 2
        assert image.attrs["simulation"] == "ASE silicon + abTEM multislice"
        local_lattice = image.attrs["lattice_parameter_angstrom"]
        assert local_lattice == pytest.approx(5.431, abs=0.01)
        assert image.attrs["nearest_neighbor_distance_angstrom"] == pytest.approx(
            local_lattice * np.sqrt(3.0) / 4.0
        )
    assert len(simulated_atoms) == 1
    assert simulated_atoms[0].get_chemical_symbols() == ["Si"] * len(simulated_atoms[0])


def test_gridbar_is_not_electron_transparent(tmp_path: Path):
    twin = twin_for_path(tmp_path)
    gridbar_x_nm = -0.5 * twin.lamella_width_nm - 0.5 * twin.gridbar_width_nm
    twin._place_beam([0.5 + gridbar_x_nm / twin.overview_fov_nm, 0.5])

    material, _local_x, _local_y = twin._beam_material()
    saved_path = Path(twin._acquire_camera_image(64, 0.1, "BM-Ceta", "Half"))

    assert material == "gridbar"
    with h5py.File(saved_path, "r") as h5:
        assert h5["image"][:].max() == 0.0


def test_scanned_overview_and_4dstem_acquisitions_are_saved(monkeypatch, tmp_path: Path):
    twin = twin_for_path(tmp_path)
    twin._set_beam_tilt([8e-3, 0.0])
    monkeypatch.setattr(
        twin,
        "_simulate_diffraction_from_atoms",
        lambda atoms, imsize: np.full(
            (imsize, imsize),
            0.5 if len(atoms) else 0.0,
            dtype=np.float32,
        ),
    )

    overview_path = Path(twin._acquire_scanned_image(48, 1e-6, ["HAADF"]))
    data_path = Path(
        twin._acquire_scanned_data_advanced(
            8,
            500e-6,
            "BM-Ceta",
            [0.0, 0.0, 1.0, 1.0],
        )
    )

    with h5py.File(overview_path, "r") as h5:
        assert h5["image/HAADF"].shape == (48, 48)
        assert h5["image/HAADF"].attrs["fov_nm"] == pytest.approx(1100.0)
    with h5py.File(data_path, "r") as h5:
        stem_data = h5["stem_data"]
        assert stem_data.shape == (8, 8, 16, 16)
        assert stem_data.attrs["convergence_angle_mrad"] == 8.0
        assert json.loads(stem_data.attrs["beam_tilt_mrad"])[0] == pytest.approx(8.0)
        lattice_parameter = h5["simulation/lattice_parameter_angstrom"][:]
        crystal_rotation = h5["simulation/crystal_rotation_deg"][:]
        assert np.nanmax(lattice_parameter) > np.nanmin(lattice_parameter)
        assert np.nanmax(crystal_rotation) > np.nanmin(crystal_rotation)


def test_tango_commands_cover_tilt_workflow(
    monkeypatch,
    tilt_twin_proxy,
    scan_proxy,
    camera_proxy,
):
    monkeypatch.setattr(
        DigitalTwinTilt,
        "_simulate_diffraction_from_atoms",
        lambda self, atoms, imsize: np.ones((imsize, imsize), dtype=np.float32),
    )
    scan_proxy.imsize = 16
    scan_proxy.dwell_time = 1e-6
    camera_proxy.imsize = 32
    tilt_twin_proxy.move_stage([2e-9, -3e-9, 4e-9, 20.0, 0.0])
    tilt_twin_proxy.place_beam([0.5, 0.5])
    tilt_twin_proxy.auto_focus()

    overview_path = Path(tilt_twin_proxy.acquire_scanned_image(["haadf"]))
    camera_path = Path(tilt_twin_proxy.acquire_camera_image())
    stem_path = Path(tilt_twin_proxy.acquire_scanned_data_advanced())

    np.testing.assert_allclose(
        tilt_twin_proxy.get_stage(),
        [2e-9, -3e-9, 4e-9, 20.0, 0.0],
    )
    assert overview_path.exists()
    assert camera_path.exists()
    with h5py.File(stem_path, "r") as h5:
        assert h5["stem_data"].shape == (16, 16, 16, 16)
