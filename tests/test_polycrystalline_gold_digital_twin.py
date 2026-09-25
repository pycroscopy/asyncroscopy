from types import SimpleNamespace

import numpy as np

from asyncroscopy.instruments.electron_microscope.polycrystalline_gold_digital_twin import PolycrystallineGoldDigitalTwin


def test_atomistic_grain_recipe_is_deterministic():
    sample_a = SimpleNamespace(volume_size_xy_nm=20.0, volume_thickness_nm=5.0, grain_size_angstrom=15.0, empty_grain_fraction=0.10, lattice_constant_angstrom=4.08, zone_axis_max_index=2, zone_axis_max_deviation_deg=2.0, acceleration_voltage_ev=200_000.0, convergence_angle_mrad=30.0)
    sample_b = SimpleNamespace(volume_size_xy_nm=20.0, volume_thickness_nm=5.0, grain_size_angstrom=15.0, empty_grain_fraction=0.10, lattice_constant_angstrom=4.08, zone_axis_max_index=2, zone_axis_max_deviation_deg=2.0, acceleration_voltage_ev=200_000.0, convergence_angle_mrad=30.0)
    PolycrystallineGoldDigitalTwin._generate_sample(sample_a, 41)
    PolycrystallineGoldDigitalTwin._generate_sample(sample_b, 41)

    assert np.array_equal(sample_a._grain_centers, sample_b._grain_centers)
    assert sample_a._grain_angles == sample_b._grain_angles
    assert all(np.array_equal(rotation_a, rotation_b) for rotation_a, rotation_b in zip(sample_a._grain_rotations, sample_b._grain_rotations))
    assert sample_a._empty_grain_indices == sample_b._empty_grain_indices
    assert sample_a._all_sample_elements == ["Au"]


def test_all_grains_are_within_two_degrees_of_a_primitive_zone_axis():
    sample = SimpleNamespace(volume_size_xy_nm=20.0, volume_thickness_nm=5.0, grain_size_angstrom=15.0, empty_grain_fraction=0.10, lattice_constant_angstrom=4.08, zone_axis_max_index=2, zone_axis_max_deviation_deg=2.0, acceleration_voltage_ev=200_000.0, convergence_angle_mrad=30.0)
    PolycrystallineGoldDigitalTwin._generate_sample(sample, 41)

    assert all(max(axis) <= 2 and np.gcd.reduce(axis) == 1 for axis in sample._grain_zone_axes)
    assert all(0.0 <= offset <= 2.0 for offset in sample._grain_zone_offsets)
    aligned_directions = [rotation @ (np.asarray(axis) / np.linalg.norm(axis)) for axis, rotation in zip(sample._grain_zone_axes, sample._grain_rotations)]
    actual_offsets = [np.degrees(np.arccos(np.clip(direction[2], -1.0, 1.0))) for direction in aligned_directions]
    assert all(offset <= 2.0 for offset in actual_offsets)
    assert all(parameter.zone_axis == sample._grain_zone_axes[index] for index, parameter in enumerate(sample._region_parameters))
    assert all(parameter.zone_offset_deg == sample._grain_zone_offsets[index] for index, parameter in enumerate(sample._region_parameters))


def test_atomistic_twin_has_no_voxel_size_property():
    assert not hasattr(PolycrystallineGoldDigitalTwin, "voxel_size_nm")


def test_raw_sample_potential_is_reused_for_an_unchanged_view():
    sample = SimpleNamespace(volume_size_xy_nm=4.0, volume_thickness_nm=1.0, grain_size_angstrom=10.0, empty_grain_fraction=0.15, lattice_constant_angstrom=4.08, zone_axis_max_index=2, zone_axis_max_deviation_deg=2.0, acceleration_voltage_ev=200_000.0, convergence_angle_mrad=30.0, _fov=2e-9, _stage_position=np.zeros(5))
    sample._sync_stage_from_proxy = lambda: None
    PolycrystallineGoldDigitalTwin._generate_sample(sample, 41)

    first = PolycrystallineGoldDigitalTwin._raw_sample_potential(sample, 32)
    second = PolycrystallineGoldDigitalTwin._raw_sample_potential(sample, 32)

    assert second is first
    assert sample._raw_potential_cache_hits == 1
