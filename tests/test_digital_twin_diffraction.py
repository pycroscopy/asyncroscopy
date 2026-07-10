from pathlib import Path

import h5py
import numpy as np
import pytest

from ase import Atoms

from asyncroscopy.instruments.electron_microscope.digital_twin_diffraction import (
    EDGE_LATTICE_CHANGE_FRACTION,
    OVAL_CENTER_LATTICE_MEAN_ANGSTROM,
    ROUND_CENTER_LATTICE_MEAN_ANGSTROM,
    DigitalTwinDiffraction,
    FOV_M,
)


def twin_for_path(tmp_path: Path) -> DigitalTwinDiffraction:
    twin = DigitalTwinDiffraction.__new__(DigitalTwinDiffraction)
    twin._tango_properties = {}
    twin._detector_proxies = {}
    twin._sample_atoms_base = Atoms()
    twin._sample_atoms_view = Atoms()
    twin._stage_position = np.zeros(5, dtype=np.float64)
    twin._beam_pos_x = 0.5
    twin._beam_pos_y = 0.5
    twin._fov = FOV_M
    twin.sample_seed = 12345
    twin.map_size = 256
    twin.overview_image_size = 64
    twin.diffraction_image_size = 64
    twin.convergence_angle_mrad = 5.0
    twin.haadf_poisson_counts = 120.0
    twin.acquisition_save_directory = str(tmp_path)
    return twin


def test_diffraction_twin_generates_dense_nanoparticle_field(tmp_path: Path):
    twin = twin_for_path(tmp_path)

    twin._generate_sample(seed=12345)

    coverage = np.count_nonzero(twin._particle_id_map) / twin._particle_id_map.size
    kinds = {record['kind'] for record in twin._particle_records_base}
    round_ratios = []
    for record in twin._particle_records_base:
        if record['kind'] != 'round':
            continue
        axis_x, axis_y = record['axes_nm']
        round_ratios.append(max(axis_x, axis_y) / min(axis_x, axis_y))
    assert 0.20 <= coverage <= 0.26
    assert len(twin._particle_records_base) >= 8
    assert kinds == {'round', 'oval'}
    assert max(round_ratios) <= 1.08
    assert all('rotation_degrees' in record for record in twin._particle_records_base)
    assert sum(record['pixel_count'] for record in twin._particle_records_base) == np.count_nonzero(twin._particle_id_map)
    assert twin._get_fov() == FOV_M


def test_lattice_parameter_varies_smoothly_from_particle_center_to_edge(tmp_path: Path):
    twin = twin_for_path(tmp_path)
    twin._generate_sample(seed=12345)

    records_by_kind = {record['kind']: record for record in twin._particle_records_base}
    for kind, expected_mean in (
        ('round', ROUND_CENTER_LATTICE_MEAN_ANGSTROM),
        ('oval', OVAL_CENTER_LATTICE_MEAN_ANGSTROM),
    ):
        record = records_by_kind[kind]
        center_lattice = record['center_lattice_parameter']
        local_values = twin._lattice_parameter_map[twin._particle_id_map == record['id']]

        assert abs(center_lattice - expected_mean) < 5 * 0.015
        if kind == 'round':
            assert center_lattice * (1.0 - EDGE_LATTICE_CHANGE_FRACTION) <= local_values.min()
            assert local_values.min() < center_lattice * 0.78
            assert center_lattice * 0.99 < local_values.max() <= center_lattice
        else:
            assert center_lattice <= local_values.min() < center_lattice * 1.01
            assert center_lattice * 1.22 < local_values.max()
            assert local_values.max() <= center_lattice * (1.0 + EDGE_LATTICE_CHANGE_FRACTION)


def test_lattice_parameter_profile_has_smooth_monotonic_endpoints():
    radius = np.linspace(0.0, 1.0, 101)
    round_profile = DigitalTwinDiffraction._local_lattice_parameter(4.078, -0.25, radius)
    oval_profile = DigitalTwinDiffraction._local_lattice_parameter(4.678, 0.25, radius)

    assert round_profile[0] == 4.078
    assert round_profile[-1] == 4.078 * 0.75
    assert np.all(np.diff(round_profile) <= 0.0)
    assert oval_profile[0] == 4.678
    assert oval_profile[-1] == 4.678 * 1.25
    assert np.all(np.diff(oval_profile) >= 0.0)


def test_beam_particle_uses_local_lattice_parameter(tmp_path: Path):
    twin = twin_for_path(tmp_path)
    twin._generate_sample(seed=12345)
    record = next(record for record in twin._particle_records_base if record['kind'] == 'round')
    particle_mask = twin._particle_id_map == record['id']
    edge_y, edge_x = np.unravel_index(
        np.argmin(np.where(particle_mask, twin._lattice_parameter_map, np.inf)),
        particle_mask.shape,
    )
    twin._beam_pos_x = edge_x / (particle_mask.shape[1] - 1)
    twin._beam_pos_y = edge_y / (particle_mask.shape[0] - 1)

    particle, rattle_value = twin._beam_particle()

    assert particle['center_lattice_parameter'] == record['center_lattice_parameter']
    assert particle['lattice_parameter'] == twin._lattice_parameter_map[edge_y, edge_x]
    assert particle['lattice_strain_fraction'] < -0.22
    assert rattle_value > 0.0


def test_particle_camera_acquisition_saves_local_lattice_metadata(monkeypatch, tmp_path: Path):
    twin = twin_for_path(tmp_path)
    particle = {
        'id': 7,
        'kind': 'oval',
        'lattice_parameter': 5.80,
        'center_lattice_parameter': 4.678,
        'lattice_strain_fraction': 5.80 / 4.678 - 1.0,
        'rotation_degrees': 23.0,
    }
    monkeypatch.setattr(twin, '_beam_particle', lambda: (particle, 0.75))
    monkeypatch.setattr(
        twin,
        '_simulate_abtem_diffraction',
        lambda particle, rattle_value, imsize: np.ones((imsize, imsize), dtype=np.float32),
    )

    saved_path = Path(twin._acquire_camera_image(32, 0.1, 'BM-Ceta', 'Full'))

    with h5py.File(saved_path, 'r') as h5:
        attrs = h5['image'].attrs
        assert attrs['particle_lattice_parameter_angstrom'] == particle['lattice_parameter']
        assert attrs['particle_center_lattice_parameter_angstrom'] == particle['center_lattice_parameter']
        assert attrs['particle_lattice_strain_fraction'] == particle['lattice_strain_fraction']
        assert attrs['particle_lattice_strain_percent'] == 100.0 * particle['lattice_strain_fraction']


def test_off_particle_camera_acquisition_saves_vacuum_diffraction(tmp_path: Path):
    pytest.importorskip('abtem', exc_type=ImportError)

    twin = twin_for_path(tmp_path)
    twin._particle_id_map = np.zeros((64, 64), dtype=np.int16)
    twin._rattle_map = np.zeros((64, 64), dtype=np.float32)
    twin._particle_records_base = []

    saved_path = Path(twin._acquire_camera_image(32, 0.1, 'BM-Ceta', 'Full'))

    assert saved_path.suffix == '.h5'
    with h5py.File(saved_path, 'r') as h5:
        image = h5['image'][()]
        assert image.shape == (32, 32)
        assert h5['image'].attrs['acquisition_type'] == 'diffraction'
        assert h5['image'].attrs['detector'] == 'BM-Ceta'
        assert h5['image'].attrs['pixel_size_mrad'] == 5.0
        assert h5['image'].attrs['max_angle_mrad'] == 80.0
        assert h5['image'].attrs['rattle_value'] == 0.0


def test_particle_atoms_include_rotation_and_rattle(monkeypatch, tmp_path: Path):
    twin = twin_for_path(tmp_path)
    twin._stage_position = np.zeros(5, dtype=np.float64)

    from asyncroscopy.instruments.electron_microscope import digital_twin_diffraction as diff

    class FakeAtoms:
        def __init__(self):
            self.calls = []

        def repeat(self, value):
            self.calls.append(('repeat', value))
            return self

        def center(self, vacuum=None):
            self.calls.append(('center', vacuum))

        def rotate(self, angle, axis, center=None):
            self.calls.append(('rotate', angle, axis, center))

        def rattle(self, stdev, seed=None):
            self.calls.append(('rattle', stdev, seed))

    fake_atoms = FakeAtoms()
    bulk_calls = []

    def fake_bulk(*args, **kwargs):
        bulk_calls.append((args, kwargs))
        return fake_atoms

    monkeypatch.setattr(diff, 'bulk', fake_bulk)

    particle = {'lattice_parameter': 4.08, 'rotation_degrees': 37.0}
    atoms = twin._particle_atoms(particle, rattle_value=0.5)

    assert atoms is fake_atoms
    assert bulk_calls[0][1]['cubic'] is True
    assert ('rotate', 37.0, 'z', 'COP') in fake_atoms.calls
    assert any(call[0] == 'rattle' and call[1] == twin._rattle_stdev(0.5) for call in fake_atoms.calls)


def test_scanned_image_stage_sync_uses_map_cache(tmp_path: Path):
    class FakeStage:
        x = 5e-9
        y = -5e-9
        z = 0.0
        alpha = 0.0
        beta = 0.0

    twin = twin_for_path(tmp_path)
    twin._generate_sample(seed=12345)
    twin._detector_proxies = {'stage': FakeStage()}

    image = twin._render_stem_image(32, 1e-6, ['HAADF'])

    assert image.shape == (32, 32)
    assert twin._stage_position.tolist() == [5e-9, -5e-9, 0.0, 0.0, 0.0]


def test_scanned_image_saves_pixel_size_metadata(tmp_path: Path):
    twin = twin_for_path(tmp_path)
    twin._generate_sample(seed=12345)

    saved_path = Path(twin._acquire_scanned_image(32, 1e-6, ['HAADF']))

    with h5py.File(saved_path, 'r') as h5:
        image = h5['image/HAADF']
        assert image.shape == (64, 64)
        assert image.attrs['pixel_size_nm'] == 500.0 / 64
        assert image.attrs['pixel_size_m'] == FOV_M / 64
        assert image.attrs['sample_pixel_size_nm'] == 500.0 / 256
