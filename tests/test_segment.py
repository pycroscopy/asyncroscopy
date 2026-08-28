from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from asyncroscopy.mcp.segment import (
    SEGMENTATION,
    _label_areas,
    _prepare_image,
    _read_first_array,
)


class ArrayNode:
    def __init__(self, array: np.ndarray):
        self.array = array

    def read(self) -> np.ndarray:
        return self.array


class GroupNode:
    def __init__(self, children: dict[str, object]):
        self.children = children

    def keys(self):
        return self.children.keys()

    def __getitem__(self, key: str):
        return self.children[key]


def test_read_first_array_walks_tiled_groups_deterministically() -> None:
    first = np.arange(4).reshape(2, 2)
    node = GroupNode(
        {
            "z-last": ArrayNode(np.ones((2, 2))),
            "a-first": GroupNode({"image": ArrayNode(first)}),
        }
    )

    np.testing.assert_array_equal(_read_first_array(node), first)


def test_prepare_image_normalizes_without_cv2_or_pillow() -> None:
    image = np.array([[0.0, 1.0], [2.0, np.nan]], dtype=np.float32)

    prepared = _prepare_image(image)

    assert prepared.shape == (2, 2, 3)
    assert prepared.dtype == np.uint8
    np.testing.assert_array_equal(prepared[..., 0], prepared[..., 1])
    np.testing.assert_array_equal(prepared[..., 1], prepared[..., 2])


def test_prepare_image_rejects_non_image_arrays() -> None:
    with pytest.raises(ValueError, match="2-D grayscale or RGB"):
        _prepare_image(np.zeros((2, 3, 4, 5)))


def test_label_areas_returns_one_based_uint32_labels() -> None:
    large = np.array([[True, True], [True, False]])
    small = np.array([[False, True], [False, False]])

    labels = _label_areas(
        [
            {"segmentation": large, "area": 3, "predicted_iou": 0.8},
            {"segmentation": small, "area": 1, "predicted_iou": 0.9},
        ],
        (2, 2),
    )

    assert labels.dtype == np.uint32
    np.testing.assert_array_equal(labels, np.array([[1, 2], [1, 0]], dtype=np.uint32))


def test_area_statistics_match_one_based_label_ids() -> None:
    device = SimpleNamespace()
    mask = np.array([[True, True], [False, False]])

    stats, centroids = SEGMENTATION.area_statistics(
        device,
        [{"segmentation": mask, "area": 2, "predicted_iou": 0.75}],
    )

    assert stats[0]["id"] == 1
    assert stats[0]["area_px"] == 2
    assert stats[0]["confidence"] == 0.75
    assert centroids == [[0.5, 0.0]]
    assert device._n_areas == 1


def test_segment_writes_labels_to_tiled_and_returns_key(monkeypatch) -> None:
    source = np.arange(16, dtype=np.float32).reshape(4, 4)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    areas = [{"segmentation": mask, "area": 4, "predicted_iou": 0.9}]

    class FakeDataProxy:
        def get_config(self) -> str:
            return '{"uri": "http://tiled.example"}'

    class FakeTiledClient:
        writes: list[tuple[np.ndarray, str, dict]] = []

        def write_array(self, labels, *, key, metadata) -> None:
            self.writes.append((labels, key, metadata))

    data_proxy = FakeDataProxy()
    tiled = FakeTiledClient()
    monkeypatch.setattr(
        "asyncroscopy.mcp.segment.from_uri",
        lambda uri, api_key=None: tiled,
    )
    device = SimpleNamespace(
        _get_data_proxy=lambda: data_proxy,
        _load_image_from_key=lambda key, proxy: source,
        segment_image=lambda image: areas,
        area_statistics=lambda masks: ([{"id": 1}], [[1.5, 1.5]]),
        model_size="facebook/sam2-hiera-large",
        error_stream=lambda message: None,
    )

    output_key = SEGMENTATION.segment(device, "source-image.h5")

    assert output_key.startswith("segmentation_")
    assert len(tiled.writes) == 1
    labels, key, metadata = tiled.writes[0]
    assert key == output_key
    assert metadata["source_data_key"] == "source-image.h5"
    assert metadata["model"] == "facebook/sam2-hiera-large"
    np.testing.assert_array_equal(
        labels,
        np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint32,
        ),
    )
