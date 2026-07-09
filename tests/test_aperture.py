import json

import pytest
import tango

from asyncroscopy.instruments.electron_microscope.hardware.aperture import ApertureInfo


EXPECTED_MECHANISMS = ["condenser", "objective", "selected_area", "projector"]


def test_aperture_info_model_is_typed() -> None:
    aperture = ApertureInfo(
        mechanism="condenser",
        name="50 um",
        aperture_type="Circular",
        diameter_m=50e-6,
        inserted=True,
        selected=True,
    )

    assert aperture.diameter_m == pytest.approx(50e-6)
    assert aperture.selected is True


def test_list_apertures(aperture_proxy: tango.DeviceProxy) -> None:
    response = json.loads(aperture_proxy.list_apertures())

    assert response["status"] == "ok"
    assert response["mechanism"] == "all"
    assert response["selected_aperture"] is None
    assert response["inserted"] is None
    assert response["error"] is None
    assert list(aperture_proxy.mechanism_names) == EXPECTED_MECHANISMS
    assert {item["mechanism"] for item in response["apertures"]} == set(EXPECTED_MECHANISMS)
    assert all("name" in item and "diameter_m" in item for item in response["apertures"])


def test_get_selected_aperture(aperture_proxy: tango.DeviceProxy) -> None:
    response = json.loads(aperture_proxy.get_selected_aperture("condenser"))

    assert response["status"] == "ok"
    assert response["mechanism"] == "condenser"
    assert response["inserted"] is True
    assert response["selected_aperture"]["name"] == "50 um"
    assert response["selected_aperture"]["selected"] is True
    assert aperture_proxy.selected_aperture_name == "50 um"
    assert aperture_proxy.selected_aperture_type == "Circular"
    assert aperture_proxy.selected_aperture_diameter_m == pytest.approx(50e-6)
    assert aperture_proxy.insertion_state == "Inserted"


def test_select_aperture_by_valid_name(aperture_proxy: tango.DeviceProxy) -> None:
    request = json.dumps({"mechanism": "condenser", "name": "70 um"})
    response = json.loads(aperture_proxy.select_aperture(request))

    assert response["status"] == "ok"
    assert response["mechanism"] == "condenser"
    assert response["selected_aperture"]["name"] == "70 um"
    assert response["selected_aperture"]["diameter_m"] == pytest.approx(70e-6)
    assert response["inserted"] is True
    assert response["error"] is None


def test_reject_invalid_mechanism(aperture_proxy: tango.DeviceProxy) -> None:
    response = json.loads(aperture_proxy.get_selected_aperture("not_a_mechanism"))

    assert response["status"] == "error"
    assert response["mechanism"] == "not_a_mechanism"
    assert response["selected_aperture"] is None
    assert response["inserted"] is None
    assert "Unknown aperture mechanism" in response["error"]
    assert aperture_proxy.last_error == response["error"]


def test_reject_invalid_aperture_name(aperture_proxy: tango.DeviceProxy) -> None:
    request = json.dumps({"mechanism": "condenser", "name": "999 um"})
    response = json.loads(aperture_proxy.select_aperture(request))

    assert response["status"] == "error"
    assert response["mechanism"] == "condenser"
    assert "Unknown aperture" in response["error"]
    assert "999 um" in response["error"]


def test_insert_and_retract_mechanism(aperture_proxy: tango.DeviceProxy) -> None:
    initial = json.loads(aperture_proxy.get_selected_aperture("objective"))
    assert initial["inserted"] is False
    assert initial["selected_aperture"] is None

    inserted = json.loads(aperture_proxy.insert_mechanism("objective"))
    assert inserted["status"] == "ok"
    assert inserted["inserted"] is True
    assert inserted["selected_aperture"]["name"] == "40 um"

    retracted = json.loads(aperture_proxy.retract_mechanism("objective"))
    assert retracted["status"] == "ok"
    assert retracted["inserted"] is False
    assert retracted["selected_aperture"] is None
    assert aperture_proxy.insertion_state == "Retracted"


def test_json_responses_are_deterministic(aperture_proxy: tango.DeviceProxy) -> None:
    first_list = aperture_proxy.list_apertures()
    second_list = aperture_proxy.list_apertures()
    first_refresh = aperture_proxy.refresh()
    second_refresh = aperture_proxy.refresh()

    assert first_list == second_list
    assert first_refresh == second_refresh
    assert first_list == json.dumps(
        json.loads(first_list),
        sort_keys=True,
        separators=(",", ":"),
    )
