import json

import pytest
import tango

from asyncroscopy.instruments.electron_microscope.hardware.corrector_ceos import CEOSCorrector


def test_digital_twin_corrector_reports_ceos_compatible_info(corrector_proxy):
    payload = json.loads(corrector_proxy.get_info())

    assert payload["jsonrpc"] == "2.0"
    assert payload["result"]["manufacturer"] == "CEOS"
    assert payload["result"]["simulation"] is True
    assert corrector_proxy.state() == tango.DevState.ON


def test_digital_twin_tableau_and_correction_round_trip(corrector_proxy):
    initial = json.loads(corrector_proxy.acquire_tableau("Fast 1"))
    initial_c1 = initial["result"]["aberrations"]["C1"][0]

    response = json.loads(corrector_proxy.correct_aberration("C1 8e-9"))
    measured = json.loads(corrector_proxy.acquire_tableau("Fast 1"))

    assert response["result"]["corrected"] is True
    assert measured["result"]["aberrations"]["C1"][0] == pytest.approx(
        initial_c1 - 8e-9
    )


def test_simulation_coefficients_are_absolute_and_validate_shape(corrector_proxy):
    coefficients = json.loads(corrector_proxy.get_aberrations_coeff_sim())
    coefficients["C1"] = [12e-9]
    corrector_proxy.set_aberrations_coeff_sim(json.dumps(coefficients))

    updated = json.loads(corrector_proxy.get_aberrations_coeff_sim())
    assert updated["C1"] == pytest.approx([12e-9])

    coefficients["C1"] = [1.0, 2.0]
    with pytest.raises(tango.DevFailed):
        corrector_proxy.set_aberrations_coeff_sim(json.dumps(coefficients))


def test_ceos_netstring_codec_round_trip():
    payload = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "acquireTableau",
        "params": {"tabType": "Fast", "angle": 1.0},
    }

    encoded = CEOSCorrector._encode_netstring(payload)
    decoded = json.loads(CEOSCorrector._decode_netstring(encoded))

    assert decoded == payload


class FragmentedSocket:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def recv(self, _bufsize):
        return self.chunks.pop(0) if self.chunks else b""


def test_ceos_netstring_receiver_handles_fragmented_payload():
    raw = CEOSCorrector._encode_netstring({"result": {"ok": True}})
    socket = FragmentedSocket([raw[:3], raw[3:9], raw[9:]])

    assert CEOSCorrector._recv_netstring(socket) == raw
