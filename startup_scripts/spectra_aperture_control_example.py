"""Jupyter-friendly example for asyncroscopy Spectra aperture control.

Run this file in a Jupyter console or an editor that supports ``# %%`` cells.
Set ``TANGO_HOST`` for the Tango database before connecting, if needed.

The example contains no Thermo Fisher reference-manual content.  It uses only
the public asyncroscopy Tango JSON commands.
"""

# %% Imports and connection
import json
import os

import tango


MICROSCOPE_DEVICE = os.getenv(
    "ASYNCROSCOPY_MICROSCOPE_DEVICE",
    "asyncroscopy/autoscriptmicroscope/default",
)

microscope = tango.DeviceProxy(MICROSCOPE_DEVICE)
microscope.ping()
print("Connected:", MICROSCOPE_DEVICE)
print("Device state:", microscope.state())


def decode(response: str) -> dict:
    """Decode and display one deterministic JSON command response."""
    payload = json.loads(response)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


# %% Safety check: always run read-only commands first
# Safety rules:
# - Confirm the microscope mode before changing any aperture.
# - Do not change the objective or selected-area aperture during live acquisition.
# - Verify the aperture name reported by the real Spectra before selecting it.
# - Run the read-only list/get commands before enabling changes.
print("STEM mode reported by microscope:", microscope.stem_mode)

all_apertures = decode(microscope.aperture_list("{}"))
mechanism_names = all_apertures.get("mechanisms", [])
print("Available mechanisms:", mechanism_names)
print("Using AutoScript hardware:", all_apertures["autoscript_available"])


# %% List apertures for the four simulation-first mechanism categories
# Real hardware discovery is authoritative. A real Spectra may report names
# such as C1, C2, C3, Objective, or SA instead of these simulation categories.
requested_mechanisms = (
    "condenser",
    "objective",
    "selected_area",
    "projector",
)

for mechanism in requested_mechanisms:
    if mechanism not in mechanism_names:
        print(f"Skipping {mechanism!r}: not reported by this microscope")
        continue
    request = json.dumps({"mechanism": mechanism})
    decode(microscope.aperture_list(request))


# %% Read the selected aperture without changing hardware
for mechanism in requested_mechanisms:
    if mechanism not in mechanism_names:
        continue
    request = json.dumps({"mechanism": mechanism})
    decode(microscope.aperture_get_selected(request))


# %% Explicit mutation example: disabled by default
# Change these only after reviewing aperture_list() output from the real scope.
TARGET_MECHANISM = "condenser"
TARGET_APERTURE_NAME = "50 um"

# Both flags must be intentionally changed after confirming the microscope is
# idle and in the intended TEM/STEM mode. Never enable this during acquisition.
CONFIRMED_MICROSCOPE_MODE_AND_IDLE = False
ENABLE_APERTURE_CHANGES = False

if CONFIRMED_MICROSCOPE_MODE_AND_IDLE and ENABLE_APERTURE_CHANGES:
    # Re-read the live inventory immediately before changing anything.
    inventory_request = json.dumps({"mechanism": TARGET_MECHANISM})
    inventory = decode(microscope.aperture_list(inventory_request))
    available_names = [item["name"] for item in inventory.get("apertures", [])]
    if TARGET_APERTURE_NAME not in available_names:
        raise ValueError(
            f"Aperture {TARGET_APERTURE_NAME!r} was not reported for "
            f"{TARGET_MECHANISM!r}. Available names: {available_names}"
        )

    # Select by the exact name returned by aperture_list().
    select_request = json.dumps(
        {
            "mechanism": TARGET_MECHANISM,
            "name": TARGET_APERTURE_NAME,
        }
    )
    decode(microscope.aperture_select(select_request))

    # Insertion and retraction are separate, explicit user actions.
    mechanism_request = json.dumps({"mechanism": TARGET_MECHANISM})
    decode(microscope.aperture_insert(mechanism_request))
    decode(microscope.aperture_retract(mechanism_request))

    # Read status again after the requested changes.
    decode(microscope.aperture_get_selected(mechanism_request))
else:
    print(
        "Aperture-changing commands were not run. Confirm microscope mode, "
        "stop live acquisition, verify the reported aperture name, and then "
        "explicitly enable both safety flags."
    )


# %% Final read-only status snapshot
decode(microscope.aperture_list("{}"))
