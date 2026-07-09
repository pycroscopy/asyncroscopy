# Spectra aperture automation

## What aperture automation means

In a Spectra TEM/STEM, an aperture is a physical element in the electron column. Selecting, inserting, retracting, or centering one changes which electrons continue through the microscope. Depending on the aperture and operating mode, this can change beam current, convergence angle, diffraction contrast, the selected specimen area, detector count rate, and image or spectrum quality.

In asyncroscopy, aperture automation means exposing those operations as explicit, inspectable Tango commands. A client can first ask what mechanisms and aperture names are available, read the current state, and then—after the microscopist confirms the microscope condition—request a specific change.

Automation does not mean that asyncroscopy should decide which physical aperture is scientifically appropriate. That remains an experimental decision made by the microscopist.

## Physical aperture roles

### Condenser apertures

Condenser apertures are part of the illumination system. They affect the beam current and illumination or probe-forming conditions. In STEM, changing a condenser aperture can alter probe current and convergence conditions, which in turn affects spatial resolution, depth of field, detector signal, and dose.

The provided AutoScript reference identifies motorized condenser mechanisms as `C1`, `C2`, and `C3`; `C3` is present only on configurations that provide it. The asyncroscopy simulator uses the broader name `condenser`, but real hardware commands must use a mechanism name returned by the microscope.

### Objective aperture

The objective aperture is used primarily in TEM imaging and diffraction-contrast workflows. It limits which scattered beams contribute to the image and can be used for bright-field or dark-field selection. Inserting the wrong objective aperture can strongly change contrast or exclude information needed by the experiment.

The provided AutoScript reference documents the motorized mechanism identifier `Objective`. Availability still depends on the installed microscope configuration.

### Selected-area aperture

The selected-area aperture defines the specimen region contributing to a selected-area diffraction pattern. Its physical size and the microscope imaging conditions determine the effective selected region at the specimen.

The provided AutoScript reference documents this mechanism as `SA`. It should not be inserted blindly: the microscopist should first establish the intended image/diffraction condition and confirm that no acquisition is running.

### Projector aperture

The supplied AutoScript aperture-mechanism documentation does **not** identify a mechanism named `Projector`. It does list `TransferLens`, but that must not be assumed to be equivalent to a projector aperture without confirmation for the specific instrument.

`projector` currently exists only as a simulation-first asyncroscopy category. Do not send that name to a real Spectra unless runtime discovery reports it. Real hardware support must always be determined from `aperture_list("{}")` and the AutoScript `get_available()` result.

## Supported automation operations

### Read-only inspection

The current asyncroscopy interface can:

- list mechanisms reported by the active backend;
- list the aperture names, types, and diameters associated with a mechanism;
- identify the selected aperture;
- report insertion state;
- report whether a mechanism is enabled with `aperture_is_enabled()`;
- report whether a mechanism is retractable with `aperture_is_retractable()`;
- indicate whether the response came from AutoScript or the simulator.

These status commands are read-only. They do not enable, disable, insert, retract, select, move, or center an aperture. Read-only inspection should always be performed before requesting a change.

### Aperture selection

An aperture is selected by its exact reported name. Diameter is metadata, not the selection key. This matters because non-circular apertures can have no diameter, and two apertures should not be treated as interchangeable merely because their nominal diameters match.

On real hardware, available names and mechanisms are discovered dynamically. Simulation names such as `"50 um"` are examples and must not be copied into a real experiment without verifying the live inventory.

`aperture_select()` now performs a read-only enabled-state precheck before assigning the selected aperture. If the mechanism reports disabled, or if the enabled-state status cannot be read, asyncroscopy rejects the request before calling AutoScript selection.

### Insertion and retraction

Insertion and retraction are separate, explicit commands. Whether a mechanism is retractable and whether the operation is allowed depend on the physical mechanism and its current AutoScript state.

`aperture_insert()` and `aperture_retract()` now perform read-only `is_enabled` and `is_retractable` prechecks before calling AutoScript mutation methods. If either status is false, unsupported, or missing, asyncroscopy rejects the request without inserting or retracting the mechanism.

The command response should be checked after every request. A software response is not a substitute for confirming the microscope state in the instrument UI when establishing a new workflow.

### Enable and disable

`aperture_enable()` and `aperture_disable()` expose AutoScript `ApertureMechanism.enable()` and `ApertureMechanism.disable()` for normal motorized aperture mechanisms only. If the mechanism does not provide the documented method, asyncroscopy reports the operation as unsupported rather than guessing.

`aperture_enable()` is allowed even when `aperture_is_enabled()` reports `false`, because enabling is the operation that recovers that state. `aperture_disable()` is conservative: it only runs when the mechanism insertion state is explicitly `"Retracted"`. If the mechanism is inserted, moving, arbitrary, in error, or its insertion state cannot be read, asyncroscopy rejects the disable request before calling AutoScript.
### Centering and position

The supplied AutoScript API documents `ApertureMechanism.position` as an XY position in meters. It can be read or set only when the mechanism is enabled, inserted, and has a selected aperture. AutoScript also documents `reset_positions()` for restoring aligned positions for all apertures on a mechanism.

The isolated asyncroscopy AutoScript adapter supports the documented position property, but the current safe Tango command set does not expose a public centering command. Centering should remain unavailable to routine clients until limits, confirmation behavior, logging, and hardware tests are defined for the specific Spectra.

## Operations that must not happen automatically

Asyncroscopy should not automatically:

- change an aperture while an image, diffraction pattern, EDS spectrum, or EELS spectrum is being acquired;
- select an aperture from diameter alone when AutoScript provides a unique name;
- insert an objective or selected-area aperture without explicit user confirmation;
- move or recenter an aperture as a side effect of image acquisition;
- assume that the simulator's mechanism names exist on the real microscope;
- treat `Moving`, `Arbitrary`, or `Error` as ready states.

Acquisition methods do not call the aperture-changing commands. Selection, insertion, retraction, enable, and disable are manual automation actions, and the current `AutoScriptMicroscope` command path rejects those aperture mutations while a microscope acquisition command is active. Read-only aperture commands and metadata reads remain allowed during acquisition.

## JSON command examples

The examples below show requests passed to the `AutoScriptMicroscope` Tango device. Responses are JSON strings and should be decoded and checked before continuing.

### List all mechanisms and apertures

Request:

```json
{}
```

Python client:

```python
import json
import tango

microscope = tango.DeviceProxy("asyncroscopy/autoscriptmicroscope/default")
response = json.loads(microscope.aperture_list("{}"))
print(json.dumps(response, indent=2))
```

The response includes `mechanisms`, `apertures`, and `autoscript_available`. If `autoscript_available` is `false`, the response comes from the simulation backend.

### List one mechanism

Simulation request:

```json
{"mechanism": "condenser"}
```

On real hardware, replace `condenser` with an exact discovered identifier such as `C2` only when it appears in the live mechanism list.

```python
request = json.dumps({"mechanism": "condenser"})
response = json.loads(microscope.aperture_list(request))
```

### Read the selected aperture

```python
request = json.dumps({"mechanism": "objective"})
response = json.loads(microscope.aperture_get_selected(request))
```

### Read enabled or retractable status

```python
request = json.dumps({"mechanism": "objective"})
enabled = json.loads(microscope.aperture_is_enabled(request))
retractable = json.loads(microscope.aperture_is_retractable(request))
```

These are status reads only. They do not expose `enable()`, `disable()`, position movement, or centering.

Expected response fields include:

```json
{
  "ok": true,
  "action": "get_selected",
  "mechanism": "objective",
  "aperture": {
    "name": "40 um",
    "type": "Circular",
    "diameter_m": 0.00004
  },
  "insertion_state": "Inserted",
  "autoscript_available": false,
  "error": null
}
```

Values shown above are simulation examples, not guaranteed Spectra hardware values.

### Select by exact name

First verify that the name occurs in the live `aperture_list` response. Then, while the microscope is idle and in the intended mode:

```python
request = json.dumps(
    {
        "mechanism": "objective",
        "name": "100 um",
    }
)
response = json.loads(microscope.aperture_select(request))
if not response["ok"]:
    raise RuntimeError(response["error"])
```

For real hardware, both strings must match the names discovered from AutoScript. The example names above belong to the simulator.

### Insert or retract explicitly

```python
request = json.dumps({"mechanism": "selected_area"})

insert_response = json.loads(microscope.aperture_insert(request))
if not insert_response["ok"]:
    raise RuntimeError(insert_response["error"])

retract_response = json.loads(microscope.aperture_retract(request))
if not retract_response["ok"]:
    raise RuntimeError(retract_response["error"])
```

The current `AutoScriptMicroscope` path rejects these mutation commands during live acquisition. On real hardware, use the exact discovered mechanism identifier, such as `SA`, rather than assuming the simulation name `selected_area`.

## Example microscopist workflows

### Before STEM imaging

1. Confirm that the microscope is in the intended STEM mode and is idle.
2. Call `aperture_list("{}")` and identify the condenser mechanism reported by the real microscope.
3. Call `aperture_get_selected()` for that mechanism.
4. Verify the aperture name, diameter metadata, and insertion state against the intended probe-current and convergence condition.
5. Make a change only through a separate explicit command.

### Before diffraction or SAED

1. Establish and verify the intended image/diffraction condition.
2. Discover whether `SA` is available.
3. Read the selected SA aperture and insertion state.
4. Confirm the desired selected area before inserting or selecting another aperture.
5. The command path rejects changes while a diffraction acquisition is active.

### Before EDS or EELS

1. Read the condenser, objective, and other relevant available aperture states.
2. Preserve the returned JSON alongside the acquisition metadata so beam-current and collection conditions are traceable.
3. AutoScriptMicroscope attaches the current aperture snapshot to saved STEM-image, camera-image, STEM-data, and EDS-spectrum metadata when it is available. A read failure is recorded under `errors` and does not stop the acquisition.
4. For EELS, note that the supplied AutoScript documentation exposes energy-filter apertures through a separate API. The motorized mechanism commands described here do not implement EELS aperture support and do not touch `microscope.optics.energy_filter.apertures`.

## Hardware-dependent behavior

AutoScript support varies with microscope configuration and software version. The provided reference documents motorized `C1`, `C2`, `C3`, `Objective`, `SA`, and `TransferLens` mechanism identifiers, but the instrument's runtime discovery result is authoritative. A mechanism, aperture name, diameter, retractability, or position operation must never be assumed solely from this document or from simulation behavior.
