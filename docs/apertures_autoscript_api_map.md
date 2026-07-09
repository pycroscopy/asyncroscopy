# AutoScript aperture API map

This note records the aperture-control API exposed by the provided Thermo Fisher AutoScript TEM HTML reference. It intentionally uses the documented names and paths without introducing wrapper method names.

## Imports and client object

The AutoScript client startup documentation imports and creates the microscope client as follows:

```python
from autoscript_tem_microscope_client import TemMicroscopeClient

microscope = TemMicroscopeClient()
```

The client must be connected using the normal AutoScript connection procedure before accessing microscope objects.

No aperture-specific import is required when selecting an aperture already returned by AutoScript. The position property accepts a `Point`, list, or tuple, so a two-value list or tuple can be used without importing `Point`.

## Motorized aperture collection

The exact collection path is:

```python
microscope.optics.aperture_mechanisms
```

Available motorized mechanisms are discovered at runtime:

```python
mechanism_names = microscope.optics.aperture_mechanisms.get_available()
```

`get_available()` returns `list[str]` identifiers. A mechanism is retrieved by identifier with:

```python
mechanism = microscope.optics.aperture_mechanisms.get_mechanism(mechanism_type)
```

Only motorized mechanisms are supported by `get_mechanism()`. An unsupported identifier raises an exception.

## Documented mechanism names

`ApertureMechanismType` documents these exact identifiers:

| Enumeration item | String value | Meaning |
|---|---|---|
| `C1` | `"C1"` | C1 condenser-lens aperture mechanism |
| `C2` | `"C2"` | C2 condenser-lens aperture mechanism |
| `C3` | `"C3"` | C3 condenser-lens aperture mechanism |
| `OBJECTIVE` | `"Objective"` | Objective-lens aperture mechanism |
| `SA` | `"SA"` | Selected-area aperture mechanism |
| `TRANSFER_LENS` | `"TransferLens"` | Transfer-lens aperture mechanism |

The collection also provides these documented convenience properties:

```python
microscope.optics.aperture_mechanisms.C1
microscope.optics.aperture_mechanisms.C2
microscope.optics.aperture_mechanisms.C3       # if present
microscope.optics.aperture_mechanisms.objective
microscope.optics.aperture_mechanisms.SA
```

The reference does not document a projector-aperture mechanism or a generic `microscope.optics.apertures` path. Code should use `get_available()` because the mechanisms physically installed on a microscope can differ.

## Aperture mechanism API

Given an `ApertureMechanism` object named `mechanism`, the exact documented interface is:

| Operation | AutoScript API | Type |
|---|---|---|
| List available apertures | `mechanism.apertures` | Read-only `list[Aperture]` |
| Get selected aperture | `mechanism.aperture` | `Aperture | None` |
| Select an aperture | `mechanism.aperture = aperture` | Assign an `Aperture` object |
| Read insertion state | `mechanism.insertion_state` | Read-only `str` |
| Read enabled state | `mechanism.is_enabled` | Read-only `bool` |
| Read retractability | `mechanism.is_retractable` | Read-only `bool` |
| Enable | `mechanism.enable()` | Method |
| Disable | `mechanism.disable()` | Method |
| Insert | `mechanism.insert()` | Method |
| Retract | `mechanism.retract()` | Method |
| Read position | `mechanism.position` | `Point | None` |
| Set position | `mechanism.position = (x, y)` | `Point`, list, or tuple; meters |
| Reset aligned positions | `mechanism.reset_positions()` | Method |

### Listing and selecting apertures

Each item returned by `mechanism.apertures` is an `Aperture` with exactly these fields:

```python
aperture.name       # str
aperture.type       # str, using an ApertureType value
aperture.diameter   # float in meters for circular apertures; otherwise None
```

The documented selection pattern is to obtain an existing object from `mechanism.apertures` and assign it to `mechanism.aperture`:

```python
available = mechanism.apertures
wanted_name = "<name reported by AutoScript>"
selected = next(item for item in available if item.name == wanted_name)
mechanism.aperture = selected
```

AutoScript documents `Aperture.name` as the unique identifier. When an aperture is selected, only its `name` is used for identification. An unmatched name causes selection to fail.

Setting `mechanism.aperture` inserts its mechanism if it is currently retracted. Reading `mechanism.aperture` returns `None` when no aperture is selected, including after retraction.

### Insertion state

`mechanism.insertion_state` maps to `ApertureMechanismInsertionState` and returns one of these exact strings:

```text
"Retracted"
"Inserted"
"Moving"
"Arbitrary"
"Error"
```

The mechanism must be enabled before reading `insertion_state`. Insertion and retraction require the mechanism to be enabled and retractable.

### Position and centering

`mechanism.position` gets or sets the XY position of the selected aperture in meters:

```python
position = mechanism.position
x = position.x
y = position.y

mechanism.position = (x, y)
```

The mechanism must be enabled and inserted before its position can be read or changed. A returned value of `None` means no aperture is selected.

The position is stored as a preset for the currently selected aperture. Selecting that aperture again recovers its stored position. `mechanism.reset_positions()` restores the default microscope-alignment positions for every aperture on that mechanism; it cannot reset only one aperture.

## Asyncroscopy read-only status commands

The current `AutoScriptMicroscope` Tango aperture path exposes read-only status commands for normal motorized aperture mechanisms:

```python
microscope.aperture_is_enabled(json.dumps({"mechanism": "C2"}))
microscope.aperture_is_retractable(json.dumps({"mechanism": "C2"}))
```

These commands read `ApertureMechanism.is_enabled` and `ApertureMechanism.is_retractable` through `microscope.optics.aperture_mechanisms`. Separate explicit mutation commands expose `ApertureMechanism.enable()` and `ApertureMechanism.disable()` when those methods are present. No status or mutation command calls position setters, centering/reset operations, or any energy-filter aperture API.

## Energy-filter aperture API

Energy-filter apertures use a separate documented path:

```python
microscope.optics.energy_filter.apertures
```

Its exact properties are:

```python
microscope.optics.energy_filter.apertures.available_values  # read-only list[str]
microscope.optics.energy_filter.apertures.value             # get/set str
```

Only a value returned by `available_values` may be assigned to `value`. This interface is distinct from the motorized `ApertureMechanism` API.

## Version notes

The provided reference was generated for AutoScript TEM 1.15.0.484. Its aperture structure exposes `name`, `type`, and `diameter`. Implementations targeting this newer structure-based API should prefer `name` for selection, because the documentation identifies `name` as unique and states that only the name is used during selection.

Availability must still be detected at runtime with `get_available()` and `mechanism.apertures`; mechanism inventory and aperture names must not be hard-coded from one Spectra configuration.

## Spectra safety rules

- Read-only discovery and status operations are safe: `get_available()`, `get_mechanism()`, `apertures`, `aperture`, `is_enabled`, `is_retractable`, and—when enabled—`insertion_state` and `position`.
- Aperture selection, insertion, retraction, enabling, disabling, position changes, and position resets must occur only through explicit user commands. Current mutation commands also perform status prechecks and reject while a microscope acquisition command is active before calling AutoScript.
- Image acquisition must not automatically select, insert, retract, center, enable, disable, or reset an aperture unless that behavior is requested and designed separately later.
- Before selection, check `is_enabled`. Before insertion or retraction, check both `is_enabled` and `is_retractable`. Before disabling, require an explicit `Retracted` insertion state. Report disabled, non-retractable, unsafe-to-disable, or unsupported/missing status clearly instead of guessing.
- Before reading or changing position, confirm that the mechanism is enabled, inserted, and has a selected aperture.
- Treat `"Moving"`, `"Arbitrary"`, and `"Error"` as non-ready states; do not issue an automatic follow-up movement.
