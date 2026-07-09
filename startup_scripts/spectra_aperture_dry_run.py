"""Read-only AutoScript aperture inventory for a Spectra microscope PC.

Run this script first on the Spectra PC and save its terminal output. The
printed mechanism and aperture names are the values to use in later
``aperture_select`` requests.

This module does not import AutoScript until ``main()`` requests a connection,
so importing it or displaying ``--help`` does not require proprietary Thermo
Fisher packages on a development machine.
"""

from __future__ import annotations

import argparse
import os
from typing import Any


DEFAULT_HOST = "10.46.217.241"
DEFAULT_PORT = 9095


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read the Spectra aperture inventory without changing hardware.",
        epilog=(
            "Run this first on the Spectra PC, save the terminal output, and use "
            "the printed aperture names in future aperture_select commands."
        ),
    )
    parser.add_argument(
        "--host",
        default=os.getenv("ASYNCROSCOPY_HARDWARE_HOST", DEFAULT_HOST),
        help="AutoScript server host (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("ASYNCROSCOPY_HARDWARE_PORT", str(DEFAULT_PORT))),
        help="AutoScript server port (default: %(default)s)",
    )
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help=(
            "Reserved for a future write-enabled script. This version remains "
            "strictly read-only even when supplied."
        ),
    )
    return parser


def connect_autoscript(host: str, port: int) -> Any:
    """Create and connect the client using AutoScriptMicroscope's host/port pattern."""
    try:
        from autoscript_tem_microscope_client import TemMicroscopeClient
    except ImportError as exc:
        raise RuntimeError(
            "AutoScript is unavailable. Run this script in the AutoScript Python "
            "environment on the Spectra microscope PC. Missing import: "
            "autoscript_tem_microscope_client.TemMicroscopeClient"
        ) from exc

    microscope = TemMicroscopeClient()
    try:
        microscope.connect(host, port)
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to the AutoScript server at {host}:{port}: {exc}"
        ) from exc
    return microscope


def format_diameter(diameter_m: float | None) -> str:
    if diameter_m is None:
        return "None"
    return f"{diameter_m:.9g} m ({diameter_m * 1e6:.6g} um)"


def print_mechanism(adapter: Any, mechanism: str) -> None:
    """Print one mechanism while containing property errors to that mechanism."""
    print(f"\nMechanism: {mechanism}")

    try:
        insertion_state = adapter.get_insertion_state(mechanism)
        print(f"  insertion_state: {insertion_state}")
    except Exception as exc:
        print(f"  ERROR reading ApertureMechanism.insertion_state: {exc}")

    try:
        apertures = adapter.list_apertures(mechanism)
    except Exception as exc:
        print(f"  ERROR reading ApertureMechanism.apertures: {exc}")
        return

    print("  available apertures:")
    if not apertures:
        print("    (none reported)")
    for aperture in apertures:
        marker = " [SELECTED]" if aperture.selected else ""
        print(f"    name: {aperture.name}{marker}")
        print(f"      type: {aperture.aperture_type}")
        print(f"      diameter: {format_diameter(aperture.diameter_m)}")

    selected = next((item for item in apertures if item.selected), None)
    if selected is None:
        print("  selected aperture: None")
    else:
        print(f"  selected aperture: {selected.name}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print("Spectra AutoScript aperture dry run")
    print("Mode: READ ONLY")
    print(f"Connection target: {args.host}:{args.port}")
    if args.confirm_write:
        print(
            "--confirm-write acknowledged, but write operations are not implemented "
            "in this script. No hardware changes will be made."
        )

    try:
        microscope = connect_autoscript(args.host, args.port)
    except RuntimeError as exc:
        print("AutoScript available: no")
        print(f"ERROR: {exc}")
        return 2

    print("AutoScript available: yes")

    # Import after connection so development machines can still import this
    # script and use --help without an AutoScript installation.
    from asyncroscopy.instruments.electron_microscope.adapters.autoscript_apertures import (
        AutoScriptApertureAdapter,
    )

    adapter = AutoScriptApertureAdapter(microscope)
    try:
        mechanisms = adapter.list_mechanisms()
    except Exception as exc:
        print(
            "ERROR reading "
            "microscope.optics.aperture_mechanisms.get_available: "
            f"{exc}"
        )
        return 3

    print(f"Discovered aperture mechanisms ({len(mechanisms)}):")
    for mechanism in mechanisms:
        print(f"  - {mechanism}")

    for mechanism in mechanisms:
        print_mechanism(adapter, mechanism)

    print("\nDry run complete. No aperture was selected, inserted, retracted, or moved.")
    print("Save this terminal output for the aperture automation configuration record.")
    print("Use the exact printed names in future aperture_select commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
