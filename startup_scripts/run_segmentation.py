#!/usr/bin/env python
"""Register and start only the SAM2 segmentation Tango device."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tango
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from asyncroscopy.utils.process_manager import ManagedProcess, ProcessManager  # noqa: E402

DEVICE_NAME = "asyncroscopy/segment/default"
INSTANCE_NAME = "segment_instance"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "configs" / "Segmentation.yaml"


@dataclass(frozen=True)
class TangoConfig:
    host: str
    port: int


@dataclass(frozen=True)
class SegmentationConfig:
    tango: TangoConfig
    data_device_address: str = "asyncroscopy/data/default"
    model_size: str = "facebook/sam2-hiera-large"
    compute_device: str = "auto"
    device_timeout_seconds: int = 120


def load_config(path: Path) -> SegmentationConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tango_config = raw.get("tango")
    if not isinstance(tango_config, dict):
        raise KeyError("Config is missing the 'tango' section")

    return SegmentationConfig(
        tango=TangoConfig(
            host=str(tango_config["host"]),
            port=int(tango_config["port"]),
        ),
        data_device_address=str(
            raw.get("data_device_address", "asyncroscopy/data/default")
        ),
        model_size=str(raw.get("model_size", "facebook/sam2-hiera-large")),
        compute_device=str(raw.get("compute_device", "auto")),
        device_timeout_seconds=int(raw.get("device_timeout_seconds", 120)),
    )


def register_device(config: SegmentationConfig) -> None:
    database = tango.Database(config.tango.host, config.tango.port)
    device_info = tango.DbDevInfo()
    device_info.server = f"SEGMENTATION/{INSTANCE_NAME}"
    device_info._class = "SEGMENTATION"
    device_info.name = DEVICE_NAME

    try:
        database.add_device(device_info)
        print(f"Registered device: {DEVICE_NAME}")
    except tango.DevFailed as exc:
        print(f"Device already registered or error: {exc}")

    database.put_device_property(
        DEVICE_NAME,
        {
            "data_device_address": [config.data_device_address],
            "model_size": [config.model_size],
            "compute_device": [config.compute_device],
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.yaml)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    tango_host = f"{config.tango.host}:{config.tango.port}"
    os.environ["TANGO_HOST"] = tango_host
    try:
        register_device(config)
    except Exception as exc:
        print(f"Could not register segmentation at {tango_host}: {exc}", file=sys.stderr)
        return 1

    try:
        with ProcessManager() as manager:
            managed: ManagedProcess = manager.start_process(
                key="segmentation",
                label="Segmentation Server",
                command=[
                    sys.executable,
                    "-m",
                    "asyncroscopy.mcp.segment",
                    INSTANCE_NAME,
                ],
                env={
                    **os.environ,
                    "TANGO_HOST": tango_host,
                    "PYTHONUNBUFFERED": "1",
                },
                stdout=None,
                stderr=None,
            )

            print("Waiting for segmentation device to initialize...")
            deadline = time.monotonic() + config.device_timeout_seconds
            while time.monotonic() < deadline:
                try:
                    proxy = tango.DeviceProxy(DEVICE_NAME)
                    proxy.ping()
                    state = proxy.state()
                    if state == tango.DevState.ON:
                        print("Segmentation device is ready. Press Ctrl+C to terminate.")
                        break
                    if state == tango.DevState.FAULT:
                        print(f"Segmentation initialization failed: {proxy.status()}")
                        return 1
                except Exception:
                    pass
                time.sleep(1)
            else:
                print("Timed out waiting for segmentation device.", file=sys.stderr)
                return 1

            while managed.running:
                try:
                    managed.process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
    except KeyboardInterrupt:
        print("\nShutting down segmentation server...")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
