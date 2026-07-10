#!/usr/bin/env python
"""Start the asyncroscopy MCP server from an explicit YAML config."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_DIR / 'configs' / 'mcp.yaml'


@dataclass(frozen=True)
class MCPConfig:
    name: str
    transport: str
    http_host: str
    http_port: int
    data_device_address: str
    quiet: bool
    blocked_classes: list[str]
    blocked_functions: dict[str, list[str]]


@dataclass(frozen=True)
class Config:
    path: Path
    tango_host: str
    tango_port: int
    mcp: MCPConfig


def _require(mapping: dict, key: str, where: str):
    if not isinstance(mapping, dict) or key not in mapping:
        raise KeyError(f"Config section '{where}' is missing required key '{key}'")
    return mapping[key]


def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    tango = _require(raw, 'tango', '(top level)')
    mcp = _require(raw, 'mcp', '(top level)')
    return Config(
        path=path,
        tango_host=_require(tango, 'host', 'tango'),
        tango_port=int(_require(tango, 'port', 'tango')),
        mcp=MCPConfig(
            name=_require(mcp, 'name', 'mcp'),
            transport=_require(mcp, 'transport', 'mcp'),
            http_host=_require(mcp, 'http_host', 'mcp'),
            http_port=int(_require(mcp, 'http_port', 'mcp')),
            data_device_address=_require(mcp, 'data_device_address', 'mcp'),
            quiet=bool(_require(mcp, 'quiet', 'mcp')),
            blocked_classes=list(_require(mcp, 'blocked_classes', 'mcp')),
            blocked_functions={key: list(value) for key, value in _require(mcp, 'blocked_functions', 'mcp').items()},
        ),
    )


def build_command(config: Config) -> list[str]:
    command = [
        'uv',
        'run',
        'python',
        '-m',
        'asyncroscopy.mcp.mcp_server',
        '--name',
        config.mcp.name,
        '--tango-host',
        config.tango_host,
        '--tango-port',
        str(config.tango_port),
        '--transport',
        config.mcp.transport,
        '--http-host',
        config.mcp.http_host,
        '--http-port',
        str(config.mcp.http_port),
        '--data-device-address',
        config.mcp.data_device_address,
        '--blocked-classes-json',
        json.dumps(config.mcp.blocked_classes),
        '--blocked-functions-json',
        json.dumps(config.mcp.blocked_functions),
    ]
    if config.mcp.quiet:
        command.append('--quiet')
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', type=Path, default=DEFAULT_CONFIG_PATH, metavar='PATH', help='MCP YAML config to start from.')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.yaml)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f'Config error: {exc}', file=sys.stderr)
        return 1

    env = {**os.environ, 'TANGO_HOST': f'{config.tango_host}:{config.tango_port}', 'PYTHONUNBUFFERED': '1'}
    command = build_command(config)
    print(f'Starting MCP server {config.mcp.name}')
    print(f'  config: {config.path}')
    print(f'  tango:  {env["TANGO_HOST"]}')
    print(f'  http:   http://{config.mcp.http_host}:{config.mcp.http_port}/mcp')
    print(f'  command: {" ".join(command)}')
    return subprocess.run(command, cwd=PROJECT_DIR, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
