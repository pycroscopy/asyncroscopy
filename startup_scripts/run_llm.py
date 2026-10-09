#!/usr/bin/env python
"""Start the asyncroscopy agent MCP server from an explicit YAML config."""

from __future__ import annotations

import os
import subprocess
import sys
import argparse

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import json
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from asyncroscopy.utils.process_manager import ManagedProcess, ProcessManager

DEFAULT_CONFIG_PATH = PROJECT_DIR / 'configs' / 'gemma-llm.yaml'


@dataclass(frozen=True)
class AgentConfig:
    name: str
    transport: str
    http_host: str
    http_port: int
    mcp_urls: list[str] = field(default_factory=list)
    model: str = "gemma4:31b"
    model_provider: str = "ollama"
    max_steps: int = 10
    use_init_chat_model: bool = False
    # Kept as plain dicts so this module never imports the agent extras, which
    # keeps --help and config errors working without them installed.
    startup_agents: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    path: Path
    agent: AgentConfig


def _require(mapping: dict, key: str, where: str):
    if not isinstance(mapping, dict) or key not in mapping:
        raise KeyError(f"Config section '{where}' is missing required key '{key}'")
    return mapping[key]


def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    agent = _require(raw, 'agent', '(top level)')
    return Config(
        path=path,
        agent=AgentConfig(
            name=_require(agent, 'name', 'agent'),
            transport=_require(agent, 'transport', 'agent'),
            http_host=_require(agent, 'http_host', 'agent'),
            http_port=int(_require(agent, 'http_port', 'agent')),
            mcp_urls=list(agent.get('mcp_urls', [])),
            model=agent.get('model', 'gemma4:31b'),
            model_provider=agent.get('model_provider', 'ollama'),
            max_steps=int(agent.get('max_steps', 10)),
            use_init_chat_model=bool(agent.get('use_init_chat_model', False)),
            startup_agents=[dict(entry) for entry in agent.get('startup_agents', [])],
        ),
    )


def build_command(config: Config) -> list[str]:
    agent = config.agent
    command = [
        'uv',
        'run',
        '--extra',
        'agent',
        '--extra',
        'ollama',
        'python',
        '-m',
        'asyncroscopy.mcp.agent_mcp_server',
        '--name',
        agent.name,
        '--transport',
        agent.transport,
        '--http-host',
        agent.http_host,
        '--http-port',
        str(agent.http_port),
        '--mcp-urls-json',
        json.dumps(agent.mcp_urls),
        '--model',
        agent.model,
        '--model-provider',
        agent.model_provider,
        '--max-steps',
        str(agent.max_steps),
        '--startup-agents-json',
        json.dumps(agent.startup_agents),
    ]
    if agent.use_init_chat_model:
        command.append('--use-init-chat-model')
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', type=Path, default=DEFAULT_CONFIG_PATH, metavar='PATH', help='Agent YAML config to start from.')
    parser.add_argument(
        '--interactive',
        action='store_true',
        default=False,
        help='Query the swarm in this terminal instead of serving it over MCP.',
    )
    return parser.parse_args(argv)


def run_interactive(config: Config) -> int:
    """Run the swarm in this process and prompt it from stdin, without serving MCP."""
    from asyncroscopy.mcp.agent import Agent, AgentSwarm

    swarm = AgentSwarm(
        model=config.agent.model,
        model_provider=config.agent.model_provider,
        max_steps=config.agent.max_steps,
        startup_agents=[Agent(**entry) for entry in config.agent.startup_agents],
        use_init_chat_model=config.agent.use_init_chat_model,
    )
    asyncio.run(swarm.start(config.agent.mcp_urls))

    print("Entering interactive mode. Type 'exit' to quit.")
    while True:
        prompt = input("Agent Prompt (or 'exit'): ")
        if prompt.lower() == 'exit':
            break
        print(f"Response: {asyncio.run(swarm.query(prompt))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.yaml)
    except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        print(f'Config error: {exc}', file=sys.stderr)
        return 1

    if args.interactive:
        return run_interactive(config)

    command = build_command(config)

    print(f'Starting agent MCP server {config.agent.name}')
    print(f'  config:  {config.path}')
    print(f'  http:    http://{config.agent.http_host}:{config.agent.http_port}/mcp')
    print(f'  tools:   {", ".join(config.agent.mcp_urls) or "(none)"}')
    print(f'  command: {" ".join(command)}')

    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}

    try:
        with ProcessManager() as manager:
            managed: ManagedProcess = manager.start_process(
                key="agent_mcp",
                label=f"Agent MCP Server ({config.agent.name})",
                command=command,
                env=env,
                stdout=None,
                stderr=None,
            )
            print("Agent MCP server started. Press Ctrl+C to terminate.")

            # Loop with a timeout so Python catches SIGINT (Ctrl+C) on Windows
            while managed.running:
                try:
                    managed.process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass

    except KeyboardInterrupt:
        print("\nShutting down agent MCP server...")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())