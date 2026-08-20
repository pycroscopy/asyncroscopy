from typing import Literal
import json
import sys
import argparse
import os
import subprocess
import yaml
from pathlib import Path
from dataclasses import dataclass
import time
import tango
from tango import DeviceProxy


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from asyncroscopy.utils.process_manager import ManagedProcess, ProcessManager
from asyncroscopy.mcp.llm import Agent

DEVICE_NAME = "asyncroscopy/llm/default"
INSTANCE_NAME = "llm_instance"
DEFAULT_CONFIG_PATH = PROJECT_DIR / 'configs' / 'gemma-llm.yaml'

@dataclass 
class TangoConfig:
    host: str
    port: int

@dataclass
class MCPConfig:
    url: str | None = None
    transport: Literal["stdio", "http", "sse", "streamable-http"] = "streamable-http"

@dataclass
class LLMConfig:
    tango: TangoConfig
    mcp_config: MCPConfig
    
    chat_model_name: str | None = None
    api_key: str | None = None
    api_base: str | None = None

    ollama_model: str | None = None
    auto_pull_model: bool = True

    startup_agents: list[Agent] | None = None


def load_config(path: Path) -> LLMConfig:
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    return LLMConfig(**raw)

def register_device(config: LLMConfig | None):
    database = tango.Database()
    try:
        device_info = tango.DbDevInfo()
        device_info.server = f"LLM/{INSTANCE_NAME}"
        device_info._class = "LLM"
        device_info.name = DEVICE_NAME
        database.add_device(device_info)
        print(f"Registered device: {DEVICE_NAME}")
    except tango.DevFailed as e:
        print(f"Device already registered or error: {e}")

    if config:
        properties = {}
        for key, value in config.__dict__.items():
            if key != "tango":
                properties[key] = value
                if key == "startup_agents":
                    properties[key] = [json.dumps(agent) for agent in value]
                elif key == "mcp_config":
                    properties[key] = json.dumps(value)
        
        database.put_device_property(DEVICE_NAME, properties)
        print(f"Set device properties: {properties}")

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', type=Path, default=DEFAULT_CONFIG_PATH, metavar='PATH', help='LLM YAML config to start from.')
    parser.add_argument('--interactive', action='store_true', default=False, help='Run in interactive mode, allowing user to send prompts to the LLM device.')
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.yaml)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f'Config error: {exc}', file=sys.stderr)
        return 1

    tango_host = f'{config.tango["host"]}:{config.tango["port"]}'
    os.environ['TANGO_HOST'] = tango_host

    register_device(config)
    
    command = [sys.executable, "-m", "asyncroscopy.mcp.llm", INSTANCE_NAME]
    env = {**os.environ, 'TANGO_HOST': tango_host, 'PYTHONUNBUFFERED': '1'}

    try:
        with ProcessManager() as manager:
            managed: ManagedProcess = manager.start_process(
                key="llm",
                label="LLM Server",
                command=command,
                env=env,
                stdout=None,
                stderr=None,
            )

            print("Waiting for LLM device to start and initialize...")

            proxy = None
            max_wait_seconds = 120
            
            for _ in range(max_wait_seconds):
                try:
                    if proxy is None:
                        proxy = DeviceProxy(DEVICE_NAME)
                        proxy.ping()
                    
                    state = proxy.state()
                    if state == tango.DevState.ON:
                        print("Device initialized and ready.")
                        break
                    elif state == tango.DevState.FAULT:
                        print(f"Device initialization failed. Status: {proxy.status()}")
                        return
                except Exception:
                    proxy = None
                
                time.sleep(1)
            else:
                print("Timeout waiting for device to initialize.")
                return

            if args.interactive:
                print("Entering interactive mode. Type 'exit' to quit.")
                while True:
                    prompt = input("LLM Prompt (or 'exit'): ")
                    if prompt.lower() == 'exit':
                        break
                    
                    try:
                        response = proxy.Query(prompt)
                        print(f"Response: {response}")
                    except Exception as e:
                        print(f"Error: {e}")
            else:
                print("Press Ctrl+C to terminate.")
                # Loop with timeout so Windows handles SIGINT / Ctrl+C cleanly
                while managed.running:
                    try:
                        managed.process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        pass    

    except KeyboardInterrupt:    
        print("\nShutting down server...")
        return 0

if __name__ == "__main__":
    main()
