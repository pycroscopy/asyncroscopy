import sys
import argparse
from gevent import os
import yaml
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import threading
import time
import tango
from asyncroscopy.mcp.llm import LLM
from tango import DeviceProxy

DEVICE_NAME = "asyncroscopy/llm/default"
INSTANCE_NAME = "llm_instance"

def register_device(config: dict | None):
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
        # Map config keys to device properties
        properties = {}
        if "mcp_url" in config:
            properties["mcp_url"] = [config["mcp_url"]]
        if "local_model_path" in config:
            properties["local_model_path"] = [config["local_model_path"]]
        if "model_provider" in config:
            properties["model_provider"] = [config["model_provider"]]
        if "model_name" in config:
            properties["model_name"] = [config["model_name"]]
        if "api_key" in config:
            properties["api_key"] = [config["api_key"]]
        
        if properties:
            database.put_device_property(DEVICE_NAME, properties)
            print(f"Set device properties: {properties}")

def run_server():
    sys.argv = ["llm.py", INSTANCE_NAME]
    LLM.run_server()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=Path, help="Path to YAML configuration")
    parser.add_argument("--interactive", type=bool, default=False, help="Run in interactive mode")
    args = parser.parse_args()

    config = None
    if args.yaml:
        with open(args.yaml, "r") as f:
            config = yaml.safe_load(f)
            print(f"Loaded config from {args.yaml}")

    os.environ['TANGO_HOST'] = f'{config["tango"]["host"]}:{config["tango"]["port"]}'

    register_device(config)
    
    # Start server in thread
    threading.Thread(target=run_server, daemon=True).start()
    
    print("Waiting for LLM device to start and initialize...")
    proxy = None
    max_wait_seconds = 120
    
    for _ in range(max_wait_seconds):
        try:
            if proxy is None:
                proxy = DeviceProxy(DEVICE_NAME)
                proxy.ping()  # Ensure server is reachable first
            
            # Check the actual device state
            state = proxy.state()
            if state == tango.DevState.ON:
                print("Device initialized and ready.")
                break
            elif state == tango.DevState.FAULT:
                print(f"Device initialization failed. Status: {proxy.status()}")
                return
            # If state is INIT, continue waiting
            
        except Exception:
            proxy = None  # Reset proxy if connection fails
        
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
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\nShutting down server...")

if __name__ == "__main__":
    main()
