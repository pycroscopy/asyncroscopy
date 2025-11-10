# Author credits - Utkarsh Pratiush <utkarshp1161@gmail.com>

import logging
import json
import time
from typing import Tuple, List, Optional, Union, Sequence
import socket

import numpy as np
from datetime import datetime
from twisted.internet import reactor, defer

from CorrectorClient import CEOSClient

class CEOSacquisitionTCP:
    def __init__(self, host="127.0.0.1", port=7072):
        self.host = host
        self.port = port
        self._next_id = 1

    def _send_recv(self, message: dict) -> dict:
        # Convert to JSON and encode in UTF-8
        json_msg = json.dumps(message, separators=(",", ":"))  # no spaces
        payload = json_msg.encode("utf-8")
        netstring = f"{len(payload)}:".encode("ascii") + payload + b","

        with socket.create_connection((self.host, self.port), timeout=3000) as sock:  # 5 minutes
            sock.sendall(netstring)

            # Read until we hit a complete netstring (ends with b",")
            buffer = b""
            while not buffer.endswith(b","):
                chunk = sock.recv(4096)
                if not chunk:
                    break  # Server closed connection
                buffer += chunk

        # Parse netstring: "length:payload,"
        try:
            length_str, rest = buffer.split(b":", 1)
            length = int(length_str)
            payload = rest[:length]
            return json.loads(payload.decode("utf-8"))
        except Exception as e:
            print("Malformed netstring or response:", buffer)
            raise e

    def _run_rpc(self, method: str, params: dict = None):
        if params is None:
            params = {}
        msg = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
            "params": params,
        }
        self._next_id += 1
        reply = self._send_recv(msg)
        if "error" in reply:
            raise RuntimeError(f"RPC Error: {reply['error']}")
        return reply.get("result", {})

    def run_tableau(self, tab_type="Fast", angle=18):
        return self._run_rpc("acquireTableau", {"tabType": tab_type, "angle": angle})

    def correct_aberration(self, name: str, value=None, target=None, select=None):
        params = {"name": name}
        if value:
            params["value"] = list(value)
        if target:
            params["target"] = list(target)
        if select:
            params["select"] = select
        return self._run_rpc("correctAberration", params)

    def measure_c1a1(self):
        return self._run_rpc("measureC1A1", {})
