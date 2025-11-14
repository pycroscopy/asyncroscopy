# Ceos_server.py

"""
digital twin
"""

import logging
import json
from typing import Tuple, List, Optional, Union, Sequence
import traceback
import socket
from twisted.internet import reactor,defer, protocol
from asyncroscopy.servers.protocols.execution_protocol import ExecutionProtocol

logging.basicConfig()
log = logging.getLogger('CEOS_acquisition')
log.setLevel(logging.INFO)

# FACTORY — holds shared state (persistent across all connections)
class CeosFactory(protocol.Factory):
    def __init__(self):
        # persistent states for all protocol instances
        self.microscope = None
        self.aberrations = {}
        self.status = "Offline"

    def buildProtocol(self, addr):
        """Create a new protocol instance and attach the factory (shared state)."""
        proto = CeosProtocol()
        proto.factory = self
        return proto


# PROTOCOL — handles per-connection command execution
class CeosProtocol(ExecutionProtocol):
    def __init__(self):
        super().__init__()
        allowed = []
        for name, value in CeosProtocol.__dict__.items():
            if callable(value) and not name.startswith("_"):
                allowed.append(name)
        self.allowed_commands = set(allowed)

    # Override stringReceived for special case of Ceos commands
    def stringReceived(self, data: bytes):
        msg = data.decode().strip()
        print(f"[Exec] Received: {msg}")
        parts = msg.split()
        cmd, *args_parts = parts
        args_dict = dict(arg.split('=', 1) for arg in args_parts if '=' in arg)

        if cmd not in self.allowed_commands:
            self.sendString(f"ERR Unknown command: {cmd}".encode())
            return

        method = getattr(self, cmd, None)
        result = method(args_dict)
        self.sendString(result)


    def getInfo(self, args_dict=None):
        """Get microscope info."""
        return b"CEOS Digital Twin Server"
    
    def uploadAberrations(self, args_dict):
        """Upload aberration data."""
        # args = aberration dictionary from pyTEMlib probe tools
        # but the values are strings
        for key in args_dict:
            args_dict[key] = float(args_dict[key])

        self.factory.aberrations.update(args_dict)
        print("args_dict:", args_dict)
        return b'Aberrations Loaded'
    
    def runTableau(self, args_dict):
        """Run a tableau acquisition."""
        # args = {"tabType": 'Fast', "angle": 18}
        # args don't matter on this one:

    def correctAberration(self, args_dict):
        """Correct an aberration."""
        # args = {"name": name, "value": [...], "target": [...], "select": ...}


if __name__ == "__main__":
    port = 9003
    print(f"[Ceos] Server running on port {port}...")
    reactor.listenTCP(port, CeosFactory())
    reactor.run()