# tem_client.py
import socket, struct, numpy as np
from concurrent.futures import ThreadPoolExecutor
import json, struct

# still needs a lot of work
class TEMClient:
    def __init__(self, host="localhost", port=9000):
        self.host = host
        self.port = port
        self.executor = ThreadPoolExecutor(max_workers=8) # arbitrary for now

    @classmethod
    def connect(cls,  host="127.0.0.1", port=9000):
        """Try to connect briefly to verify central server is up.
        Returns TEMClient(host, port) on success, or None on failure.
        """
        print(f"Connecting to central server {host}:{port}...")
        try:
            with socket.create_connection((host, port), timeout=5) as s:
                print("Connected to central server.")
            return cls(host, port)
        except (ConnectionRefusedError, socket.timeout):
            print(f"Could not connect to central server at {host}:{port}")
            return None

    def send_command(self, command: dict, timeout: float | None = None) -> bytes:
        """
        Send a length-prefixed command and receive a length-prefixed response.
        Accepts a dictionary in the form {command: str, args: list}
        """

        command_name = command.get("name")
        args = command.get("args", [])

        args_str = " ".join(str(arg) for arg in args)
        command_str = f"{command_name} {args_str}"

        print("[client] sending:", command_str)
        try:
            # Encode the command
            payload = command_str.encode()
            header = struct.pack("!I", len(payload))

            with socket.create_connection((self.host, self.port), timeout=timeout) as sock:
                # Send length-prefixed message
                sock.sendall(header + payload)
                print("[client] sent:", command)

                # Read the 4-byte response header
                resp_hdr = self._recv_exact(sock, 4)
                resp_len = struct.unpack("!I", resp_hdr)[0]

                # Read exactly that many bytes
                data = self._recv_exact(sock, resp_len)

            return data

        except (ConnectionRefusedError, socket.timeout):
            print(f"Could not connect to {self.host}:{self.port} after {timeout} seconds")
            return None

        except Exception as e:
            print(f"Error communicating with server: {e}")
            return None

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes or raise ConnectionError if socket closes early."""
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed early while receiving data")
            buf += chunk
        return buf