# tem_client.py
'''Client for TEM central server.'''
import socket
import struct
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# still needs a lot of work
class TEMClient:
    """Client for TEM central server."""
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

    def send_command(self, command: str, timeout: float | None = None) -> bytes:
        """Send a length-prefixed command and receive a length-prefixed response."""
        print("[client] sending:", command)
        try:
            # Encode the command
            payload = command.encode()
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


    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes or raise ConnectionError if socket closes early."""
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed early while receiving data")
            buf += chunk
        return buf

    def send_command_beta(self, destination: str, command: str,
                          args: dict, timeout: float | None = None) -> bytes:
        """Send a length-prefixed command with args and receive a length-prefixed response."""
        cmd = f"{destination}_{command} " + " ".join(f"{k}={v}" for k, v in args.items())
        print("[client] sending:", cmd)
        try:
            # Encode the command
            payload = cmd.encode()
            header = struct.pack("!I", len(payload))

            with socket.create_connection((self.host, self.port), timeout=timeout) as sock:
                # Send length-prefixed message
                sock.sendall(header + payload)
                print("[client] sent:", cmd)

                # Read the 4-byte response header
                resp_hdr = self._recv_exact(sock, 4)
                resp_len = struct.unpack("!I", resp_hdr)[0]

                # Read exactly that many bytes
                data = self._recv_exact(sock, resp_len)

            return data.decode()

        except (ConnectionRefusedError, socket.timeout):
            print(f"Could not connect to {self.host}:{self.port} after {timeout} seconds")
            return None

    #  AS_server.py methods
    # ======================
    def get_status(self):
        """Get the current status of the AS server."""
        cmd = "AS_get_status"
        data = self.send_command(cmd)
        return data.decode()

    def get_scanned_image(self, scanning_detector: str, size: int, dwell_time: float) -> bytes:
        """Get a scanned image from the AS server."""
        cmd = f"AS_get_scanned_image {scanning_detector} {size} {dwell_time}"
        data = self.send_command(cmd)

        image = np.frombuffer(data, dtype=np.uint8).reshape(size, size)
        return image

    def get_spectrum(self, size):
        """Get a spectrum from the Gatan server."""
        cmd = f"Gatan_get_spectrum {size}"
        data = self.send_command(cmd)
        spectrum = np.frombuffer(data, dtype=np.float32)
        return spectrum

    def get_stage(self):
        """Get the current stage positions from the AS server."""
        cmd = "AS_get_stage"
        data = self.send_command(cmd)
        positions = np.frombuffer(data, dtype=np.float32)
        return positions

    #  Ceos_server.py methods
    # ======================
    def run_tableau(self, tab_type="Fast", angle=18):
        """Run a tableau acquisition on the CEOS corrector."""
        cmd = f"Ceos_run_tableau {tab_type} {angle}"
        data = self.send_command(cmd)
        return data.decode()

    def correct_aberration(self, name: str, value=None, target=None, select=None):
        """Correct an aberration on the CEOS corrector."""
        cmd = f"Ceos_correct_aberration {name} {value} {target} {select}"
        data = self.send_command(cmd)
        return data.decode()

    def measure_c1a1(self):
        """Measure C1 and A1 aberrations on the CEOS corrector."""
        cmd = "Ceos_measure_c1a1"
        data = self.send_command(cmd)
        return data.decode()


    # Unique methods, including concurrent acquisitions
    # ================================================================
    def get_image_and_spectrum(self, image_size: int, image_dwell_time: float,
                               spectrum_size: int) -> bytes:
        """Run both acquisitions concurrently and return results."""
        future_img = self.executor.submit(self.get_scanned_image,
                                          'Haadf', image_size, image_dwell_time)
        future_spec = self.executor.submit(self.get_spectrum, spectrum_size)
        img = future_img.result()
        spec = future_spec.result()
        return img, spec
