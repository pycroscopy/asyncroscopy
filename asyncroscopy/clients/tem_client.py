# tem_client.py
import socket, struct, numpy as np
from concurrent.futures import ThreadPoolExecutor

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


    # Below should mirror ASProtocol methods in AS_server.py
    # ================================================================
    def connect_AS(self, host: str, port: int) -> bytes:
        command = f"AS_connect_AS {host} {port}"
        response = self.send_command(command, timeout=5)
        return response.decode()

    def get_status(self):
        cmd = "AS_get_status"
        data = self.send_command(cmd)
        return data.decode()

    def get_scanned_image(self, scanning_detector: str, size: int, dwell_time: float) -> bytes:
        cmd = f"AS_get_scanned_image {scanning_detector} {size} {dwell_time}"
        data = self.send_command(cmd)
        image = np.frombuffer(data, dtype=np.uint8).reshape(size, size)
        return image

    def get_spectrum(self, size):
        cmd = f"Gatan_get_spectrum {size}"
        data = self.send_command(cmd)
        spectrum = np.frombuffer(data, dtype=np.float32)
        return spectrum

    def get_stage(self):
        cmd = "AS_get_stage"
        data = self.send_command(cmd)
        positions = np.frombuffer(data, dtype=np.float32)
        return positions

    # Unique methods, including concurrent acquisitions
    # ================================================================
    def get_image_and_spectrum(self, image_size, spectrum_size):
        """Run both acquisitions concurrently and return results."""
        future_img = self.executor.submit(self.get_image, image_size)
        future_spec = self.executor.submit(self.get_spectrum, spectrum_size)
        img = future_img.result()
        spec = future_spec.result()
        return img, spec
