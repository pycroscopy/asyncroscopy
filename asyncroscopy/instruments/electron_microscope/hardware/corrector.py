"""
Corrector device server.

Acts as a translator between the Tango control system and the real CEOS
hardware server (JSON-RPC over netstring-framed TCP).

Communication pattern
---------------------
Commands sent to this device are forwarded to the CEOS hardware server as
JSON-RPC 2.0 requests, encoded in netstring format::

    <length>:<json_payload>,

The response is decoded the same way and returned to the Tango client.

Return convention
-----------------
All commands return DevString containing the raw JSON-RPC result payload
so that the caller can parse it as needed.

Client-side example::

    import json
    proxy = tango.DeviceProxy("asyncroscopy/ceos/default")

    info        = json.loads(proxy.get_info())
    aberrations = json.loads(proxy.get_aberrations())
    proxy.run_tableau("Fast 18")          # tabType, angle — space-separated
    proxy.correct_aberration("A1 0.0")    # name, value  — space-separated
"""

import json
import logging
import socket
from typing import Optional

import tango
from tango import AttrWriteType, DevState, DevEncoded, DevString
from tango.server import Device, attribute, command, device_property

log = logging.getLogger("CEOS_corrector")
log.setLevel(logging.INFO)


class CORRECTOR(Device):
    """
    Tango device that wraps the CEOS corrector hardware server.

    Each command opens a short-lived TCP connection to the CEOS server,
    sends a JSON-RPC 2.0 request, waits for the response, and returns the
    result as a JSON string.
    """

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    ceos_host = device_property(
        dtype=str,
        default_value="10.46.217.241",
        doc="Hostname or IP address of the CEOS hardware server",
    )

    ceos_port = device_property(
        dtype=int,
        default_value=9092,
        doc="TCP port of the CEOS hardware server",
    )

    socket_timeout = device_property(
        dtype=float,
        default_value=120000.0,
        doc="Socket timeout in seconds for CEOS communication",
    )

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    status_message = attribute(
        label="Status message",
        dtype=str,
        access=AttrWriteType.READ,
        doc="Last status string received from the CEOS server",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.INIT)

        self._message_id: int = 1
        self._last_status: str = "Uninitialised"

        self._connect()

    def _connect(self) -> None:
        """Verify TCP connectivity to the CEOS server and transition to ON."""
        try:
            with socket.create_connection(
                (self.ceos_host, self.ceos_port),
                timeout=self.socket_timeout,
            ):
                pass
            self.info_stream(
                f"CEOS server reachable at {self.ceos_host}:{self.ceos_port}"
            )
            self._last_status = "Connected"
            self.set_state(DevState.ON)
        except OSError as exc:
            self.error_stream(f"Cannot reach CEOS server: {exc}")
            self._last_status = f"Connection failed: {exc}"
            self.set_state(DevState.FAULT)

    # ------------------------------------------------------------------
    # Attribute read methods
    # ------------------------------------------------------------------

    def read_status_message(self) -> str:
        return self._last_status

    # ------------------------------------------------------------------
    # Public commands
    # ------------------------------------------------------------------

    @command(dtype_out=DevString)
    def get_info(self) -> str:
        """Return general information about the CEOS corrector."""
        return self._call("getInfo")


    @command(dtype_in=str, dtype_out=DevString)
    def acquire_tableau(self, args: str) -> str:
        """
        Run a correction tableau on the CEOS corrector.

        Parameters are packed into one space-separated string because
        Tango scalar commands accept only one input argument::

            proxy.run_tableau("Fast 18")
            proxy.run_tableau("Full 0")
        """

        parts = args.strip().split()
        tab_type, angle_str = parts
        angle = float(angle_str)
        return self._call("acquireTableau", {"tabType": tab_type, "angle": angle})

    @command(dtype_out=DevString)
    def measure_c1a1(self) -> str:
        return self._call("measureC1A1")

    @command(dtype_in=str, dtype_out=DevString)
    def correct_aberration(self, args: str) -> str:
        """
        Set a single aberration to the given value.

        Parameters are packed into one space-separated string.
        Some aberrations (C3) take one value:
            proxy.correct_aberration("C3 -0.00034e-6")
        Some aberrations take two values:
            proxy.correct_aberration("A1 0.00078 0.00027")
        """
        parts = args.strip().split()
        name = parts[0]
        values = [float(p) for p in parts[1:]]

        # Send as list — matches what the TCP hardware server expects
        return self._call("correctAberration", {"name": name, "value": values})

    @command
    def reconnect(self) -> None:
        """Re-attempt the TCP connection to the CEOS server."""
        self._connect()


    # ------------------------------------------------------------------
    # Public commands pertaining to simulation
    # ------------------------------------------------------------------

    @command(dtype_in=str)
    def set_aberrations_coeff_sim(self, json_aberrations_string: str):
        self.ab = json.loads(json_aberrations_string)
        pass

    @command(dtype_out=str)
    def get_aberrations_coeff_sim(self):
        if self.ab == None:
            return
        return json.dumps(self.ab)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call(self, method: str, params: Optional[dict] = None) -> dict:
        """
        Send a JSON-RPC 2.0 request to the CEOS server and return the
        decoded response JSON string.

        Parameters
        ----------
        method:
            JSON-RPC method name (e.g. ``"GetInfo"``).
        params:
            Optional dict of named parameters.

        Returns
        -------
        str
            Inner JSON string from the CEOS netstring response.

        Raises
        ------
        tango.DevFailed
            On any communication or protocol error.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": self._message_id,
            "method": method,
            "params": params or {},
        }
        self._message_id += 1

        netstring = self._encode_netstring(payload)
        self.debug_stream(f"[CEOS] -> {netstring}")

        try:
            with socket.create_connection(
                (self.ceos_host, self.ceos_port),
                timeout=self.socket_timeout,
            ) as sock:
                sock.sendall(netstring)
                raw_response = self._recv_netstring(sock)

            self.debug_stream(f"[CEOS] <- {raw_response}")
            result = self._decode_netstring(raw_response)
            self._last_status = "OK"
            return result

        except OSError as exc:
            self.error_stream(f"CEOS communication error: {exc}")
            self._last_status = f"Error: {exc}"
            self.set_state(DevState.FAULT)
            tango.Except.throw_exception(
                "CeosCommError",
                str(exc),
                "CeosCorrector._call()",
            )

    def _encode_netstring(self, payload: dict) -> bytes:
        """Serialize *payload* dict as a length-prefixed JSON-RPC netstring."""
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return f"{len(body)}:".encode("ascii") + body + b","

    def _decode_netstring(self, raw: bytes) -> str:
        """Strip the netstring length prefix and trailing comma."""
        s = raw.decode("utf-8").strip()
        if ":" in s and s.split(":", 1)[0].isdigit():
            _, s = s.split(":", 1)
        return s.rstrip(",")

    def _recv_netstring(self, sock: socket.socket, bufsize: int = 4096) -> bytes:
        """Read from *sock* until a complete netstring (ending with b',') arrives."""
        buffer = b""
        while not buffer.endswith(b","):
            chunk = sock.recv(bufsize)
            if not chunk:
                break
            buffer += chunk
        return buffer


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CORRECTOR.run_server()
