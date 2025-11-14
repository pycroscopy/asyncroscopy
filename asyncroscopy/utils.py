import subprocess
import sys
import os

def start_server(script_path, host, port):
    """
    Start a Python script as a completely detached background process
    on Windows, macOS, and Linux, passing host and port args.
    """

    # Convert port to string (subprocess requires list of strings)
    cmd = [sys.executable, script_path, host, str(port)]

    if os.name == "nt":  # Windows
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200

        return subprocess.Popen(
            cmd,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL
        )

    else:  # Linux / macOS / Unix
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setpgrp
        )