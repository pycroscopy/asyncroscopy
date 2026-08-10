import json
import os
import signal
import subprocess
import sys
import time

import pytest

from asyncroscopy.utils import process_manager
from asyncroscopy.utils.process_manager import ProcessManager, ManagedProcess


def test_process_manager_init(tmp_path):
    manager = ProcessManager(name="test_run", state_dir=tmp_path)
    assert manager.name == "test_run"
    assert manager.state_dir == tmp_path
    assert manager.state_file == tmp_path / "test_run.json"


def test_start_process_tracks_process_group(tmp_path, monkeypatch):
    manager = ProcessManager(name="test_run", state_dir=tmp_path)
    calls = {}

    class FakePopen:
        stdout = None
        stderr = None
        pid = 9999

        def __init__(self, command, **kwargs):
            calls["command"] = command
            calls["kwargs"] = kwargs

        def poll(self):
            return None

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(process_manager.subprocess, "Popen", FakePopen)

    with manager:
        proc = manager.start_process("test_key", "Test Label", ["uv", "run", "dummy"])
        assert proc.pid == 9999
        assert proc.key == "test_key"
        assert proc.label == "Test Label"
        assert proc.command == ["uv", "run", "dummy"]
        assert proc in manager.active_processes

        # State file must contain active pids
        assert manager.state_file.exists()
        pids = json.loads(manager.state_file.read_text(encoding="utf-8"))
        assert pids == [9999]

        if os.name == "nt":
            assert "creationflags" in calls["kwargs"]
        else:
            assert calls["kwargs"]["start_new_session"] is True


def test_stop_process_graceful_and_force(tmp_path, monkeypatch):
    manager = ProcessManager(name="test_run", state_dir=tmp_path, timeout=0.1)
    signals = []

    class FakeProcess:
        pid = 12345
        _poll_val = None
        wait_calls = 0

        def poll(self):
            return self._poll_val

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                # First wait times out to trigger force kill escalation
                raise subprocess.TimeoutExpired(["cmd"], timeout)
            self._poll_val = 0
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

    def mock_killpg(pgid, sig):
        signals.append((pgid, sig))

    taskkill_calls = []

    if os.name != "nt":
        monkeypatch.setattr(process_manager.os, "killpg", mock_killpg)
        monkeypatch.setattr(process_manager.os, "getpgid", lambda pid: pid)
    else:
        # Capture taskkill invocations without spawning a real process
        class FakePopenResult:
            pass

        def mock_popen(cmd, **kwargs):
            taskkill_calls.append(cmd)
            return FakePopenResult()

        monkeypatch.setattr(process_manager.subprocess, "Popen", mock_popen)

    proc = ManagedProcess("test_key", "Test Label", FakeProcess(), ["cmd"])
    manager.active_processes.append(proc)
    manager.save()

    manager.stop_process(proc)

    assert proc not in manager.active_processes
    # The state file must have removed the PID
    pids = json.loads(manager.state_file.read_text(encoding="utf-8"))
    assert pids == []

    if os.name != "nt":
        assert (12345, signal.SIGTERM) in signals
        assert (12345, signal.SIGKILL) in signals
    else:
        assert taskkill_calls == [["taskkill", "/F", "/T", "/PID", "12345"]]


def test_cleanup_stale_state_on_enter(tmp_path, monkeypatch):
    state_file = tmp_path / "test_run.json"
    state_file.write_text("[12345, 67890]", encoding="utf-8")

    killed_pids = []

    def mock_kill(pid, sig):
        killed_pids.append((pid, sig))

    monkeypatch.setattr(process_manager.os, "kill", mock_kill)
    if os.name != "nt":
        monkeypatch.setattr(process_manager.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(process_manager.os, "killpg", mock_kill)

    time_vals = [100.0, 102.0, 100.0, 102.0]
    monkeypatch.setattr(process_manager.time, "time", lambda: time_vals.pop(0) if time_vals else 200.0)
    monkeypatch.setattr(process_manager.time, "sleep", lambda x: None)

    manager = ProcessManager(name="test_run", state_dir=tmp_path)
    with manager:
        # State file should be cleaned up on entering context
        assert not state_file.exists()

    if os.name != "nt":
        # Check both SIGTERM and SIGKILL were sent due to mock time not incrementing
        assert (12345, signal.SIGTERM) in killed_pids
        assert (12345, signal.SIGKILL) in killed_pids
        assert (67890, signal.SIGTERM) in killed_pids
        assert (67890, signal.SIGKILL) in killed_pids


def test_wipe_databases(tmp_path, monkeypatch):
    monkeypatch.setattr(process_manager, "PROJECT_DIR", tmp_path)

    lowercase = tmp_path / "tango_database.db"
    uppercase = tmp_path / "Tango_database.db"
    lowercase.write_text("old db", encoding="utf-8")
    uppercase.write_text("old db", encoding="utf-8")

    manager = ProcessManager(name="test_run", state_dir=tmp_path)
    manager.wipe_databases()

    assert not lowercase.exists()
    assert not uppercase.exists()


def test_stop_processes_on_port(tmp_path, monkeypatch):
    manager = ProcessManager(name="test_run", state_dir=tmp_path)

    if os.name == "nt":
        calls = []

        class FakeCompletedProcess:

            def __init__(self, stdout="", returncode=0):
                self.stdout = stdout
                self.returncode = returncode

        def mock_run(cmd, **kwargs):
            calls.append(cmd)
            if "netstat" in cmd[0]:
                return FakeCompletedProcess(
                    stdout="  TCP    0.0.0.0:9094           0.0.0.0:0              LISTENING       5555\n"
                )
            return FakeCompletedProcess()

        monkeypatch.setattr(subprocess, "run", mock_run)

        count = manager.stop_processes_on_port(9094)
        assert count == 1
        assert any("taskkill" in cmd[0] and "5555" in cmd for cmd in calls)

    else:
        class FakeCompletedProcess:
            stdout = "1001\n1002\n"

        killed_pids = []

        def mock_kill(pid, sig):
            if sig == 0:
                raise ProcessLookupError()
            killed_pids.append(pid)

        monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: FakeCompletedProcess())
        monkeypatch.setattr(os, "kill", mock_kill)
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg", mock_kill)

        count = manager.stop_processes_on_port(9094)
        assert count == 2
        assert 1001 in killed_pids
        assert 1002 in killed_pids


def test_start_process_streams_output_via_callback(tmp_path):
    manager = ProcessManager(name="test_run", state_dir=tmp_path)
    received = []

    with manager:
        managed = manager.start_process(
            "echo_key",
            "Echo",
            [sys.executable, "-c", "print('hello-callback')"],
            on_output=received.append,
        )
        managed.process.wait(timeout=5)
        deadline = time.time() + 2
        while not received and time.time() < deadline:
            time.sleep(0.01)

    assert received == ["hello-callback"]
    # The callback is additive - lines still land in the regular buffer too.
    assert list(managed.stdout_lines) == ["hello-callback"]


def test_install_shutdown_signal_handler_converts_sigterm_to_keyboard_interrupt(monkeypatch):
    installed = {}
    monkeypatch.setattr(process_manager.signal, "signal", lambda sig, handler: installed.setdefault(sig, handler))

    process_manager.install_shutdown_signal_handler()

    assert signal.SIGTERM in installed
    with pytest.raises(KeyboardInterrupt):
        installed[signal.SIGTERM](signal.SIGTERM, None)

    # A second delivery is a no-op once shutdown is already underway.
    installed[signal.SIGTERM](signal.SIGTERM, None)