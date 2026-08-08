# Process Manager

`ProcessManager` (`asyncroscopy/utils/process_manager.py`) provides the infrastructure for spawning and managing configuration-defined subprocesses in Asyncroscopy. It is used by `startup_scripts/run_servers.py`, `startup_scripts/run_mcp.py`, and `startup_scripts/run_llm.py`.

## Responsibilities

- **Lifecycle Management:** Orchestrates starting, tracking, and stopping child processes.
- **Cleanup:** Ensures all processes are terminated cleanly on exit (even if the parent crashes).
- **State Tracking:** Records active PIDs in a state file within `.processes/` to detect and kill stale processes from previous crashed sessions during startup.
- **Cross-Platform Support:** Handles differences in process management between Windows and POSIX systems.

## Architecture

### `ManagedProcess`

A dataclass representing a single child subprocess. It contains identifiers for the process, the underlying `subprocess.Popen` handle, and buffers to store the most recent output lines. 

### `ProcessManager`

The controller orchestrating multiple `ManagedProcess` instances. It should be initialized with standard context manager syntax (e.g. `with ProcessManager() as manager`). Includes:

- **`start_process(...)`**: Launches a process and tracks the process handle
- **`stop_process(managed)`**: Attempts a graceful termination (`SIGTERM`) followed by a forced kill (`SIGKILL`) if the process exceeds the configured `timeout`.
- **`shutdown_all()`**: Concurrently initiates shutdown of all tracked processes, following the `timeout` period before escalating to force kills.

## Example Usage in Launcher Scripts

`run_servers.py`:

1. **Initialization**: `ProcessManager` is instantiated, clearing any stale processes found in `.processes/`.
2. **Spawning**: It iterates through the configuration (devices, tiled, instrument) and calls `start_process` for each service.
3. **Execution**: The manager maintains the `active_processes` list.
4. **Shutdown**: Upon exit (or `Ctrl+C`), the manager's context manager (`__exit__`) invokes `shutdown_all()` to ensure a clean state for the next run.