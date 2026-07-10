# Troubleshooting

Symptom-first answers to problems hit during live deployment. Most startup
issues come down to one of three things: a **stale server still running**, a
**stale `.db` file**, or a **slow network**. See
[run-servers.md](run-servers.md) for normal startup.

### Newly acquired data can't be found in Tiled, even though the file was written to disk

A stale Tiled server is still running and serving an **old** save directory while
new data is written somewhere else. Only one Tiled server should be active, bound
to the current save path.

Find and kill the stale Tiled process (default port `9091`), then restart it on
the right path:

```powershell
netstat -ano | findstr :9091
taskkill /PID <PID> /F
```

The cleaner fix is to let `run_servers.py` manage Tiled — it starts it as a
tracked process bound to the save path you enter, and stops it on Ctrl+C.
Changing `data.save_path` on the DATA device also restarts the managed Tiled
server on the new path, so the server and the save path can't drift apart.

### Tiled (or the whole stack) fails to start after pulling new changes

Delete the stale `.db` files — both the Tiled catalog db and the Tango database
db — then start fresh:

```bash
uv run startup_scripts/run_servers.py
```

A clean run rebuilds both databases.

### Startup gives a weird error or an unexplained timeout

First suspect: **old servers are still running.** On the Windows microscope PC
the PID changes constantly, so killing by PID is unreliable.

- Re-run `run_servers.py` and answer **yes** to *Clear old processes first*.
- To kill one specific Tango device server by name instead:

  ```python
  import tango
  db = tango.Database()
  print(db.get_server_list("CORRECTOR/*"))
  d = tango.DeviceProxy("dserver/CORRECTOR/corrector_instance")
  d.command_inout("Kill")
  ```

### Database won't start, or "database did not become ready"

`tango.Database()` only **connects** to an existing database — it does not start
one. The Tango database server must be started first (`run_servers.py` does this
for you), or manually:

```bash
TANGO_HOST=localhost:9094 uv run python -m tango.databaseds.database 2
```

If the database server still won't come up within ~2 minutes, enable
`tango.reset_database_file: true` in the server YAML or check **Delete
tango_database.db before start** in the startup GUI, then retry.

### Servers stop responding (e.g. after repeated `place_beam` calls)

Reported recovery: kill the server notebook kernel, enable the Tango database
file reset option, then start the servers again.

### Unexplained timeouts when connecting from a remote location

Usually slow or unstable internet rather than a code fault. It has shown up
alongside an unusably laggy Teams call, and once after a power flicker /
thunderstorm — both times from a remote connection.

- Just retry — re-run the cell or command.
- Or remote into the microscope PC with TeamViewer and use VS Code there, so the
  connection to the servers stays local to that machine.

### `pandas` version mismatch(observed - while using PyTemLib)

Recreate the environment:

```bash
# delete .venv, then:
uv sync
```
