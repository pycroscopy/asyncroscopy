# Running Servers in a GUI

## How to Run the GUI

```bash
uv run scripts/run_server_gui.py
```

Users can then click on the config file to select a microscope to run the servers for. It is initially set to digital twin to prevent any mistakes with activating servers for the wrong device. 

## Current verified, working OS
- Windows
- MacOS

## Issues
- Database server script does not exist 
- Starting solely the mcp server fails to connect most likely due to a lack of a delay 
