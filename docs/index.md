# Asyncroscopy Documentation
:::::{grid} 2
:gutter: 0

::::{grid-item}
:::{image} ./images/architecturev1.png
:height: 200px
:::
::::

::::{grid-item}
:::{image} ./images/architecturev2.png
:height: 200px
:::
::::

:::::
---
Welcome to the Asyncroscopy documentation site.
Use this site to navigate contributor guidance, microscope architecture notes, hardware extension docs, MCP server references, and upcoming changes.

## Start Here

- [Contributing Guide](dev_guide.md): project engineering principles and pull request expectations.
- [Base Electron Microscope Extension Notes](Microscopy/modify_base_electron_microscope.md): where to add or change core microscope behavior.
- [Thermo Microscope Extension Notes](Microscopy/modify_auto_script_microscope.md): detector integration and orchestration guidance.

## Hardware and Integrations

- [Add a Detector](Adding_New_Hardware/add_detector.md): detector onboarding checklist and implementation notes.
- [Data Integration (Tiled)](Tiled_server/data_integration.md): how acquisitions are saved, registered, and served via the DATA device and Tiled.
- [MCP Server Documentation](MCP/mcp_server.md): how Tango commands are exposed to MCP-compatible agents.

## Operation

- [Starting Servers - CLI](Operation/run-servers.md)
- [Starting Servers - GUI](Operation/run-server-gui.md)

## Roadmap

- [Upcoming Changes](upcoming_changes.md): planned areas and deferred work.
