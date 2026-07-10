__all__ = ["MCPServer"]


def __getattr__(name):
    if name == "MCPServer":
        from .mcp_server import MCPServer

        return MCPServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
