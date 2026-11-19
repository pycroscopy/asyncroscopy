__all__ = ["MCPServer", "AgentSwarm", "AgentMCPServer"]

def __getattr__(name):
    if name == "MCPServer":
        from .mcp_server import MCPServer
        return MCPServer
    if name == "AgentSwarm":
        from .agent import AgentSwarm
        return AgentSwarm
    if name == "AgentMCPServer":
        from .agent_mcp_server import AgentMCPServer
        return AgentMCPServer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")