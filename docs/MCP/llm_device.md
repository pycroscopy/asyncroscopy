# LLM Device

The `LLM` device is a Tango device that wraps an LangChain AI agent swarm. It enables AI agents to interact with the hardware tools exposed via the MCP server. It is started with `startup_scripts/run_llm.py`.

## Overview

The device initializes a LangChain/LangGraph-based swarm. It exposes Tango Commands to connect to multiple MCP servers, spawn specialized worker agents, and take queries. A central Supervisor agent routes tasks to the appropriate worker based on user input.

This Device may be initialized with a local model, via Ollama, or more generally via Langchain's `init_chat_model` method that takes a model and provider, and optionally other args/kwargs like an API key. With Ollama, the model is automatically served if the server isn't already running and optionally pulled if not already downloaed.

## Commands

- **`SpawnAgent(config: str)`**
  Creates a new worker agent. Expects a JSON-serialized string:
  ```json
  {
    "name": "agent_name",
    "system_prompt": "Agent role and instructions.",
    "description": "An optional description to help the Supervisor with routing",
    "model": "optional_model_override",
    "tools": ["glob_pattern_1", "glob_pattern_2"]
  }
  ```
  Glob patterns can be used for giving agents multiple related tools, such as "*image" to give it all tools whose names end in the word "image".

- **`ConnectMCP(config: str)`**
  Connects to an MCP server and inherits its tools. Expects a JSON string:
  ```json
  {
    "url": "http://127.0.0.1:8000/mcp",
    "transport": "streamable_http"
  }
  ```

- **`Query(prompt: str)`**
  Queries the swarm. If more than one agent is in the swarm, a Supervisor routes this prompt to the appropriate agent. Otherwise, the single agent handles the request like a normal chatbot.
