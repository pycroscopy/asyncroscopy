"""Tango device wrapping an Ollama/LangChain AI agent that connects to an MCP server."""

import operator
from typing import Annotated, Sequence, TypedDict
from dataclasses import dataclass

import sys
import asyncio
import signal
import time
import subprocess

import urllib.request
import urllib.error

import fnmatch
import json
import re

import tango
from tango.server import Device, attribute, command, device_property

try:
    from langchain.chat_models import init_chat_model
    from langchain_core.tools import BaseTool
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
    from langchain_mcp_adapters.client import MultiServerMCPClient
    
    from langgraph.graph import END, START, StateGraph
    from langchain.agents import create_agent
except ImportError:
    print("Missing dependencies! Please run:")
    print("uv sync --extra agent")
    sys.exit(1)


@dataclass
class Agent:
    name: str
    system_prompt: str
    model: str
    tools: list[str]  # List of tool names, supporting glob patterns (e.g., ["math_*", "read_file"])
    description: str = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str


class LLM(Device):
    mcp_url = device_property(dtype=str, default_value="http://127.0.0.1:8000/mcp")
    ollama_model = device_property(dtype=str, default_value="gemma4:31b")
    use_init_chat_model = device_property(dtype=bool, default_value=False)
    model_provider = device_property(dtype=str, default_value="ollama")

    max_steps = attribute(label="Max Steps", dtype=int, access=tango.AttrWriteType.READ_WRITE)

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(tango.DevState.INIT)
        self._max_steps = 5

        # Registries
        self._agents: list[Agent] = []
        self._tools: list[BaseTool] = []
        self._mcp_clients: list[MultiServerMCPClient] = [] # Prevents client GC and connection dropping

        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            if not self.use_init_chat_model or self.model_provider == "ollama":
                self.ensure_ollama_running()

            if self.use_init_chat_model:
                self.info_stream(f"Initializing via init_chat_model")
                self._model = init_chat_model(
                    model=self.ollama_model,
                    model_provider=self.model_provider,
                    temperature=0
                )
            else:
                from langchain_ollama import ChatOllama
                self.info_stream(f"Initializing via ChatOllama")
                self._model = ChatOllama(
                    model=self.ollama_model,
                    temperature=0,
                    reasoning=False,
                )
            
            print("\n[SYSTEM]: Pre-warming model into VRAM (Cold Start)...")
            sys.stdout.flush()
            start_warmup = time.time()
            self._loop.run_until_complete(self._model.ainvoke([HumanMessage(content=" ")]))
            print(f"[SYSTEM]: Model pre-warmed in {time.time() - start_warmup:.2f}s!")

            if self.mcp_url:
                if not self.ConnectMCP([self.mcp_url, "streamable_http"]):
                    print(f"[SYSTEM]: Failed to connect to MCP Server at {self.mcp_url}.")
            
            self.set_state(tango.DevState.ON)
        except Exception as e:
            self.set_state(tango.DevState.FAULT)
            self.set_status(f"Initialization failed: {e}")
            self.error_stream(f"Failed to start: {e}")

    def delete_device(self) -> None:
        """Clean shutdown hook called when Tango server stops."""
        try:
            if hasattr(self, "_loop") and not self._loop.is_closed():
                self._loop.close()
        except Exception:
            pass
        super().delete_device()

    def read_max_steps(self) -> int:
        return self._max_steps

    def write_max_steps(self, value: int) -> None:
        if value < 1:
            raise ValueError("max_steps must be at least 1.")
        self._max_steps = value

    def ensure_ollama_running(self, host: str = "http://localhost:11434", timeout: int = 10) -> None:
        tags_url = f"{host.rstrip('/')}/api/tags"
        try:
            with urllib.request.urlopen(tags_url, timeout=1):
                return
        except (urllib.error.URLError, TimeoutError, ConnectionRefusedError):
            pass

        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True 
            )
        except FileNotFoundError:
            raise RuntimeError("Ollama binary not found on PATH.")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with urllib.request.urlopen(tags_url, timeout=1):
                    return
            except (urllib.error.URLError, TimeoutError, ConnectionRefusedError):
                time.sleep(0.5)

        raise RuntimeError(f"Ollama endpoint '{tags_url}' did not respond.")

    @command(dtype_in=str, dtype_out=str)
    def Query(self, prompt: str) -> str:
        self.set_state(tango.DevState.RUNNING)
        try:
            return self._loop.run_until_complete(self._run_swarm(prompt))
        except Exception as e:
            print(f"\n[CRITICAL ERROR]: {e}")
            return str(e)
        finally:
            self.set_state(tango.DevState.ON)

    @command(
        dtype_in=[str],
        doc_in="List of strings: [url, transport_type]",
        dtype_out=bool,
        doc_out="Success status"
    )
    def ConnectMCP(self, args: list[str]) -> bool:
        if len(args) < 1:
            raise ValueError("ConnectMCP requires at least one argument: [url, transport_type]")
        elif len(args) == 1: # No transport type provided, default to streamable_http
            args.append("streamable_http")
        elif len(args) >= 2:
            if args[1] not in ["streamable_http", "websocket"]:
                raise ValueError("Invalid transport type. Must be 'streamable_http' or 'websocket'.")
        
        async def connect_and_fetch_tools():
            server_id = f"server_{len(self._mcp_clients)}"
            client = MultiServerMCPClient({
                server_id: {"url": args[0], "transport": args[1]}
            })
            tools = await client.get_tools()
            return client, tools

        print(f"\n[SYSTEM]: Connecting to MCP Server at {args[0]}...")
        client, tools = self._loop.run_until_complete(connect_and_fetch_tools())
        
        self._mcp_clients.append(client)
        self._tools.extend(tools)
        print(f"[SYSTEM]: Connected. Inherited {len(tools)} tools.")
        
        return True

    @command(
        dtype_in=str,
        doc_in="JSON configuration of the Agent: {'name': '...', 'system_prompt': '...', 'model': '...', 'tools': ['*']}",
        dtype_out=bool,
        doc_out="Success status"
    )
    def SpawnAgent(self, agent_config_json: str) -> bool:
        """Dynamically creates a new AI worker node in the swarm."""
        try:
            config = json.loads(agent_config_json)
            agent = Agent(
                name=config["name"],
                system_prompt=config["system_prompt"],
                model=config.get("model", self.ollama_model),
                tools=config.get("tools", ["*"]),
                description=config.get("description", "")
            )
            self._agents.append(agent)
            print(f"\n[SYSTEM]: Successfully spawned agent '{agent.name}'")
            return True
        except Exception as e:
            self.error_stream(f"Failed to spawn agent: {e}")
            raise RuntimeError(f"SpawnAgent Error: {e}")

    # Helper to filter tools via glob patterns
    def _get_agent_tools(self, allowed_patterns: list[str]) -> list:
        if "*" in allowed_patterns: return self._tools
        filtered = []
        for t in self._tools:
            if any(fnmatch.fnmatch(t.name, pat) for pat in allowed_patterns):
                filtered.append(t)
        return filtered

    def _extract_json(self, text: str) -> str:
        """Strip markdown code fences (```json ... ``` or ``` ... ```) if present."""
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()

    async def _run_swarm(self, prompt: str) -> str:
        if not self._agents:
            return "Swarm Error: No agents available. Please use the SpawnAgent command to create at least one worker before querying."

        # Fast path: no need for supervisor/routing overhead with a single agent
        if len(self._agents) == 1:
            agent = self._agents[0]
            agent_tools = self._get_agent_tools(agent.tools)  # pull filter logic into a helper
            print(f"[SYSTEM]: Binding {len(agent_tools)} tools to {agent.name}")
            agent_executor = create_agent(
                model=self._model,
                tools=agent_tools,
                system_prompt=agent.system_prompt
            )
            print(f"\n[{agent.name}] is working...")
            result = await agent_executor.ainvoke({"messages": [HumanMessage(content=prompt)]})
            return result["messages"][-1].content


        builder = StateGraph(AgentState)
        agent_names = [a.name for a in self._agents]

        # Creates a ReAct sub-graph for each Agent
        def create_agent_node(agent: Agent):
            agent_tools = self._get_agent_tools(agent.tools)
            print(f"[SYSTEM]: Binding {len(agent_tools)} tools to {agent.name}")
            
            agent_executor = create_agent(
                model=self._model, 
                tools=agent_tools, 
                system_prompt=agent.system_prompt
            )

            async def node(state: AgentState):
                print(f"\n[{agent.name}] is working...")
                # Pass the conversation history into the agent's internal loop
                result = await agent_executor.ainvoke({"messages": state["messages"]})
                
                # Extract the final answer from this agent
                last_msg = result["messages"][-1]
                
                # We return it as a HumanMessage so the Supervisor reads it as standard text 
                # rather than seeing messy internal tool-call logs. This saves context window
                return {
                    "messages": [
                        HumanMessage(content=f"[{agent.name}]: {last_msg.content}", name=agent.name)
                    ]
                }
            return node

        # Register workers
        for agent in self._agents:
            builder.add_node(agent.name, create_agent_node(agent))

        # Supervisor Node
        options = agent_names + ["FINISH"]
        
        async def supervisor_node(state: AgentState):
            print("\n[Supervisor] Evaluating routing...")

            # An agent has "contributed" if there's an AI/Human message beyond the original user prompt
            has_delegated = len(state["messages"]) > 1

            if not has_delegated:
                # First turn: force a delegation, don't even ask the model whether to finish
                available = [n for n in agent_names]
                agent_roster = "\n".join(
                    f"- {a.name}: {a.description or a.system_prompt}" for a in self._agents
                )

                sys_prompt = SystemMessage(
                    content=(
                        "You are the Swarm Supervisor. Below are the available agents and what each is for:\n"
                        f"{agent_roster}\n\n"
                        "Based on the conversation, decide which agent should act next to progress the user's request.\n"
                        "Only output FINISH if the user's request has been fully and concretely answered — "
                        "not if an agent asked a question or said it couldn't complete the task; in that case, "
                        "route to a different agent who might be able to help instead.\n"
                        f"Respond with JSON containing a single key 'next' mapping to one of: {options}"
                    )
                )               
                response = await self._model.ainvoke([sys_prompt] + state["messages"])
                try:    
                    decision = json.loads(self._extract_json(response.content))
                    next_agent = decision.get("next")
                    if next_agent not in available:
                        next_agent = available[0]  # fallback: just pick the first agent
                except Exception as e:
                    next_agent = available[0]
                    print(f"[SUPERVISOR ERROR]: {e}")
                return {"next_agent": next_agent}

            # Later turns: normal routing, FINISH is a valid choice
            sys_prompt = SystemMessage(
                content=(
                    f"You are the Swarm Supervisor. Active agents: {agent_names}.\n"
                    "Based on the conversation, decide who should act next.\n"
                    "If the user's request is fully resolved, output FINISH.\n"
                    f"Respond with JSON containing a single key 'next' mapping to one of: {options}"
                )
            )
            response = await self._model.ainvoke([sys_prompt] + state["messages"])
            try:
                decision = json.loads(self._extract_json(response.content))
                next_agent = decision.get("next", "FINISH")
                if next_agent not in options:
                    next_agent = "FINISH"
            except Exception:
                next_agent = "FINISH"
            return {"next_agent": next_agent}

        builder.add_node("Supervisor", supervisor_node)
        builder.add_edge(START, "Supervisor")

        for name in agent_names:
            builder.add_edge(name, "Supervisor")

        def route(state: AgentState):
            return "FINISH" if state["next_agent"] == "FINISH" else state["next_agent"]

        mapping = {name: name for name in agent_names}
        mapping["FINISH"] = END
        builder.add_conditional_edges("Supervisor", route, mapping)

        graph = builder.compile()

        # Graph execution loop
        print(f"\n{'='*50}\n[NEW REQUEST]: {prompt}\n{'='*50}")

        last_response = None
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=prompt)]},
            config={"recursion_limit": self._max_steps}
        ):
            for node_name, state_update in chunk.items():
                if node_name != "Supervisor" and "messages" in state_update:
                    msg = state_update["messages"][-1]
                    print(f"{msg.content}")
                    last_response = msg.content

        return last_response if last_response is not None else "Swarm Error: No agent produced a response before routing finished."

if __name__ == "__main__":
    LLM.run_server()