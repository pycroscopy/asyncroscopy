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
    from langchain.agents import create_agent
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from langgraph.graph import END, START, StateGraph
except ImportError:
    print("Missing dependencies! Please run:")
    print("uv sync --extra agent")
    sys.exit(1)


@dataclass
class Agent:
    """Represents a single AI agent in the swarm."""
    name: str
    system_prompt: str
    tools: list[str]  # List of tool names, supporting glob patterns (e.g., ["math_*", "read_file"])
    model: str | None = None
    description: str = ""


class AgentState(TypedDict):
    """State dictionary for each Agent node in the swarm graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str


class LLM(Device):
    mcp_url = device_property(dtype=str, default_value="http://127.0.0.1:8000/mcp")
    startup_agents = device_property(dtype=(str,), default_value=())
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
        self._mcp_clients: list[MultiServerMCPClient] = []

        if self.startup_agents:
            self._agents = [Agent(**json.loads(agent_json)) for agent_json in self.startup_agents]
            print(f"[SYSTEM]: Loaded startup agents: {self._agents}")

        # Setup asyncio event loop
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            if not self.use_init_chat_model or self.model_provider == "ollama":
                self.ensure_ollama_running()

            if self.use_init_chat_model: # Initialize from most model providers (e.g., OpenAI)
                self.info_stream("Initializing via init_chat_model")
                self._model = init_chat_model(
                    model=self.ollama_model,
                    model_provider=self.model_provider,
                    temperature=0
                )
            else: # Initialize locally via Ollama
                from langchain_ollama import ChatOllama
                self.info_stream("Initializing via ChatOllama")
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

            # Connect to an MCP server initially if specified
            if self.mcp_url:
                config = json.dumps({"url": self.mcp_url, "transport": "streamable_http"})
                if not self.ConnectMCP(config):
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

    @attribute(dtype=(str,), max_dim_x=100)
    def agents(self) -> list[str]:
        """Return a list of the names of all currently spawned agents."""
        return [agent.name for agent in self._agents]

    def ensure_ollama_running(self, host: str = "http://localhost:11434", timeout: int = 10) -> None:
        """Check if Ollama server is running, starting it if necessary."""
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
        """Query the agent swarm with a prompt, returning the final response."""
        self.set_state(tango.DevState.RUNNING)
        try:
            return self._loop.run_until_complete(self._run_swarm(prompt))
        except Exception as e:
            print(f"\n[CRITICAL ERROR]: {e}")
            return str(e)
        finally:
            self.set_state(tango.DevState.ON)

    @command(
        dtype_in=str,
        doc_in="JSON config of the MCP server: {'url': '...', 'transport': '...'}",
        dtype_out=bool,
        doc_out="Success status"
    )
    def ConnectMCP(self, config: str) -> bool:
        """Connect to an MCP server and inherit its tools. Returns true for success."""
        try:
            args = json.loads(config)
            url = args.get("url")
            transport = args.get("transport", "streamable_http")

            async def connect_and_fetch_tools():
                server_id = f"server_{len(self._mcp_clients)}"
                client = MultiServerMCPClient({server_id: {"url": url, "transport": transport}})
                tools = await client.get_tools()
                return client, tools

            print(f"\n[SYSTEM]: Connecting to MCP Server at {url}...")
            client, tools = self._loop.run_until_complete(connect_and_fetch_tools())

            self._mcp_clients.append(client)
            self._tools.extend(tools)
            print(f"[SYSTEM]: Connected. Inherited {len(tools)} tools.")
        except Exception as e:
            self.error_stream(f"Failed to connect to MCP server: {e}")
            return False

        return True

    @command(
        dtype_in=str,
        doc_in="JSON config of the Agent: {'name': '...', 'system_prompt': '...', 'model': '...', 'tools': ['*']}",
        dtype_out=bool,
        doc_out="Success status"
    )
    def SpawnAgent(self, config: str) -> bool:
        """Creates a new agent in the swarm."""
        try:
            args = json.loads(config)
            agent = Agent(
                name=args["name"],
                system_prompt=args["system_prompt"],
                model=args.get("model", self.ollama_model),
                tools=args.get("tools", ["*"]),
                description=args.get("description", "")
            )
            self._agents.append(agent)
            print(f"\n[SYSTEM]: Successfully spawned agent '{agent.name}'")
            return True
        except Exception as e:
            self.error_stream(f"Failed to spawn agent: {e}")
            return False

    def _get_agent_tools(self, allowed_patterns: list[str]) -> list:
        """Returnes a list of filtered tools based on the allowed glob patterns."""
        if "*" in allowed_patterns:
            return self._tools
        return [t for t in self._tools if any(fnmatch.fnmatch(t.name, pat) for pat in allowed_patterns)]

    def _build_agent_executor(self, agent: Agent):
        """Filter this agent's tools and construct its ReAct executor."""
        agent_tools = self._get_agent_tools(agent.tools)
        print(f"[SYSTEM]: Binding {len(agent_tools)} tools to {agent.name}")
        return create_agent(model=self._model, tools=agent_tools, system_prompt=agent.system_prompt)

    def _extract_json(self, text: str) -> str:
        """Strip markdown code fences (```json ... ``` or ``` ... ```) if present."""
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()

    def _parse_routing_decision(self, content: str, valid_options: list[str], fallback: str) -> str:
        """Parse a supervisor response's {'next': ...} decision, falling back on any error or invalid value."""
        try:
            decision = json.loads(self._extract_json(content))
            next_agent = decision.get("next", fallback)
            return next_agent if next_agent in valid_options else fallback
        except Exception as e:
            print(f"[SUPERVISOR ERROR]: {e}")
            return fallback

    async def _stream_agent(self, agent_executor, messages, agent_label: str = "") -> str:
        """Run a create_agent executor while streaming tokens and tool calls to stdout."""
        prefix = f"[{agent_label}] " if agent_label else ""
        start_time = time.time()
        first_token_received = False
        final_content = ""

        async for event in agent_executor.astream_events({"messages": messages}, version="v2"):
            kind = event["event"]

            if kind == "on_chat_model_start":
                # A new generation round is starting (could be a tool-call round or the final answer)
                start_time = time.time()
                first_token_received = False

            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if not first_token_received:
                    ttft = time.time() - start_time
                    print(f"\n{prefix}[DIAGNOSTIC]: Time to first token: {ttft:.2f}s")
                    print(f"{prefix}[GENERATION]: ", end="")
                    first_token_received = True
                if chunk.content:
                    print(chunk.content, end="")
                    sys.stdout.flush()

            elif kind == "on_chat_model_end":
                output = event["data"]["output"]
                tool_calls = getattr(output, "tool_calls", None) or []
                if not tool_calls:
                    # This round produced no tool calls -> it's the final answer
                    final_content = (output.content or "").strip()
                print()  # newline after this round's streamed text

            elif kind == "on_tool_start":
                tool_name = event["name"]
                tool_input = event["data"].get("input")
                print(f"{prefix}[EXECUTING TOOL]: {tool_name}({tool_input})")

            elif kind == "on_tool_end":
                output = event["data"].get("output")
                print(f"{prefix}[TOOL RESULT]: {output}")

        if final_content:
            print(f"{prefix}[FINAL ANSWER RETURNED]:\n{final_content}\n{'=' * 50}")
        return final_content

    async def _run_swarm(self, prompt: str) -> str:
        """Run the agent swarm with a given prompt, returning the final response."""
        if not self._agents:
            return "Swarm Error: No agents available. Please use the SpawnAgent command to create at least one worker before querying."

        # Fast path: no need for supervisor/routing overhead with a single agent
        if len(self._agents) == 1:
            agent = self._agents[0]
            agent_executor = self._build_agent_executor(agent)
            print(f"\n[{agent.name}] is working...")
            return await self._stream_agent(
                agent_executor, [HumanMessage(content=prompt)], agent_label=agent.name
            )

        builder = StateGraph(AgentState)
        agent_names = [a.name for a in self._agents]
        options = agent_names + ["FINISH"]

        # Creates a ReAct sub-graph for each Agent
        def create_agent_node(agent: Agent):
            agent_executor = self._build_agent_executor(agent)

            async def node(state: AgentState):
                print(f"\n[{agent.name}] is working...")
                content = await self._stream_agent(agent_executor, state["messages"], agent_label=agent.name)
                print(f"[{agent.name}] finished.\n")
                return {
                    "messages": [
                        HumanMessage(content=f"[{agent.name}]: {content}", name=agent.name)
                    ]
                }
            return node

        # Register workers
        for agent in self._agents:
            builder.add_node(agent.name, create_agent_node(agent))

        agent_roster = "\n".join(
            f"- {a.name}: {a.description or a.system_prompt}" for a in self._agents
        )

        async def supervisor_node(state: AgentState):
            print("\n[Supervisor] Evaluating routing...")

            # Check if agent has contributed if there's another AI/Human message beyond the original user prompt
            has_delegated = len(state["messages"]) > 1

            if not has_delegated:
                # First turn forces subagent routing
                instructions = (
                    f"Below are the available agents and what each is for:\n{agent_roster}\n\n"
                    "Based on the conversation, decide which agent should act next to progress the user's request.\n"
                    "Only output FINISH if the user's request has been fully and concretely answered — "
                    "not if an agent asked a question or said it couldn't complete the task; in that case, "
                    "route to a different agent who might be able to help instead."
                )
                valid_options, fallback = agent_names, agent_names[0]
            else:
                # Later turns either do normal routing or FINISH
                instructions = (
                    f"Active agents: {agent_names}.\n"
                    "Based on the conversation, decide who should act next.\n"
                    "If the user's request is fully resolved, output FINISH."
                )
                valid_options, fallback = options, "FINISH"

            sys_prompt = SystemMessage(
                content=(
                    f"You are the Swarm Supervisor. {instructions}\n"
                    f"Respond with JSON containing a single key 'next' mapping to one of: {options}"
                )
            )
            response = await self._model.ainvoke([sys_prompt] + state["messages"])
            next_agent = self._parse_routing_decision(response.content, valid_options, fallback)
            if next_agent == "FINISH":
                print("[Supervisor] Decision: FINISH\n")
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
        print(f"\n{'='*50}\n[NEW REQUEST]: {prompt}\n{'=' * 50}")

        last_response = None
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=prompt)]},
            config={"recursion_limit": self._max_steps}
        ):
            for node_name, state_update in chunk.items():
                if node_name != "Supervisor" and "messages" in state_update:
                    msg = state_update["messages"][-1]
                    last_response = msg.content

        return last_response if last_response is not None else "Swarm Error: No agent produced a response before routing finished."

# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    LLM.run_server()