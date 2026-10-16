"""Tests for the Tango-free agent swarm, with the LangChain stack stubbed."""

import asyncio
import json
import sys
import types

from unittest.mock import AsyncMock, MagicMock

import pytest

def setup_llm_stubs():
    """Stub every import in agent.py's try block so the module loads without the real agent deps."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if "langgraph.graph" in sys.modules:
        return

    base_msg_cls = type("BaseMessage", (), {})
    human_msg_cls = type("HumanMessage", (base_msg_cls,), {
        "__init__": lambda self, content, name=None: (
            setattr(self, "content", content) or setattr(self, "name", name)
        ),
    })
    system_msg_cls = type("SystemMessage", (base_msg_cls,), {
        "__init__": lambda self, content: setattr(self, "content", content),
    })
    ai_msg_cls = type("AIMessage", (base_msg_cls,), {
        "__init__": lambda self, content, tool_calls=None: (
            setattr(self, "content", content) or setattr(self, "tool_calls", tool_calls or [])
        ),
    })
    tool_msg_cls = type("ToolMessage", (base_msg_cls,), {
        "__init__": lambda self, content, tool_call_id=None: (
            setattr(self, "content", content) or setattr(self, "tool_call_id", tool_call_id)
        ),
    })

    langchain_core = types.ModuleType("langchain_core")
    lc_tools = types.ModuleType("langchain_core.tools")
    lc_tools.BaseTool = type("BaseTool", (), {})
    lc_messages = types.ModuleType("langchain_core.messages")
    lc_messages.BaseMessage = base_msg_cls
    lc_messages.HumanMessage = human_msg_cls
    lc_messages.SystemMessage = system_msg_cls
    lc_messages.AIMessage = ai_msg_cls
    lc_messages.ToolMessage = tool_msg_cls
    langchain_core.tools = lc_tools
    langchain_core.messages = lc_messages

    langchain = types.ModuleType("langchain")
    lc_cm = types.ModuleType("langchain.chat_models")
    lc_cm.init_chat_model = MagicMock()
    langchain.chat_models = lc_cm
    
    lc_agents = types.ModuleType("langchain.agents")
    lc_agents.create_agent = MagicMock()
    langchain.agents = lc_agents

    lc_mcp = types.ModuleType("langchain_mcp_adapters")
    lc_mcp_client = types.ModuleType("langchain_mcp_adapters.client")
    lc_mcp_client.MultiServerMCPClient = MagicMock()
    lc_mcp.client = lc_mcp_client

    lg = types.ModuleType("langgraph")
    lg_graph = types.ModuleType("langgraph.graph")
    lg_graph.END = "__end__"
    lg_graph.START = "__start__"
    lg_graph.StateGraph = MagicMock()
    lg.graph = lg_graph

    sys.modules.update({
        "langchain_core": langchain_core,
        "langchain_core.tools": lc_tools,
        "langchain_core.messages": lc_messages,
        "langchain": langchain,
        "langchain.chat_models": langchain.chat_models,
        "langchain.agents": langchain.agents,
        "langchain_mcp_adapters": lc_mcp,
        "langchain_mcp_adapters.client": lc_mcp_client,
        "langgraph": lg,
        "langgraph.graph": lg_graph,
    })


setup_llm_stubs()

from asyncroscopy.mcp.agent import Agent, AgentSwarm
from asyncroscopy.mcp.agent_mcp_server import AgentMCPServer
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_agent_swarm(**kwargs) -> AgentSwarm:
    """Return a bare AgentSwarm with its model replaced by a mock."""
    swarm = AgentSwarm(
        max_steps=kwargs.get("max_steps", 5),
        startup_agents=kwargs.get("agents", []),
        model="mock-model",
        verbose=False,
    )
    swarm._tools = kwargs.get("tools", [])
    swarm._model = kwargs.get("model", None)
    swarm._mcp_clients = []
    return swarm


def _make_tool(name: str) -> MagicMock:
    t = MagicMock()
    t.name = name
    return t


def _make_agent(name="worker", system_prompt="You are helpful.", tools=None) -> Agent:
    return Agent(name=name, system_prompt=system_prompt, tools=tools or ["*"])


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

class TestGetAgentTools:
    def test_wildcard_returns_all(self):
        tools = [_make_tool("math_add"), _make_tool("read_file"), _make_tool("write_file")]
        swarm = _make_agent_swarm(tools=tools)
        assert swarm._get_agent_tools(["*"]) is tools

    def test_exact_name_match(self):
        t_read = _make_tool("read_file")
        t_write = _make_tool("write_file")
        swarm = _make_agent_swarm(tools=[t_read, t_write])
        result = swarm._get_agent_tools(["read_file"])
        assert result == [t_read]

    def test_glob_prefix(self):
        tools = [_make_tool("math_add"), _make_tool("math_sub"), _make_tool("read_file")]
        swarm = _make_agent_swarm(tools=tools)
        result = swarm._get_agent_tools(["math_*"])
        assert len(result) == 2
        assert all(t.name.startswith("math_") for t in result)

    def test_multiple_patterns(self):
        tools = [_make_tool("math_add"), _make_tool("read_file"), _make_tool("write_file")]
        swarm = _make_agent_swarm(tools=tools)
        result = swarm._get_agent_tools(["math_*", "read_file"])
        assert len(result) == 2

    def test_no_match_returns_empty(self):
        swarm = _make_agent_swarm(tools=[_make_tool("math_add")])
        assert swarm._get_agent_tools(["nonexistent"]) == []

    def test_empty_tool_list(self):
        swarm = _make_agent_swarm(tools=[])
        assert swarm._get_agent_tools(["*"]) == []


class TestExtractJson:
    def test_bare_json_passthrough(self):
        swarm = _make_agent_swarm()
        raw = '{"next": "worker", "task": "do something"}'
        assert swarm._extract_json(raw) == raw

    def test_strips_json_code_fence(self):
        swarm = _make_agent_swarm()
        fenced = '```json\n{"next": "worker"}\n```'
        assert swarm._extract_json(fenced) == '{"next": "worker"}'

    def test_strips_plain_code_fence(self):
        swarm = _make_agent_swarm()
        fenced = '```\n{"next": "FINISH"}\n```'
        assert swarm._extract_json(fenced) == '{"next": "FINISH"}'

    def test_whitespace_stripped(self):
        swarm = _make_agent_swarm()
        assert swarm._extract_json('  {"a": 1}  ') == '{"a": 1}'

    def test_non_json_text_passthrough(self):
        swarm = _make_agent_swarm()
        text = "Just some plain text"
        assert swarm._extract_json(text) == text


class TestParseRoutingDecision:
    def test_valid_next_returned(self):
        swarm = _make_agent_swarm()
        content = '{"next": "worker", "task": "scan the sample"}'
        next_agent, subtask = swarm._parse_routing_decision(content, ["worker", "FINISH"], "FINISH")
        assert next_agent == "worker"
        assert subtask == "scan the sample"

    def test_invalid_next_falls_back(self):
        swarm = _make_agent_swarm()
        content = '{"next": "nonexistent_agent", "task": "whatever"}'
        next_agent, _ = swarm._parse_routing_decision(content, ["worker", "FINISH"], "FINISH")
        assert next_agent == "FINISH"

    def test_missing_next_key_falls_back(self):
        swarm = _make_agent_swarm()
        content = '{"task": "do something"}'
        # decision.get("next", fallback) returns fallback when key is absent
        next_agent, _ = swarm._parse_routing_decision(content, ["worker"], "worker")
        assert next_agent == "worker"

    def test_malformed_json_falls_back(self):
        swarm = _make_agent_swarm()
        next_agent, subtask = swarm._parse_routing_decision("{broken json!!}", ["worker"], "worker")
        assert next_agent == "worker"
        assert subtask == ""

    def test_fenced_json_parsed(self):
        swarm = _make_agent_swarm()
        content = '```json\n{"next": "FINISH", "task": ""}\n```'
        next_agent, _ = swarm._parse_routing_decision(content, ["worker", "FINISH"], "worker")
        assert next_agent == "FINISH"

    def test_missing_task_key_returns_empty_string(self):
        swarm = _make_agent_swarm()
        content = '{"next": "worker"}'
        _, subtask = swarm._parse_routing_decision(content, ["worker"], "worker")
        assert subtask == ""


class TestSpawnAgent:
    def test_spawn_adds_agent(self):
        swarm = _make_agent_swarm()
        config = json.dumps({"name": "Alpha", "system_prompt": "You scan.", "tools": ["scan_*"]})
        assert swarm.spawn_agent(config) == "Alpha"
        assert len(swarm._agents) == 1
        assert swarm._agents[0].name == "Alpha"
        assert swarm._agents[0].tools == ["scan_*"]

    def test_spawn_multiple_agents(self):
        swarm = _make_agent_swarm()
        for i in range(3):
            swarm.spawn_agent(json.dumps({"name": f"Agent{i}", "system_prompt": "help", "tools": ["*"]}))
        assert len(swarm._agents) == 3

    def test_spawn_defaults_tools_to_wildcard(self):
        swarm = _make_agent_swarm()
        swarm.spawn_agent(json.dumps({"name": "Beta", "system_prompt": "help"}))
        assert swarm._agents[0].tools == ["*"]

    def test_spawn_missing_required_field_fails(self):
        swarm = _make_agent_swarm()
        # "name" is required by Agent dataclass
        assert swarm.spawn_agent(json.dumps({"system_prompt": "missing name"})) == ""
        assert swarm._agents == []

    def test_spawn_invalid_json_fails(self):
        swarm = _make_agent_swarm()
        assert swarm.spawn_agent("{bad json") == ""
        assert swarm._agents == []

    def test_spawn_preserves_description(self):
        swarm = _make_agent_swarm()
        swarm.spawn_agent(json.dumps({"name": "Gamma", "system_prompt": "help", "description": "does science"}))
        assert swarm._agents[0].description == "does science"


class TestMaxSteps:
    def test_default(self):
        assert _make_agent_swarm().max_steps == 5

    def test_set_valid_returns_new_value(self):
        swarm = _make_agent_swarm()
        assert swarm.set_max_steps(10) == 10
        assert swarm.max_steps == 10

    def test_set_zero_raises(self):
        with pytest.raises(ValueError):
            _make_agent_swarm().set_max_steps(0)

    def test_set_negative_raises(self):
        with pytest.raises(ValueError):
            _make_agent_swarm().set_max_steps(-3)

    def test_set_one_is_valid(self):
        swarm = _make_agent_swarm()
        assert swarm.set_max_steps(1) == 1


class TestRunSwarm:
    def test_no_agents_returns_error_message(self):
        swarm = _make_agent_swarm()
        result = asyncio.run(swarm._run_swarm("hello"))
        assert "No agents available" in result

    def test_single_agent_calls_stream_agent(self):
        """Single-agent path skips the supervisor graph entirely."""
        agent = _make_agent(name="Solo")
        swarm = _make_agent_swarm(agents=[agent])

        # Replace internal helpers so no LangChain objects are needed
        swarm._build_agent_executor = MagicMock(return_value=MagicMock())
        swarm._stream_agent = AsyncMock(return_value="42 is the answer.")

        result = asyncio.run(swarm._run_swarm("What is the answer?"))

        assert result == "42 is the answer."
        swarm._build_agent_executor.assert_called_once_with(agent)
        swarm._stream_agent.assert_called_once()

    def test_single_agent_receives_prompt_in_message(self):
        """The prompt must be forwarded as the HumanMessage content."""
        agent = _make_agent()
        swarm = _make_agent_swarm(agents=[agent])
        swarm._build_agent_executor = MagicMock(return_value=MagicMock())

        captured_messages = []

        async def fake_stream(executor, messages, agent_label="", transcript=None):
            captured_messages.extend(messages)
            return "done"

        swarm._stream_agent = fake_stream

        asyncio.run(swarm._run_swarm("scan now"))
        assert len(captured_messages) == 1
        assert captured_messages[0].content == "scan now"


class TestOpenAIMessagesToLangchain:
    def test_user_message_becomes_human_message(self):
        [msg] = AgentSwarm._openai_messages_to_langchain([{"role": "user", "content": "hi"}])
        assert isinstance(msg, HumanMessage)
        assert msg.content == "hi"

    def test_system_message_becomes_system_message(self):
        [msg] = AgentSwarm._openai_messages_to_langchain([{"role": "system", "content": "be careful"}])
        assert isinstance(msg, SystemMessage)
        assert msg.content == "be careful"

    def test_assistant_message_without_tool_calls(self):
        [msg] = AgentSwarm._openai_messages_to_langchain([{"role": "assistant", "content": "done"}])
        assert isinstance(msg, AIMessage)
        assert msg.content == "done"
        assert msg.tool_calls == []

    def test_assistant_message_with_tool_calls_parses_arguments(self):
        [msg] = AgentSwarm._openai_messages_to_langchain([{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "acquire_image", "arguments": '{"detector": "haadf"}'},
            }],
        }])
        assert isinstance(msg, AIMessage)
        assert msg.tool_calls == [{"name": "acquire_image", "args": {"detector": "haadf"}, "id": "call_1"}]

    def test_tool_message_becomes_tool_message(self):
        [msg] = AgentSwarm._openai_messages_to_langchain([{
            "role": "tool", "tool_call_id": "call_1", "content": "stem_image_HAADF_x.h5",
        }])
        assert isinstance(msg, ToolMessage)
        assert msg.content == "stem_image_HAADF_x.h5"
        assert msg.tool_call_id == "call_1"

    def test_unknown_role_falls_back_to_human_message(self):
        [msg] = AgentSwarm._openai_messages_to_langchain([{"role": "weird", "content": "??"}])
        assert isinstance(msg, HumanMessage)

    def test_missing_content_defaults_to_empty_string(self):
        [msg] = AgentSwarm._openai_messages_to_langchain([{"role": "user"}])
        assert msg.content == ""


class TestLangchainMessageToOpenAI:
    def test_plain_text_message(self):
        message = AIMessage(content="hello there")
        result = AgentSwarm._langchain_message_to_openai(message)
        assert result == {"role": "assistant", "content": "hello there"}

    def test_message_with_tool_calls_encodes_arguments_as_json_string(self):
        message = AIMessage(content="", tool_calls=[{"name": "acquire_image", "args": {"n": 1}, "id": "call_9"}])
        result = AgentSwarm._langchain_message_to_openai(message)
        assert result["tool_calls"] == [{
            "id": "call_9",
            "type": "function",
            "function": {"name": "acquire_image", "arguments": '{"n": 1}'},
        }]

    def test_tool_call_missing_id_gets_a_fallback(self):
        message = AIMessage(content="", tool_calls=[{"name": "acquire_image", "args": {}, "id": ""}])
        result = AgentSwarm._langchain_message_to_openai(message)
        assert result["tool_calls"][0]["id"] == "call_0"


class TestComplete:
    def test_returns_tool_call_decision(self):
        swarm = _make_agent_swarm()
        response = AIMessage(content="", tool_calls=[{"name": "acquire_image", "args": {}, "id": "call_1"}])
        bound_model = AsyncMock()
        bound_model.ainvoke.return_value = response
        swarm._model = MagicMock()
        swarm._model.bind_tools.return_value = bound_model

        request = {
            "messages": [{"role": "user", "content": "acquire an image"}],
            "tools": [{"type": "function", "function": {"name": "acquire_image"}}],
        }
        result = json.loads(asyncio.run(swarm.complete(json.dumps(request))))

        assert result["message"]["tool_calls"][0]["function"]["name"] == "acquire_image"
        swarm._model.bind_tools.assert_called_once_with(request["tools"])

    def test_skips_bind_tools_when_no_tools_given(self):
        swarm = _make_agent_swarm()
        response = AIMessage(content="hi there")
        swarm._model = AsyncMock()
        swarm._model.ainvoke.return_value = response

        request = {"messages": [{"role": "user", "content": "hi"}]}
        result = json.loads(asyncio.run(swarm.complete(json.dumps(request))))

        assert result["message"]["content"] == "hi there"
        swarm._model.ainvoke.assert_called_once()

    def test_invalid_json_returns_error_payload(self):
        swarm = _make_agent_swarm()
        result = json.loads(asyncio.run(swarm.complete("not json")))
        assert "error" in result
        assert "message" in result["error"]

    def test_model_exception_returns_error_payload(self):
        swarm = _make_agent_swarm()
        swarm._model = AsyncMock()
        swarm._model.ainvoke.side_effect = RuntimeError("model unavailable")

        request = {"messages": [{"role": "user", "content": "hi"}]}
        result = json.loads(asyncio.run(swarm.complete(json.dumps(request))))

        assert result["error"]["message"] == "model unavailable"

class TestAgentIntrospection:
    def test_agents_lists_names(self):
        swarm = _make_agent_swarm(agents=[_make_agent(name="Alpha"), _make_agent(name="Beta")])
        assert swarm.agents == ["Alpha", "Beta"]

    def test_agents_empty_by_default(self):
        assert _make_agent_swarm().agents == []

    def test_tools_is_json_list_of_names(self):
        swarm = _make_agent_swarm(tools=[_make_tool("a_tool"), _make_tool("b_tool")])
        assert json.loads(swarm.tools) == [{"name": "a_tool"}, {"name": "b_tool"}]


class TestAgentMCPServerTools:
    """The swarm is served over MCP, so each Tango command needs a tool wrapper."""

    @staticmethod
    def _server(**kwargs) -> "AgentMCPServer":
        server = AgentMCPServer(
            name="test_agent",
            mcp_urls=[],
            startup_agents=[],
            verbose=False,
            **kwargs,
        )
        # Replace the model so no Ollama/LangChain call can be made by a test.
        server.swarm._model = MagicMock()
        return server

    def test_setup_registers_all_tools(self) -> None:
        server = self._server()
        registered = []
        server.mcp.add_tool = registered.append
        server.setup()

        assert [tool.__name__ for tool in registered] == [
            "query_agent",
            "spawn_agent",
            "list_agents",
            "list_agent_tools",
            "complete",
            "set_max_steps",
        ]

    def test_spawn_agent_tool_delegates_to_swarm(self) -> None:
        server = self._server()
        config = json.dumps({"name": "eds", "system_prompt": "spectra", "tools": ["*spectrum"]})

        assert server.spawn_agent(config) == "eds"
        assert server.list_agents() == ["eds"]

    def test_spawn_agent_tool_returns_empty_on_bad_config(self) -> None:
        assert self._server().spawn_agent("{bad json") == ""

    def test_list_agent_tools_tool_returns_json(self) -> None:
        server = self._server()
        server.swarm._tools = [_make_tool("DigitalTwin_acquire_camera_image")]
        assert json.loads(server.list_agent_tools()) == [{"name": "DigitalTwin_acquire_camera_image"}]

    def test_set_max_steps_tool_delegates_to_swarm(self) -> None:
        assert self._server().set_max_steps(7) == 7

    def test_query_agent_tool_calls_swarm_query(self) -> None:
        server = self._server()
        server.swarm.query = AsyncMock(return_value="42")

        assert asyncio.run(server.query_agent("what is the answer?")) == "42"
        server.swarm.query.assert_awaited_once_with("what is the answer?", include_transcript=True)

    def test_complete_tool_calls_swarm_complete(self) -> None:
        server = self._server()
        server.swarm.complete = AsyncMock(return_value='{"message": {}}')

        result = asyncio.run(server.complete('{"messages": []}'))
        assert json.loads(result) == {"message": {}}
