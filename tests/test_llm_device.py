import json
from unittest.mock import MagicMock

import pytest
import tango

# ======================================================================
# Mocks
# ======================================================================

class ToolMock:
    def __init__(self, name="test_tool", description="A dummy tool for testing."):
        self.name = name
        self.description = description
        self.called = False
        self.call_args = None

    async def ainvoke(self, args_dict):
        self.called = True
        self.call_args = args_dict
        return "Observation: Tool executed successfully."

    def reset(self):
        self.called = False
        self.call_args = None


class ModelMock:
    def __init__(self):
        self.call_count = 0
        self.side_effects = []
        self.default_content = "Final Answer: Default mock response."

    async def ainvoke(self, *args, **kwargs):
        self.call_count += 1
        if self.side_effects:
            return self.side_effects.pop(0)
        msg = MagicMock()
        msg.content = self.default_content
        return msg

    def reset(self):
        self.call_count = 0
        self.side_effects = []

# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="module")
def mock_tool():
    return ToolMock()

@pytest.fixture(scope="module")
def model_mock():
    return ModelMock()

@pytest.fixture(scope="module")
def llm_proxy(tango_ctx):
    """Creates a DeviceProxy connection to the LLM device in tango_ctx."""
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/llm/default"))

@pytest.fixture(autouse=True)
def setup_and_reset_llm_mocks(tango_ctx, mock_tool, model_mock):
    """
    Injects the mocks into the live Python LLM instance hosted by 
    MultiDeviceTestContext and resets mock states before each test.
    """
    mock_tool.reset()
    model_mock.reset()

    llm_instance = tango_ctx.get_device("asyncroscopy/llm/default")

    # Get actual Device instance
    util = tango.Util.instance()
    llm_server_device = util.get_device_by_name("asyncroscopy/llm/default")

    llm_server_device.set_state(tango.DevState.ON)
    llm_server_device._tools = [mock_tool]
    llm_server_device._model = model_mock


# ======================================================================
# Tests
# ======================================================================

def test_device_initialization_state(llm_proxy):
    """Test that the device initializes successfully and turns ON."""
    assert llm_proxy.state() == tango.DevState.ON


def test_tools_attribute(llm_proxy):
    """Test that the tools attribute properly returns the JSON tools schema."""
    tools_json = llm_proxy.tools
    tools = json.loads(tools_json)
    
    assert len(tools) == 1
    assert tools[0]["name"] == "test_tool"
    assert tools[0]["description"] == "A dummy tool for testing."


def test_query_direct_answer(llm_proxy, model_mock):
    """Test a simple query where the model provides an immediate final answer."""
    response = llm_proxy.Query("Hello!")
    
    assert response == "Default mock response."
    assert model_mock.call_count == 1


def test_query_with_tool_execution(llm_proxy, model_mock, mock_tool):
    """Test the agent loop when the model decides to use a tool before answering."""
    tool_request_msg = MagicMock()
    tool_request_msg.content = 'Action: test_tool\nArguments: {"test_arg": 1}'
    
    final_answer_msg = MagicMock()
    final_answer_msg.content = "Final Answer: The tool gave me the data."
    
    model_mock.side_effects = [tool_request_msg, final_answer_msg]

    response = llm_proxy.Query("Run the test tool.")
    
    assert response == "The tool gave me the data."
    assert model_mock.call_count == 2
    assert mock_tool.called is True
    assert mock_tool.call_args == {"test_arg": 1}


def test_query_max_steps_limit(llm_proxy, model_mock):
    """Test that the loop exits gracefully if it hits max_steps without a final answer."""
    infinite_tool_msg = MagicMock()
    infinite_tool_msg.content = "Action: test_tool\nArguments: {}"
    
    model_mock.side_effects = [infinite_tool_msg] * 10
    
    response = llm_proxy.Query("Do an infinite loop.")
    
    assert "Action: test_tool" in response
    assert model_mock.call_count == 5