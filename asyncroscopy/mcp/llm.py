"""Tango device wrapping a LangChain AI agent that connects to the MCP server."""

import asyncio
import json
import warnings
from threading import Thread

import tango
from tango.server import Device, attribute, command, device_property

import urllib
import subprocess
import time

try:
    from langchain.chat_models import init_chat_model
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_core.tools import BaseTool
except ImportError:
    print("Install the optional AI extra with uv sync --extra agent")


class LLM(Device):
    mcp_url = device_property(dtype=str, default_value="http://127.0.0.1:8000/mcp")
    model_provider = device_property(dtype=str, default_value="openai")
    model_name = device_property(dtype=str, default_value="gpt-4o")
    api_key = device_property(dtype=str, default_value="")
    local_model_path = device_property(dtype=str, default_value="")
    ollama_model = device_property(dtype=str, default_value="gemma4:31b")
    max_steps = device_property(dtype=int, default_value=5)

    def init_device(self) -> None:
        """Initialize the Tango device."""
        Device.init_device(self)
        self.set_state(tango.DevState.INIT)
        self.set_change_event("token_stream", True, False)

        self._model = None
        self._tools: list[BaseTool] = []
        self._is_hf_pipeline = False  # Flag to handle streaming limitations of HF

        try:
            asyncio.run(self.start_session())
            self.set_state(tango.DevState.ON)
            self.info_stream("LLM device initialized")
        except Exception as e:
            self.set_state(tango.DevState.FAULT)
            self.set_status(f"Initialization failed: {e}")

    async def start_session(self):
        """Initialize MCP server connection and load model."""
        try:
            client = MultiServerMCPClient(
                {
                    "asyncroscopy": {
                        "url": self.mcp_url,
                        "transport": "streamable_http",
                    }
                }
            )
            self._tools = await client.get_tools()
            
            # 1. Check for Ollama First
            if self.ollama_model:
                try:
                    from langchain_ollama import ChatOllama
                except ImportError:
                    raise ImportError("Please run: uv pip install langchain-ollama")
                
                self.ensure_ollama_running()  # Ensure Ollama server is running before loading the model
                
                self.info_stream(f"Loading Ollama model: {self.ollama_model}")
                self._model = ChatOllama(
                    model=self.ollama_model,
                    temperature=0.2,
                )
                self._is_hf_pipeline = False

            # 2. Check for Local HuggingFace Path
            elif self.local_model_path:
                self.info_stream(f"Loading local HF model: {self.local_model_path}")
                try:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
                    import transformers.utils.logging as transformer_logging
                except ImportError:
                    raise ImportError("Failed to import local AI dependencies. Please run: uv sync --extra agent --extra localagent")

                transformer_logging.set_verbosity_error()
                warnings.simplefilter(action='ignore', category=FutureWarning)

                raw_path = str(self.local_model_path)
                tokenizer = AutoTokenizer.from_pretrained(raw_path)
                hf_model = AutoModelForCausalLM.from_pretrained(
                    raw_path,
                    device_map="cuda",
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    local_files_only=True,
                )

                hf_pipeline = pipeline(
                    "text-generation",
                    model=hf_model,
                    tokenizer=tokenizer,
                    max_new_tokens=256,
                    temperature=0.2,
                    return_full_text=False,
                )

                self._model = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=hf_pipeline))
                self._is_hf_pipeline = True

            # 3. Fallback to API Models
            else:
                self.info_stream(f"Loading API model: {self.model_name} via {self.model_provider}")
                self._model = init_chat_model(
                    model=self.model_name,
                    model_provider=self.model_provider,
                    api_key=self.api_key or None,
                )
                self._is_hf_pipeline = False

        except Exception as e:
            self.error_stream(f"Failed to start session: {e}")
            raise

    @attribute(dtype=str, doc="List of tools the LLM has")
    def tools(self) -> str:
        tool_data = [{"name": t.name, "description": t.description} for t in self._tools]
        return json.dumps(tool_data)

    @attribute(dtype=str)
    def token_stream(self):
        return getattr(self, "_current_token", "")

    @command(dtype_in=str, dtype_out=None)
    def Stream_Query(self, prompt):
        def background_task():
            # Tell clients we are busy
            self.set_state(tango.DevState.RUNNING)
            try:
                self.Query(prompt)
            finally:
                # Tell clients we are done, even if it crashed
                self.set_state(tango.DevState.ON)
                
        Thread(target=background_task, daemon=True).start()

    @command(dtype_in=str, dtype_out=str)
    def Query(self, prompt: str) -> str:
        """Run a query using the langchain AI agent, executing it in a synchronous event loop wrapper."""
        return asyncio.run(self._run_agent(prompt))

    async def _run_agent(self, prompt: str) -> str:
        """Run the async agent query loop to invoke tools from the MCP server."""
        if not self._model:
            return "Model not initialized."

        tools_string = "\n".join([f"- Name: {t.name}\n  Description: {t.description}\n" for t in self._tools])

        agent_context = f"""You are an AI Agent with access to these tools:
        {tools_string}

        To use a tool, you MUST respond using this exact format:
        Action: <tool_name>
        Arguments: <JSON_object_or_string>

        When you have the final answer, respond with:
        Final Answer: <your_response>

        User Request: {prompt}"""

        self._current_token = ""

        for step in range(self.max_steps):
            response = ""
            
            if self._is_hf_pipeline:
                # HF Pipeline doesn't support async streaming out of the box, block on invoke
                response_msg = self._model.invoke(agent_context)
                raw_response = response_msg.content if hasattr(response_msg, 'content') else str(response_msg)
                response = raw_response.replace("<turn|>", "").strip()
                
                self._current_token = response
                self.push_change_event("token_stream", response)
            else:
                # Ollama and API providers fully support native async streaming
                async for chunk in self._model.astream(agent_context):
                    chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    response += chunk_text
                    
                    self._current_token = response
                    self.push_change_event("token_stream", response)
                    
                response = response.replace("<turn|>", "").strip()
            
            # Check if the model wants to call a tool
            if "Action:" in response and "Arguments:" in response:
                try:
                    # Parse out the tool execution details
                    tool_name = response.split("Action:")[1].split("\n")[0].strip()
                    tool_args_raw = response.split("Arguments:")[1].split("\n")[0].strip()
                    
                    # Convert args to dict if it's JSON, otherwise keep as string
                    try:
                        tool_args = json.loads(tool_args_raw)
                    except json.JSONDecodeError:
                        tool_args = tool_args_raw

                    # Find the matching tool object from loaded MCP tools
                    active_tool = next((t for t in self._tools if t.name == tool_name), None)
                    
                    if active_tool:
                        self.info_stream(f"[Executing Tool]: Calling {tool_name}({tool_args})...")
                        observation = await active_tool.ainvoke(tool_args)
                        self.info_stream(f"[Tool Result]: {observation}\n")
                        
                        # Feed the observation back to the model's memory context
                        agent_context += f"\n{response}\nObservation: {observation}"
                    else:
                        self.error_stream(f"Error: Model tried to call unknown tool '{tool_name}'\n")
                        break
                except Exception as e:
                    self.error_stream(f"Parsing/Execution error: {e}")
                    break
                    
            elif "Final Answer:" in response:
                return response.split("Final Answer:", 1)[1].strip()
            else:
                self.info_stream("\nModel responded with plain text instead of using a tool format.")
                return response
        return response
    
    def ensure_ollama_running(self, host: str = "http://localhost:11434", timeout: int = 10) -> None:
        """Checks if Ollama server is responding. If not, attempts to start `ollama serve` in the background."""
        tags_url = f"{host.rstrip('/')}/api/tags"
        
        # 1. Check if server is already running
        try:
            with urllib.request.urlopen(tags_url, timeout=1):
                return  # Ollama is live and responding
        except (urllib.error.URLError, TimeoutError, ConnectionRefusedError):
            pass

        # 2. Launch Ollama in the background if server is down
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True  # Detaches from parent process lifetime
            )
        except FileNotFoundError:
            raise RuntimeError("Ollama binary not found on PATH. Please ensure Ollama is installed.")

        # 3. Poll until service becomes available
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with urllib.request.urlopen(tags_url, timeout=1):
                    return  # Ollama started successfully
            except (urllib.error.URLError, TimeoutError, ConnectionRefusedError):
                time.sleep(0.5)

        raise RuntimeError(f"Started 'ollama serve' but endpoint '{tags_url}' did not respond within {timeout}s.")

# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    LLM.run_server()