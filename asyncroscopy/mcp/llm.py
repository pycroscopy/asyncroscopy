"""Tango device wrapping an Ollama/LangChain AI agent that connects to an MCP server."""

import asyncio
import signal
import sys
import time
import urllib.request
import urllib.error
import subprocess

import tango
from tango.server import Device, attribute, command, device_property

try:
    from langchain.chat_models import init_chat_model
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, ToolMessage
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    print("Missing dependencies! Please run:")
    print("uv pip install langchain langchain-core langchain-ollama langchain-mcp-adapters")
    sys.exit(1)


class LLM(Device):
    mcp_url = device_property(dtype=str, default_value="http://127.0.0.1:8000/mcp")
    
    # Standard Ollama setup
    ollama_model = device_property(dtype=str, default_value="gemma4:31b")
    
    # Options for dynamic model initialization via `init_chat_model`
    use_init_chat_model = device_property(dtype=bool, default_value=False)
    model_provider = device_property(dtype=str, default_value="ollama")

    max_steps = attribute(label="Max Steps", dtype=int, access=tango.AttrWriteType.READ_WRITE)

    def init_device(self) -> None:
        """Initialize the Tango device and pre-warm model into VRAM."""
        Device.init_device(self)
        self.set_state(tango.DevState.INIT)
        self._max_steps = 5

        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # Create one persistent loop attached to the device instance
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            if not self.use_init_chat_model or self.model_provider == "ollama":
                self.ensure_ollama_running()

            # Instantiate model
            if self.use_init_chat_model:
                self.info_stream(f"Initializing via init_chat_model (Provider: {self.model_provider})")
                self._model = init_chat_model(
                    model=self.ollama_model,
                    model_provider=self.model_provider,
                    temperature=0
                )
            else:
                self.info_stream(f"Initializing via ChatOllama")
                self._model = ChatOllama(
                    model=self.ollama_model,
                    temperature=0,
                    reasoning=False
                )
            
            # Pre-warm the model into GPU VRAM before setting state to ON
            print("\n[SYSTEM]: Pre-warming model into VRAM (Cold Start)...")
            sys.stdout.flush()
            start_warmup = time.time()
            
            self._loop.run_until_complete(self._model.ainvoke([HumanMessage(content=" ")]))
            
            print(f"[SYSTEM]: Model pre-warmed in {time.time() - start_warmup:.2f}s! Device ready.")
            self.set_state(tango.DevState.ON)
            self.info_stream(f"LLM initialized with model: {self.ollama_model}")
            
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
            return self._loop.run_until_complete(self._run_agent(prompt))
        except Exception as e:
            err_msg = f"Agent execution failed: {e}"
            print(f"\n[CRITICAL ERROR]: {err_msg}")
            return err_msg
        finally:
            self.set_state(tango.DevState.ON)

    async def _run_agent(self, prompt: str) -> str:
        print("\n[SYSTEM]: Connecting to MCP Server...")
        client = MultiServerMCPClient(
            {
                "local_mcp": {
                    "url": self.mcp_url,
                    "transport": "streamable_http",
                }
            }
        )

        try:
            tools = await client.get_tools()
        except Exception as e:
            print(f"\n[CRITICAL ERROR]: Failed to retrieve tools: {e}")
            raise

        # Bind native tools to the model
        llm_with_tools = self._model.bind_tools(tools)
        
        # Initialize conversation history with the prompt
        messages = [HumanMessage(content=prompt)]

        print(f"\n{'='*50}\n[NEW REQUEST]: {prompt}\n{'='*50}")

        for step in range(self._max_steps):
            print(f"\n--- [STEP {step + 1}/{self._max_steps}] ---")
            print("[WAITING FOR MODEL...]: ", end="")
            sys.stdout.flush()
            
            start_time = time.time()
            first_token_received = False
            ai_message = None
            
            # Stream the model response
            async for chunk in llm_with_tools.astream(messages):
                if not first_token_received:
                    ttft = time.time() - start_time
                    print(f"\n[DIAGNOSTIC]: Time to first token: {ttft:.2f} seconds.")
                    print("[GENERATION]: ", end="")
                    first_token_received = True

                # Print text as it streams
                if chunk.content:
                    print(chunk.content, end="")
                    sys.stdout.flush() 
                
                if ai_message is None:
                    ai_message = chunk
                else:
                    ai_message += chunk
                
            print("\n") 

            # Check if tools need to be called
            if not ai_message.tool_calls:
                final_ans = ai_message.content.strip()
                print(f"[FINAL ANSWER RETURNED]:\n{final_ans}\n{'='*50}")
                return final_ans

            messages.append(ai_message)

            # Execute tool calls
            for tool_call in ai_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                print(f"[EXECUTING TOOL]: {tool_name}({tool_args})")
                active_tool = next((t for t in tools if t.name == tool_name), None)
                
                if active_tool:
                    try:
                        observation = await active_tool.ainvoke(tool_args)
                        obs_str = str(observation)
                    except Exception as e:
                        obs_str = f"Error executing tool: {e}"
                else:
                    obs_str = f"Model tried to call unknown tool '{tool_name}'"
                
                print(f"[TOOL RESULT]: {obs_str}")
                
                messages.append(ToolMessage(
                    name=tool_name,
                    content=obs_str,
                    tool_call_id=tool_id
                ))
                
        return "Max steps reached without a final answer."


if __name__ == "__main__":
    LLM.run_server()