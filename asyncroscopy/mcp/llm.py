"""Tango device wrapping an Ollama LangChain AI agent that connects to an MCP server."""

import asyncio
import json
import signal
import sys
import time
import urllib.request
import urllib.error
import subprocess

import tango
from tango.server import Device, attribute, command, device_property

try:
    from langchain_ollama import ChatOllama
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    print("Missing dependencies! Please run:")
    print("uv pip install langchain-ollama langchain-mcp-adapters")
    sys.exit(1)


class LLM(Device):
    mcp_url = device_property(dtype=str, default_value="http://127.0.0.1:8000/mcp")
    ollama_model = device_property(dtype=str, default_value="gemma4:31b")
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
            self.ensure_ollama_running()
            self._model = ChatOllama(
                model=self.ollama_model,
                streaming=True,
                reasoning=False,
            )
            
            # Pre-warm the model into GPU VRAM before setting state to ON
            print("\n[SYSTEM]: Pre-warming Ollama model into VRAM (Cold Start)...")
            sys.stdout.flush()
            start_warmup = time.time()
            self._loop.run_until_complete(self._model.ainvoke(" "))
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

        tools_string = "\n".join([f"- Name: {t.name}\n  Description: {t.description}" for t in tools])

        agent_context = f"""You are an AI Agent with access to these tools:
        {tools_string}

        To use a tool, respond exactly like this:
        Action: <tool_name>
        Arguments: <JSON_object_or_string>

        When you have the final answer, respond exactly like this:
        Final Answer: <your_response>

        User Request: {prompt}"""

        print(f"\n{'='*50}\n[NEW REQUEST]: {prompt}\n{'='*50}")

        for step in range(self._max_steps):
            print(f"\n--- [STEP {step + 1}/{self._max_steps}] ---")
            print("[WAITING FOR MODEL...]: ", end="")
            sys.stdout.flush()
            
            response = ""
            start_time = time.time()
            first_token_received = False
            
            async for chunk in self._model.astream(agent_context):
                if not first_token_received:
                    ttft = time.time() - start_time
                    print(f"\n[DIAGNOSTIC]: Time to first token: {ttft:.2f} seconds.")
                    print("[GENERATION]: ", end="")
                    first_token_received = True

                chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                response += chunk_text
                print(chunk_text, end="")
                sys.stdout.flush() 
                
            print("\n") 
            response = response.strip()

            if "Action:" in response and "Arguments:" in response:
                try:
                    tool_name = response.split("Action:")[1].split("\n")[0].strip()
                    tool_args_raw = response.split("Arguments:")[1].split("\n")[0].strip()
                    
                    try:
                        tool_args = json.loads(tool_args_raw)
                    except json.JSONDecodeError:
                        tool_args = tool_args_raw

                    active_tool = next((t for t in tools if t.name == tool_name), None)
                    
                    if active_tool:
                        print(f"[EXECUTING TOOL]: {tool_name}({tool_args})")
                        observation = await active_tool.ainvoke(tool_args)
                        print(f"[TOOL RESULT]: {observation}")
                        
                        agent_context += f"\n{response}\nObservation: {observation}"
                    else:
                        err = f"Model tried to call unknown tool '{tool_name}'"
                        print(f"[ERROR]: {err}")
                        agent_context += f"\n{response}\nObservation: {err}"
                        
                except Exception as e:
                    err = f"Parsing error: {e}"
                    print(f"[ERROR]: {err}")
                    agent_context += f"\n{response}\nObservation: {err}"
                    
            elif "Final Answer:" in response:
                final_ans = response.split("Final Answer:", 1)[1].strip()
                print(f"[FINAL ANSWER RETURNED]:\n{final_ans}\n{'='*50}")
                return final_ans
            else:
                print("[WARNING]: Model replied with standard text, bypassing final answer format.")
                return response
                
        return "Max steps reached without a final answer."


if __name__ == "__main__":
    LLM.run_server()