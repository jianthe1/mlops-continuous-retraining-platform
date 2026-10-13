import os
import sys

# 1. Sanitize invalid system SSL variables before loading network libs
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]
if "SSL_CERT_DIR" in os.environ and not os.path.exists(os.environ["SSL_CERT_DIR"]):
    del os.environ["SSL_CERT_DIR"]

import asyncio
import json
from fastmcp import Client
from google import genai
from google.genai import types

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")
VERBOSE = False

def clean_schema_for_gemini(schema):
    """Recursively removes OpenAPI attributes unsupported by the Gemini API."""
    if isinstance(schema, dict):
        banned = {"additional_properties", "additionalProperties", "$schema", "title"}
        return {k: clean_schema_for_gemini(v) for k, v in schema.items() if k not in banned}
    elif isinstance(schema, list):
        return [clean_schema_for_gemini(item) for item in schema]
    return schema

async def run_tool_call(mcp_client: Client, call) -> types.Part:
    """Executes a single MCP tool call and returns a GenAI FunctionResponse part."""
    tool_name = call.name
    tool_args = dict(call.args or {})
    print(f"  -> Executing: {tool_name}({tool_args})")
    
    try:
        mcp_result = await mcp_client.call_tool(name=tool_name, arguments=tool_args)
        raw_text = mcp_result.content[0].text if mcp_result.content else "{}"
        try:
            structured = json.loads(raw_text)
        except json.JSONDecodeError:
            structured = {"output": raw_text}
    except Exception as e:
        structured = {"error": f"Tool execution failed: {str(e)}"}

    if VERBOSE:
        print(f"     [Raw Response]: {json.dumps(structured, indent=2)[:300]}...")

    return types.Part.from_function_response(
        name=tool_name,
        response={"result": structured},
    )

async def run_interactive_agent():
    global VERBOSE
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    gemini_client = genai.Client(api_key=api_key)

    print(f"Connecting to MCP Server at: {MCP_SERVER_URL} ...")
    try:
        async with Client(MCP_SERVER_URL) as mcp_client:
            mcp_tools = await mcp_client.list_tools()
            print(f"Discovered {len(mcp_tools)} tools from MCP server.")

            func_declarations = [
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description or "",
                    parameters=clean_schema_for_gemini(getattr(t, "input_schema", {})),
                )
                for t in mcp_tools
            ]

            gemini_tools = [types.Tool(function_declarations=func_declarations)]

            system_instruction = (
                "You are an expert Kubernetes Site Reliability Engineer (SRE).\n"
                "Diagnostic Rules:\n"
                "1. Always correlate warning events with live pod statuses from `list_pods`.\n"
                "2. If a warning event targets a pod absent from `list_pods`, flag it as a historical artifact.\n"
                "3. If a running pod has 0 restarts, past readiness failures were transient startup delays, not active outages.\n"
                "4. When diagnosing nodes, verify conditions (MemoryPressure, DiskPressure, PIDPressure) and allocatable headroom.\n"
                "5. Present findings concisely: Active Incidents first, then Historical Artifacts, followed by Actionable Recommendations."
            )

            chat = gemini_client.chats.create(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    tools=gemini_tools,
                    system_instruction=system_instruction,
                ),
            )

            print("\n=== Kubernetes MCP Diagnostic Agent Ready ===")
            print("Commands: 'exit' to quit | ':v' to toggle raw payload inspection\n")

            while True:
                try:
                    user_query = input("SRE-Agent> ").strip()
                except (KeyboardInterrupt, EOFError):
                    break

                if not user_query:
                    continue
                if user_query.lower() in ("exit", "quit"):
                    break
                if user_query == ":v":
                    VERBOSE = not VERBOSE
                    print(f"[*] Verbose raw tool output: {'ON' if VERBOSE else 'OFF'}")
                    continue

                response = chat.send_message(user_query)

                while response.function_calls:
                    tasks = [run_tool_call(mcp_client, call) for call in response.function_calls]
                    function_responses = await asyncio.gather(*tasks)
                    response = chat.send_message(function_responses)

                print(f"\n{response.text}\n")

    except Exception as err:
        print(f"\n[Connection Error]: {err}")
        print("Tip: Check if your 'kubectl port-forward svc/k8s-mcp-service 8000:8000' session is still running.")

if __name__ == "__main__":
    asyncio.run(run_interactive_agent())