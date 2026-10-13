import os
# Sanitize stale SSL environment variables
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]
if "SSL_CERT_DIR" in os.environ and not os.path.exists(os.environ["SSL_CERT_DIR"]):
    del os.environ["SSL_CERT_DIR"]

import asyncio
import json
import sys
from fastmcp import Client
from google import genai
from google.genai import types

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")

def clean_schema_for_gemini(schema):
    """Recursively removes OpenAPI attributes unsupported by the Gemini API."""
    if isinstance(schema, dict):
        banned = {"additional_properties", "additionalProperties", "$schema", "title"}
        return {k: clean_schema_for_gemini(v) for k, v in schema.items() if k not in banned}
    elif isinstance(schema, list):
        return [clean_schema_for_gemini(item) for item in schema]
    return schema

async def run_interactive_agent():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    gemini_client = genai.Client(api_key=api_key)

    print(f"Connecting to MCP Server at: {MCP_SERVER_URL} ...")
    async with Client(MCP_SERVER_URL) as mcp_client:
        mcp_tools = await mcp_client.list_tools()
        print(f"Discovered {len(mcp_tools)} tools from MCP server.")

        func_declarations = []
        for t in mcp_tools:
            raw_schema = getattr(t, "input_schema", {})
            func_declarations.append(
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description or "",
                    parameters=clean_schema_for_gemini(raw_schema),
                )
            )

        gemini_tools = [types.Tool(function_declarations=func_declarations)]

        system_instruction = (
            "You are a Senior Kubernetes Platform and SRE Engineer.\n"
            "Diagnostic Rules:\n"
            "1. Correlate all warning events with the live pod list from `list_pods`.\n"
            "2. If a warning belongs to a pod that is no longer listed, treat it as a past artifact from a completed deployment, NOT an active failure.\n"
            "3. If a running pod has 0 restarts, a single past readiness probe warning was a transient startup delay, not an active outage.\n"
            "4. Clearly separate ACTIVE issues from HISTORICAL artifacts in your diagnostic summary."
        )

        chat = gemini_client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                system_instruction=system_instruction,
            ),
        )

        print("\n=== Kubernetes MCP Diagnostic Agent Ready (Type 'exit' to quit) ===\n")
        
        while True:
            try:
                user_query = input("SRE-Agent> ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_query:
                continue
            if user_query.lower() in ("exit", "quit"):
                break

            response = chat.send_message(user_query)

            # Handle tool call loop
            while response.function_calls:
                for call in response.function_calls:
                    print(f"  -> Executing: {call.name}({dict(call.args)})")
                    mcp_result = await mcp_client.call_tool(
                        name=call.name, arguments=dict(call.args)
                    )
                    raw_text = mcp_result.content[0].text if mcp_result.content else "{}"
                    try:
                        structured = json.loads(raw_text)
                    except json.JSONDecodeError:
                        structured = {"output": raw_text}

                    response = chat.send_message(
                        types.Part.from_function_response(
                            name=call.name,
                            response={"result": structured},
                        )
                    )

            print(f"\n{response.text}\n")

if __name__ == "__main__":
    asyncio.run(run_interactive_agent())