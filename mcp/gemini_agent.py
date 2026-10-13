import asyncio
import json
import os
import sys
from fastmcp import Client
from google import genai
from google.genai import types

# Default points to local port-forward or in-cluster service DNS
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")


async def run_diagnostics(user_query: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    gemini_client = genai.Client(api_key=api_key)

    print(f"Connecting to MCP Server at: {MCP_SERVER_URL} ...")
    async with Client(MCP_SERVER_URL) as mcp_client:
        # 1. Fetch tool definitions from MCP Server
        mcp_tools = await mcp_client.list_tools()
        print(f"Discovered {len(mcp_tools)} cluster tools from MCP server.")

        # 2. Convert MCP schemas into Gemini Function Declarations
        func_declarations = [
            types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=t.inputSchema,
            )
            for t in mcp_tools
        ]
        gemini_tools = [types.Tool(function_declarations=func_declarations)]

        # 3. Create the chat session
        chat = gemini_client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                system_instruction=(
                    "You are a Senior Kubernetes Platform and SRE Engineer. "
                    "Use the available cluster inspection tools to inspect resources, "
                    "read logs, identify crashes, and explain root causes clearly."
                ),
            ),
        )

        print(f"\nPrompt: {user_query}\n")
        response = chat.send_message(user_query)

        # 4. Handle tool call loop
        while response.function_calls:
            for call in response.function_calls:
                print(
                    f"-> Gemini executing tool: {call.name}({dict(call.args)})"
                )

                mcp_result = await mcp_client.call_tool(
                    name=call.name, arguments=dict(call.args)
                )

                raw_output = (
                    mcp_result.content[0].text if mcp_result.content else "{}"
                )
                try:
                    structured_output = json.loads(raw_output)
                except json.JSONDecodeError:
                    structured_output = {"output": raw_output}

                response = chat.send_message(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"result": structured_output},
                    )
                )

        print("\n=== SRE Diagnostic Summary ===")
        print(response.text)


if __name__ == "__main__":
    query = (
        "Check the default namespace. Are any pods crashing, restarting, "
        "or showing warning events? If so, fetch their logs and tell me what is wrong."
    )
    asyncio.run(run_diagnostics(query))