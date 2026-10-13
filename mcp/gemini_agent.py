import os

# Clean up stale SSL environment variables
if "SSL_CERT_FILE" in os.environ and not os.path.exists(
    os.environ["SSL_CERT_FILE"]
):
  del os.environ["SSL_CERT_FILE"]
if "SSL_CERT_DIR" in os.environ and not os.path.exists(
    os.environ["SSL_CERT_DIR"]
):
  del os.environ["SSL_CERT_DIR"]

import asyncio
import json
import sys
from fastmcp import Client
from google import genai
from google.genai import types

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")


def clean_schema_for_gemini(schema):
  """Recursively strips keys that the Gemini API does not allow in OpenAPI schemas."""
  if isinstance(schema, dict):
    banned_keys = {
        "additional_properties",
        "additionalProperties",
        "$schema",
        "title",
    }
    return {
        k: clean_schema_for_gemini(v)
        for k, v in schema.items()
        if k not in banned_keys
    }
  elif isinstance(schema, list):
    return [clean_schema_for_gemini(item) for item in schema]
  return schema


async def run_diagnostics(user_query: str):
  api_key = os.getenv("GEMINI_API_KEY")
  if not api_key:
    print("ERROR: GEMINI_API_KEY environment variable is not set.")
    sys.exit(1)

  gemini_client = genai.Client(api_key=api_key)

  print(f"Connecting to MCP Server at: {MCP_SERVER_URL} ...")
  async with Client(MCP_SERVER_URL) as mcp_client:
    mcp_tools = await mcp_client.list_tools()
    print(f"Discovered {len(mcp_tools)} cluster tools from MCP server.")

    func_declarations = []
    for t in mcp_tools:
      # Support both new input_schema and legacy inputSchema
      raw_schema = getattr(t, "input_schema", {})
      sanitized_schema = clean_schema_for_gemini(raw_schema)

      func_declarations.append(
          types.FunctionDeclaration(
              name=t.name,
              description=t.description or "",
              parameters=sanitized_schema,
          )
      )

    gemini_tools = [types.Tool(function_declarations=func_declarations)]

    chat = gemini_client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            tools=gemini_tools,
            system_instruction=(
                "You are a Senior Kubernetes Platform and SRE Engineer. "
                "Use the available cluster inspection tools to inspect"
                " resources, "
                "read logs, identify crashes, and explain root causes clearly."
            ),
        ),
    )

    print(f"\nPrompt: {user_query}\n")
    response = chat.send_message(user_query)

    # Tool invocation loop
    while response.function_calls:
      for call in response.function_calls:
        print(f"-> Gemini executing tool: {call.name}({dict(call.args)})")

        mcp_result = await mcp_client.call_tool(
            name=call.name, arguments=dict(call.args)
        )

        raw_output = mcp_result.content[0].text if mcp_result.content else "{}"
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
      "or showing warning events? If so, fetch their logs and tell me what is"
      " wrong."
  )
  asyncio.run(run_diagnostics(query))