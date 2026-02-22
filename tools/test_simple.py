"""Test groksearch-simple with grok-4.20-beta via stdio."""
import os, sys, asyncio, json

server_cmd = [
    sys.executable, "-c",
    "import sys;sys.path.insert(0,'src');from grok_search.server import mcp;mcp.run(transport='stdio')"
]

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

async def main():
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["GROK_API_URL"] = "http://127.0.0.1:8000/v1"
    env["GROK_API_KEY"] = "test"
    env["NO_PROXY"] = "127.0.0.1,localhost"

    params = StdioServerParameters(command=server_cmd[0], args=server_cmd[1:], env=env)

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            print("=== Session initialized ===\n")

            # tools/list
            tools = await s.list_tools()
            print(f"Tools: {[t.name for t in tools.tools]}\n")

            # switch to grok-4.20-beta
            r = await s.call_tool("switch_model", {"model": "grok-4.20-beta"})
            print(f"switch_model: {r.content[0].text[:200]}\n")

            # web_search
            print("--- web_search (grok-4.20-beta) ---")
            r = await s.call_tool("web_search", {"query": "今日热点新闻"})
            out = r.content[0].text
            print(f"len={len(out)}")
            print(out[:500])
            print()

            # Check for <think> tags
            if "<think>" in out:
                print("WARNING: <think> tags NOT stripped!")
            else:
                print("OK: no <think> tags in output")

            # web_fetch
            print("\n--- web_fetch ---")
            r = await s.call_tool("web_fetch", {"url": "https://docs.python.org/3/"})
            out = r.content[0].text
            print(f"len={len(out)}")
            print(out[:200])

            print("\n=== ALL DONE ===")

if __name__ == "__main__":
    asyncio.run(main())
