"""MCP Memory Server entry point.

Run with:
    python -m mcp_server.server
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import FastMCP app
from core.db import init_db
import mcp_server.memory_resources  # noqa: F401  # Registers MCP resources
from mcp_server.memory_tools import mcp as tools_mcp

# The main app - run this
app = tools_mcp

if __name__ == "__main__":
    asyncio.run(init_db())
    # Run with stdio transport
    tools_mcp.run()
