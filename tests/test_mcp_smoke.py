from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str, db_path: Path, env_overrides: dict[str, str] | None = None) -> str:
    env = os.environ.copy()
    env["OMNIMIND_DB_PATH"] = str(db_path)
    if env_overrides:
        env.update(env_overrides)

    root_str = str(PROJECT_ROOT)
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root_str if not py_path else root_str + os.pathsep + py_path

    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"python failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return proc.stdout.strip()


def test_memory_tools_lazy_db_init_and_roundtrip(tmp_path: Path) -> None:
    code = (
        "import asyncio, json\n"
        "from mcp_server.memory_tools import memory_upsert, memory_get\n"
        "res1 = asyncio.run(memory_upsert('smoke.key', 'smoke value', 'lesson'))\n"
        "res2 = asyncio.run(memory_get('smoke.key', 'lesson'))\n"
        "print(json.dumps({'upsert_ok': bool(res1.get('ok')), 'get_ok': bool(res2.get('ok')), "
        "'value': (res2.get('lesson') or {}).get('lesson')}))\n"
    )
    out = _run_python(code, tmp_path / "lazy_init.db")
    payload = json.loads(out.splitlines()[-1])

    assert payload["upsert_ok"] is True
    assert payload["get_ok"] is True
    assert payload["value"] == "smoke value"


def test_mcp_resources_registered_and_health_rendered(tmp_path: Path) -> None:
    code = (
        "import asyncio\n"
        "from mcp_server.memory_tools import mcp as tools_mcp, memory_upsert\n"
        "from mcp_server.memory_resources import mcp as resources_mcp, lessons_resource, health_resource\n"
        "assert tools_mcp is resources_mcp\n"
        "asyncio.run(memory_upsert('resource.key', 'resource lesson', 'lesson'))\n"
        "lessons_txt = asyncio.run(lessons_resource())\n"
        "health_txt = asyncio.run(health_resource())\n"
        "ok = ('resource.key' in lessons_txt) and ('Memory Health' in health_txt) and ('Vectors:' in health_txt)\n"
        "print('OK' if ok else 'BAD')\n"
    )
    out = _run_python(code, tmp_path / "resources.db")
    assert out.splitlines()[-1] == "OK"


def test_key_modules_importable(tmp_path: Path) -> None:
    code = (
        "import importlib\n"
        "mods = [\n"
        "  'core.memory',\n"
        "  'mcp_server.memory_tools',\n"
        "  'mcp_server.memory_resources',\n"
        "  'api.routes_memory',\n"
        "  'plugins.builtins.memory_tool',\n"
        "]\n"
        "for m in mods:\n"
        "  importlib.import_module(m)\n"
        "print('OK')\n"
    )
    out = _run_python(code, tmp_path / "imports.db")
    assert out.splitlines()[-1] == "OK"


def test_new_mcp_tools_procedural_semantic_and_metrics(tmp_path: Path) -> None:
    code = (
        "import asyncio, json\n"
        "from mcp_server.memory_tools import (\n"
        "  memory_add_procedure, memory_get_procedure, memory_add_entity,\n"
        "  memory_add_relation, memory_get_relations, memory_metrics\n"
        ")\n"
        "async def main():\n"
        "  await memory_add_procedure('how.test', 'How Test', ['step1', 'step2'])\n"
        "  p = await memory_get_procedure('how.test')\n"
        "  e1 = await memory_add_entity('svc-a', 'service', {'lang':'py'})\n"
        "  e2 = await memory_add_entity('svc-b', 'service', {'lang':'py'})\n"
        "  await memory_add_relation(e1['id'], 'calls', e2['id'])\n"
        "  rels = await memory_get_relations(e1['id'])\n"
        "  mets = await memory_metrics()\n"
        "  print(json.dumps({'p_ok': p.get('ok'), 'rels': len(rels), 'm_ok': mets.get('ok')}))\n"
        "asyncio.run(main())\n"
    )
    out = _run_python(code, tmp_path / "new_tools.db")
    payload = json.loads(out.splitlines()[-1])
    assert payload["p_ok"] is True
    assert payload["rels"] >= 1
    assert payload["m_ok"] is True


def test_health_reports_db_backend_selection(tmp_path: Path) -> None:
    code = (
        "import asyncio, json\n"
        "from mcp_server.memory_tools import memory_health\n"
        "out = asyncio.run(memory_health())\n"
        "print(json.dumps(out.get('db_backend') or {}))\n"
    )
    out = _run_python(
        code,
        tmp_path / "backend.db",
        env_overrides={"OMNIMIND_DB_TYPE": "postgres"},
    )
    payload = json.loads(out.splitlines()[-1])
    assert payload.get("requested") == "postgres"
    assert payload.get("effective") == "sqlite"
