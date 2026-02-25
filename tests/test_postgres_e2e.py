from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _postgres_driver_available() -> bool:
    return bool(importlib.util.find_spec("psycopg2") or importlib.util.find_spec("psycopg"))


def _docker_daemon_available() -> bool:
    if shutil.which("docker") is None:
        return False
    proc = subprocess.run(
        ["docker", "info"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _external_pg_requested() -> bool:
    return os.getenv("POSTGRES_E2E_EXTERNAL") == "1"


def _e2e_mode() -> str | None:
    if os.getenv("RUN_POSTGRES_E2E") != "1":
        return None
    if _external_pg_requested():
        return "external"
    if _docker_daemon_available():
        return "docker"
    return None


def _can_run_e2e() -> bool:
    return _e2e_mode() is not None and _postgres_driver_available()


pytestmark = pytest.mark.skipif(
    not _can_run_e2e(),
    reason=(
        "set RUN_POSTGRES_E2E=1 with postgres driver and either "
        "docker daemon (auto container) or POSTGRES_E2E_EXTERNAL=1"
    ),
)


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def _wait_postgres_ready(timeout_s: int = 90) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        proc = _run(
            ["docker", "exec", "ai_postgres", "pg_isready", "-U", "memory_user", "-d", "memory"],
            check=False,
        )
        if proc.returncode == 0:
            return
        time.sleep(2)
    raise AssertionError("postgres container did not become ready in time")


def _wait_external_postgres_ready(
    *, host: str, port: int, db: str, user: str, password: str, timeout_s: int = 90
) -> None:
    import psycopg2

    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                dbname=db,
                user=user,
                password=password,
                connect_timeout=5,
            )
            conn.close()
            return
        except Exception as e:  # pragma: no cover
            last_error = str(e)
            time.sleep(2)
    raise AssertionError(f"external postgres did not become ready in time: {last_error}")


def test_postgres_backend_roundtrip_via_memory_tools(tmp_path: Path) -> None:
    mode = _e2e_mode()
    assert mode in ("docker", "external")

    host = os.getenv("OMNIMIND_POSTGRES_HOST", "127.0.0.1")
    port = int(os.getenv("OMNIMIND_POSTGRES_PORT", "5442"))
    db = os.getenv("OMNIMIND_POSTGRES_DB", "memory")
    user = os.getenv("OMNIMIND_POSTGRES_USER", "memory_user")
    password = os.getenv("OMNIMIND_POSTGRES_PASSWORD", "SecureP@ssw0rd_2024!MemoryDB")

    if mode == "docker":
        _run(["docker", "compose", "up", "-d", "ai_postgres"])
    try:
        if mode == "docker":
            _wait_postgres_ready()
        else:
            _wait_external_postgres_ready(
                host=host,
                port=port,
                db=db,
                user=user,
                password=password,
            )

        env = os.environ.copy()
        env.update(
            {
                "OMNIMIND_DB_TYPE": "postgres",
                "OMNIMIND_POSTGRES_ENABLED": "true",
                "OMNIMIND_SQLITE_ENABLED": "false",
                "OMNIMIND_DB_STRICT_BACKEND": "true",
                "OMNIMIND_POSTGRES_HOST": host,
                "OMNIMIND_POSTGRES_PORT": str(port),
                "OMNIMIND_POSTGRES_DB": db,
                "OMNIMIND_POSTGRES_USER": user,
                "OMNIMIND_POSTGRES_PASSWORD": password,
                "OMNIMIND_DB_PATH": str(tmp_path / "unused_sqlite.db"),
            }
        )
        root_str = str(PROJECT_ROOT)
        py_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = root_str if not py_path else root_str + os.pathsep + py_path

        code = (
            "import asyncio, json\n"
            "from mcp_server.memory_tools import memory_health, memory_upsert, memory_get\n"
            "async def main():\n"
            "  h = await memory_health()\n"
            "  await memory_upsert('pg.e2e.key', 'pg e2e value', 'lesson')\n"
            "  g = await memory_get('pg.e2e.key', 'lesson')\n"
            "  out = {'effective': (h.get('db_backend') or {}).get('effective'), 'value': ((g.get('lesson') or {}).get('lesson'))}\n"
            "  print(json.dumps(out))\n"
            "asyncio.run(main())\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        payload = json.loads(proc.stdout.splitlines()[-1])
        assert payload["effective"] == "postgres"
        assert payload["value"] == "pg e2e value"
    finally:
        if mode == "docker":
            _run(["docker", "compose", "stop", "ai_postgres"], check=False)
