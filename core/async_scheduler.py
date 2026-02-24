"""Async task scheduler for memory operations.

Provides background task scheduling for memory operations like
indexing, consolidation, etc.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional
import uuid


class TaskStatus(Enum):
    """Task status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority (lower number = higher priority)."""

    HIGH = 1
    NORMAL = 5
    LOW = 10


@dataclass
class ScheduledTask:
    """A scheduled task."""

    id: str
    name: str
    func_name: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Any = None
    error: Optional[str] = None


class AsyncScheduler:
    """Simple async task scheduler.

    Provides basic task scheduling for memory operations.
    For production, consider using Celery or Redis queues.
    """

    def __init__(self, max_concurrent: int = 3):
        """Initialize scheduler.

        Args:
            max_concurrent: Maximum concurrent tasks
        """
        self.max_concurrent = max_concurrent
        self._tasks: dict[str, ScheduledTask] = {}
        self._running: set[str] = set()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._background_task: Optional[asyncio.Task] = None
        self._running_flag = False

    async def schedule(
        self,
        name: str,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs,
    ) -> str:
        """Schedule a task for async execution.

        Args:
            name: Task name
            func: Async function to run
            args: Positional arguments
            priority: Task priority
            kwargs: Keyword arguments

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())[:12]

        task = ScheduledTask(
            id=task_id,
            name=name,
            func_name=func.__name__,
            args=args,
            kwargs=kwargs,
            priority=priority,
        )

        self._tasks[task_id] = task

        # Start background runner if not running
        if not self._running_flag:
            await self._start()

        return task_id

    async def _start(self) -> None:
        """Start background task processor."""
        if self._running_flag:
            return

        self._running_flag = True
        self._background_task = asyncio.create_task(self._process_tasks())

    async def _process_tasks(self) -> None:
        """Process queued tasks."""
        while self._running_flag:
            # Find pending task with highest priority
            pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]

            if not pending:
                await asyncio.sleep(0.5)
                continue

            # Sort by priority
            pending.sort(key=lambda t: t.priority.value)
            task = pending[0]

            # Wait for semaphore
            async with self._semaphore:
                if task.status != TaskStatus.PENDING:
                    continue

                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now(timezone.utc).isoformat()
                self._running.add(task.id)

                try:
                    # Execute task - this is a simplified version
                    # In production, you'd call the actual function
                    result = f"Task {task.id} would execute {task.func_name}"
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.error = str(e)
                    task.status = TaskStatus.FAILED
                finally:
                    task.completed_at = datetime.now(timezone.utc).isoformat()
                    self._running.discard(task.id)

    async def get_task(self, task_id: str) -> Optional[dict]:
        """Get task status and result."""
        task = self._tasks.get(task_id)
        if not task:
            return None

        return {
            "id": task.id,
            "name": task.name,
            "func_name": task.func_name,
            "priority": task.priority.name,
            "status": task.status.value,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "result": task.result,
            "error": task.error,
        }

    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 50,
    ) -> list[dict]:
        """List tasks."""
        tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]

        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return [
            {
                "id": t.id,
                "name": t.name,
                "func_name": t.func_name,
                "status": t.status.value,
                "created_at": t.created_at,
            }
            for t in tasks[:limit]
        ]

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False

        task.status = TaskStatus.CANCELLED
        return True

    async def shutdown(self) -> None:
        """Shutdown scheduler."""
        self._running_flag = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass


# Singleton
_scheduler: Optional[AsyncScheduler] = None


def get_scheduler() -> AsyncScheduler:
    """Get scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncScheduler()
    return _scheduler
