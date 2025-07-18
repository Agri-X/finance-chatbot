from dataclasses import dataclass, field
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage


import asyncio
import concurrent.futures
import logging
import threading
import time
from datetime import datetime
from threading import Thread
from typing import Dict, List, Optional


@dataclass
class ScheduledTask:
    """Represents a scheduled task with all necessary execution parameters."""

    task_id: str
    execution_time: datetime
    prompt: str
    timezone: str = "local"
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    last_executed: Optional[datetime] = None
    execution_count: int = 0


class TaskScheduler:
    """In-memory task scheduler that manages scheduled LLM executions."""

    def __init__(self, graph):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.graph = graph
        self.running = False
        self.scheduler_thread: Optional[Thread] = None
        self.lock = threading.Lock()
        self.loop = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

    def start(self):
        """Start the task scheduler thread."""
        if not self.running:
            self.running = True
            try:
                self.loop = asyncio.get_event_loop()
            except RuntimeError:
                self.loop = None

            self.scheduler_thread = Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            logging.info("Task scheduler started")

    def stop(self):
        """Stop the task scheduler thread."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        if self.executor:
            self.executor.shutdown(wait=True)
        logging.info("Task scheduler stopped")

    def add_task(self, task: ScheduledTask) -> bool:
        """Add a new task to the scheduler."""
        with self.lock:
            self.tasks[task.task_id] = task
            logging.info(
                f"Added task {task.task_id} scheduled for {task.execution_time}"
            )
            return True

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the scheduler."""
        with self.lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                logging.info(f"Removed task {task_id}")
                return True
            return False

    def get_pending_tasks(self) -> List[ScheduledTask]:
        """Get all pending tasks."""
        with self.lock:
            return [task for task in self.tasks.values() if task.status == "pending"]

    def get_all_tasks(self) -> List[ScheduledTask]:
        """Get all tasks."""
        with self.lock:
            return list(self.tasks.values())

    def _run_scheduler(self):
        """Main scheduler loop that runs in a separate thread."""
        while self.running:
            try:
                current_time = datetime.now()
                tasks_to_execute = []

                with self.lock:
                    for task in list(self.tasks.values()):
                        if task.status == "pending":
                            task_time = task.execution_time
                            if task_time.tzinfo is not None:
                                if current_time.tzinfo is None:
                                    current_time = current_time.replace(
                                        tzinfo=task_time.tzinfo
                                    )
                            else:
                                if current_time.tzinfo is not None:
                                    current_time = current_time.replace(tzinfo=None)

                            if current_time >= task_time:
                                tasks_to_execute.append(task)

                for task in tasks_to_execute:
                    self._execute_task(task)

                time.sleep(1)  # Check every second

            except Exception as e:
                logging.error(f"Error in scheduler loop: {e}")
                time.sleep(5)

    def _execute_task(self, task: ScheduledTask):
        """Execute a single task by calling the LLM with the stored prompt."""
        try:
            logging.info(f"Executing task {task.task_id}: {task.prompt[:50]}...")

            with self.lock:
                task.status = "executing"
                task.last_executed = datetime.now()
                task.execution_count += 1

            self.executor.submit(self._execute_task_sync, task)

        except Exception as e:
            logging.error(f"Error executing task {task.task_id}: {e}")
            with self.lock:
                task.status = "failed"

    def _execute_task_sync(self, task: ScheduledTask):
        """Execute task synchronously in thread pool."""
        try:
            self._run_llm_prompt(task.prompt, task.task_id)
            with self.lock:
                task.status = "completed"
                logging.info(f"Task {task.task_id} completed")

        except Exception as e:
            logging.error(f"Error in task execution: {e}")
            with self.lock:
                task.status = "failed"

    def _run_llm_prompt(self, prompt: str, task_id: str):
        """Run LLM prompt synchronously."""
        try:
            logging.info(f"Running LLM prompt for task {task_id}: {prompt[:100]}...")

            if not self.graph:
                logging.error(f"Chatbot graph not initialized for task {task_id}")
                return

            thread_id = f"scheduled_task_{task_id}"
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 100,
            }

            messages = [HumanMessage(content=prompt)]

            if self.loop and self.loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._execute_llm_async(messages, config, task_id), self.loop
                )
                future.result(timeout=300)
            else:
                asyncio.run(self._execute_llm_async(messages, config, task_id))

        except Exception as e:
            logging.error(f"Error executing LLM prompt for task {task_id}: {e}")

    async def _execute_llm_async(self, messages, config, task_id):
        """Execute LLM prompt asynchronously."""
        try:
            result_messages = []
            async for message, metadata in self.graph.astream(
                {"messages": messages},
                stream_mode="messages",
                config=RunnableConfig(**config),
            ):
                if message.content and not isinstance(message, HumanMessage):
                    result_messages.append(message.content)

            final_result = "".join(result_messages)
            logging.info(
                f"Task {task_id} completed. Result length: {len(final_result)} characters"
            )

            if final_result:
                logging.info(f"Task {task_id} result preview: {final_result[:200]}...")

        except Exception as e:
            logging.error(f"Error in async LLM execution for task {task_id}: {e}")
            raise


task_scheduler: Optional[TaskScheduler] = None
