import concurrent
import concurrent.futures
import chainlit as cl
import logging
import os
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Literal, Optional, Dict, List
from dataclasses import dataclass, field
from threading import Thread
import threading
import time

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client as LangSmithClient
from langchain_mcp_adapters.client import MultiServerMCPClient

import pytz
from langchain.tools import tool
from tzlocal import get_localzone
from dotenv import load_dotenv

# Import your tools and utilities
from tools.time import time_tools
from tools.email import email_tools
from utils.auth import authenticate_email_user
from utils.graph import render_chart, should_render_chart
from utils.prompt import system_template

load_dotenv()


@dataclass
class ScheduledTask:
    """
    Represents a scheduled task with all necessary execution parameters.
    """

    task_id: str
    execution_time: datetime
    prompt: str
    after_finish_prompt: Optional[str] = None
    repeat_interval_seconds: Optional[int] = None
    timezone: str = "local"
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    last_executed: Optional[datetime] = None
    execution_count: int = 0


class TaskScheduler:
    """
    In-memory task scheduler that manages scheduled LLM executions.
    Runs in a separate thread to handle task execution timing.
    """

    def __init__(self, chatbot_instance):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.chatbot = chatbot_instance
        self.running = False
        self.scheduler_thread: Optional[Thread] = None
        self.lock = threading.Lock()
        self.loop = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

    def start(self):
        """Start the task scheduler thread."""
        if not self.running:
            self.running = True
            # Try to get the current event loop
            try:
                self.loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop in current thread, will create one when needed
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
        """
        Add a new task to the scheduler.

        Args:
            task: The ScheduledTask to add

        Returns:
            bool: True if task was added successfully
        """
        with self.lock:
            self.tasks[task.task_id] = task
            logging.info(
                f"Added task {task.task_id} scheduled for {task.execution_time}"
            )
            return True

    def remove_task(self, task_id: str) -> bool:
        """
        Remove a task from the scheduler.

        Args:
            task_id: ID of the task to remove

        Returns:
            bool: True if task was removed successfully
        """
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

    def _run_scheduler(self):
        """Main scheduler loop that runs in a separate thread."""
        while self.running:
            try:
                current_time = datetime.now()
                tasks_to_execute = []

                with self.lock:
                    for task in list(self.tasks.values()):
                        if task.status == "pending":
                            # Ensure both datetimes have the same timezone awareness
                            task_time = task.execution_time
                            if task_time.tzinfo is not None:
                                # Task time is timezone-aware, make current_time aware
                                if current_time.tzinfo is None:
                                    current_time = current_time.replace(
                                        tzinfo=task_time.tzinfo
                                    )
                            else:
                                # Task time is naive, make current_time naive
                                if current_time.tzinfo is not None:
                                    current_time = current_time.replace(tzinfo=None)

                            if current_time >= task_time:
                                tasks_to_execute.append(task)

                for task in tasks_to_execute:
                    self._execute_task(task)

                time.sleep(1)  # Check every second

            except Exception as e:
                logging.error(f"Error in scheduler loop: {e}")
                time.sleep(5)  # Wait before retrying

    def _execute_task(self, task: ScheduledTask):
        """
        Execute a single task by calling the LLM with the stored prompt.

        Args:
            task: The ScheduledTask to execute
        """
        try:
            logging.info(f"Executing task {task.task_id}: {task.prompt[:50]}...")

            with self.lock:
                task.status = "executing"
                task.last_executed = datetime.now()
                task.execution_count += 1

            # Execute the main prompt - run in thread pool to avoid blocking
            self.executor.submit(self._execute_task_sync, task)

        except Exception as e:
            logging.error(f"Error executing task {task.task_id}: {e}")
            with self.lock:
                task.status = "failed"

    def _execute_task_sync(self, task: ScheduledTask):
        """Execute task synchronously in thread pool."""
        try:
            # Execute the main prompt
            self._run_llm_prompt(task.prompt, task.task_id)

            # Execute after-finish prompt if provided
            if task.after_finish_prompt:
                self._run_llm_prompt(task.after_finish_prompt, task.task_id)

            # Handle task completion and repetition
            with self.lock:
                if task.repeat_interval_seconds and task.repeat_interval_seconds > 0:
                    # Schedule next execution maintaining timezone awareness
                    if task.execution_time.tzinfo is not None:
                        # Use timezone-aware datetime
                        next_time = datetime.now(
                            task.execution_time.tzinfo
                        ) + timedelta(seconds=task.repeat_interval_seconds)
                    else:
                        # Use naive datetime
                        next_time = datetime.now() + timedelta(
                            seconds=task.repeat_interval_seconds
                        )

                    task.execution_time = next_time
                    task.status = "pending"
                    logging.info(
                        f"Task {task.task_id} rescheduled for {task.execution_time}"
                    )
                else:
                    # One-time task completed
                    task.status = "completed"
                    logging.info(f"Task {task.task_id} completed")

        except Exception as e:
            logging.error(f"Error in task execution: {e}")
            with self.lock:
                task.status = "failed"

    def _run_llm_prompt(self, prompt: str, task_id: str):
        """
        Run LLM prompt synchronously.

        Args:
            prompt: The prompt to execute
            task_id: ID of the task for logging
        """
        try:
            logging.info(f"Running LLM prompt for task {task_id}: {prompt[:100]}...")

            if not self.chatbot.graph:
                logging.error(f"Chatbot graph not initialized for task {task_id}")
                return

            # Create a system thread_id for scheduled tasks
            thread_id = f"scheduled_task_{task_id}"

            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 100,
            }

            # Execute the prompt through the chatbot's graph
            messages = [HumanMessage(content=prompt)]

            # Run synchronously using asyncio
            if self.loop and self.loop.is_running():
                # If there's an event loop running, schedule the coroutine
                future = asyncio.run_coroutine_threadsafe(
                    self._execute_llm_async(messages, config, task_id), self.loop
                )
                future.result(timeout=300)  # 5 minute timeout
            else:
                # No event loop, create a new one
                asyncio.run(self._execute_llm_async(messages, config, task_id))

        except Exception as e:
            logging.error(f"Error executing LLM prompt for task {task_id}: {e}")

    async def _execute_llm_async(self, messages, config, task_id):
        """Execute LLM prompt asynchronously."""
        try:
            result_messages = []
            async for message, metadata in self.chatbot.graph.astream(
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


# --- Global Task Scheduler ---
task_scheduler: Optional[TaskScheduler] = None


@tool
async def schedule_task_tool(
    time_to_execute: str,
    prompt: str,
    after_finish_prompt: Optional[str] = None,
    repeat_interval_seconds: Optional[int] = None,
    timezone: str = "local",
) -> str:
    """
    Schedule a task to be executed by the AI agent at a specific future time.

    This tool allows users to schedule automated actions like:
    - "get info on nvidia and send to my email"
    - "check portfolio performance daily"
    - "send weekly market summary"

    Args:
        time_to_execute: Scheduled time in "YYYY-MM-DD HH:MM:SS" format
        prompt: Main instruction for the agent to execute
        after_finish_prompt: Optional follow-up instruction after main task
        repeat_interval_seconds: Optional interval in seconds for recurring tasks
        timezone: Timezone for execution time (e.g., 'UTC', 'America/New_York')

    Returns:
        str: Status message about the scheduled task
    """
    global task_scheduler

    if not task_scheduler:
        return "Error: Task scheduler not initialized."

    if repeat_interval_seconds is not None and repeat_interval_seconds <= 0:
        return "Error: repeat_interval_seconds must be a positive integer."

    try:
        # Parse timezone
        if timezone == "local":
            target_tz = pytz.timezone(str(get_localzone()))
        else:
            target_tz = pytz.timezone(timezone)

        # Parse execution time
        execution_dt_naive = datetime.strptime(time_to_execute, "%Y-%m-%d %H:%M:%S")
        execution_dt = target_tz.localize(execution_dt_naive)
        now_dt = datetime.now(target_tz)

        if execution_dt < now_dt:
            return f"Error: The time '{time_to_execute}' is in the past."

        # Create task
        task_id = f"task_{int(datetime.now().timestamp())}"
        task = ScheduledTask(
            task_id=task_id,
            execution_time=execution_dt,
            prompt=prompt,
            after_finish_prompt=after_finish_prompt,
            repeat_interval_seconds=repeat_interval_seconds,
            timezone=timezone,
        )

        # Add to scheduler
        if task_scheduler.add_task(task):
            # Update UI task list
            task_list = get_task_list()
            new_task = cl.Task(
                forId=task_id,
                title=f"Scheduled: {prompt[:40]}...",
                status=cl.TaskStatus.READY,
            )
            task_list.tasks.append(new_task)
            await task_list.send()

            scheduled_time_str = execution_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            return f"Task '{prompt[:30]}...' scheduled successfully for {scheduled_time_str}."
        else:
            return "Error: Failed to schedule task."

    except (ValueError, pytz.UnknownTimeZoneError) as e:
        return f"Error scheduling task: {e}"
    except Exception as e:
        logging.exception(f"Unexpected error in schedule_task_tool: {e}")
        return "An unexpected error occurred while scheduling the task."


def get_task_list() -> cl.TaskList:
    """
    Retrieve or create the TaskList for the current user session.

    Returns:
        cl.TaskList: The task list for the current session
    """
    task_list = cl.user_session.get("task_list")
    if not task_list:
        task_list = cl.TaskList()
        task_list.name = "Scheduled Tasks"
        cl.user_session.set("task_list", task_list)
    return task_list


# --- MCP Client Configuration ---

mcp_client = MultiServerMCPClient(
    {
        "yf_server": {
            "command": "python",
            "args": [str(Path("mcp-server/yf_server/server.py").resolve())],
            "env": {
                "PYTHONPATH": str(Path(__file__).resolve().parent),
            },
            "transport": "stdio",
        },
        "finance": {
            "command": "python",
            "args": [str(Path("mcp-server/investor_agent/server.py").resolve())],
            "env": {
                "PYTHONPATH": str(Path(__file__).resolve().parent),
            },
            "transport": "stdio",
        },
        "news_agent": {
            "command": "python",
            "args": [str(Path("mcp-server/news_agent/server.py").resolve())],
            "env": {
                "PYTHONPATH": str(Path(__file__).resolve().parent),
                "NEWS_API_KEY": os.environ.get("NEWS_API_KEY", ""),
            },
            "transport": "stdio",
        },
        "trading_agent": {
            "command": "uv",
            "args": [
                "--directory",
                str(Path("mcp-server/alpaca_agent").resolve()),
                "run",
                "server.py",
            ],
            "env": {
                "PYTHONPATH": str(Path(__file__).resolve().parent),
                "ALPACA_PAPER_API_KEY": os.environ.get("ALPACA_PAPER_API_KEY", ""),
                "ALPACA_PAPER_API_SECRET": os.environ.get(
                    "ALPACA_PAPER_API_SECRET", ""
                ),
            },
            "transport": "stdio",
        },
    }  # type: ignore
)


class FinanceChatbot:
    """
    Main chatbot class that handles LLM interactions and manages the conversation flow.
    Integrates with MCP servers for financial data and supports scheduled task execution.
    """

    def __init__(self):
        self.system_prompt = None
        self.main_model = None
        self.graph = None
        self.memory = MemorySaver()
        self.list_tools = []
        self.tool_node = None

    async def initialize_tools(self):
        """Initialize MCP client, tools, model, and build the processing graph."""
        await self._initialize_client_and_tools()
        self._initialize_model()
        self._build_graph()

    async def _initialize_client_and_tools(self):
        """Initialize MCP client and aggregate all available tools."""
        logging.info("Initializing MCP client and tools...")
        mcp_tools = await mcp_client.get_tools()
        self.list_tools = mcp_tools + time_tools + [schedule_task_tool] + email_tools
        self.tool_node = ToolNode(tools=self.list_tools)
        logging.info("Tools initialized successfully.")

    def _initialize_model(self):
        """Initialize the language model with system prompt from LangSmith or defaults."""
        logging.info("Initializing model and system prompt...")
        try:
            langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
            if not langsmith_api_key:
                raise ValueError("LANGSMITH_API_KEY not set.")

            client = LangSmithClient(api_key=langsmith_api_key)
            prompt_template = client.pull_prompt("finance-chatbot", include_model=True)
            prompt_str = prompt_template.steps[0].format(
                date=datetime.now().strftime("%Y-%m-%d")
            )
            param = prompt_template.steps[1].bound._get_ls_params()
            model_name = prompt_template.steps[1].bound.model.split("/")[-1]
            model_provider = param["ls_provider"]
            self.system_prompt = SystemMessage(content=prompt_str)

        except Exception as e:
            logging.warning(
                f"Could not pull from LangSmith: {e}. Using default prompt."
            )
            model_name = None
            model_provider = None
            rendered = system_template.format(date=datetime.now().strftime("%Y-%m-%d"))
            self.system_prompt = SystemMessage(content=rendered)

        self.main_model = init_chat_model(
            os.getenv("MODEL_NAME", model_name),
            model_provider=os.getenv("MODEL_PROVIDER", model_provider),
        ).bind_tools(self.list_tools)
        logging.info("Model initialized successfully.")

    def _build_graph(self):
        """Build and compile the LangGraph for processing conversations."""
        logging.info("Building state graph...")
        builder = StateGraph(MessagesState)

        builder.add_node("agent", self.call_main_model)
        builder.add_node("tools", self.tool_node)
        builder.add_node("render_chart", render_chart)
        builder.add_node("should_render_chart_decision", lambda state: state)

        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent", self.should_use_tools, {"tools": "tools", "END": END}
        )
        builder.add_conditional_edges(
            "tools",
            self.after_tools,
            {"agent": "agent", "render_or_end": "should_render_chart_decision"},
        )
        builder.add_conditional_edges(
            "should_render_chart_decision",
            should_render_chart,
            {"render_chart": "render_chart", "agent": "agent"},
        )
        builder.add_edge("render_chart", "agent")

        self.graph = builder.compile(checkpointer=self.memory)
        logging.info("Graph built and compiled successfully.")

    def call_main_model(self, state: MessagesState):
        """
        Node function to invoke the main language model.

        Args:
            state: Current conversation state

        Returns:
            dict: Updated state with model response
        """
        messages = [self.system_prompt] + state["messages"]
        response = self.main_model.invoke(messages)
        return {"messages": [response]}

    def should_use_tools(self, state: MessagesState) -> Literal["tools", "END"]:
        """
        Determine whether to use tools or end the conversation.

        Args:
            state: Current conversation state

        Returns:
            str: Next node to execute
        """
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", []) else "END"

    def after_tools(self, state: MessagesState) -> Literal["agent", "render_or_end"]:
        """
        Determine next action after tool execution.

        Args:
            state: Current conversation state

        Returns:
            str: Next node to execute
        """
        last_message = state["messages"][-1]
        return "render_or_end" if last_message.type == "tool" else "agent"

    async def process_message(
        self, message_content: str, user_id: str, session_id: str
    ):
        """
        Process a user message through the chatbot system.

        Args:
            message_content: The user's message
            user_id: User identifier
            session_id: Session identifier

        Returns:
            async generator: Streaming response from the model
        """
        thread_id = f"{user_id}_{session_id}"

        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 100,
            "user_id": user_id,
        }

        cb = cl.LangchainCallbackHandler(stream_final_answer=True)

        async for message, metadata in self.graph.astream(
            {"messages": [HumanMessage(content=message_content)]},
            stream_mode="messages",
            config=RunnableConfig(callbacks=[cb], **config),
        ):
            if (
                message.content
                and not isinstance(message, HumanMessage)
                and not metadata["langgraph_node"] == "tools"
            ):
                yield message.content


# --- Global Instances ---
app = FinanceChatbot()


# --- Chainlit Event Handlers ---


@cl.on_app_startup
def on_app_startup():
    """Initialize logging and start the task scheduler."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("App is starting up...")


def initialize_scheduler():
    """Initialize and start the task scheduler."""
    global task_scheduler
    if task_scheduler is None:
        task_scheduler = TaskScheduler(app)
        task_scheduler.start()
        logging.info("Task scheduler started")


@cl.on_stop
def on_stop():
    """Gracefully shutdown services when the app stops."""
    logging.info("Application stopping. Shutting down services.")
    global task_scheduler
    if task_scheduler:
        task_scheduler.stop()


@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    """
    Handle user authentication.

    Args:
        username: User's email
        password: User's password

    Returns:
        cl.User or None: User object if authenticated, None otherwise
    """
    is_logged_in = await authenticate_email_user(email=username, password=password)
    if is_logged_in:
        return cl.User(identifier=username, metadata={"provider": "credentials"})
    return None


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new chat session."""
    user = cl.user_session.get("user")

    # Initialize scheduler if not already done
    initialize_scheduler()

    await app.initialize_tools()

    if not user:
        await cl.Message(content="Authentication required. Please log in.").send()
        return

    # Initialize task list for the session
    task_list = cl.user_session.get("task_list")
    if not task_list:
        task_list = cl.TaskList(name="Scheduled Tasks", status="Running")
        cl.user_session.set("task_list", task_list)


@cl.on_message
async def on_message(msg: cl.Message):
    """
    Handle incoming user messages.

    Args:
        msg: The incoming message from the user
    """
    user = cl.user_session.get("user")
    if not user:
        await cl.Message(content="Please log in to continue chatting.").send()
        return

    final_answer = cl.Message(content="")

    async for content in app.process_message(
        msg.content, user.identifier, cl.context.session.id
    ):
        await final_answer.stream_token(content)

    await final_answer.send()


@cl.set_starters  # type: ignore
async def set_starters():
    """
    Define starter suggestions for new chat sessions.

    Returns:
        list: List of starter suggestions
    """
    return [
        cl.Starter(
            label="📈 Apple Investment Analysis",
            message="Provide a full investment analysis of Apple (AAPL).",
        ),
        cl.Starter(
            label="📰 Semiconductor Industry News",
            message="Show me the latest news impacting the semiconductor industry.",
        ),
        cl.Starter(
            label="⏰ Schedule Daily Market Summary",
            message="Schedule a daily market summary to be sent to my email at 9 AM.",
        ),
        cl.Starter(
            label="🤖 What Can You Do?",
            message="List all your financial and analytical capabilities in detail.",
        ),
    ]


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
