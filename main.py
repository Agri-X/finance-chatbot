import logging
import os

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.tools import BaseTool
from langsmith import Client as LangSmithClient

from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from chainlit_local import chainlit_global as cl

from tools.economic_calendar import scrape_trading_economics_calendar
from tools.time import time_tools
from tools.email import email_tools
from tools.scheduler import scheduler_tools

global task_scheduler

from utils.scheduler import TaskScheduler, task_scheduler
from utils.auth import authenticate_email_user
from utils.prompt import system_template

load_dotenv()

logger = logging.getLogger(__name__)


DEFAULT_RECURSION_LIMIT = 100
BASE_PATH = Path(__file__).resolve().parent


def create_mcp_client():
    """Initialize MCP client with all required servers."""
    servers_config = {
        "yf_server": {
            "command": "python",
            "args": [str(BASE_PATH / "mcp-server/yf_server/server.py")],
            "env": {"PYTHONPATH": str(BASE_PATH)},
            "transport": "stdio",
        },
        "finance": {
            "command": "python",
            "args": [str(BASE_PATH / "mcp-server/investor_agent/server.py")],
            "env": {"PYTHONPATH": str(BASE_PATH)},
            "transport": "stdio",
        },
        "news_agent": {
            "command": "python",
            "args": [str(BASE_PATH / "mcp-server/news_agent/server.py")],
            "env": {
                "PYTHONPATH": str(BASE_PATH),
                "NEWS_API_KEY": os.environ.get("NEWS_API_KEY", ""),
            },
            "transport": "stdio",
        },
        "trading_agent": {
            "command": "uv",
            "args": [
                "--directory",
                str(BASE_PATH / "mcp-server/alpaca_agent"),
                "run",
                "server.py",
            ],
            "env": {
                "PYTHONPATH": str(BASE_PATH),
                "ALPACA_PAPER_API_KEY": os.environ.get("ALPACA_PAPER_API_KEY", ""),
                "ALPACA_PAPER_API_SECRET": os.environ.get(
                    "ALPACA_PAPER_API_SECRET", ""
                ),
            },
            "transport": "stdio",
        },
    }

    return MultiServerMCPClient(servers_config)  # type: ignore


async def load_system_prompt():
    """Load system prompt from LangSmith or use default."""
    try:
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        if not langsmith_api_key:
            raise ValueError("LANGSMITH_API_KEY not set")

        client = LangSmithClient(api_key=langsmith_api_key)
        prompt_template = client.pull_prompt("finance-chatbot", include_model=True)

        prompt_str = prompt_template.steps[0].format(
            date=datetime.now().strftime("%Y-%m-%d")
        )

        param = prompt_template.steps[1].bound._get_ls_params()
        model_name = prompt_template.steps[1].bound.model.split("/")[-1]

        model_provider = param["ls_provider"]

        logger.info(f"Loaded system prompt from LangSmith with model: {model_name}")
        return SystemMessage(content=prompt_str), model_name, model_provider

    except Exception as e:
        logger.warning(f"Could not pull from LangSmith: {e}. Using default prompt.")

        rendered = system_template.format(date=datetime.now().strftime("%Y-%m-%d"))
        return SystemMessage(content=rendered), None, None


mcp_client = create_mcp_client()


async def initialize_tools_and_model():
    """Initialize all tools and the main model."""
    try:
        prompt, session_model_name, session_model_provider = await load_system_prompt()

        model_name = os.getenv("MODEL_NAME", session_model_name)
        model_provider = os.getenv("MODEL_PROVIDER", session_model_provider)

        global all_tools

        try:
            assert all_tools
        except:
            await populate_tools()

        memory = MemorySaver()

        model = init_chat_model(f"{model_provider}:{model_name}")

        main_model = create_react_agent(
            model, all_tools, prompt=prompt, checkpointer=memory
        )

        logger.info(f"Initialized {len(all_tools)} tools and model: {model_name}")
        return main_model, model

    except Exception as e:
        logger.error(f"Failed to initialize tools and model: {e}")
        raise


def initialize_scheduler(graph: CompiledStateGraph) -> None:
    """Initialize and start the task scheduler."""
    try:
        global task_scheduler

        if not task_scheduler:
            task_scheduler = TaskScheduler(graph)
            task_scheduler.start()
        logger.info("Task scheduler started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize scheduler: {e}")
        raise


async def get_thread_config(user: cl.User) -> Dict:
    """Generate thread configuration for a user."""
    thread_id = f"{user.identifier}_{cl.context.session.id}"
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": DEFAULT_RECURSION_LIMIT,
        "user_id": user.identifier,
    }


@cl.on_app_startup
async def on_app_startup():
    """Initialize logging and application startup."""
    logger.info("Financial chatbot is starting up...")

    await populate_tools()

    main_model, model = await initialize_tools_and_model()

    initialize_scheduler(main_model)

    logger.info("Financial chatbot is started")


async def populate_tools():
    mcp_tools = await mcp_client.get_tools()

    global all_tools

    all_tools = (
        mcp_tools
        + time_tools
        + scheduler_tools
        + email_tools
        + [scrape_trading_economics_calendar]
    )

    logger.info(f"Populated {len(all_tools)} tools.")


@cl.on_stop
async def on_stop():
    """Gracefully shutdown services when the app stops."""
    logger.info("Application stopping. Shutting down services...")

    global task_scheduler

    if task_scheduler:
        try:
            task_scheduler.stop()
            logger.info("Task scheduler stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")


@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Handle user authentication."""
    try:
        is_logged_in = await authenticate_email_user(email=username, password=password)
        if is_logged_in:
            logger.info(f"User {username} authenticated successfully")
            return cl.User(identifier=username, metadata={"provider": "credentials"})
        logger.warning(f"Authentication failed for user: {username}")
        return None
    except Exception as e:
        logger.error(f"Authentication error for {username}: {e}")
        return None


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new chat session."""
    logger.info("Starting Chat")

    user = cl.user_session.get("user")
    global task_scheduler

    if not user:
        await cl.Message(content="🔐 Authentication required. Please log in.").send()
        return

    try:
        logger.info(f"Starting chat session for user: {user.identifier}")

        main_model, model = await initialize_tools_and_model()

        cl.user_session.set("main_model", main_model)
        cl.user_session.set("model", model)
        cl.user_session.set("task_scheduler", task_scheduler)

        task_scheduler = TaskScheduler(main_model)
        task_scheduler.start()

    except Exception as e:
        logger.error(f"Failed to start chat session for {user.identifier}: {e}")
        await cl.Message(
            content="❌ Sorry, there was an error initializing your session. Please try again."
        ).send()
    logger.info("Starting Chat Finished")


@cl.on_message
async def on_message(msg: cl.Message):
    """Handle incoming user messages."""
    user = cl.user_session.get("user")
    model = cl.user_session.get("main_model")

    if not user or not isinstance(user, cl.User):
        await cl.Message(content="🔐 Please log in to continue chatting.").send()
        return

    if not isinstance(model, CompiledStateGraph):
        await cl.Message(
            content="❌ System not properly initialized. Please refresh and try again."
        ).send()
        return

    try:
        logger.info(f"Processing message from {user.identifier}: {msg.content}...")

        config = await get_thread_config(user)
        cb = cl.LangchainCallbackHandler(stream_final_answer=True)
        final_answer = cl.Message(content="")

        async for message, metadata in model.astream(
            {"messages": [HumanMessage(content=msg.content)]},
            stream_mode="messages",
            config=RunnableConfig(callbacks=[cb], **config),
        ):
            if (
                not isinstance(message, HumanMessage)
                and metadata.get("langgraph_node") != "tools"
            ):
                await final_answer.stream_token(message.content)

        await final_answer.send()

    except Exception as e:
        logger.error(f"Error processing message from {user.identifier}: {e}")
        await cl.Message(
            content="❌ Sorry, I encountered an error processing your request. Please try again."
        ).send()


@cl.set_starters
async def set_starters(user: Optional[cl.User]) -> List[cl.Starter]:
    """Define starter suggestions for new chat sessions."""
    return [
        cl.Starter(
            label="📈 Apple Investment Analysis",
            message="Provide a comprehensive investment analysis of Apple (AAPL) including financials, technical indicators, and market outlook.",
        ),
        cl.Starter(
            label="📰 Semiconductor Industry News",
            message="Show me the latest news and trends impacting the semiconductor industry and related stocks.",
        ),
        cl.Starter(
            label="⏰ Schedule Market Summary",
            message="Schedule a daily market summary to be sent to my email in 5 minutes.",
        ),
        cl.Starter(
            label="💵 Buy TSLA Stock",
            message="Execute a buy order for 10 shares of TSLA at current market price.",
        ),
        cl.Starter(
            label="💰 Portfolio Summary",
            message="Show me my current portfolio performance, holdings, and P&L summary.",
        ),
        cl.Starter(
            label="📋 NVDA Options Analysis",
            message="Analyze call options for NVDA expiring on 2025-01-17 with strike prices near current market.",
        ),
        cl.Starter(
            label="📊 Tech Stock Comparison",
            message="Compare the financial performance and valuation metrics of GOOGL vs MSFT.",
        ),
        cl.Starter(
            label="🗓️ Task Management",
            message="Show me all my scheduled tasks and their current status.",
        ),
        cl.Starter(
            label="🤖 System Capabilities",
            message="What are your complete financial analysis and trading capabilities?",
        ),
    ]


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
