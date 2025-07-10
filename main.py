from datetime import datetime
import json
import logging
import os
import asyncio
from pathlib import Path
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from matplotlib import pyplot as plt
import chainlit as cl
import pandas as pd

from utils.prompt import system_prompt

memory = MemorySaver()

client = MultiServerMCPClient(
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
        # "fetch": {"transport": "stdio", "command": "uvx", "args": ["mcp-server-fetch"]},
        # "memory": {
        #     "transport": "stdio",
        #     "command": "npx",
        #     "args": ["-y", "@modelcontextprotocol/server-memory"],
        # },
    }  # type: ignore
)


async def initialize_tools():
    return await client.get_tools()


tools = asyncio.run(initialize_tools())

# Models
main_model = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_BASE", "gemini-2.0-pro"),
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

main_model = main_model.bind_tools(tools)
tool_node = ToolNode(tools=tools)


# Node Functions
def call_main_model(state: MessagesState):
    """Main model that decides which tools to use"""
    messages = [system_prompt] + state["messages"]
    response = main_model.invoke(messages)
    return {"messages": [response]}


def analyze_response_need(state: MessagesState):
    """Analyze if the response needs formatting"""
    messages = state["messages"]
    last_message = messages[-1]

    # Simple heuristic: if response is long or contains tool results, it might need formatting
    needs_formatting = len(last_message.content) > 500 or any(  # Long response
        hasattr(msg, "tool_calls") and msg.tool_calls for msg in messages[-3:]
    )  # Recent tool usage

    # Store decision in state
    new_state = state.copy()
    new_state["needs_formatting"] = needs_formatting
    return new_state


def get_trailing_tool_messages_with_indices(arr):
    result = []
    for i in range(len(arr) - 1, -1, -1):
        item = arr[i]
        if hasattr(item, "type") and item.type == "tool":
            result.append((item, i))
        else:
            break
    return list(reversed(result))


async def render_chart(state: MessagesState) -> MessagesState:
    messages = state["messages"]
    trailing_tool_messages = get_trailing_tool_messages_with_indices(messages)

    if not trailing_tool_messages:
        logging.info("No trailing 'tool' messages found for chart generation.")
        return {"messages": messages}

    for tool_message_obj, original_index in trailing_tool_messages:
        try:
            tool_content = json.loads(tool_message_obj.content)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding error at index {original_index}: {e}")
            continue

        if "chart" in tool_content:
            original_message_text = tool_content.get("message", "")
            try:
                charts = await generate_charts_from_embedded_data(tool_content["chart"])
                del tool_content["chart"]
                log_message = (
                    f"LOG: Chart(s) for {', '.join(charts)} have been sent to client."
                )
                tool_content["message"] = (
                    f"{original_message_text}\n{log_message}".strip()
                )
                logging.info(log_message)

            except Exception as e:
                logging.error(f"Chart generation error at index {original_index}: {e}")
                if "chart" in tool_content:
                    del tool_content["chart"]
                error_message = f"Error generating charts: {e}. Please check the chart configuration."
                tool_content["message"] = (
                    f"{original_message_text}\n{error_message}".strip()
                )

        messages[original_index].content = json.dumps(tool_content)

    return {"messages": messages}


# Conditional Edge Functions
def should_use_tools(state: MessagesState) -> Literal["tools", "END"]:
    """Route to tools if model made tool calls, otherwise analyze response"""
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "END"


def should_format_response(state: MessagesState) -> Literal["format", "END"]:
    """Decide whether to format the response based on analysis"""
    if state.get("needs_formatting", False):
        return "format"
    return "END"


def after_tools(
    state: MessagesState,
) -> Literal["agent", "render_or_end"]:  # Renamed to better reflect its decision
    """After using tools, decide if we need to call the agent again or render/end"""
    messages = state["messages"]

    if hasattr(messages[-1], "type") and messages[-1].type == "tool":
        return "render_or_end"
    return "agent"


def should_render_chart(state: MessagesState) -> Literal["render_chart", "agent"]:
    messages = state["messages"]
    if not messages:
        logging.info("No messages found in state.")
        return "agent"

    trailing_tool_messages = get_trailing_tool_messages_with_indices(messages)

    if not trailing_tool_messages:
        logging.info("No trailing 'tool' messages found for chart generation.")
        return "agent"

    for tool_message_obj, original_index in trailing_tool_messages:
        try:
            tool_content = json.loads(tool_message_obj.content)
            if isinstance(tool_content, dict) and "chart" in tool_content:
                logging.info(
                    f"Chart detected in tool message at index {original_index}."
                )
                return "render_chart"
        except json.JSONDecodeError as e:
            logging.warning(
                f"JSON decoding error in tool message at index {original_index}: {e}. Skipping this message."
            )
        except TypeError as e:
            logging.warning(
                f"Type error for tool message content at index {original_index}: {e}. Content: {tool_message_obj.content}. Skipping this message."
            )
        except Exception as e:
            logging.error(
                f"Unexpected error processing tool message at index {original_index}: {e}. Skipping this message."
            )

    logging.info("No chart content found in any trailing tool messages.")
    return "agent"


# Build the StateGraph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("agent", call_main_model)
builder.add_node("tools", tool_node)
builder.add_node("render_chart", render_chart)
builder.add_node("should_render_chart_decision", lambda state: state)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_use_tools, {"tools": "tools", "END": END})
builder.add_conditional_edges(
    "tools",
    after_tools,
    {"agent": "agent", "render_or_end": "should_render_chart_decision"},
)
builder.add_conditional_edges(
    "should_render_chart_decision",
    should_render_chart,
    {"render_chart": "render_chart", "agent": "agent"},
)
builder.add_edge("render_chart", "agent")
graph = builder.compile(checkpointer=memory)

@cl.on_app_startup
def on_app_startup():
    logging.info("App is starting up...")


@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session - requires authentication"""
    user = cl.user_session.get("user")

    if not user:
        await cl.Message(
            content="Authentication required. Please log in to continue."
        ).send()
        return


async def generate_charts_from_embedded_data(chart_configs):
    """
    Generates and sends multiple plots from a configuration file
    that includes embedded data.

    Args:
        chart_configs (list): A list of dictionaries, where each dict
                              defines a chart and contains its own data.
    """
    logging.info("chart drawing called")

    for config in chart_configs:
        logging.info("creating chart for config: %s", config["title"])
        logging.info(config)
        # 1. Create a DataFrame from the embedded data
        df = pd.DataFrame(config["data"])

        # Set the index column (e.g., 'Date') and convert it to datetime
        if "index_column" in config:
            df.set_index(config["index_column"], inplace=True)
            df.index = pd.to_datetime(df.index)

        # 2. Create the plot
        figsize = tuple(config.get("figsize", (20, 10)))
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the primary data lines
        if "plots" in config:
            for plot_info in config["plots"]:
                plot_kwargs = {
                    "label": plot_info.get("label"),
                    "color": plot_info.get("color"),
                    "linestyle": plot_info.get("linestyle"),
                }
                plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}
                ax.plot(df[plot_info["data_key"]], **plot_kwargs)

        # Add any horizontal lines
        if "hlines" in config:
            for hline_info in config["hlines"]:
                ax.axhline(**hline_info)

        # 3. Finalize and send the plot
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

        await cl.Message(
            content=config["title"],
            elements=[
                cl.Pyplot(name="plot", figure=fig, display="inline"),
            ],
        ).send()
    return [x["title"] for x in chart_configs if "title" in x]


@cl.set_starters  # type: ignore
async def set_starters():
    """Chat starter suggestions with better labels and a prompt for capabilities overview."""

    starters = [
        (
            "📈 Apple Investment Analysis",
            "Provide a full investment analysis of Apple (AAPL).",
        ),
        (
            "📉 Tesla Buy, Sell, or Hold?",
            "Should I buy, sell, or hold Tesla (TSLA)?",
        ),
        (
            "📊 Nvidia Technical Summary",
            "Give me a technical summary for Nvidia (NVDA).",
        ),
        (
            "🧾 Amazon Financial Results",
            "What are the latest financial results for Amazon (AMZN)?",
        ),
        (
            "📊 Microsoft vs Google Stock Comparison",
            "Compare the stock performance of Microsoft (MSFT) and Google (GOOGL).",
        ),
        (
            "📰 Semiconductor Industry News",
            "Show me the latest news impacting the semiconductor industry.",
        ),
        (
            "😨 Fear & Greed Index Now",
            "What is the current Fear & Greed Index?",
        ),
        (
            "🏦 Johnson & Johnson Institutional Holders",
            "Who are the biggest institutional holders of Johnson & Johnson (JNJ)?",
        ),
        (
            "💵 Starbucks Price Check",
            "What is the current stock price of Starbucks?",
        ),
        (
            "📉 Netflix 1-Year Price Chart",
            "Generate a 1-year price chart for Netflix (NFLX).",
        ),
        (
            "🤖 What Can You Do?",
            "List all your financial and analytical capabilities in detail, including data sources and supported analysis types.",
        ),
    ]

    return [cl.Starter(label=label, message=message) for label, message in starters]


@cl.on_message
async def on_message(msg: cl.Message):
    """Handle incoming messages - authentication required"""
    user = cl.user_session.get("user")

    # Block chat if user not authenticated
    if not user:
        await cl.Message(content="Please log in to continue chatting.").send()
        return

    # Use user-specific thread ID for persistent memory
    thread_id = f"{user.identifier}_{cl.context.session.id}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    final_answer = cl.Message(content="")

    async for message, metadata in graph.astream(
        {"messages": [HumanMessage(content=msg.content)]},
        stream_mode="messages",
        config=RunnableConfig(callbacks=[cb], **config),
    ):
        if (
            message.content
            and not isinstance(message, HumanMessage)
            and not metadata["langgraph_node"] == "tools"
        ):
            await final_answer.stream_token(message.content)

    await final_answer.send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
