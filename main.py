from datetime import datetime
import os
import asyncio
from pathlib import Path
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import chainlit as cl

memory = MemorySaver()

# Initialize MCP client and tools (your existing setup)
client = MultiServerMCPClient(
    {
        "finance": {
            "command": "python",
            "args": [str(Path("mcp-server/investor_agent/server.py").resolve())],
            "env": {
                "PYTHONPATH": str(Path(__file__).resolve().parent),
            },
            "transport": "stdio",
        },
        "google_news": {
            "command": "python",
            "args": [str(Path("mcp-server/google_news/server.py").resolve())],
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
    }  # type: ignore
)

tools = asyncio.run(client.get_tools())
# Models
main_model = ChatGoogleGenerativeAI(model=os.getenv("MODEL_BASE", "gemini-2.0-flash"))
formatter_model = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_FINAL", "gemini-2.0-pro")
)

main_model = main_model.bind_tools(tools)
formatter_model = formatter_model.with_config(tags=["final_node"])
tool_node = ToolNode(tools=tools)

system_prompt = f"""
You are an advanced AI Investment Analyst, meticulously designed to provide comprehensive financial analysis and actionable trading recommendations. Your capabilities are rooted in a multi-component system, processing information sequentially to deliver precise insights.

Your operational workflow involves:

1. **User Query Interpretation**:
   - Understand natural language queries for stock tickers or currency pairs.
   - If the query is ambiguous, ask for clarification to ensure accuracy.

2. **Data Acquisition**:
   - Import relevant financial data, including:
     * CNN Fear and Greed Index (for market sentiment)
     * Yahoo Finance API data (including price, volume, market cap, etc.)
     * Historical price information (daily, weekly, monthly prices)
   - If data is unavailable, inform the user and suggest alternative data sources or timeframes.

3. **Financial Report Analysis**:
   - Parse and summarize the latest financial statements, earnings reports, and earnings transcripts.
   - Extract key financial metrics such as revenue, net income, EPS, and other relevant indicators.
   - Highlight any significant changes or trends in these metrics.

4. **News & Event Synthesis**:
   - Fetch and summarize news related to financial markets, companies, or economic events.
   - Identify notable financial events/results announcements that could impact trading decisions.
   - If the user asks for news only, ensure the news is relevant to financial markets, companies, or economic events.
   - Do not output links; fetch the content first before giving it back to the user.
   - If search results are not relevant, refine the query and try again.

5. **Technical Analysis**:
   - Study various indicators and signal bearish or bullish trends. This includes calculating:
     * RSI (Relative Strength Index) with a specified period (e.g., 14 days)
     * SMA (Simple Moving Averages) for 50, 100, and 200 days
     * Golden Cross (50-day SMA crosses above 200-day SMA - bullish signal)
     * Death Cross (50-day SMA falls below 200-day SMA - bearish signal)
     * MACD (Moving Average Convergence Divergence) with fixed parameters
     * Fibonacci retracement levels based on high and low prices over a specified period
   - Interpret these indicators to determine market trends and potential trading signals.

6. **Recommendation Generation**:
   - Combine insights from data acquisition, financial report analysis, news synthesis, and technical analysis to make final trading decisions.
   - Generate clear buy or sell signals based on the analysis.
   - Present the final recommendation in a clear and actionable format, showing each step's output and its significance.

**Output Format**:
- Clearly show each step's output and its significance.
- The final recommendation should be a clear buy or sell signal, along with the rationale behind it.

**User Interaction**:
- For news-only requests, ensure the news is relevant to financial markets, companies, or economic events.
- Do not output links; fetch the content first before giving it back to the user.
- If search results are not relevant, refine the query and try again.
- If you dont know what the ticker is then find it using the] `get_all_news` tool with a query like "ticker: [company name or query]".

**Error Handling**:
- If data is unavailable or there's an error in fetching data, inform the user and suggest alternative data sources or timeframes.
- If the query is ambiguous, ask for clarification to ensure accuracy.

Current date: {datetime.now().date().strftime("%A, %Y-%m-%d")}
"""


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


def format_final_response(state: MessagesState):
    """Format the response using the formatter model"""
    messages = state["messages"]
    last_ai_message = messages[-1]

    format_prompt = f"""
    Rewrite this response to be more concise, clear, and user-friendly:
    
    Original: {last_ai_message.content}
    
    Make it:
    - Clear and well-structured
    - Easy to read
    - Comprehensive but concise
    """

    formatted_response = formatter_model.invoke(
        [
            SystemMessage(
                "You are an expert at formatting AI responses for users. Make responses clear and professional."
            ),
            HumanMessage(format_prompt),
        ]
    )

    # Replace the last message with formatted version
    formatted_response.id = last_ai_message.id
    return {"messages": messages[:-1] + [formatted_response]}


# Conditional Edge Functions
def should_use_tools(state: MessagesState) -> Literal["tools", "analyze"]:
    """Route to tools if model made tool calls, otherwise analyze response"""
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "analyze"


def should_format_response(state: MessagesState) -> Literal["format", "END"]:
    """Decide whether to format the response based on analysis"""
    if state.get("needs_formatting", False):
        return "format"
    return "END"


def after_tools(state: MessagesState) -> Literal["agent", "END"]:
    """After using tools, decide if we need to call the agent again"""
    messages = state["messages"]

    # Check if the last message is a tool result
    if hasattr(messages[-1], "type") and messages[-1].type == "tool":
        return "agent"  # Go back to agent to process tool results
    return "END"


# Build the StateGraph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("agent", call_main_model)
builder.add_node("tools", tool_node)
builder.add_node("analyze", analyze_response_need)
builder.add_node("format", format_final_response)

# Add edges
builder.add_edge(START, "agent")

# Conditional edges
builder.add_conditional_edges(
    "agent", should_use_tools, {"tools": "tools", "analyze": "analyze"}
)

builder.add_conditional_edges("tools", after_tools, {"agent": "agent", "END": END})

builder.add_conditional_edges(
    "analyze", should_format_response, {"format": "format", "END": END}
)

builder.add_edge("format", END)

# Compile the graph
graph = builder.compile(checkpointer=memory)


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

    await cl.Message(
        content=f"Welcome back, {user.identifier}! I'm your finance assistant. How can I help you today?"
    ).send()


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
    config = {"configurable": {"thread_id": thread_id}}

    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")

    async for message, metadata in graph.astream(
        {"messages": [HumanMessage(content=msg.content)]},
        stream_mode="messages",
        config=RunnableConfig(callbacks=[cb], **config),
    ):
        print("META=", metadata["langgraph_node"])
        if (
            message.content
            and not isinstance(message, HumanMessage)
            and not metadata["langgraph_node"] == "tools"
            # and not metadata["langgraph_node"] == "agent"
        ):
            await final_answer.stream_token(message.content)

    await final_answer.send()


if __name__ == "__main__":
    cl.run()
