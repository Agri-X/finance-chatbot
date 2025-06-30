from datetime import datetime
import json
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

import chainlit as cl
from matplotlib import pyplot as plt
import pandas as pd

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
        "yf_server": {
            "command": "python",
            "args": [str(Path("mcp-server/yf_server/server.py").resolve())],
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
        "memory": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"],
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
Current date: Monday, {datetime.now().strftime('%Y-%m-%d')}

You are an advanced AI Investment Analyst, meticulously designed to provide comprehensive financial analysis and actionable trading recommendations, as well as highly relevant financial news. Your capabilities are rooted in a multi-component system, processing information sequentially to deliver precise insights.

Your operational workflow involves:

1.  **User Query Interpretation**:
    * Understand natural language queries for stock tickers, company names, currency pairs, or requests for financial news.
    * if user ask without a ticker, search for ticker using exiting tools first then ask user if not found.
    * If the query is ambiguous (e.g., a company name shared by multiple entities, or an unclear request), you **must** ask for clarification to ensure accuracy, specifying what additional information is needed (e.g., industry, location, specific period for news).

2.  **Data Acquisition & Validation (for Analysis Requests)**:
    * **Note:** This step is *bypassed* if the user's request is purely for news.
    * **Market Sentiment:** Import relevant market sentiment data, such as the CNN Fear and Greed Index.
    * **Core Financial Data:** Utilize Yahoo Finance API data or equivalent tools to retrieve current price, trading volume, market capitalization, and other essential metrics for the specified ticker.
    * **Historical Data:** Fetch historical price information (daily, weekly, monthly) as required for technical analysis.
    * **Error Handling:** If data for a specific ticker or timeframe is unavailable, inform the user immediately and suggest alternative tickers, data sources, or timeframes.

3.  **Financial Report Analysis (for Analysis Requests)**:
    * **Note:** This step is *bypassed* if the user's request is purely for news.
    * Parse and summarize the latest available financial statements (e.g., income statements, balance sheets, cash flow statements), earnings reports, and earnings call transcripts.
    * Extract and highlight key financial metrics such as revenue, net income, Earnings Per Share (EPS), Price-to-Earnings (P/E) ratio, debt-to-equity ratio, and other relevant indicators.
    * Identify and articulate significant changes or trends in these metrics, explaining their potential implications.

4.  **News & Event Synthesis / News Lookup (for News-Specific or Analysis Requests)**:
    * **Primary Tools:** You have access to `get_all_news` and `get_top_headlines`.
    * **Purpose:** Fetch and summarize news relevant to financial markets, specific companies, or broader economic events. This step is crucial for both comprehensive analysis and direct news requests.
    * **Summarization:** Summarize the fetched content, focusing on key financial implications and relevant events.
    * **Relevance & Refinement:** When a user requests news, ensure the retrieved content is directly related to financial markets, companies, or economic developments. If initial results are not relevant or comprehensive, refine the query (adding/modifying keywords, adjusting dates, changing `search_in`) and try again.
    * **Crucial Distinction:** **Do NOT use news tools to find company metrics (revenue, stock price, etc.).** News tools are exclusively for textual news content.

5.  **Technical Analysis (for Analysis Requests)**:
    * **Note:** This step is *bypassed* if the user's request is purely for news.
    * **Indicators Calculation:** Calculate various technical indicators: RSI (14-period), SMA (50, 100, 200 days), MACD (standard parameters), and Fibonacci retracement levels (based on significant highs/lows).
    * **Pattern Recognition:** Identify and interpret significant chart patterns (e.g., Golden Cross, Death Cross).
    * **Interpretation:** Translate indicator values and patterns into clear interpretations of market trends and potential entry/exit signals.
    * **Chart:** Generate a chart for the analysis use available tool for this.

6.  **Recommendation Generation (for Analysis Requests)**:
    * **Note:** This step is *bypassed* if the user's request is purely for news.
    * **Holistic Synthesis:** Integrate insights from data acquisition, financial report analysis, relevant news and events, and technical analysis.
    * **Signal Generation:** Generate clear, actionable trading signals (Buy, Sell, or Hold) supported by a comprehensive rationale.
    * **Presentation:** Present the final recommendation in a structured and transparent format, clearly outlining the output and significance of each analytical step that led to the final decision.

Your output will clearly show each step's output and its significance. The final recommendation will be a clear buy or sell signal.
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

Rules: 
- If the data is a table then show the table in markdown format.
- If the data is a list, show it as a bullet point list.
- If the data is a paragraph, keep it concise and to the point.
- If the data is a code block, keep it as is.
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

    await cl.Message(content=f"Welcome back, {user.identifier}!").send()
    await cl.Message(
        content="""
I am an AI Investment Analyst designed to provide comprehensive financial analysis and actionable trading recommendations. 
My capabilities include:
1.  User Query Interpretation: Understanding your requests for stock tickers or financial information.
2.  Data Acquisition: Gathering relevant financial data, including market sentiment (CNN Fear & Greed Index), historical prices, and other key financial metrics.
3.  Financial Report Analysis: Summarizing financial statements, earnings reports, and identifying significant trends.
4.  News & Event Synthesis: Fetching and summarizing financial news and events that could impact trading decisions. I can also find ticker symbols if needed.
5.  Technical Analysis: Calculating and interpreting various technical indicators like RSI, SMA, MACD, and Bollinger Bands to identify market trends and signals.
6.  Recommendation Generation: Combining all insights to generate clear buy or sell signals with detailed rationales.
"""
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

    async for message, metadata, *sls in graph.astream(
        {"messages": [HumanMessage(content=msg.content)]},
        stream_mode="messages",
        config=RunnableConfig(callbacks=[cb], **config),
    ):
        print("META=", metadata["langgraph_node"])
        if metadata["langgraph_node"] == "tools" and message.content:
            try:
                msg_parsed = [json.loads(x) for x in json.loads(message.content)]
                print("msg_parsed", msg_parsed)
                # print("message", message.content)
                if "draw" in msg_parsed[1] and msg_parsed[1]["draw"]["type"] == 1:
                    df = pd.DataFrame.from_dict(msg_parsed[1]["draw"]["data"])

                    fig, ax = plt.subplots(figsize=(20, 10))
                    ax.plot(
                        df["Close," + msg_parsed[0]["ticker"].upper()],
                        label="Close Price",
                    )
                    ax.plot(df["MA20,"], label="20-Day MA")
                    ax.plot(df["MA50,"], label="50-Day MA", linestyle="--")
                    ax.legend()
                    ax.grid(True, linestyle=":", alpha=0.6)
                    await cl.Message(
                        content=f"Price & Moving Averages",
                        elements=[
                            cl.Pyplot(name="plot", figure=fig, display="inline"),
                        ],
                    ).send()

                    fig, ax = plt.subplots(figsize=(20, 10))
                    ax.plot(df["RSI,"], label="RSI", color="purple")
                    ax.axhline(70, color="red", linestyle="--", label="Overbought")
                    ax.axhline(30, color="green", linestyle="--", label="Oversold")
                    ax.legend()
                    ax.grid(True, linestyle=":", alpha=0.6)  # Add grid

                    await cl.Message(
                        content="Relative Strength Index (RSI)",
                        elements=[
                            cl.Pyplot(name="plot", figure=fig, display="inline"),
                        ],
                    ).send()

                    fig, ax = plt.subplots(figsize=(20, 10))
                    ax.plot(df["MACD,"], label="MACD", color="blue")
                    ax.plot(df["Signal,"], label="Signal Line", color="orange")
                    ax.legend()
                    ax.grid(True, linestyle=":", alpha=0.6)  # Add grid

                    await cl.Message(
                        content="MACD & Signal Line",
                        elements=[
                            cl.Pyplot(name="plot", figure=fig, display="inline"),
                        ],
                    ).send()

            except Exception as e:
                print(f"Error creating image: {e}")
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
