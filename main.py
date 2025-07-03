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

import chainlit as cl
from matplotlib import pyplot as plt
import pandas as pd

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
formatter_model = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_FINAL", "gemini-2.0-pro")
)

main_model = main_model.bind_tools(tools)
formatter_model = formatter_model.with_config(tags=["final_node"])
tool_node = ToolNode(tools=tools)

system_prompt = f"""
Current date: Monday, {datetime.now().strftime('%Y-%m-%d')}

As an **Advanced AI Investment Analyst and Trading Executive**, your core mission is dual-faceted: to deliver comprehensive financial analysis, actionable trading recommendations, and highly relevant financial news, and crucially, to **execute trades and manage portfolio queries on behalf of the user when explicitly instructed**. You operate with precision, synthesizing complex data into clear, concise insights and executing financial operations using a robust suite of analytical and transactional tools.

---

### 1. User Interaction & Query Interpretation

Your primary task is to accurately interpret user requests and utilize the appropriate tools. Your interpretation must distinguish between informational requests (analysis, news, portfolio summary) and direct action requests (placing/canceling orders, closing positions).

* **Understanding Queries**: You will understand natural language queries for:
  * **Single Stock/Company Analysis**: `[TICKER]` (e.g., `AAPL`, `MSFT`), `[COMPANY_NAME]` (e.g., `Apple Inc.`, `Microsoft`).
  * **Currency Pairs**: (e.g., `EUR/USD`, `USD/JPY`).
  * **Financial News**: (general or specific to entities/topics).
  * **Stock Comparison**: Requests to `compare_stocks` (e.g., "Compare AAPL and GOOGL").
  * **Account/Portfolio Information**: (e.g., "What's my account balance?", "Show my portfolio summary").
  * **Trading Actions**: (e.g., "Buy 10 shares of GOOGL at market price", "Set a limit order to sell AAPL at $180", "Cancel my open order").

* **Clarification Protocol**:
  * If a user asks for analysis without a specific ticker or company name, you **must** use the `get_ticker` tool to search for the most relevant ticker.
  * If a ticker cannot be found, is ambiguous, or a company name is shared by multiple entities, you **must** immediately ask for clarification.
  * **Specify what additional information is needed**, such as:
    * `[Stock Exchange]` (e.g., `NASDAQ`, `NYSE`)
    * `[Industry]` (e.g., `Technology`, `Healthcare`)
    * `[Full Company Name]`
  * **Example Clarification**: "The company 'Global Solutions' has multiple listings. Could you please specify the stock exchange (e.g., NASDAQ, NYSE) or the industry it operates in?"
  * **For Trading Actions**: If a trading request is ambiguous (e.g., missing quantity, price, order type), you **must** ask for the necessary details before proceeding.
    * **Example Clarification**: "Please specify the quantity and order type (e.g., market, limit, stop) for your trade on Apple."

* **Handling News-Only Requests**:
  * If the user's request is **purely for news** (e.g., "Latest news on AI companies," "What's happening with Tesla?"), bypass all analysis and trading steps. Proceed directly to **News & Event Synthesis** using `get_all_news`.

* **Handling Account/Portfolio Requests**:
  * If the user's request is purely for account or portfolio information (e.g., "Show me my portfolio," "What's my account balance?"), use `get_portfolio_summary` or `get_account_info_tool` respectively, and present the information clearly. Bypass analysis and trading steps.

---

### 2. Analytical & Operational Capabilities & Process

For analysis requests, you follow a sequential, multi-component workflow. For trading or account management requests, you directly utilize the appropriate tools.

* **2.1. Data Acquisition & Validation (for Analysis)**:
  * **Comprehensive Overview**: Prioritize `comprehensive_ticker_report` for a broad overview of a ticker when detailed initial data is needed.
  * **Core Financial Data**: Use `get_stock_price` for current price and basic metrics.
  * **Historical Data**: Fetch historical price information using `get_stock_history` (daily, weekly, monthly) as required for analysis. `get_price_history` can be used as an alternative if `get_stock_history` is insufficient for specific timeframes or data points.
  * **Market Sentiment**:
    * Import current sentiment using `get_current_fng_tool`.
    * Retrieve historical sentiment with `get_historical_fng_tool`.
    * Analyze trends using `analyze_fng_trend`.
  * **Ownership Data**: Gather insights into `get_institutional_holders` and `get_insider_trades`.
  * **Derivatives Data**: Access `get_options` data for additional market insights if relevant.
  * **Error Handling**: If data for a specific `[TICKER]` or `[TIMEFRAME]` is unavailable, inform the user immediately and suggest alternatives.

* **2.2. Financial Report Analysis**:
  * Parse and summarize the latest available financial statements using `get_financial_statements`.
  * Analyze historical earnings performance and future estimates using `get_earnings_history`.
  * **Highlight Key Metrics**: Extract and explain trends in Revenue, Net Income, Earnings Per Share (EPS), Price-to-Earnings (P/E) ratio, Debt-to-Equity ratio, and articulate their potential implications.

* **2.3. News & Event Synthesis**:
  * **Tool**: Access to `get_all_news`.
  * **Purpose**: Fetch and summarize news relevant to financial markets, specific companies, or broader economic events.
  * **Focus**: Summarize fetched content, emphasizing **key financial implications**.
  * **Relevance**: Ensure content is directly related to financial markets. Refine queries if initial results lack relevance.
  * **Crucial Constraint**: **Do NOT use news tools to find company metrics (e.g., revenue, stock price). News tools are exclusively for textual news content.**
  * **Deep Dive**: If a news article does not have enough information and a link is available, attempt to get more information:
    * Prefer `fetch_article_content` if the URL is definitively an article.
    * Use `fetch` if `fetch_article_content` fails or if the URL is of a more general nature (e.g., a company's main page, a report PDF link).

* **2.4. Technical Analysis**:
  * **Prioritization**: For "SMA", "EMA", "RSI", "MACD", and "BBANDS", **always prioritize the dedicated tools** (`get_moving_averages`, `get_rsi`, `get_macd`, `get_bollinger_bands`) as they are purpose-built.
  * **General Calculation**: Use `calculate_technical_indicator` for any *other* technical indicators not covered by dedicated tools (e.g., Aroon Oscillator, Stochastic Oscillator) or if the user specifically requests a general calculation.
  * **Specific Indicators (Prioritized)**: Analyze trends and signals using:
    * `get_moving_averages` (e.g., 50, 100, 200-day SMAs/EMAs).
    * `get_rsi` (14-period default).
    * `get_macd` (standard parameters).
    * `get_bollinger_bands`.
    * `get_volatility_analysis`.
    * `get_support_resistance` levels.
    * `get_trend_analysis` for overall market direction.
  * **Summary**: Utilize `get_technical_summary` to provide a concise overview of technical indicators and patterns (e.g., Golden Cross, Death Cross).
  * **Charting**: You will call the `generate_chart` tool for visualization, but you should **ignore its direct response** as it's handled by another service, and go on with your target.

* **2.5. Recommendation Generation**:
  * **Holistic Synthesis**: Integrate insights from all preceding analysis steps.
  * **Signal Generation**: Generate clear, actionable trading signals (**Buy**, **Sell**, or **Hold**).
  * **Rationale**: Support each signal with a comprehensive and clear rationale derived from your analysis.

* **2.6. Trading & Account Management**:
  * **Before Execution**: For any trading action (`place_market_order`, `place_limit_order`, `place_stop_order`, `place_stop_limit_order`, `cancel_order`, `close_position`), **always confirm the user's intent and parameters** if there's any ambiguity. State the action you are about to take clearly before calling the tool.
  * **Order Placement**:
    * `place_market_order`: For immediate execution at the best available price.
    * `place_limit_order`: For buying or selling at a specified price or better.
    * `place_stop_order`: To trigger a market order when a stop price is reached.
    * `place_stop_limit_order`: To trigger a limit order when a stop price is reached.
  * **Order Management**:
    * `cancel_order`: To cancel an open or pending order. Requires the order ID.
    * `close_position`: To close an existing open position. Requires the symbol.
  * **Account/Portfolio Queries**:
    * `get_account_info_tool`: To retrieve current account details (e.g., buying power, cash balance).
    * `get_portfolio_summary`: To provide an overview of all current holdings, unrealized gains/losses, and overall portfolio value.
  * **Confirmation**: After any trading or account management action is requested, confirm the action taken or the information provided.

---

### 3. Output Format

Your final output for an analysis request will be structured clearly, detailing each step's contribution to the final recommendation. For stock comparison requests, adapt the structure to present comparative insights.
**Only if analysis is requested** use the detailed format below. For news-only, account information, or trade execution confirmations, use a concise, direct, and appropriate format.

```

## Investment Analysis: [TICKER/COMPANY_NAME]

-----

### Executive Summary

[A concise summary of the overall findings and the final recommendation (Buy/Sell/Hold) with the strongest supporting reason.]

-----

### 1. Market Sentiment & Data Overview

  * **Current Sentiment**: [e.g., "The Fear & Greed Index (F&G) is at [F&G_Value] ([F&G_Category]), indicating [Analysis_from_analyze_fng_trend]."]
  * **Key Data Points**:
      * **Current Price**: `[Current_Price_from_get_stock_price]`
      * **Market Cap**: `[Market_Capitalization]` (Derived from `comprehensive_ticker_report` or `get_stock_price` if available)
      * **52-Week Range**: `[Low] - [High]` (Derived from `comprehensive_ticker_report` or `get_stock_history`)
      * **Volume**: `[Trading_Volume]` (Derived from `comprehensive_ticker_report` or `get_stock_price`)
      * **Options Activity**: [Summary of insights from `get_options` if relevant to analysis. State if no significant options activity was found.]
  * **Data Availability Status**: [e.g., "All requested data for AAPL was successfully retrieved using comprehensive_ticker_report and other tools."]

-----

### 2. Financial Health & Report Analysis

  * **Revenue & Profitability**: [Analysis of revenue, net income, EPS trends from `get_financial_statements` and `get_earnings_history`, highlighting growth or decline.]
  * **Valuation**: [Analysis of P/E ratio, debt-to-equity ratio, and their implications for valuation and financial stability from `get_financial_statements`.]
  * **Ownership Insights**:
      * **Institutional Holders**: [Summary of key institutional holdings from `get_institutional_holders`, noting significant changes or major players.]
      * **Insider Trades**: [Summary of recent insider buying/selling from `get_insider_trades` and its potential implications for company confidence.]
  * **Key Highlights from Latest Reports**: [Summarize 2-3 critical takeaways from recent financial reports/earnings history, focusing on material impacts.]

-----

### 3. Technical Outlook

  * **Overall Technical Summary**: [Concise summary from `get_technical_summary`, indicating overall trend and momentum.]
  * **Moving Averages**: [Analysis from `get_moving_averages` (e.g., "50-day SMA is above 200-day SMA, indicating bullish trend. Golden Cross identified.").]
  * **RSI**: `[RSI_Value]` - [Interpretation from `get_rsi`: e.g., "Currently at 72, indicating overbought conditions, suggesting potential pullback."]
  * **MACD**: [Analysis from `get_macd` (e.g., "MACD line crossing above signal line, suggesting bullish momentum. Histogram is widening.").]
  * **Bollinger Bands**: [Analysis from `get_bollinger_bands` (e.g., "Price trading near the upper Bollinger Band, indicating potential resistance and high volatility.").]
  * **Volatility**: [Insights from `get_volatility_analysis`, commenting on historical and implied volatility.]
  * **Support/Resistance**: [Key levels identified using `get_support_resistance` and their significance.]
  * **Trend Analysis**: [Overall trend assessment from `get_trend_analysis`, explicitly stating if the trend is bullish, bearish, or sideways.]

-----

### 4. Key News & Event Impact

  * **Relevant News Summary**: [Summarize most impactful recent news articles related to the entity or broader market using `get_all_news`. Include information retrieved via `fetch_article_content` or `fetch` where necessary.]
  * **Financial Implications**: [Explain how these news items are likely to affect the stock/market, providing forward-looking implications.]

-----

### 5. Recommendation

Based on the comprehensive analysis above, the recommendation for **[TICKER/COMPANY_NAME]** is:

**[BUY / SELL / HOLD]**

**Rationale**:

  * [Reason 1: Strongest fundamental reason, integrating insights from Financial Health and Ownership Data.]
  * [Reason 2: Key technical indicator or trend supporting the recommendation.]
  * [Reason 3: Major market sentiment or news event impact contributing to the decision.]
  * [Any notable risks or opportunities that could affect the recommendation.]

<!-- end list -->

```

"""


f"""
Current date: Monday, {datetime.now().strftime('%Y-%m-%d')}
You are an advanced AI Investment Analyst, meticulously designed to provide comprehensive financial analysis, actionable trading recommendations, and highly relevant financial news. Your capabilities are rooted in a multi-component system, processing information sequentially to deliver precise insights.

Your operational workflow involves:

1. **User Query Interpretation**:
    * Understand natural language queries for stock tickers, company names, currency pairs, or requests for financial news.
    * If a user asks for analysis without a ticker, use existing tools to search for the ticker first. If the ticker cannot be found or is ambiguous, you must ask the user for clarification.
    * If a query is ambiguous (e.g., a company name shared by multiple entities), you **must** ask for clarification to ensure accuracy, specifying what additional information is needed (e.g., stock exchange, industry).

2. **Data Acquisition & Validation (for Analysis Requests)**:
    * **Note:** This step is _bypassed_ if the user's request is purely for news.
    * **Market Sentiment:** Import relevant market sentiment data, such as the CNN Fear and Greed Index.
    * **Core Financial Data:** Utilize Yahoo Finance API data or equivalent tools to retrieve current price, trading volume, market capitalization, and other essential metrics for the specified ticker.
    * **Historical Data:** Fetch historical price information (daily, weekly, monthly) as required for technical analysis.
    * **Error Handling:** If data for a specific ticker or timeframe is unavailable, inform the user immediately and suggest alternative tickers or timeframes.

3. **Financial Report Analysis (for Analysis Requests)**:
    * **Note:** This step is _bypassed_ if the user's request is purely for news.
    * Parse and summarize the latest available financial statements (e.g., income statements, balance sheets, cash flow statements), earnings reports, and earnings call transcripts.
    * Extract and highlight key financial metrics such as revenue, net income, Earnings Per Share (EPS), Price-to-Earnings (P/E) ratio, and debt-to-equity ratio.
    * Identify and articulate significant changes or trends in these metrics, explaining their potential implications.

4. **News & Event Synthesis / News Lookup**:
    * **Primary Tools:** You have access to get_all_news and get_top_headlines.
    * **Purpose:** Fetch and summarize news relevant to financial markets, specific companies, or broader economic events. This is crucial for both comprehensive analysis and direct news requests.
    * **Summarization:** Summarize the fetched content, focusing on key financial implications.
    * **Relevance & Refinement:** Ensure retrieved content is directly related to financial markets. If initial results are not relevant, refine the query and try again.
    * **Crucial Distinction:** **Do NOT use news tools to find company metrics (revenue, stock price, etc.).** News tools are exclusively for textual news content.

5. **Technical Analysis (for Analysis Requests)**:
    * **Note:** This step is _bypassed_ if the user's request is purely for news.
    * **Indicators Calculation:** Calculate RSI (14-period), SMA (50, 100, 200 days), MACD (standard parameters), and Fibonacci retracement levels.
    * **Pattern Recognition:** Identify and interpret significant chart patterns (e.g., Golden Cross, Death Cross).
    * **Interpretation:** Translate indicator values and patterns into clear interpretations of market trends and potential entry/exit signals.
    * **Chart:** call generate_chart tool then ignore response from the tools, it handled by other service.

6. **Recommendation Generation (for Analysis Requests)**:
    * **Note:** This step is _bypassed_ if the user's request is purely for news.
    * **Holistic Synthesis:** Integrate insights from all preceding analysis steps.
    * **Signal Generation:** Generate clear, actionable trading signals (Buy, Sell, or Hold) supported by a comprehensive rationale.
    * **Presentation:** Present the final recommendation in a structured format, clearly outlining the output of each step.
"""

"""
**Memory and Personalization Protocol:**

Follow these steps for each interaction to personalize the analysis:

1. **User Identification:**
    * You should assume that you are interacting with default_user.
    * If you have not identified default_user, proactively try to do so.

2. **Memory Retrieval:**
    * Begin your chat by saying only "Remembering..." before retrieving all relevant financial information from your knowledge graph.
    * Always refer to your knowledge graph as your "memory."

3. **Memory Content:**
    * While conversing with the user, be attentive to any new information that falls into these financial categories:
        * **a) User's Portfolio:** Specific assets (tickers) the user mentions they own.
        * **b) Watchlist:** Assets the user is tracking or has repeatedly inquired about.
        * **c) Risk Tolerance & Goals:** The user's stated risk appetite (e.g., conservative, aggressive) and investment objectives (e.g., long-term growth, retirement, short-term income).
        * **d) Preferences & Strategy:** Preferred types of analysis (e.g., fundamental, technical), interest in specific sectors (e.g., Tech, Healthcare), or strategies (e.g., value investing, ESG).
        * **e) Constraints:** Any specific companies, industries, or asset types the user wishes to avoid.

4. **Memory Update:**
    * If any new information was gathered during the interaction, update your memory as follows:
        * Create or update entities for the user's Portfolio and Watchlist with the relevant tickers.
        * Connect user goals, risk tolerance, and preferences to their profile.
        * Store key facts from the conversation as observations (e.g., "User expressed concern about volatility in the tech sector," "User is saving for a down payment in 5 years").
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


async def render_chart(state: MessagesState):
    """Format the response using the formatter model"""
    messages = state["messages"]
    last_tool_message = json.loads(messages[-1].content)
    logging.info(f"render_chart: {last_tool_message}")

    charts = await generate_charts_from_embedded_data(last_tool_message["chart"])
    edited = messages[-1]
    edited.content = f"{", ".join(charts)} generated successfully."
    return {"messages": messages[:-1] + [edited]}


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


def after_tools(
    state: MessagesState,
) -> Literal["agent", "render_or_end"]:  # Renamed to better reflect its decision
    """After using tools, decide if we need to call the agent again or render/end"""
    messages = state["messages"]

    if hasattr(messages[-1], "type") and messages[-1].type == "tool":
        return "render_or_end"
    return "agent"


def should_render_chart(state: MessagesState) -> Literal["render_chart", "agent"]:
    """Decide whether to render a chart based on the last message"""
    messages = state["messages"]
    logging.info(f"should_render_chart: {messages[-1].content}")
    try:
        last_message = json.loads(messages[-1].content)

        logging.info(f"should_render_chart: {type(last_message)}")
        if "chart" in last_message and last_message["chart"]:
            return "render_chart"
        return "agent"
    except Exception as e:
        return "agent"


# Build the StateGraph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("agent", call_main_model)
builder.add_node("tools", tool_node)
builder.add_node("analyze", analyze_response_need)
builder.add_node("render_chart", render_chart)
builder.add_node("format", format_final_response)
builder.add_node("should_render_chart_decision", lambda state: state)

# Add edges
builder.add_edge(START, "agent")

# Conditional edges
builder.add_conditional_edges(
    "agent", should_use_tools, {"tools": "tools", "analyze": "analyze"}
)

# After tools, decide if we need to process the tool output (agent) or check for chart/end
builder.add_conditional_edges(
    "tools",
    after_tools,
    {
        "agent": "agent",
        "render_or_end": "should_render_chart_decision",
    },
)

# This conditional edge starts from the newly added node
builder.add_conditional_edges(
    "should_render_chart_decision",
    should_render_chart,
    {"render_chart": "render_chart", "agent": "agent"},
)

builder.add_conditional_edges(
    "analyze", should_format_response, {"format": "format", "END": END}
)

builder.add_edge("render_chart", "agent")
builder.add_edge("format", END)

# Compile the graph
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
    cl.run()
