# Advanced AI Investment Analyst and Trading Executive: README

## Changelog: Version 0.2.2

This version significantly enhances the AI's ability to provide comprehensive financial calendar information, consolidating earnings and economic events into a unified view.

### **🚀 New Features**

* **Comprehensive Financial Calendar**: The AI can now fetch and display a combined view of corporate earnings announcements and economic events for specified dates or date ranges, offering a holistic overview of market-moving events.

### **🛠️ Tool Updates & Removals**

* **Replaced Tools**: The previous `scrape_trading_economics_calendar` and `get_and_display_earnings_by_range` tools have been replaced by more robust and integrated calendar functions.
* **New Tools Added**:
  * `fetch_earnings_calendar`: To retrieve detailed corporate earnings announcements.
  * `fetch_economic_calendar`: To retrieve detailed economic events and indicators.
  * `get_comprehensive_financial_calendar`: To provide a unified, formatted view of both earnings and economic events.

## Changelog: Version 0.2.1

This version introduces a new capability to fetch economic calendar data, expanding the AI&amp;#039;s ability to incorporate macroeconomic indicators into its analysis.

### **🚀 New Features**

* **Economic Calendar Integration**: The AI can now scrape and display economic calendar data from Trading Economics, allowing for analysis of upcoming events that may impact the market.

### **🛠️ New Tools Added**

* `scrape_trading_economics_calendar`: A new tool to fetch data directly from the Trading Economics calendar.

## Changelog: Version 0.1.1

This version focuses on refining the documented capabilities and correcting the list of available tools to accurately reflect the AI&amp;#039;s current functions.

### **🛠️ Tool &amp;amp; Documentation Refinements**

* **Tool List Correction**: Removed `get_earnings_calendar`, `get_sp500_tickers`, and `fetch_article_content` from the documented tool list as they are not currently available.
* **New Tool Documentation**: Added `get_and_display_earnings_by_range` to the list of available tools.

***

## Changelog: Version 0.1.0

This version introduces a major new capability focused on scheduling, automation, and reporting, significantly expanding the assistant&amp;#039;s utility beyond real-time analysis and trading.

### **🚀 New Features**

* **Task Scheduling &amp;amp; Reminders**: You can now schedule tasks and reminders for the AI to execute in the future. This allows for automated, time-based actions like generating a portfolio summary every morning or getting news alerts at a specific time.
* **Automated Email Reporting**: The AI can now send analysis, reports, or any requested content directly to your email address. This can be done on-demand or as part of a scheduled task.

### **🛠️ New Tools Added**

A new suite of tools has been added to support the scheduling and reporting features:

* **Time Management**:
  * `get_current_time`: To get the precise current time in any timezone.
  * `calculate_relative_time`: To understand and calculate future or past times (e.g., &amp;quot;in 2 hours&amp;quot;, &amp;quot;at 5 PM EST&amp;quot;).
* **Task Management**:
  * `schedule_task_tool`: The core function to schedule a prompt for future execution.
  * `list_scheduled_tasks`: To view all pending automated tasks.
  * `cancel_scheduled_task`: To cancel a previously scheduled task.
* **Reporting**:
  * `send_markdown_email`: To send formatted content to an email address.

### **📄 Documentation &amp;amp; Tool Updates**

* **README Update**: The main README has been updated to include the new &amp;quot;Scheduling, Reminders, &amp;amp; Reporting&amp;quot; section.
* The `generate_chart` tool mentioned in the v0.0.1 README was removed as it is not an available function.

***

## Introduction

As an Advanced AI Investment Analyst and Trading Executive, my core mission is multifaceted: to deliver comprehensive financial analysis, actionable trading recommendations, and highly relevant financial news; to **execute trades and manage portfolio queries on behalf of the user when explicitly instructed**; and to automate and schedule financial reporting tasks. I operate with precision, synthesizing complex data into clear insights, executing financial operations, and managing tasks using a robust suite of analytical, transactional, and scheduling tools. This document outlines my capabilities, workflow, and the tools I use to provide you with market intelligence and portfolio management.

***

### Core Capabilities

* **Comprehensive Stock Analysis**: I can perform in-depth analysis of individual stocks or companies. Just provide a ticker symbol (e.g., `AAPL`) or a company name (e.g., `Apple Inc.`).
* **Stock Comparison**: I can compare two stocks to help you make informed decisions (e.g., &amp;quot;Compare GOOGL and MSFT&amp;quot;).
* **Financial News Synthesis**: I can fetch, summarize, and analyze the latest financial news relevant to specific companies, industries, or the market as a whole.
* **Financial Calendar Integration**: I can provide a unified view of upcoming corporate earnings announcements and economic events, helping you stay informed about market-moving catalysts.
* **Actionable Recommendations**: Based on a holistic analysis, I generate clear **Buy**, **Sell**, or **Hold** recommendations, supported by a detailed rationale.
* **Trade Execution &amp;amp; Portfolio Management**: I can place, cancel, and manage orders, as well as provide summaries of your account and portfolio holdings upon your direct instruction.
* **Scheduling &amp;amp; Task Automation**: I can schedule tasks and reminders for the future. For example, you can ask me to &amp;quot;provide a market summary tomorrow at 9 AM&amp;quot; or &amp;quot;remind me to check on my GOOGL position in 3 hours.&amp;quot;
* **Email Reporting**: I can send the results of my analysis or other content directly to an email address, either immediately or at a scheduled time.

***

### Analytical Workflow

For analysis requests, I follow a structured, multi-step process to ensure every analysis is thorough and well-rounded.

#### 1. User Interaction &amp;amp; Query Interpretation

My first step is to understand your request accurately.

* **Ticker Identification**: If you provide a company name, I use the `get_ticker` tool to find the correct stock symbol.
* **Clarification**: If a company name is ambiguous or has multiple listings, I will ask for more information to ensure accuracy.
* **News-Only Requests**: For requests purely about news, I will directly proceed to the News &amp;amp; Event Synthesis step.

#### 2. Data Acquisition &amp;amp; Validation

I gather a wide range of data points from various sources to build a complete picture.

* **Broad Overview**: `comprehensive_ticker_report`
* **Core Financials**: `get_stock_price`, `get_price_history`
* **Market Sentiment**: `get_current_fng_tool`, `get_historical_fng_tool`, `analyze_fng_trend`
* **Ownership &amp;amp; Derivatives**: `get_institutional_holders`, `get_insider_trades`, `get_options`
* **Earnings Data**: `fetch_earnings_calendar`, `get_earnings_history`
* **Economic Data**: `fetch_economic_calendar`, `get_comprehensive_financial_calendar`

#### 3. Financial Report Analysis

I dive deep into a company&amp;#039;s financial health.

* **Financial Statements**: `get_financial_statements` (Income, Balance Sheet, Cash Flow)
* **Earnings Performance**: `get_earnings_history` to analyze past performance and future estimates.
* **Key Metrics**: I highlight trends in Revenue, Net Income, EPS, P/E Ratio, and Debt-to-Equity.

#### 4. Technical Analysis

I analyze price and volume data to identify market trends and patterns.

* **Indicator Calculation**: `calculate_technical_indicator` for custom computations.
* **Specific Indicators**:
  * `get_moving_averages` (e.g., 50, 100, 200-day)
  * `get_rsi` (Relative Strength Index)
  * `get_macd` (Moving Average Convergence Divergence)
  * `get_bollinger_bands`
  * `get_volatility_analysis`
  * `get_support_resistance` levels
  * `get_trend_analysis`
* **Summary**: I use `get_technical_summary` for a quick overview of key technical signals.

#### 5. News &amp;amp; Event Synthesis

I stay on top of the latest events that could impact your investments.

* **News Aggregation**: `get_all_news` to search millions of articles from various sources.
* **Content Analysis**: I summarize the most impactful news and explain its financial implications.

#### 6. Recommendation Generation

This is the culmination of my analysis. I synthesize all the data from the previous steps to provide a clear, actionable trading signal (**Buy**, **Sell**, or **Hold**) with a robust, evidence-based rationale.

***

### Trading &amp;amp; Account Management

For direct action requests, I bypass the analytical workflow and use transactional tools to manage your portfolio.

* **Intent Confirmation**: Before executing any trade, I will confirm the parameters of your request (e.g., quantity, price, order type) to ensure accuracy.
* **Order Placement**: I can execute various order types based on your instructions:
  * `place_market_order`
  * `place_limit_order`
  * `place_stop_order`
  * `place_stop_limit_order`
* **Position &amp;amp; Order Management**:
  * `close_position`: To liquidate an existing holding.
  * `cancel_order`: To cancel a pending order.
* **Account Information**:
  * `get_account_info_tool`: For details on your account balance and status.
  * `get_portfolio_summary`: For a complete overview of your current holdings.

***

### Scheduling, Reminders, &amp;amp; Reporting

I can automate tasks and deliver information at a time that suits you.

* **Time-Based Triggers**: Using `get_current_time` and `calculate_relative_time`, I can understand requests like &amp;quot;in 5 hours&amp;quot; or &amp;quot;tomorrow at 10 AM EST.&amp;quot;
* **Task Scheduling**: With `schedule_task_tool`, I can set a reminder or schedule a complex task for a future time.
* **Task Management**: You can view all pending tasks with `list_scheduled_tasks` and cancel any with `cancel_scheduled_task`.
* **Email Delivery**: I can use `send_markdown_email` to send reports, analysis, or news summaries to a specified email address, either instantly or as part of a scheduled task.

***

### Here are the tools I have available

* **Financial Data &amp;amp; Analysis:**
  * `comprehensive_ticker_report`: Get a full report on a stock (overview, metrics, news).
  * `get_ticker`: Fetch a company&amp;#039;s stock ticker symbol.
  * `get_stock_price`: Retrieve the current stock price.
  * `get_price_history`: Retrieve historical price data for various periods.
  * `get_financial_statements`: Get income, balance, or cash flow statements.
  * `get_institutional_holders`: List major institutional and mutual fund holders.
  * `get_earnings_history`: Provide a history of earnings reports.
  * `fetch_earnings_calendar`: Fetch corporate earnings announcements for a single date or date range.
  * `get_insider_trades`: Show recent insider trading activity.
  * `get_options`: Fetch options contract data.
  * `compare_stocks`: Compare key metrics for two stocks.

* **Technical Analysis:**
  * `calculate_technical_indicator`: Calculate various indicators (SMA, EMA, RSI, MACD, BBANDS).
  * `get_moving_averages`: Calculate multiple moving averages.
  * `get_rsi`: Calculate the Relative Strength Index (RSI).
  * `get_macd`: Calculate the Moving Average Convergence Divergence (MACD).
  * `get_bollinger_bands`: Calculate Bollinger Bands.
  * `get_volatility_analysis`: Analyze stock volatility.
  * `get_support_resistance`: Identify support and resistance levels.
  * `get_trend_analysis`: Analyze the stock&amp;#039;s trend.
  * `get_technical_summary`: Provide a summary of all key technical indicators.

* **Market Sentiment:**
  * `get_current_fng_tool`: Get the current Fear &amp;amp; Greed Index.
  * `get_historical_fng_tool`: Retrieve historical Fear &amp;amp; Greed data.
  * `analyze_fng_trend`: Analyze the trend of the Fear &amp;amp; Greed Index.

* **News &amp;amp; Web Content:**
  * `get_all_news`: Searches for financial news articles.
  * `fetch_economic_calendar`: Fetch economic events and indicators for a single date or date range.
  * `get_comprehensive_financial_calendar`: Get complete financial calendar combining earnings and economic events for date(s).

* **Trading &amp;amp; Account Management:**
  * `get_account_info_tool`: Get your current account information.
  * `get_portfolio_summary`: Provide a summary of your investment portfolio.
  * `place_market_order`: Place a market order.
  * `place_limit_order`: Place a limit order.
  * `place_stop_order`: Place a stop order.
  * `place_stop_limit_order`: Place a stop-limit order.
  * `cancel_order`: Cancel an open order by ID.
  * `close_position`: Close an open position for a symbol.

* **Scheduling &amp;amp; Productivity:**
  * `get_current_time`: Get the current time in any timezone.
  * `calculate_relative_time`: Calculate future/past dates (e.g., &amp;quot;in 5 hours&amp;quot;).
  * `schedule_task_tool`: Schedule a task for future execution.
  * `list_scheduled_tasks`: List all pending scheduled tasks.
  * `cancel_scheduled_task`: Cancel a scheduled task by ID.
  * `send_markdown_email`: Send an email with Markdown formatted content.

***

### Disclaimer

*The information and analysis provided by the Advanced AI Investment Analyst and Trading Executive is for informational purposes only and should not be considered financial advice. All investment decisions should be made with the consultation of a qualified financial professional.*
