from dataclasses import dataclass
import dataclasses
from datetime import date, datetime, timedelta
import logging
from venv import logger
from literalai import Dict
import pandas as pd
from tabulate import tabulate
import yfinance as yf
from mcp.server.fastmcp import FastMCP
from typing import List, Optional, Union
from requests import get

from utils.technical_indicator import TechnicalIndicators


mcp = FastMCP("Stock Price Server")


# --- Utility Functions ---
def fetch_ticker(symbol: str):
    """Helper to safely fetch a yfinance Ticker."""
    return yf.Ticker(symbol.upper())


def safe_get_price(ticker):
    """Attempt to retrieve the current price of a stock."""
    try:
        data = ticker.history(period="1d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
        price = ticker.info.get("regularMarketPrice")
        if price is not None:
            return float(price)
        raise ValueError("Price data not available.")
    except Exception as e:
        raise ValueError(f"Error retrieving stock price: {e}")


ti = TechnicalIndicators()


@mcp.tool()
def get_ticker(company_name: str):
    """
    Fetches ticker based on company name.
    Args:
        company_name (str): The name of the company to search for.
    """

    res = yf.search.Search(
        query=company_name,
        max_results=1,
    )
    logger.info(f"Searching for ticker symbol for company: {company_name}")
    logger.info(f"Search results: {res.quotes[0]}")

    data = res.quotes[0]["symbol"]
    return data


@mcp.tool()
def get_stock_price(symbol: str):
    """
    Retrieve the current stock price for the given ticker symbol.
    Returns the latest closing price as a float.
    """
    symbol = symbol.upper()
    ticker = fetch_ticker(symbol)
    return safe_get_price(ticker)


@mcp.tool()
def get_moving_averages(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
    windows: List[int] = [20, 50, 200],
):
    """
    Calculate multiple moving averages for a stock.

    Args:
        symbol: Stock ticker symbol
        period: Data period (e.g., "6mo", "1y", "max")
        interval: Data interval (e.g., "1d", "1wk")
        windows: List of MA periods to calculate

    Returns:
        Dictionary with moving average values and chart data
    """
    try:
        data = ti.get_stock_data(symbol, period, interval)
        result = {}

        # Initialize chart_data structure
        chart_data_plots = [
            {
                "data_key": f"Close,{symbol}",
                "label": "Close Price",
                "color": "blue",
            }
        ]
        chart_data_values = {
            "Date": data.index.strftime("%Y-%m-%d").tolist(),
            f"Close,{symbol}": data["Close"].tolist(),
        }

        # Dynamically add SMA and EMA to results and chart_data
        for window in windows:
            ma = ti.calculate_moving_average(data, window)
            ema = ti.calculate_exponential_moving_average(data, window)

            # Ensure lists are of the same length by dropping NaNs and aligning
            # Find the minimum non-NaN index across all series for alignment
            first_valid_index_ma = ma.first_valid_index()
            first_valid_index_ema = ema.first_valid_index()

            # If any MA or EMA is entirely NaN, skip it for that window
            if first_valid_index_ma is None and first_valid_index_ema is None:
                continue

            # Determine the starting index for slicing to align all series
            start_index_for_alignment = 0
            if first_valid_index_ma is not None:
                start_index_for_alignment = max(
                    start_index_for_alignment, data.index.get_loc(first_valid_index_ma)
                )
            if first_valid_index_ema is not None:
                start_index_for_alignment = max(
                    start_index_for_alignment, data.index.get_loc(first_valid_index_ema)
                )

            aligned_dates = (
                data.index[start_index_for_alignment:].strftime("%Y-%m-%d").tolist()
            )
            aligned_close = data["Close"][start_index_for_alignment:].tolist()
            aligned_ma = (
                ma[start_index_for_alignment:].tolist()
                if first_valid_index_ma is not None
                else [None] * len(aligned_dates)
            )
            aligned_ema = (
                ema[start_index_for_alignment:].tolist()
                if first_valid_index_ema is not None
                else [None] * len(aligned_dates)
            )

            result[f"SMA_{window}"] = aligned_ma
            result[f"EMA_{window}"] = aligned_ema

            # Add to chart data, ensuring alignment
            chart_data_values[f"SMA_{window},"] = aligned_ma
            chart_data_values[f"EMA_{window},"] = aligned_ema

            chart_data_plots.append(
                {
                    "data_key": f"SMA_{window},",
                    "label": f"SMA {window}",
                    "color": "green",
                    "linestyle": "--",
                }
            )
            chart_data_plots.append(
                {
                    "data_key": f"EMA_{window},",
                    "label": f"EMA {window}",
                    "color": "purple",
                    "linestyle": "-",
                }
            )

        # Update dates and close to be aligned with the latest start of any MA/EMA
        # This makes sure the chart's 'Date' and 'Close' lists match the length of the MAs/EMAs if they are shorter due to NaNs.
        # If there are no MAs/EMAs calculated (e.g. invalid window or data), we ensure at least 'Date' and 'Close' are present.
        if (
            chart_data_values
        ):  # Check if chart_data_values has been populated beyond just dates/close initially
            # Find the minimum length of all lists in chart_data_values
            min_len = min(len(v) for v in chart_data_values.values())

            # Trim all lists to this minimum length
            for key, value_list in chart_data_values.items():
                chart_data_values[key] = value_list[-min_len:]

            # The 'result' dictionary also needs to have its lists trimmed to the same min_len for consistency,
            # especially for "dates" and "close" which were populated earlier based on full data.
            result["dates"] = chart_data_values["Date"]
            result["close"] = chart_data_values[f"Close,{symbol}"]
            for key in result:
                if key not in ["dates", "close"] and isinstance(result[key], list):
                    result[key] = result[key][-min_len:]
        else:
            # If no MAs were calculated (e.g., empty windows list), ensure base data is still available.
            result["dates"] = data.index.strftime("%Y-%m-%d").tolist()
            result["close"] = data["Close"].tolist()
            chart_data_values["Date"] = result["dates"]
            chart_data_values[f"Close,{symbol}"] = result["close"]

        chart_data = {
            "title": f"{symbol.upper()} - Moving Averages",
            "figsize": [20, 10],
            "index_column": "Date",
            "data": chart_data_values,
            "plots": chart_data_plots,
        }

        result["chart"] = [chart_data]

        return result
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_rsi(symbol: str, period: str = "6mo", interval: str = "1d", window: int = 14):
    """
    Calculate RSI for a stock.

    Args:
        symbol: Stock ticker symbol
        period: Data period
        interval: Data interval
        window: RSI period

    Returns:
        Dictionary with RSI values, dates, close prices, and chart data
    """
    try:
        data = ti.get_stock_data(symbol, period, interval)
        rsi = ti.calculate_rsi(data, window)

        # Align data for plotting
        combined_data = data[["Close"]].copy()
        combined_data["RSI"] = rsi
        combined_data = combined_data.dropna()

        dates = combined_data.index.strftime("%Y-%m-%d").tolist()
        rsi_values = combined_data["RSI"].tolist()
        close_prices = combined_data["Close"].tolist()

        # Generate data for horizontal RSI levels
        # These lists will be filled with the constant value for each date
        overbought_line = [70.0] * len(dates)
        oversold_line = [30.0] * len(dates)
        mid_line = [50.0] * len(dates)

        chart_data = {
            "title": f"{symbol.upper()} - Relative Strength Index (RSI)",
            "figsize": [20, 15],  # Increased height to accommodate two subplots
            "index_column": "Date",
            "data": {
                "Date": dates,
                f"Close,{symbol}": close_prices,
                "RSI,": rsi_values,
                "Overbought_Line,": overbought_line,  # New data key
                "Oversold_Line,": oversold_line,  # New data key
                "Mid_Line,": mid_line,  # New data key
            },
            "plots": [
                {
                    "data_key": f"Close,{symbol}",
                    "label": "Close Price",
                    "color": "blue",
                    "subplot": "top",  # Indicate this plot should be in the top subplot
                },
                {
                    "data_key": "RSI,",
                    "label": "RSI",
                    "color": "orange",
                    "subplot": "bottom",  # Indicate this plot should be in the bottom subplot
                },
                # Add horizontal lines for RSI levels using the new data keys
                {
                    "data_key": "Overbought_Line,",  # Now refers to the new list
                    "label": "Overbought (70)",
                    "color": "red",
                    "linestyle": "--",
                    "subplot": "bottom",
                },
                {
                    "data_key": "Oversold_Line,",  # Now refers to the new list
                    "label": "Oversold (30)",
                    "color": "green",
                    "linestyle": "--",
                    "subplot": "bottom",
                },
                {
                    "data_key": "Mid_Line,",  # Now refers to the new list
                    "label": "Mid (50)",
                    "color": "gray",
                    "linestyle": ":",
                    "subplot": "bottom",
                },
            ],
            "subplots": ["top", "bottom"],  # Define the subplot names
        }

        return {
            "dates": dates,
            "rsi": rsi_values,
            "close": close_prices,
            "chart": [chart_data],
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_macd(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
):
    """
    Calculate MACD for a stock.

    Args:
        symbol: Stock ticker symbol
        period: Data period
        interval: Data interval
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period

    Returns:
        Dictionary with MACD values, dates, close prices, and chart data
    """
    try:
        data = ti.get_stock_data(symbol, period, interval)
        macd_data_series = ti.calculate_macd(
            data, fast_period, slow_period, signal_period
        )

        # Align data for plotting
        combined_data = data[["Close"]].copy()
        combined_data["MACD"] = macd_data_series["macd"]
        combined_data["Signal"] = macd_data_series["signal"]
        combined_data["Histogram"] = macd_data_series["histogram"]
        combined_data = combined_data.dropna()

        dates = combined_data.index.strftime("%Y-%m-%d").tolist()
        macd_values = combined_data["MACD"].tolist()
        signal_values = combined_data["Signal"].tolist()
        histogram_values = combined_data["Histogram"].tolist()
        close_prices = combined_data["Close"].tolist()

        chart_data = {
            "title": f"{symbol.upper()} - Moving Average Convergence Divergence (MACD)",
            "figsize": [20, 15],  # Increased height for two subplots
            "index_column": "Date",
            "data": {
                "Date": dates,
                f"Close,{symbol}": close_prices,
                "MACD,": macd_values,
                "Signal,": signal_values,
                "Histogram,": histogram_values,
            },
            "plots": [
                {
                    "data_key": f"Close,{symbol}",
                    "label": "Close Price",
                    "color": "blue",
                    "subplot": "top",
                },
                {
                    "data_key": "MACD,",
                    "label": "MACD Line",
                    "color": "orange",
                    "subplot": "bottom",
                },
                {
                    "data_key": "Signal,",
                    "label": "Signal Line",
                    "color": "purple",
                    "linestyle": "--",
                    "subplot": "bottom",
                },
                {
                    "data_key": "Histogram,",
                    "label": "Histogram",
                    "color": "gray",
                    "type": "bar",  # Specify bar chart for histogram
                    "subplot": "bottom",
                },
                {
                    "data_key": None,  # Zero line for MACD
                    "label": "Zero Line",
                    "color": "black",
                    "linestyle": ":",
                    "value": 0,
                    "subplot": "bottom",
                },
            ],
            "subplots": ["top", "bottom"],
        }

        return {
            "dates": dates,
            "macd": macd_values,
            "signal": signal_values,
            "histogram": histogram_values,
            "close": close_prices,
            "chart": [chart_data],
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_bollinger_bands(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
    window: int = 20,
    num_std: float = 2.0,
):
    """
    Calculate Bollinger Bands for a stock.

    Args:
        symbol: Stock ticker symbol
        period: Data period
        interval: Data interval
        window: Moving average period
        num_std: Number of standard deviations

    Returns:
        Dictionary with Bollinger Bands values and dates
    """
    try:
        data = ti.get_stock_data(symbol, period, interval)
        bb_data = ti.calculate_bollinger_bands(data, window, num_std)

        bb_data = {
            "dates": data.index.strftime("%Y-%m-%d").tolist(),
            "upper": bb_data["upper"].dropna().tolist(),
            "middle": bb_data["middle"].dropna().tolist(),
            "lower": bb_data["lower"].dropna().tolist(),
            "close": data["Close"].tolist(),
        }

        min_len = min(
            len(bb_data["dates"]),
            len(bb_data["close"]),
            len(bb_data["upper"]),
            len(bb_data["middle"]),
            len(bb_data["lower"]),
        )
        dates = bb_data["dates"][-min_len:]
        close = bb_data["close"][-min_len:]
        upper = bb_data["upper"][-min_len:]
        middle = bb_data["middle"][-min_len:]
        lower = bb_data["lower"][-min_len:]

        chart_data = {
            "title": f"{symbol.upper()} - Bollinger Bands",
            "figsize": [20, 10],
            "index_column": "Date",
            "data": {
                "Date": dates,
                f"Close,{symbol}": close,
                "Upper Band,": upper,
                "Middle Band,": middle,
                "Lower Band,": lower,
            },
            "plots": [
                {
                    "data_key": f"Close,{symbol}",
                    "label": "Close Price",
                    "color": "blue",
                },
                {
                    "data_key": "Upper Band,",
                    "label": "Upper Band",
                    "color": "green",
                    "linestyle": "--",
                },
                {
                    "data_key": "Middle Band,",
                    "label": "Middle Band (MA)",
                    "color": "orange",
                    "linestyle": "--",
                },
                {
                    "data_key": "Lower Band,",
                    "label": "Lower Band",
                    "color": "red",
                    "linestyle": "--",
                },
            ],
        }

        bb_data["chart"] = [chart_data]

        return bb_data

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_volatility_analysis(symbol: str, period: str = "1y", interval: str = "1d"):
    """
    Calculate volatility metrics for a stock.

    Args:
        symbol: Stock ticker symbol
        period: Data period
        interval: Data interval

    Returns:
        Dictionary with volatility metrics and dates
    """
    try:
        data = ti.get_stock_data(symbol, period, interval)

        # Calculate various volatility metrics
        vol_20d = ti.calculate_volatility(data, window=20)
        vol_50d = ti.calculate_volatility(data, window=50)
        atr = ti.calculate_atr(data)

        # Calculate daily returns
        data["Returns"] = data["Close"].pct_change()

        return {
            "dates": data.index.strftime("%Y-%m-%d").tolist(),
            "volatility_20d": vol_20d.dropna().tolist(),
            "volatility_50d": vol_50d.dropna().tolist(),
            "atr": atr.dropna().tolist(),
            "daily_returns": data["Returns"].dropna().tolist(),
            "close": data["Close"].tolist(),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_support_resistance(
    symbol: str, period: str = "1y", interval: str = "1d", window: int = 20
):
    """
    Find support and resistance levels for a stock.

    Args:
        symbol: Stock ticker symbol
        period: Data period
        interval: Data interval
        window: Lookback period for pivot points

    Returns:
        Dictionary with support and resistance levels
    """
    try:
        data = ti.get_stock_data(symbol, period, interval)
        levels = ti.detect_support_resistance(data, window)

        latest_close = data["Close"].iloc[-1]

        return {
            "support_levels": levels["support"],
            "resistance_levels": levels["resistance"],
            "latest_close": float(latest_close),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_trend_analysis(symbol: str, period: str = "1y", interval: str = "1d"):
    """
    Complete trend analysis for a stock.

    Args:
        symbol: Stock ticker symbol
        period: Data period
        interval: Data interval

    Returns:
        Dictionary with trend analysis results
    """
    try:
        data = ti.get_stock_data(symbol, period, interval)

        # Calculate various trend indicators
        trends = ti.detect_trends(data)
        patterns = ti.calculate_pattern_recognition(data)
        rsi = ti.calculate_rsi(data)

        # Detect divergence between price and RSI
        divergences = ti.detect_divergence(data, rsi)

        # Filter signals to the last 10 days
        last_10_days = -10

        # Compile signals
        signals = []
        dates = data.index[last_10_days:].strftime("%Y-%m-%d").tolist()

        for i, date in enumerate(dates):
            idx = i + len(data) + last_10_days
            if idx >= len(data):
                continue

            day_signals = []

            # Check for trend changes
            if trends["signal"].iloc[idx] == 1:
                day_signals.append("Bullish trend change")
            elif trends["signal"].iloc[idx] == -1:
                day_signals.append("Bearish trend change")

            # Check for patterns
            for pattern, signal in patterns.items():
                if signal.iloc[idx] == 1:
                    day_signals.append(f"{pattern.replace('_', ' ').title()} pattern")

            # Check for divergences
            if divergences["bullish_divergence"].iloc[idx] == 1:
                day_signals.append("Bullish divergence")
            elif divergences["bearish_divergence"].iloc[idx] == 1:
                day_signals.append("Bearish divergence")

            if day_signals:
                signals.append({"date": date, "signals": day_signals})

        # Determine overall trend
        latest_trend = trends["trend"].iloc[-1]
        if latest_trend > 0:
            overall_trend = "Bullish"
        elif latest_trend < 0:
            overall_trend = "Bearish"
        else:
            overall_trend = "Neutral"

        return {
            "overall_trend": overall_trend,
            "signals": signals,
            "trends": trends["trend"].iloc[last_10_days:].tolist(),
            "close": data["Close"].iloc[last_10_days:].tolist(),
            "dates": dates,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_technical_summary(symbol: str):
    """
    Generate a complete technical analysis summary for a stock.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with technical analysis summary
    """
    logger.info(f"Starting technical summary generation for symbol: {symbol}")
    try:
        # Get data with different timeframes
        logger.debug(f"Fetching daily data for {symbol} (6 months, 1 day interval)")
        data_daily = ti.get_stock_data(symbol, period="6mo", interval="1d")
        if data_daily.empty:
            logger.warning(f"No daily data retrieved for {symbol}. Exiting.")
            return {"error": f"No daily data available for {symbol}"}
        logger.debug(f"Fetched {len(data_daily)} daily data points.")

        logger.debug(f"Fetching weekly data for {symbol} (2 years, 1 week interval)")
        data_weekly = ti.get_stock_data(symbol, period="2y", interval="1wk")
        if data_weekly.empty:
            logger.warning(
                f"No weekly data retrieved for {symbol}. Proceeding with daily only."
            )
        logger.debug(f"Fetched {len(data_weekly)} weekly data points.")

        latest_price = data_daily["Close"].iloc[-1]
        logger.info(f"Latest price for {symbol}: {latest_price:.2f}")

        # --- FIX 1: Calculate full SMAs first for Golden/Death Cross check ---
        logger.debug("Calculating SMA 50 and SMA 200 for daily data.")
        sma_50_series = ti.calculate_moving_average(data_daily, 50)
        sma_200_series = ti.calculate_moving_average(data_daily, 200)

        # Calculate other indicators and get the latest value
        logger.debug("Calculating other technical indicators.")
        sma_20 = ti.calculate_moving_average(data_daily, 20).iloc[-1]
        sma_50 = sma_50_series.iloc[-1]
        sma_200 = sma_200_series.iloc[-1]
        logger.debug(
            f"SMAs: SMA20={sma_20:.2f}, SMA50={sma_50:.2f}, SMA200={sma_200:.2f}"
        )

        ema_12 = ti.calculate_exponential_moving_average(data_daily, 12).iloc[-1]
        ema_26 = ti.calculate_exponential_moving_average(data_daily, 26).iloc[-1]
        logger.debug(f"EMAs: EMA12={ema_12:.2f}, EMA26={ema_26:.2f}")

        rsi_14 = ti.calculate_rsi(data_daily).iloc[-1]
        logger.debug(f"RSI(14): {rsi_14:.2f}")

        macd_data = ti.calculate_macd(data_daily)
        macd = macd_data["macd"].iloc[-1]
        macd_signal = macd_data["signal"].iloc[-1]
        logger.debug(f"MACD: {macd:.2f}, MACD Signal: {macd_signal:.2f}")

        bb_data = ti.calculate_bollinger_bands(data_daily)
        bb_upper = bb_data["upper"].iloc[-1]
        bb_middle = bb_data["middle"].iloc[-1]
        bb_lower = bb_data["lower"].iloc[-1]
        logger.debug(
            f"Bollinger Bands: Upper={bb_upper:.2f}, Middle={bb_middle:.2f}, Lower={bb_lower:.2f}"
        )

        volatility = ti.calculate_volatility(data_daily)
        logger.debug(f"Volatility: {volatility}")
        if volatility is not None and not pd.isna(volatility.iloc[-1]):
            # Annualize volatility based on the data frequency
            volatility = volatility.iloc[-1]

        # Support and resistance
        logger.debug("Detecting support and resistance levels.")
        levels = ti.detect_support_resistance(data_daily)
        supports = [level for level in levels["support"] if level < latest_price]
        resistances = [level for level in levels["resistance"] if level > latest_price]
        nearest_support = max(supports) if supports else None
        nearest_resistance = min(resistances) if resistances else None

        # Trend analysis
        logger.debug("Detecting daily and weekly trends.")
        daily_trend_series = ti.detect_trends(data_daily)["trend"]
        daily_trend = (
            daily_trend_series.iloc[-1] if not daily_trend_series.empty else 0
        )  # Handle empty series

        weekly_trend_series = ti.detect_trends(data_weekly)["trend"]
        weekly_trend = (
            weekly_trend_series.iloc[-1] if not weekly_trend_series.empty else 0
        )  # Handle empty series

        logger.debug(
            f"Daily Trend: {daily_trend:.2f}, Weekly Trend: {weekly_trend:.2f}"
        )

        # Generate signals
        signals = []
        logger.info("Generating technical signals.")

        # Moving average signals
        if latest_price > sma_20:
            signals.append("Price above SMA(20) - short-term bullish")
            logger.debug("Signal: Price above SMA(20)")
        else:
            signals.append("Price below SMA(20) - short-term bearish")
            logger.debug("Signal: Price below SMA(20)")

        if latest_price > sma_50:
            signals.append("Price above SMA(50) - medium-term bullish")
            logger.debug("Signal: Price above SMA(50)")
        else:
            signals.append("Price below SMA(50) - medium-term bearish")
            logger.debug("Signal: Price below SMA(50)")

        if latest_price > sma_200:
            signals.append("Price above SMA(200) - long-term bullish")
            logger.debug("Signal: Price above SMA(200)")
        else:
            signals.append("Price below SMA(200) - long-term bearish")
            logger.debug("Signal: Price below SMA(200)")

        # --- FIX 1 (cont.): Check for cross using the full series ---
        if (
            not sma_50_series.empty
            and not sma_200_series.empty
            and len(sma_50_series) >= 2
            and len(sma_200_series) >= 2
        ):
            if (
                sma_50_series.iloc[-1] > sma_200_series.iloc[-1]
                and sma_50_series.iloc[-2] <= sma_200_series.iloc[-2]
            ):
                signals.append(
                    "Recent Golden Cross (SMA50 crossed above SMA200) - major bullish signal"
                )
                logger.info("Signal: Recent Golden Cross detected.")
            if (
                sma_50_series.iloc[-1] < sma_200_series.iloc[-1]
                and sma_50_series.iloc[-2] >= sma_200_series.iloc[-2]
            ):
                signals.append(
                    "Recent Death Cross (SMA50 crossed below SMA200) - major bearish signal"
                )
                logger.info("Signal: Recent Death Cross detected.")
        else:
            logger.warning("Not enough data for SMA Golden/Death Cross check.")

        # RSI signals
        if rsi_14 > 70:
            signals.append("RSI > 70 - Overbought")
            logger.info("Signal: RSI Overbought.")
        elif rsi_14 < 30:
            signals.append("RSI < 30 - Oversold")
            logger.info("Signal: RSI Oversold.")
        else:
            logger.debug(f"RSI({rsi_14:.2f}) is neutral (30-70).")

        # MACD signals
        logger.debug("Checking MACD signals.")
        if (
            macd > macd_signal
            and macd_data["macd"].iloc[-2] <= macd_data["signal"].iloc[-2]
        ):
            signals.append("MACD Bullish Crossover")
            logger.info("Signal: MACD Bullish Crossover detected.")
        elif (
            macd < macd_signal
            and macd_data["macd"].iloc[-2] >= macd_data["signal"].iloc[-2]
        ):
            signals.append("MACD Bearish Crossover")
            logger.info("Signal: MACD Bearish Crossover detected.")
        else:
            logger.debug("No MACD crossover detected.")

        # Bollinger Bands signals
        logger.debug("Checking Bollinger Bands signals.")
        if latest_price > bb_upper:
            signals.append("Price above Upper Bollinger Band")
            logger.info("Signal: Price above Upper Bollinger Band.")
        elif latest_price < bb_lower:
            signals.append("Price below Lower Bollinger Band")
            logger.info("Signal: Price below Lower Bollinger Band.")
        else:
            logger.debug("Price is within Bollinger Bands.")

        if bb_middle > 0:  # Avoid division by zero
            band_width = (bb_upper - bb_lower) / bb_middle
            logger.debug(f"Bollinger Band Width: {band_width:.4f}")
            # Check if current bandwidth is in the lower 10th percentile of the last 120 days
            historical_bw = (bb_data["upper"] - bb_data["lower"]) / bb_data["middle"]
            if not historical_bw.empty and band_width < historical_bw.quantile(0.1):
                signals.append(
                    "Bollinger Bands Squeeze - potential for high volatility"
                )
                logger.info("Signal: Bollinger Bands Squeeze detected.")
            else:
                logger.debug("No Bollinger Bands Squeeze detected.")
        else:
            logger.warning(
                "Bollinger Middle Band is zero, cannot calculate band width."
            )

        # Determine overall bias based on multiple timeframes
        logger.info("Determining overall bias.")
        if daily_trend > 0 and weekly_trend > 0:
            overall_bias = "Strong Bullish"
        elif daily_trend > 0 and weekly_trend <= 0:
            overall_bias = "Moderately Bullish"
        elif daily_trend <= 0 and weekly_trend > 0:
            overall_bias = "Neutral with Bullish Bias"
        else:
            overall_bias = "Bearish"
        logger.info(f"Overall bias for {symbol}: {overall_bias}")

        # Format results for return
        summary = {
            "symbol": symbol,
            "last_price": float(latest_price),
            "overall_bias": overall_bias,
            "signals": signals,
            "indicators": {
                "sma_20": float(sma_20),
                "sma_50": float(sma_50),
                "sma_200": float(sma_200),
                "ema_12": float(ema_12),
                "ema_26": float(ema_26),
                "rsi_14": float(rsi_14),
                "macd": float(macd),
                "macd_signal": float(macd_signal),
                "bb_upper": float(bb_upper),
                "bb_middle": float(bb_middle),
                "bb_lower": float(bb_lower),
                "volatility_annualized": (
                    float(volatility * 100)
                    if volatility and not pd.isna(volatility)
                    else None
                ),  # Ensure it's not NaN
            },
            "support_resistance": {
                "nearest_support": float(nearest_support) if nearest_support else None,
                "nearest_resistance": (
                    float(nearest_resistance) if nearest_resistance else None
                ),
            },
        }
        logger.info(f"Successfully generated technical summary for {symbol}.")
        logger.debug(f"Summary: {summary}")
        return summary
    except Exception as e:
        logger.exception(f"An unexpected error occurred for symbol {symbol}: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


@mcp.resource("stock://{symbol}")
def stock_resource(symbol: str):
    """
    Expose stock price data as a resource.
    Returns a formatted string with the current stock price.
    """
    try:
        price = get_stock_price(symbol)
        return f"The current price of {symbol.upper()} is ${price:.2f}"
    except ValueError as e:
        return f"[{symbol.upper()}] Error: {e}"


# @mcp.tool()
def get_stock_history(symbol: str, period: str = "1mo"):
    """
    Retrieve historical stock data in CSV format.
    """
    symbol = symbol.upper()
    try:
        ticker = fetch_ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return f"[{symbol}] No historical data found for period '{period}'."
        return data.to_csv()
    except Exception as e:
        return f"[{symbol}] Error fetching historical data: {e}"


@mcp.tool()
def compare_stocks(symbol1: str, symbol2: str):
    """
    Compare two stock prices.
    """
    symbol1, symbol2 = symbol1.upper(), symbol2.upper()
    try:
        price1 = get_stock_price(symbol1)
        price2 = get_stock_price(symbol2)
        if price1 > price2:
            return (
                f"{symbol1} (${price1:.2f}) is higher than {symbol2} (${price2:.2f})."
            )
        elif price1 < price2:
            return f"{symbol1} (${price1:.2f}) is lower than {symbol2} (${price2:.2f})."
        else:
            return f"{symbol1} and {symbol2} have the same price (${price1:.2f})."
    except Exception as e:
        return f"Error comparing stocks: {e}"


def clean_column_name(col_name):
    """
    Cleans a single column name by removing parentheses and the ticker part.
    Assumes the ticker part is ', TICKER)' where TICKER is a string like 'AMD'.
    """
    if isinstance(col_name, tuple):
        col_name = ", ".join(col_name)
    col_name = col_name.replace("(", "").replace(")", "")
    import re

    col_name = re.sub(r",\s*[A-Z]+\s*$", "", col_name)
    col_name = re.sub(r",\s*[A-Z]+\s*\)", "", col_name)
    return col_name.strip()


# @mcp.tool()
async def generate_chart(ticker, period="5mo", interval="1d"):
    """
    Generate a chart for the given stock ticker symbol.

    Parameters:
        period : str
            Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            Default: 1mo
            Either Use period parameter or use start and end
        interval : str
            Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            Intraday data cannot extend last 60 days
    """
    df = yf.download(ticker, period=period, interval=interval)

    assert isinstance(df, pd.DataFrame)

    if df.empty:
        return {
            "summary": f"❌ No data found for {ticker.upper()}. Please check the ticker symbol.",
            "plot": None,
            "ticker": ticker.upper(),
        }

    df.columns = [clean_column_name(col) for col in df.columns]
    df.dropna(inplace=True)

    # Moving Averages
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = -loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df.dropna(inplace=True)

    # Ensure Date column
    df.index = pd.to_datetime(df.index)
    df = df.reset_index()
    if "Date" not in df.columns:
        df.rename(columns={"index": "Date"}, inplace=True)

    return {
        "chart": [
            {
                "title": "Price & Moving Averages",
                "figsize": [20, 10],
                "index_column": "Date",
                "data": {
                    "Date": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
                    f"Close,{ticker.upper()}": df["Close"].round(2).tolist(),
                    "MA20,": df["MA20"].round(2).tolist(),
                    "MA50,": df["MA50"].round(2).tolist(),
                },
                "plots": [
                    {"data_key": f"Close,{ticker.upper()}", "label": "Close Price"},
                    {"data_key": "MA20,", "label": "20-Day MA"},
                    {"data_key": "MA50,", "label": "50-Day MA", "linestyle": "--"},
                ],
            },
            {
                "title": "Relative Strength Index (RSI)",
                "figsize": [20, 10],
                "index_column": "Date",
                "data": {
                    "Date": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
                    "RSI,": df["RSI"].round(2).tolist(),
                },
                "plots": [{"data_key": "RSI,", "label": "RSI", "color": "purple"}],
                "hlines": [
                    {"y": 70, "color": "red", "linestyle": "--", "label": "Overbought"},
                    {"y": 30, "color": "green", "linestyle": "--", "label": "Oversold"},
                ],
            },
            {
                "title": "MACD & Signal Line",
                "figsize": [20, 10],
                "index_column": "Date",
                "data": {
                    "Date": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
                    "MACD,": df["MACD"].round(2).tolist(),
                    "Signal,": df["Signal"].round(2).tolist(),
                },
                "plots": [
                    {"data_key": "MACD,", "label": "MACD", "color": "blue"},
                    {
                        "data_key": "Signal,",
                        "label": "Signal Line",
                        "color": "orange",
                        "linestyle": "--",
                    },
                ],
            },
        ]
    }


@dataclass
class StockData:
    """
    A dataclass to represent stock earnings data for a company.
    """

    ticker: str
    company: str
    total: int
    nextEPSDate: Optional[str] = None
    releaseTime: Optional[str] = None
    qDate: Optional[str] = None
    q1RevEst: Optional[float] = None
    q1EstEPS: Optional[float] = None
    confirmDate: Optional[str] = None
    epsTime: Optional[str] = None
    quarterDate: Optional[str] = None
    qSales: Optional[float] = None


def human_format(num):
    """
    Formats a number to a human-readable string with K, M, B suffixes.
    Handles NaN/None values by returning them as is.
    """
    if pd.isna(num):  # Check for pandas NaN or None
        return num
    num = float(num)  # Ensure it's a float for calculations
    if abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    if abs(num) >= 1_000:
        return f"{num / 1_000:.2f}K"
    return f"{num:.2f}"  # Format smaller numbers to two decimal places


@mcp.tool()
def getEarnings(day: Union[str, datetime, date]) -> Dict:
    """
    Fetches earnings data from for a given date.

    Args:
        day: The date for which to fetch earnings. Can be a string in ISO format (YYYY-MM-DD),
             a datetime object, or a date object.

    Returns:
        A dictionary containing the earnings data.

    Raises:
        ValueError: If the input 'day' is not a valid date type or is a weekend.
        RequestException: If there's an issue making the HTTP request.
        Exception: For any other unexpected errors during data processing.
    """
    MAIN_URL = "https://www.earningswhispers.com"

    try:
        # Convert input to a date object
        if isinstance(day, str):
            try:
                parsed_day = date.fromisoformat(day)
                logging.debug(f"Converted string '{day}' to date object: {parsed_day}")
            except ValueError:
                logging.error(
                    f"Invalid date string format: '{day}'. Expected YYYY-MM-DD."
                )
                raise ValueError(
                    f"Invalid date string format: '{day}'. Expected YYYY-MM-DD."
                )
        elif isinstance(day, datetime):
            parsed_day = day.date()
            logging.debug(
                f"Converted datetime object '{day}' to date object: {parsed_day}"
            )
        elif isinstance(day, date):
            parsed_day = day
            logging.debug(f"Using provided date object: {parsed_day}")
        else:
            logging.error(
                f"Invalid type for 'day': {type(day)}. Expected str, datetime, or date."
            )
            raise TypeError(
                f"Invalid type for 'day': {type(day)}. Expected str, datetime, or date."
            )

        if parsed_day.weekday() in [5, 6]:
            logging.warning(
                f"Attempted to fetch earnings for a weekend: {parsed_day.isoformat()}"
            )
            raise ValueError("Cannot fetch earnings for a weekend.")

        api_url = f"{MAIN_URL}/api/caldata/{parsed_day.isoformat().replace('-', '')}"
        logging.info(f"Attempting to fetch earnings data from: {api_url}")

        r = get(url=api_url, headers={"Referer": MAIN_URL}, timeout=10)
        r.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        data = r.json()
        logging.info(f"Successfully fetched earnings data for {parsed_day.isoformat()}")
        return data

    except ValueError as ve:
        # Re-raise ValueError as it's a specific input validation error
        raise ve
    except Exception as e:
        logging.exception(f"An unexpected error occurred while fetching earnings: {e}")
        # Using logging.exception() logs the traceback automatically
        raise e


@mcp.tool()
async def get_and_display_earnings_by_range(start_date: date, end_date: date):
    """
    Fetches stock earnings data for a specified date range.

    Args:
        start_date (date): The starting date for fetching earnings data (inclusive).
        end_date (date): The ending date for fetching earnings data (inclusive).

    Returns:
        Optional[str]: An error message string if an error occurs during fetching or processing,
                       otherwise None if the operation completes successfully.
    """
    try:
        if start_date > end_date:
            return "Error: start_date cannot be after end_date."

        all_earnings: List[StockData] = []

        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
                logging.info(f"Skipping {current_date.strftime('%Y-%m-%d')} (Weekend)")
                current_date += timedelta(days=1)
                continue  # Skip to the next day

            logging.info(
                f"\n--- Attempting to fetch earnings for {current_date.strftime('%Y-%m-%d')} ---"
            )
            try:
                raw_earnings_data = getEarnings(current_date)

                logging.info("raw_earnings_data")
                logging.info(raw_earnings_data)

                for item in raw_earnings_data:
                    filtered_item = {
                        k: v
                        for k, v in item.items()
                        if k in StockData.__dataclass_fields__
                    }
                    all_earnings.append(StockData(**filtered_item))

            except Exception as e:
                logging.error(
                    f"Error fetching earnings for {current_date.strftime('%Y-%m-%d')}: {e}"
                )
                return f"Error fetching earnings for {current_date.strftime('%Y-%m-%d')}: {e}"

            current_date += timedelta(days=1)

        if not all_earnings:
            return f"No earnings data found for the period from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}. This might be due to weekends, holidays, or no scheduled earnings."

        df = pd.DataFrame.from_records(
            [dataclasses.asdict(stock) for stock in all_earnings]
        )

        columns_to_format = ["q1RevEst", "q1EstEPS", "qSales"]
        for col in columns_to_format:
            if col in df.columns:
                df.loc[:, col] = df[col].apply(human_format)

        date_columns = ["nextEPSDate", "confirmDate", "epsTime", "quarterDate"]
        for col in date_columns:
            if col in df.columns:

                df.loc[:, col] = pd.to_datetime(
                    df[col], format="mixed", errors="coerce"
                ).dt.strftime("%d-%m-%Y")

        return tabulate(
            df.to_dict("list"),
            headers="keys",
            tablefmt="github",
        )
    except Exception as e:
        logging.error(e)
        return f"error occurred: {e}"


# Run the server
if __name__ == "__main__":
    mcp.run()
