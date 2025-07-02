import logging
from venv import logger
import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP
from typing import List

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
        Dictionary with moving average values
    """
    try:
        data = ti.get_stock_data(symbol, period, interval)
        result = {}

        for window in windows:
            ma = ti.calculate_moving_average(data, window)
            ema = ti.calculate_exponential_moving_average(data, window)

            result[f"SMA_{window}"] = ma.dropna().tolist()
            result[f"EMA_{window}"] = ema.dropna().tolist()

        # Also include dates for reference
        result["dates"] = data.index.strftime("%Y-%m-%d").tolist()
        result["close"] = data["Close"].tolist()

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
        Dictionary with RSI values and dates
    """
    try:
        data = ti.get_stock_data(symbol, period, interval)
        rsi = ti.calculate_rsi(data, window)

        return {
            "dates": data.index.strftime("%Y-%m-%d").tolist(),
            "rsi": rsi.dropna().tolist(),
            "close": data["Close"].tolist(),
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
        Dictionary with MACD values and dates
    """
    try:
        data = ti.get_stock_data(symbol, period, interval)
        macd_data = ti.calculate_macd(data, fast_period, slow_period, signal_period)

        return {
            "dates": data.index.strftime("%Y-%m-%d").tolist(),
            "macd": macd_data["macd"].dropna().tolist(),
            "signal": macd_data["signal"].dropna().tolist(),
            "histogram": macd_data["histogram"].dropna().tolist(),
            "close": data["Close"].tolist(),
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

        return {
            "dates": data.index.strftime("%Y-%m-%d").tolist(),
            "upper": bb_data["upper"].dropna().tolist(),
            "middle": bb_data["middle"].dropna().tolist(),
            "lower": bb_data["lower"].dropna().tolist(),
            "close": data["Close"].tolist(),
        }
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


@mcp.tool()
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


@mcp.tool()
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


# Run the server
if __name__ == "__main__":
    mcp.run()
