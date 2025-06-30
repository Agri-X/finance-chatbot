import pandas as pd
import numpy as np
import logging

# from typing import Dict, List
import yfinance as yf

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Class that provides various technical indicators and analysis tools for stock data.
    """

    @staticmethod
    def get_stock_data(symbol: str, period: str = "6mo", interval: str = "1d"):
        """
        Retrieve historical stock data for technical analysis.

        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with historical stock data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            return data
        except Exception as e:
            raise ValueError(f"Error retrieving data for {symbol}: {e}")

    @staticmethod
    def calculate_moving_average(
        data: pd.DataFrame, window: int, column: str = "Close"
    ):
        """
        Calculate simple moving average.

        Args:
            data: DataFrame with price data
            window: Period for moving average
            column: Column name to calculate MA for (default: Close)

        Returns:
            Series with moving average values
        """
        return data[column].rolling(window=window).mean()

    @staticmethod
    def calculate_exponential_moving_average(
        data: pd.DataFrame, window: int, column: str = "Close"
    ):
        """
        Calculate exponential moving average.

        Args:
            data: DataFrame with price data
            window: Period for EMA
            column: Column name to calculate EMA for (default: Close)

        Returns:
            Series with EMA values
        """
        return data[column].ewm(span=window, adjust=False).mean()

    @staticmethod
    def calculate_rsi(data: pd.DataFrame, window: int = 14, column: str = "Close"):
        """
        Calculate Relative Strength Index (RSI).

        Args:
            data: DataFrame with price data
            window: RSI period (default: 14)
            column: Column name to calculate RSI for (default: Close)

        Returns:
            Series with RSI values
        """
        delta = data[column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        for i in range(window, len(delta)):
            if i > window:
                avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain[i]) / window
                avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss[i]) / window

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(
        data: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "Close",
    ):
        """
        Calculate Moving Average Convergence Divergence (MACD).

        Args:
            data: DataFrame with price data
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
            column: Column name to calculate MACD for (default: Close)

        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' Series
        """
        fast_ema = TechnicalIndicators.calculate_exponential_moving_average(
            data, fast_period, column
        )
        slow_ema = TechnicalIndicators.calculate_exponential_moving_average(
            data, slow_period, column
        )

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    @staticmethod
    def calculate_bollinger_bands(
        data: pd.DataFrame,
        window: int = 20,
        num_std: float = 2.0,
        column: str = "Close",
    ):
        """
        Calculate Bollinger Bands.

        Args:
            data: DataFrame with price data
            window: Moving average period (default: 20)
            num_std: Number of standard deviations (default: 2.0)
            column: Column name for calculation (default: Close)

        Returns:
            Dictionary with 'upper', 'middle', and 'lower' bands as Series
        """
        middle_band = TechnicalIndicators.calculate_moving_average(data, window, column)
        std_dev = data[column].rolling(window=window).std()

        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)

        return {"upper": upper_band, "middle": middle_band, "lower": lower_band}

    @staticmethod
    def calculate_atr(data: pd.DataFrame, window: int = 14):
        """
        Calculate Average True Range (ATR).

        Args:
            data: DataFrame with price data
            window: ATR period (default: 14)

        Returns:
            Series with ATR values
        """
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()

        return atr

    @staticmethod
    def calculate_volatility(
        data: pd.DataFrame,
        window: int = 20,
        column: str = "Close",
        annualize: bool = True,
        trading_days_per_year: int = 252,
        hours_per_trading_day: float = 6.5,
    ) -> pd.Series:
        """
        Calculate rolling volatility.

        Args:
            data: DataFrame with price data. Must have a DatetimeIndex.
            window: Period for volatility calculation (default: 20).
            column: Column name to calculate volatility for (default: "Close").
            annualize: Whether to annualize the volatility (default: True).
            trading_days_per_year: Number of trading days in a year for annualization. (default: 252)
            hours_per_trading_day: Number of trading hours in a day, for hourly data annualization. (default: 6.5)


        Returns:
            Series with volatility values. Returns an empty Series if input data is invalid.
        """
        logger.info(
            f"Starting volatility calculation for column '{column}' with window={window}."
        )

        if not isinstance(data, pd.DataFrame) or data.empty:
            logger.warning(
                "Input 'data' is not a valid DataFrame or is empty. Returning empty Series."
            )
            return pd.Series(dtype=float)

        if column not in data.columns:
            logger.error(
                f"Column '{column}' not found in the input DataFrame. Available columns: {data.columns.tolist()}"
            )
            return pd.Series(dtype=float)

        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning(
                "DataFrame index is not a DatetimeIndex. Frequency inference might be inaccurate for annualization."
            )

        # Calculate logarithmic returns
        # Add a small epsilon to avoid log(0) for safety, though prices should ideally be positive
        prices = data[column].astype(float)
        if (prices <= 0).any():
            logger.warning(
                f"Column '{column}' contains non-positive values. Replacing with NaN for log returns."
            )
            prices = prices.replace(0, np.nan)  # Replace 0 with NaN

        # Calculate logarithmic returns, handling potential division by zero or NaN results
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # Suppress warnings for division by zero/log of zero
            log_returns = np.log(prices / prices.shift(1))

        log_returns = log_returns.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()  # Handle infinite values and NaNs

        if log_returns.empty:
            logger.warning(
                "Logarithmic returns could not be calculated or resulted in empty series after dropping NaNs. Returning empty Series."
            )
            return pd.Series(dtype=float)

        # Calculate rolling standard deviation
        if len(log_returns) < window:
            logger.warning(
                f"Not enough data points ({len(log_returns)}) for the specified window ({window}). Returning empty Series."
            )
            return pd.Series(dtype=float)

        volatility = log_returns.rolling(window=window).std()
        logger.debug(f"Calculated raw rolling volatility using window {window}.")

        # Annualize if requested
        if annualize:
            freq = pd.infer_freq(data.index)
            annualization_factor = None

            if freq:
                logger.debug(f"Inferred frequency: {freq}")
                if "D" in freq:  # Daily data (e.g., 'D', 'B', 'C')
                    annualization_factor = np.sqrt(trading_days_per_year)
                    logger.debug(
                        f"Annualizing daily data with factor sqrt({trading_days_per_year})."
                    )
                elif "H" in freq:  # Hourly data
                    annualization_factor = np.sqrt(
                        trading_days_per_year * hours_per_trading_day
                    )
                    logger.debug(
                        f"Annualizing hourly data with factor sqrt({trading_days_per_year} * {hours_per_trading_day})."
                    )
                elif "T" in freq:  # Minute data (e.g., 'T' for minute, 'min')
                    # Assuming 60 minutes per hour for trading_hours_per_day
                    annualization_factor = np.sqrt(
                        trading_days_per_year * hours_per_trading_day * 60
                    )
                    logger.debug(
                        f"Annualizing minute data with factor sqrt({trading_days_per_year} * {hours_per_trading_day} * 60)."
                    )
                elif "W" in freq:  # Weekly data
                    annualization_factor = np.sqrt(
                        trading_days_per_year / 5
                    )  # Approximately 5 trading days per week
                    logger.debug(
                        f"Annualizing weekly data with factor sqrt({trading_days_per_year} / 5)."
                    )
                elif "M" in freq or "MS" in freq:  # Monthly data
                    annualization_factor = np.sqrt(
                        trading_days_per_year / 21
                    )  # Approximately 21 trading days per month
                    logger.debug(
                        f"Annualizing monthly data with factor sqrt({trading_days_per_year} / 21)."
                    )
                else:
                    logger.warning(
                        f"Could not determine standard annualization factor for inferred frequency '{freq}'. No annualization applied."
                    )
            else:
                logger.warning(
                    "Could not infer data frequency from index. No annualization applied without explicit frequency information."
                )

            if annualization_factor:
                volatility = volatility * annualization_factor
            else:
                logger.info(
                    "Annualization was requested but skipped due to unknown or unhandled frequency."
                )
        else:
            logger.info("Annualization was not requested.")

        logger.info("Volatility calculation complete.")
        return volatility

    @staticmethod
    def detect_support_resistance(
        data: pd.DataFrame, window: int = 20, sensitivity: float = 0.03
    ):
        """
        Detect support and resistance levels using local minima and maxima.

        Args:
            data: DataFrame with price data
            window: Lookback period for finding pivots (default: 20)
            sensitivity: Minimum price change percentage to consider (default: 0.03)

        Returns:
            Dictionary with 'support' and 'resistance' levels
        """
        high = data["High"]
        low = data["Low"]

        resistance_levels = []
        support_levels = []

        # Find pivot highs (local maxima)
        for i in range(window, len(high) - window):
            if all(high[i] > high[i - j] for j in range(1, window + 1)) and all(
                high[i] > high[i + j] for j in range(1, window + 1)
            ):
                # Check if significantly different from previously found resistance levels
                if not any(
                    abs(high[i] - level) / level < sensitivity
                    for level in resistance_levels
                ):
                    resistance_levels.append(high[i])

        # Find pivot lows (local minima)
        for i in range(window, len(low) - window):
            if all(low[i] < low[i - j] for j in range(1, window + 1)) and all(
                low[i] < low[i + j] for j in range(1, window + 1)
            ):
                # Check if significantly different from previously found support levels
                if not any(
                    abs(low[i] - level) / level < sensitivity
                    for level in support_levels
                ):
                    support_levels.append(low[i])

        return {
            "support": sorted(support_levels),
            "resistance": sorted(resistance_levels),
        }

    @staticmethod
    def detect_trends(
        data: pd.DataFrame,
        short_window: int = 20,
        long_window: int = 50,
        column: str = "Close",
    ):
        """
        Detect trends using moving average crossovers.

        Args:
            data: DataFrame with price data
            short_window: Short-term MA period (default: 20)
            long_window: Long-term MA period (default: 50)
            column: Column name to detect trends for (default: Close)

        Returns:
            Dictionary with 'trend' and 'signal' Series
        """
        short_ma = TechnicalIndicators.calculate_moving_average(
            data, short_window, column
        )
        long_ma = TechnicalIndicators.calculate_moving_average(
            data, long_window, column
        )

        # Create trend indicator (1: uptrend, -1: downtrend, 0: neutral/undefined)
        trend = pd.Series(0, index=data.index)
        trend[short_ma > long_ma] = 1  # Uptrend
        trend[short_ma < long_ma] = -1  # Downtrend

        # Create signal for trend changes
        signal = pd.Series(0, index=data.index)
        signal[(trend.shift(1) <= 0) & (trend > 0)] = (
            1  # Buy signal (trend turning positive)
        )
        signal[(trend.shift(1) >= 0) & (trend < 0)] = (
            -1
        )  # Sell signal (trend turning negative)

        return {"trend": trend, "signal": signal}

    @staticmethod
    def calculate_pattern_recognition(data: pd.DataFrame):
        """
        Basic pattern recognition for common candlestick patterns.

        Args:
            data: DataFrame with price data (must have Open, High, Low, Close)

        Returns:
            Dictionary with pattern signals (1 where pattern is detected)
        """
        pattern_signals = {}

        # Doji pattern (open and close are very close)
        doji = pd.Series(0, index=data.index)
        body_size = abs(data["Close"] - data["Open"])
        avg_body = body_size.rolling(window=14).mean()
        shadow_size = data["High"] - data["Low"]
        doji[(body_size < 0.1 * shadow_size) & (body_size < 0.25 * avg_body)] = 1
        pattern_signals["doji"] = doji

        # Hammer pattern (long lower shadow, small body at the top)
        hammer = pd.Series(0, index=data.index)
        lower_shadow = pd.Series(0, index=data.index)
        upper_shadow = pd.Series(0, index=data.index)

        # For days with close > open (bullish)
        bullish = data["Close"] > data["Open"]
        lower_shadow[bullish] = data["Open"][bullish] - data["Low"][bullish]
        upper_shadow[bullish] = data["High"][bullish] - data["Close"][bullish]

        # For days with open > close (bearish)
        bearish = data["Open"] > data["Close"]
        lower_shadow[bearish] = data["Close"][bearish] - data["Low"][bearish]
        upper_shadow[bearish] = data["High"][bearish] - data["Open"][bearish]

        # Hammer criteria
        body_height = abs(data["Close"] - data["Open"])
        hammer[
            (lower_shadow > 2 * body_height) & (upper_shadow < 0.2 * body_height)
        ] = 1
        pattern_signals["hammer"] = hammer

        # Engulfing pattern (current candle completely engulfs previous candle)
        bullish_engulfing = pd.Series(0, index=data.index)
        bearish_engulfing = pd.Series(0, index=data.index)

        # Bullish engulfing
        bullish_engulfing[
            (data["Open"] < data["Close"].shift(1))
            & (data["Close"] > data["Open"].shift(1))
            & (data["Close"] > data["Open"])
            & (data["Open"].shift(1) > data["Close"].shift(1))
        ] = 1

        # Bearish engulfing
        bearish_engulfing[
            (data["Open"] > data["Close"].shift(1))
            & (data["Close"] < data["Open"].shift(1))
            & (data["Close"] < data["Open"])
            & (data["Open"].shift(1) < data["Close"].shift(1))
        ] = 1

        pattern_signals["bullish_engulfing"] = bullish_engulfing
        pattern_signals["bearish_engulfing"] = bearish_engulfing

        return pattern_signals

    @staticmethod
    def detect_divergence(data: pd.DataFrame, indicator: pd.Series, window: int = 14):
        """
        Detect divergence between price and indicator (e.g., RSI).

        Args:
            data: DataFrame with price data
            indicator: Series with indicator values (e.g., RSI)
            window: Lookback period for finding pivots (default: 14)

        Returns:
            Dictionary with 'bullish_divergence' and 'bearish_divergence' Series
        """
        close = data["Close"]

        bullish_divergence = pd.Series(0, index=data.index)
        bearish_divergence = pd.Series(0, index=data.index)

        # Find local price lows and indicator lows
        for i in range(window, len(close) - window):
            # Check for price making lower low
            if (
                (close[i] < close[i - 1])
                and (close[i] < close[i + 1])
                and (close[i] < min(close[i - window : i]))
                and (close[i] < min(close[i + 1 : i + window + 1]))
            ):

                # But indicator making higher low (bullish divergence)
                if (indicator[i] > indicator[i - window]) and (
                    indicator[i] > indicator[i - window // 2]
                ):
                    bullish_divergence[i] = 1

        # Find local price highs and indicator highs
        for i in range(window, len(close) - window):
            # Check for price making higher high
            if (
                (close[i] > close[i - 1])
                and (close[i] > close[i + 1])
                and (close[i] > max(close[i - window : i]))
                and (close[i] > max(close[i + 1 : i + window + 1]))
            ):

                # But indicator making lower high (bearish divergence)
                if (indicator[i] < indicator[i - window]) and (
                    indicator[i] < indicator[i - window // 2]
                ):
                    bearish_divergence[i] = 1

        return {
            "bullish_divergence": bullish_divergence,
            "bearish_divergence": bearish_divergence,
        }
