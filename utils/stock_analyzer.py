import pandas as pd
import numpy as np
from tabulate import tabulate
import talib as ta
import yfinance as yf
from scipy.signal import argrelextrema
import warnings

warnings.filterwarnings("ignore")


class StockAnalyzer:
    def __init__(self, stock_list, period="6mo"):
        """
        Initialize the StockAnalyzer with a list of stock symbols.

        Args:
            stock_list (list): List of stock symbols to analyze
            period (str): Period for historical data (default: "6mo")
        """
        self.stock_list = stock_list
        self.period = period
        self.tickers = yf.Tickers(stock_list)
        self.ticker_summary = {}
        self.history = None

    def get_stock_info(self):
        """Fetch basic stock information for all tickers."""
        print("--- Fetching stock information ---")

        info_keys = [
            "volume",
            "averageVolume",
            "averageVolume10days",
            "averageDailyVolume10Day",
            "52WeekChange",
            "fiftyTwoWeekLow",
            "fiftyTwoWeekHigh",
            "fiftyTwoWeekLowChange",
            "fiftyTwoWeekLowChangePercent",
            "fiftyTwoWeekRange",
            "fiftyTwoWeekHighChange",
            "fiftyTwoWeekHighChangePercent",
            "fiftyTwoWeekChangePercent",
            "fiftyDayAverage",
            "fiftyDayAverageChange",
            "fiftyDayAverageChangePercent",
            "twoHundredDayAverage",
            "currentPrice",
            "regularMarketVolume",
            "regularMarketChangePercent",
            "regularMarketPrice",
            "twoHundredDayAverageChange",
            "twoHundredDayAverageChangePercent",
        ]

        for ticker in self.stock_list:
            self.ticker_summary[ticker] = {}
            ticker_info = self.tickers.tickers[ticker].info

            for key in info_keys:
                self.ticker_summary[ticker][key] = ticker_info.get(key, None)

        print(f"Successfully fetched information for {len(self.stock_list)} stocks.")

    def get_historical_data(self):
        """Fetch historical price data."""
        print("--- Fetching historical data ---")

        self.history = self.tickers.history(period=self.period, group_by="ticker")
        self.history.index = self.history.index.date
        self.history.index.name = "Date"

        # Clean up columns
        columns_to_drop_level_1 = ["Dividends", "Stock Splits"]
        columns_to_drop = [
            col for col in self.history.columns if col[1] in columns_to_drop_level_1
        ]
        self.history = self.history.drop(columns=columns_to_drop)

        print(f"Historical data fetched for period: {self.period}")

    def calculate_support_resistance(self, ticker, lookback=20, num_levels=3):
        """
        Calculate support and resistance levels using pivot points method.

        Args:
            ticker (str): Stock symbol
            lookback (int): Number of periods to look back for pivot points
            num_levels (int): Number of support/resistance levels to identify

        Returns:
            dict: Dictionary containing support and resistance levels
        """
        try:
            high_prices = self.history[(ticker, "High")].values
            low_prices = self.history[(ticker, "Low")].values
            close_prices = self.history[(ticker, "Close")].values

            # Check if we have enough data
            if (
                len(high_prices) < lookback
                or len(low_prices) < lookback
                or len(close_prices) < lookback
            ):
                print(
                    f"Warning: Insufficient data for {ticker} support/resistance calculation. Need at least {lookback} data points."
                )
                # Return default values
                return {
                    "pivot_support": [0.0, 0.0, 0.0],
                    "pivot_resistance": [0.0, 0.0, 0.0],
                    "pivot_point": 0.0,
                    "local_support": [0.0] * min(num_levels, 1),
                    "local_resistance": [0.0] * min(num_levels, 1),
                }

            # Find local maxima (resistance) and minima (support)
            try:
                high_indices = argrelextrema(
                    high_prices, np.greater, order=max(1, lookback // 2)
                )[0]
                low_indices = argrelextrema(
                    low_prices, np.less, order=max(1, lookback // 2)
                )[0]
            except:
                high_indices = []
                low_indices = []

            # Get resistance levels from local maxima
            if len(high_indices) > 0:
                resistance_levels = high_prices[high_indices]
                resistance_levels = sorted(resistance_levels, reverse=True)[:num_levels]
            else:
                resistance_levels = (
                    [np.max(high_prices[-lookback:])] if len(high_prices) > 0 else [0.0]
                )

            if len(low_indices) > 0:
                support_levels = low_prices[low_indices]
                support_levels = sorted(support_levels)[:num_levels]
            else:
                support_levels = (
                    [np.min(low_prices[-lookback:])] if len(low_prices) > 0 else [0.0]
                )

            # Pad lists to ensure we have the required number of levels
            while len(resistance_levels) < num_levels:
                resistance_levels.append(
                    resistance_levels[-1] if resistance_levels else 0.0
                )

            while len(support_levels) < num_levels:
                support_levels.append(support_levels[-1] if support_levels else 0.0)

            # Calculate pivot point levels (traditional method)
            try:
                recent_high = np.max(high_prices[-lookback:])
                recent_low = np.min(low_prices[-lookback:])
                recent_close = close_prices[-1]

                pivot = (recent_high + recent_low + recent_close) / 3
                r1 = 2 * pivot - recent_low
                r2 = pivot + (recent_high - recent_low)
                r3 = recent_high + 2 * (pivot - recent_low)

                s1 = 2 * pivot - recent_high
                s2 = pivot - (recent_high - recent_low)
                s3 = recent_low - 2 * (recent_high - pivot)
            except:
                # Fallback values if calculation fails
                pivot = r1 = r2 = r3 = s1 = s2 = s3 = 0.0

            return {
                "pivot_support": [s3, s2, s1],
                "pivot_resistance": [r1, r2, r3],
                "pivot_point": pivot,
                "local_support": support_levels,
                "local_resistance": resistance_levels,
            }

        except Exception as e:
            print(f"Error calculating support/resistance for {ticker}: {str(e)}")
            return {
                "pivot_support": [0.0, 0.0, 0.0],
                "pivot_resistance": [0.0, 0.0, 0.0],
                "pivot_point": 0.0,
                "local_support": [0.0] * num_levels,
                "local_resistance": [0.0] * num_levels,
            }

    def calculate_technical_indicators(self):
        """Calculate technical indicators for all stocks."""
        print("--- Calculating technical indicators ---")

        indicators = {}

        for ticker in self.stock_list:
            print(f"Calculating indicators for {ticker}...")

            close_price = self.history[(ticker, "Close")].values.astype(float)
            high_price = self.history[(ticker, "High")].values.astype(float)
            low_price = self.history[(ticker, "Low")].values.astype(float)
            volume = self.history[(ticker, "Volume")].values.astype(float)

            # Moving Averages
            indicators[(ticker, "SMA_20")] = ta.SMA(close_price, timeperiod=20)
            indicators[(ticker, "SMA_50")] = ta.SMA(close_price, timeperiod=50)
            indicators[(ticker, "EMA_12")] = ta.EMA(close_price, timeperiod=12)
            indicators[(ticker, "EMA_26")] = ta.EMA(close_price, timeperiod=26)

            # RSI
            indicators[(ticker, "RSI_14")] = ta.RSI(close_price, timeperiod=14)

            # MACD
            macd, macdsignal, macdhist = ta.MACD(
                close_price, fastperiod=12, slowperiod=26, signalperiod=9
            )
            indicators[(ticker, "MACD")] = macd
            indicators[(ticker, "MACDsignal")] = macdsignal
            indicators[(ticker, "MACDhist")] = macdhist

            # Bollinger Bands
            upperband, middleband, lowerband = ta.BBANDS(
                close_price, timeperiod=20, nbdevup=2, nbdevdn=2
            )
            indicators[(ticker, "BB_upper")] = upperband
            indicators[(ticker, "BB_middle")] = middleband
            indicators[(ticker, "BB_lower")] = lowerband

            # Stochastic Oscillator
            slowk, slowd = ta.STOCH(high_price, low_price, close_price)
            indicators[(ticker, "STOCH_K")] = slowk
            indicators[(ticker, "STOCH_D")] = slowd

            # Volume indicators
            indicators[(ticker, "OBV")] = ta.OBV(close_price, volume)
            indicators[(ticker, "AD")] = ta.AD(
                high_price, low_price, close_price, volume
            )

            # Support and Resistance
            sr_levels = self.calculate_support_resistance(ticker)

            # Add support/resistance as constant arrays
            data_length = len(close_price)
            for i, level in enumerate(sr_levels["pivot_support"][:3]):
                indicators[(ticker, f"Support_{i+1}")] = np.full(data_length, level)

            for i, level in enumerate(sr_levels["pivot_resistance"][:3]):
                indicators[(ticker, f"Resistance_{i+1}")] = np.full(data_length, level)

            indicators[(ticker, "Pivot_Point")] = np.full(
                data_length, sr_levels["pivot_point"]
            )

        # Create indicators DataFrame
        indicators_df = pd.DataFrame(indicators, index=self.history.index)

        # Combine with historical data
        self.history = pd.concat([self.history, indicators_df], axis=1).dropna()

        print("Technical indicators calculation complete.")

    def generate_signals(self):
        """Generate trading signals based on technical indicators."""
        print("--- Generating trading signals ---")

        for ticker in self.stock_list:
            # Trend signals
            self.history[(ticker, "Above_SMA_20")] = (
                self.history[(ticker, "Close")] > self.history[(ticker, "SMA_20")]
            )

            self.history[(ticker, "Above_SMA_50")] = (
                self.history[(ticker, "Close")] > self.history[(ticker, "SMA_50")]
            )

            self.history[(ticker, "Golden_Cross")] = (
                self.history[(ticker, "SMA_20")] > self.history[(ticker, "SMA_50")]
            )

            # MACD signals
            self.history[(ticker, "MACD_Bullish_Crossover")] = (
                self.history[(ticker, "MACD")].shift(1)
                < self.history[(ticker, "MACDsignal")].shift(1)
            ) & (self.history[(ticker, "MACD")] > self.history[(ticker, "MACDsignal")])

            self.history[(ticker, "MACD_Bearish_Crossover")] = (
                self.history[(ticker, "MACD")].shift(1)
                > self.history[(ticker, "MACDsignal")].shift(1)
            ) & (self.history[(ticker, "MACD")] < self.history[(ticker, "MACDsignal")])

            # RSI signals
            self.history[(ticker, "RSI_Signal")] = "Neutral"
            self.history.loc[
                self.history[(ticker, "RSI_14")] > 70, (ticker, "RSI_Signal")
            ] = "Overbought"
            self.history.loc[
                self.history[(ticker, "RSI_14")] < 30, (ticker, "RSI_Signal")
            ] = "Oversold"

            # Bollinger Bands signals
            self.history[(ticker, "BB_Signal")] = "Between Bands"
            self.history.loc[
                self.history[(ticker, "Close")] > self.history[(ticker, "BB_upper")],
                (ticker, "BB_Signal"),
            ] = "Above Upper Band"
            self.history.loc[
                self.history[(ticker, "Close")] < self.history[(ticker, "BB_lower")],
                (ticker, "BB_Signal"),
            ] = "Below Lower Band"

            # Support/Resistance signals
            current_price = self.history[(ticker, "Close")].iloc[-1]

            # Check if price is near support/resistance (within 2%)
            tolerance = 0.02

            self.history[(ticker, "Near_Support")] = False
            self.history[(ticker, "Near_Resistance")] = False

            for i in range(1, 4):
                support_level = self.history[(ticker, f"Support_{i}")].iloc[-1]
                resistance_level = self.history[(ticker, f"Resistance_{i}")].iloc[-1]

                # Check if current price is near support or resistance
                if abs(current_price - support_level) / current_price <= tolerance:
                    self.history.loc[
                        self.history.index[-1], (ticker, "Near_Support")
                    ] = True

                if abs(current_price - resistance_level) / current_price <= tolerance:
                    self.history.loc[
                        self.history.index[-1], (ticker, "Near_Resistance")
                    ] = True

            # Overall signal
            latest_data = self.history.iloc[-1]

            bullish_signals = sum(
                [
                    latest_data[(ticker, "Above_SMA_20")],
                    latest_data[(ticker, "Above_SMA_50")],
                    latest_data[(ticker, "Golden_Cross")],
                    latest_data[(ticker, "MACD_Bullish_Crossover")],
                    latest_data[(ticker, "RSI_Signal")] == "Oversold",
                    latest_data[(ticker, "BB_Signal")] == "Below Lower Band",
                    latest_data[(ticker, "Near_Support")],
                ]
            )

            bearish_signals = sum(
                [
                    not latest_data[(ticker, "Above_SMA_20")],
                    not latest_data[(ticker, "Above_SMA_50")],
                    not latest_data[(ticker, "Golden_Cross")],
                    latest_data[(ticker, "MACD_Bearish_Crossover")],
                    latest_data[(ticker, "RSI_Signal")] == "Overbought",
                    latest_data[(ticker, "BB_Signal")] == "Above Upper Band",
                    latest_data[(ticker, "Near_Resistance")],
                ]
            )

            if bullish_signals >= 4:
                signal = "Strong Buy"
            elif bullish_signals >= 3:
                signal = "Buy"
            elif bearish_signals >= 4:
                signal = "Strong Sell"
            elif bearish_signals >= 3:
                signal = "Sell"
            else:
                signal = "Hold"

            self.history[(ticker, "Overall_Signal")] = signal

        print("Signal generation complete.")

    def get_latest_analysis(self, ticker):
        """Get the latest analysis for a specific ticker."""
        if self.history is None:
            return "No data available. Run full analysis first."

        latest = self.history.iloc[-1]

        analysis = {
            "ticker": ticker,
            "date": self.history.index[-1],
            "price": latest[(ticker, "Close")],
            "volume": latest[(ticker, "Volume")],
            "sma_20": latest[(ticker, "SMA_20")],
            "sma_50": latest[(ticker, "SMA_50")],
            "rsi": latest[(ticker, "RSI_14")],
            "macd": latest[(ticker, "MACD")],
            "macd_signal": latest[(ticker, "MACDsignal")],
            "bb_upper": latest[(ticker, "BB_upper")],
            "bb_lower": latest[(ticker, "BB_lower")],
            "support_levels": [
                latest[(ticker, "Support_1")],
                latest[(ticker, "Support_2")],
                latest[(ticker, "Support_3")],
            ],
            "resistance_levels": [
                latest[(ticker, "Resistance_1")],
                latest[(ticker, "Resistance_2")],
                latest[(ticker, "Resistance_3")],
            ],
            "pivot_point": latest[(ticker, "Pivot_Point")],
            "overall_signal": latest[(ticker, "Overall_Signal")],
            "rsi_signal": latest[(ticker, "RSI_Signal")],
            "bb_signal": latest[(ticker, "BB_Signal")],
            "near_support": latest[(ticker, "Near_Support")],
            "near_resistance": latest[(ticker, "Near_Resistance")],
        }

        return analysis

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("=== Starting Full Stock Analysis ===")
        self.get_stock_info()
        self.get_historical_data()
        self.calculate_technical_indicators()
        self.generate_signals()
        print("=== Analysis Complete ===\n")

        # Print summary for each stock
        for ticker in self.stock_list:
            analysis = self.get_latest_analysis(ticker)
            print(f"\n--- {ticker} Analysis Summary ---")
            print(f"Current Price: ${analysis['price']:.2f}")
            print(f"Overall Signal: {analysis['overall_signal']}")
            print(f"RSI Signal: {analysis['rsi_signal']}")
            print(f"BB Signal: {analysis['bb_signal']}")
            print(
                f"Support Levels: {[f'${level:.2f}' for level in analysis['support_levels']]}"
            )
            print(
                f"Resistance Levels: {[f'${level:.2f}' for level in analysis['resistance_levels']]}"
            )
            print(f"Pivot Point: ${analysis['pivot_point']:.2f}")
            print(f"Near Support: {analysis['near_support']}")
            print(f"Near Resistance: {analysis['near_resistance']}")

    def get_table_for_ticker(self, ticker: str):
        try:
            found = self.history[ticker]
            return tabulate(found.round(3), headers="keys", tablefmt="pipe")
        except:
            return ""

    def get_summary_for_ticker(self, ticker):
        try:
            found = self.ticker_summary[ticker]
            return tabulate(found.items(), tablefmt="plain")
        except:
            return ""


# Usage Example
if __name__ == "__main__":
    # Define your stock list
    stock_list = [
        "AMZN",
        # "AAPL",
        # "AVGO",
        # "BRK-B",
        # "GOOG",
        # "INTC",
        # "JPM",
        # "LLY",
        # "L",
        # "MA",
        # "META",
        # "MSFT",
        # "NVDA",
        # "ORCL",
        # "QCOM",
        # "TDOC",
        # "TSLA",
        # "V",
        # "WMT",
    ]

    # Initialize analyzer
    analyzer = StockAnalyzer(stock_list, period="6mo")

    # Run full analysis
    analyzer.run_full_analysis()

    # Access the processed data
    print(f"\nDataFrame shape: {analyzer.history.shape}")
    print(
        f"Available columns for AMZN: {[col for col in analyzer.history.columns if col[0] == 'AMZN']}"
    )
