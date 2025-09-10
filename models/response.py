from pydantic import BaseModel, Field
from typing import List, Literal


class MarketSnapshot(BaseModel):
    """
    Data related to the current market price and 52-week range.
    """

    current_price: float = Field(description="The current trading price of the stock.")
    fifty_two_week_low: float = Field(
        description="The lowest price the stock has traded at in the past 52 weeks."
    )
    fifty_two_week_high: float = Field(
        description="The highest price the stock has traded at in the past 52 weeks."
    )
    percent_from_high: float = Field(
        description="Percentage difference between current price and 52-week high."
    )
    percent_from_low: float = Field(
        description="Percentage difference between current price and 52-week low."
    )


class TechnicalIndicators(BaseModel):
    """
    Analysis and interpretation of key technical indicators.
    """

    moving_averages: str = Field(
        description="A summary of the stock's trend based on moving averages."
    )
    rsi: str = Field(
        description="The Relative Strength Index (RSI) value and its interpretation (e.g., 'Overbought', 'Oversold', 'Neutral')."
    )
    macd: str = Field(
        description="The Moving Average Convergence Divergence (MACD) signal and trend."
    )
    bollinger_bands: str = Field(
        description="A summary of the stock's volatility based on Bollinger Bands."
    )
    complete_trend_analysis: str = Field(
        description="A summary of the stock's complete trend analysis."
    )


class KeyLevels(BaseModel):
    """
    Important support and resistance price levels.
    """

    support_levels: List[float] = Field(
        description="A list of key support price levels. These are potential price floors where a downtrend might pause."
    )
    resistance_levels: List[float] = Field(
        description="A list of key resistance price levels. These are potential price ceilings where an uptrend might pause."
    )


class Guidance(BaseModel):
    suggested_action: Literal["Buy", "Sell", "Hold", "Neutral"]
    target_prices: List[float] = Field(
        description="One or more price targets for executing the suggested action (e.g., multiple sell levels)."
    )
    stop_loss: float | None = Field(
        default=None, description="Suggested stop-loss level for risk management."
    )


class SummaryRecommendation(BaseModel):
    rating: Literal["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
    rationale: str = Field(
        description="A concise rationale combining all indicators into a clear trading signal."
    )


class TechnicalAnalysisReport(BaseModel):
    """
    A comprehensive technical analysis report for a given stock.
    """

    ticker: str = Field(description="The stock ticker symbol (e.g., 'AAPL', 'TSLA').")
    email: str = Field(description="Target report email address")
    brought_price: float = Field(
        description="The price at which the stock was acquired."
    )
    market_snapshot: MarketSnapshot = Field(
        description="Current and historical price data for the stock."
    )
    technical_indicators: TechnicalIndicators = Field(
        description="Analysis of key technical indicators."
    )
    key_levels: KeyLevels = Field(
        description="Analysis of key support and resistance levels."
    )
    guidance: Guidance = Field(
        description="The final recommendation and optimal entry/exit price."
    )
    summary_recommendation: SummaryRecommendation
