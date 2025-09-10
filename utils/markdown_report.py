from models.response import TechnicalAnalysisReport


def create_markdown_report_from_pydantic(report: TechnicalAnalysisReport) -> str:
    """
    Converts a TechnicalAnalysisReport Pydantic model instance into a markdown string.

    Args:
        report: An instance of the TechnicalAnalysisReport model.

    Returns:
        A markdown-formatted string of the report.
    """
    # Create an empty list to store markdown lines
    md_lines = []

    # Report Header
    md_lines.append(f"# Technical Analysis Report for {report.ticker}\n")
    if report.brought_price:
        md_lines.append(f"Brought at {report.brought_price}\n")
    md_lines.append(f"**Recommendation:** {report.summary_recommendation.rating}\n")
    md_lines.append(f"**Rationale:** {report.summary_recommendation.rationale}\n")
    md_lines.append("---\n")

    # Market Snapshot Section
    md_lines.append("## Market Snapshot\n")
    snapshot = report.market_snapshot
    md_lines.append(f"- **Current Price:** ${snapshot.current_price:.2f}")
    md_lines.append(f"- **52-Week High:** ${snapshot.fifty_two_week_high:.2f}")
    md_lines.append(f"- **52-Week Low:** ${snapshot.fifty_two_week_low:.2f}")
    md_lines.append(
        f"- **Percent from 52-Week High:** {snapshot.percent_from_high:.2f}%"
    )
    md_lines.append(f"- **Percent from 52-Week Low:** {snapshot.percent_from_low:.2f}%")
    md_lines.append("\n---\n")

    # Technical Indicators Section
    md_lines.append("## Technical Indicators\n")
    indicators = report.technical_indicators
    md_lines.append(f"- **Moving Averages:** {indicators.moving_averages}")
    md_lines.append(f"- **RSI:** {indicators.rsi}")
    md_lines.append(f"- **MACD:** {indicators.macd}")
    md_lines.append(f"- **Bollinger Bands:** {indicators.bollinger_bands}")
    md_lines.append(f"- **Trend Analysis:** {indicators.complete_trend_analysis}")
    md_lines.append("\n---\n")

    # Key Levels Section
    md_lines.append("## Key Levels\n")
    levels = report.key_levels
    support_str = ", ".join([f"${price:.2f}" for price in levels.support_levels])
    resistance_str = ", ".join([f"${price:.2f}" for price in levels.resistance_levels])
    md_lines.append(f"- **Support:** {support_str}")
    md_lines.append(f"- **Resistance:** {resistance_str}")
    md_lines.append("\n---\n")

    # Guidance Section
    md_lines.append("## Guidance\n")
    guidance = report.guidance
    md_lines.append(f"- **Suggested Action:** **{guidance.suggested_action}**")
    md_lines.append(
        f"- **Target Prices:** {', '.join([f'${p:.2f}' for p in guidance.target_prices])}"
    )
    if guidance.stop_loss is not None:
        md_lines.append(f"- **Stop Loss:** ${guidance.stop_loss:.2f}")
    md_lines.append("\n")

    # Join all lines and return
    return "\n".join(md_lines)
