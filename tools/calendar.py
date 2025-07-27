import dataclasses
import logging
import re
from datetime import date, timedelta
from typing import Union, List

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from requests import get
from requests.cookies import RequestsCookieJar
from tabulate import tabulate


@dataclasses.dataclass
class FinancialEvent:
    """Combined financial event dataclass for both earnings and economic events."""

    date: str
    time: str
    type: str  # "earnings" or "economic"
    event: str
    company_ticker: str = ""
    company_name: str = ""
    country: str = ""
    actual: str = ""
    previous: str = ""
    consensus: str = ""
    forecast: str = ""
    eps_estimate: str = ""
    revenue_estimate: str = ""


def human_format(num):
    """Formats a number to a human-readable string with K, M, B suffixes.

    Args:
        num: Number to format (float, int, or None/NaN).

    Returns:
        str: Formatted number string with appropriate suffix (K/M/B) or empty string if invalid.
    """
    if pd.isna(num) or num is None:
        return ""
    try:
        num = float(num)
        if abs(num) >= 1_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        if abs(num) >= 1_000_000:
            return f"{num / 1_000_000:.2f}M"
        if abs(num) >= 1_000:
            return f"{num / 1_000:.2f}K"
        return f"{num:.2f}"
    except (ValueError, TypeError):
        return str(num)


def _parse_date_range(
    start_date: Union[date, str], end_date: Union[date, str, None] = None
) -> List[date]:
    """Parse date range input and return list of dates.

    Args:
        start_date: Start date or single date
        end_date: End date (optional, if None returns single date)

    Returns:
        List of date objects in the range
    """
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)

    if end_date is None:
        return [start_date]

    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    return dates


def _get_single_day_earnings(day: date) -> List[dict]:
    """Fetches earnings data for a single date from EarningsWhispers API.

    Args:
        day: The date to fetch earnings data for.

    Returns:
        List of earnings data dictionaries, empty list if no data or error.
    """
    MAIN_URL = "https://www.earningswhispers.com"

    if day.weekday() in [5, 6]:
        return []

    api_url = f"{MAIN_URL}/api/caldata/{day.isoformat().replace('-', '')}"

    try:
        r = get(url=api_url, headers={"Referer": MAIN_URL}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def _get_single_day_economic_events(target_date: date) -> List[dict]:
    """Fetches economic calendar events for a single date from Trading Economics.

    Args:
        target_date: The date to fetch economic events for.

    Returns:
        List of economic event dictionaries with time, country, event details,
        and actual/previous/consensus values. Empty list if no data or error.
    """
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    }

    try:
        jar = RequestsCookieJar()
        jar.set(
            "calendar-countries", "usa", domain="tradingeconomics.com", path="/calendar"
        )

        response = requests.get(
            "https://tradingeconomics.com/calendar", headers=headers, cookies=jar
        )
        response.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    calendar_table = soup.find("table", {"id": "calendar"})

    if not calendar_table:
        return []

    events = []
    current_date = None
    target_date_str = target_date.strftime("%A %B %d %Y")

    for element in calendar_table.find_all(["thead", "tr"]):
        if element.name == "thead" and "table-header" in element.get("class", []):
            date_text = element.get_text(strip=True)
            date_match = re.search(r"(\w+\s+\w+\s+\d+\s+\d+)", date_text)
            if date_match:
                current_date = date_match.group(1)

        elif (
            element.name == "tr"
            and element.get("data-url")
            and current_date == target_date_str
        ):
            cells = element.find_all("td")
            if len(cells) >= 7:
                time_span = cells[0].find("span")
                time_val = time_span.get_text(strip=True) if time_span else ""

                country_iso_span = cells[1].find("span", class_="calendar-iso")
                country_val = (
                    country_iso_span.get_text(strip=True)
                    if country_iso_span
                    else cells[1].get_text(strip=True)
                )

                event_link = cells[4].find("a", class_="calendar-event")
                event_val = (
                    event_link.get_text(strip=True)
                    if event_link
                    else cells[2].get_text(strip=True).strip()
                )

                actual_span = cells[5].find("span", {"id": "actual"})
                actual_val = actual_span.get_text(strip=True) if actual_span else ""

                prev_span = cells[6].find("span", {"id": "previous"})
                previous_val = prev_span.get_text(strip=True) if prev_span else ""

                consensus_elem = cells[7].find(["span", "a"], {"id": "consensus"})
                consensus_val = (
                    consensus_elem.get_text(strip=True) if consensus_elem else ""
                )

                forecast_elem = element.find(["span", "a"], {"id": "forecast"})
                forecast_val = (
                    forecast_elem.get_text(strip=True) if forecast_elem else ""
                )

                events.append(
                    {
                        "time": time_val,
                        "country": country_val,
                        "event": event_val,
                        "actual": actual_val,
                        "previous": previous_val,
                        "consensus": consensus_val,
                        "forecast": forecast_val,
                    }
                )

    return events


@tool
def fetch_earnings_calendar(
    start_date: Union[date, str], end_date: Union[date, str, None] = None
) -> List[dict]:
    """Fetch corporate earnings announcements for a single date or date range.

    Retrieves earnings data from EarningsWhispers API including company names, tickers,
    EPS estimates, revenue estimates, and release times.

    Args:
        start_date: Start date (date object or ISO string like '2024-01-15')
        end_date: End date (optional). If None, fetches single day. If provided,
                 fetches all dates from start_date to end_date inclusive.

    Returns:
        List of earnings event dictionaries with keys:
        - date: Event date (YYYY-MM-DD)
        - company: Company name
        - ticker: Stock ticker symbol
        - time: Release time (e.g., 'AMC', 'BMO', 'DMT')
        - q1EstEPS: Estimated earnings per share
        - q1RevEst: Estimated revenue

    Examples:
        Single day: fetch_earnings_calendar('2024-01-15')
        Date range: fetch_earnings_calendar('2024-01-15', '2024-01-17')
    """
    dates = _parse_date_range(start_date, end_date)
    all_earnings = []

    for single_date in dates:
        try:
            daily_earnings = _get_single_day_earnings(single_date)
            for earning in daily_earnings:
                earning["date"] = single_date.strftime("%Y-%m-%d")
            all_earnings.extend(daily_earnings)
        except Exception as e:
            logging.warning(f"Error fetching earnings for {single_date}: {e}")

    return all_earnings


@tool
def fetch_economic_calendar(
    start_date: Union[date, str], end_date: Union[date, str, None] = None
) -> List[dict]:
    """Fetch economic events and indicators for a single date or date range.

    Retrieves economic calendar data from Trading Economics including GDP releases,
    employment data, inflation reports, Fed announcements, and other macro events.

    Args:
        start_date: Start date (date object or ISO string like '2024-01-15')
        end_date: End date (optional). If None, fetches single day. If provided,
                 fetches all dates from start_date to end_date inclusive.

    Returns:
        List of economic event dictionaries with keys:
        - date: Event date (YYYY-MM-DD)
        - time: Release time (e.g., '8:30 AM', '2:00 PM')
        - country: Country code (e.g., 'USA', 'EUR', 'GBP')
        - event: Event name/description
        - actual: Actual released value
        - previous: Previous period value
        - consensus: Market consensus/forecast
        - forecast: Additional forecast data

    Examples:
        Single day: fetch_economic_calendar('2024-01-15')
        Date range: fetch_economic_calendar('2024-01-15', '2024-01-17')
    """
    dates = _parse_date_range(start_date, end_date)
    all_events = []

    for single_date in dates:
        try:
            daily_events = _get_single_day_economic_events(single_date)
            for event in daily_events:
                event["date"] = single_date.strftime("%Y-%m-%d")
            all_events.extend(daily_events)
        except Exception as e:
            logging.warning(f"Error fetching economic events for {single_date}: {e}")

    return all_events


@tool
def get_comprehensive_financial_calendar(
    start_date: Union[date, str], end_date: Union[date, str, None] = None
) -> str:
    """Get complete financial calendar combining earnings and economic events for date(s).

    Provides a unified view of all financial market events including corporate earnings
    announcements and economic data releases. Data is formatted as a readable table
    with time-based sorting.

    Args:
        start_date: Start date (date object or ISO string like '2024-01-15')
        end_date: End date (optional). If None, shows single day. If provided,
                 shows all dates from start_date to end_date inclusive.

    Returns:
        Formatted string table containing:
        - Date, Time, Type (Earnings/Economic)
        - Event details (company earnings or economic indicator)
        - Ticker symbols for earnings
        - EPS/Revenue estimates for earnings
        - Country, Actual/Previous/Consensus values for economic events

    Examples:
        Single day: get_comprehensive_financial_calendar('2024-01-15')
        Date range: get_comprehensive_financial_calendar('2024-01-15', '2024-01-17')
        Weekend handling: Returns message for weekend dates
        No events: Returns "No events found" message
    """
    try:
        dates = _parse_date_range(start_date, end_date)

        # Filter out weekends
        weekday_dates = [d for d in dates if d.weekday() < 5]
        if not weekday_dates:
            date_range_str = (
                f"{dates[0].strftime('%Y-%m-%d')}"
                if len(dates) == 1
                else f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
            )
            return f"No financial events available for {date_range_str} (weekend days only)."

        all_events = []

        # Fetch earnings data
        try:
            for single_date in weekday_dates:
                earnings_data = _get_single_day_earnings(single_date)
                for item in earnings_data:
                    event = FinancialEvent(
                        date=single_date.strftime("%Y-%m-%d"),
                        time=item.get("releaseTime", ""),
                        type="Earnings",
                        event=f"{item.get('company', '')} Earnings",
                        company_ticker=item.get("ticker", ""),
                        company_name=item.get("company", ""),
                        eps_estimate=human_format(item.get("q1EstEPS")),
                        revenue_estimate=human_format(item.get("q1RevEst")),
                    )
                    all_events.append(event)
        except Exception as e:
            logging.warning(f"Error fetching earnings: {e}")

        # Fetch economic events
        try:
            for single_date in weekday_dates:
                econ_events = _get_single_day_economic_events(single_date)
                for item in econ_events:
                    event = FinancialEvent(
                        date=single_date.strftime("%Y-%m-%d"),
                        time=item.get("time", ""),
                        type="Economic",
                        event=item.get("event", ""),
                        country=item.get("country", ""),
                        actual=item.get("actual", ""),
                        previous=item.get("previous", ""),
                        consensus=item.get("consensus", ""),
                        forecast=item.get("forecast", ""),
                    )
                    all_events.append(event)
        except Exception as e:
            logging.warning(f"Error fetching economic events: {e}")

        if not all_events:
            date_range_str = (
                f"{weekday_dates[0].strftime('%Y-%m-%d')}"
                if len(weekday_dates) == 1
                else f"{weekday_dates[0].strftime('%Y-%m-%d')} to {weekday_dates[-1].strftime('%Y-%m-%d')}"
            )
            return f"No financial events found for {date_range_str}."

        # Sort by date, then time, then type
        all_events.sort(key=lambda x: (x.date, str(x.time) or "ZZZ", x.type))

        # Create display data
        display_data = []
        for event in all_events:
            if event.type == "Earnings":
                display_data.append(
                    {
                        "Date": event.date,
                        "Type": event.type,
                        "Event": event.event,
                        "Ticker": event.company_ticker,
                        "EPS Est": event.eps_estimate,
                        "Rev Est": event.revenue_estimate,
                        "Country": "",
                        "Actual": "",
                        "Previous": "",
                        "Consensus": "",
                    }
                )
            else:  # Economic
                display_data.append(
                    {
                        "Date": event.date,
                        "Type": event.type,
                        "Event": event.event,
                        "Ticker": "",
                        "EPS Est": "",
                        "Rev Est": "",
                        "Country": event.country,
                        "Actual": event.actual,
                        "Previous": event.previous,
                        "Consensus": event.consensus,
                    }
                )

        df = pd.DataFrame(display_data)

        date_range_str = (
            f"{weekday_dates[0].strftime('%Y-%m-%d')}"
            if len(weekday_dates) == 1
            else f"{weekday_dates[0].strftime('%Y-%m-%d')} to {weekday_dates[-1].strftime('%Y-%m-%d')}"
        )

        return f"Financial Calendar for {date_range_str}:\n\n" + tabulate(
            df, headers="keys", tablefmt="github", showindex=False
        )

    except Exception as e:
        logging.exception(e)
        return f"Error occurred: {e}"


# Legacy tool names for backward compatibility
@tool
def get_earnings(day: date):
    """Legacy function - use fetch_earnings_calendar instead."""
    return _get_single_day_earnings(day)


@tool
def get_economic_events(target_date: date):
    """Legacy function - use fetch_economic_calendar instead."""
    return _get_single_day_economic_events(target_date)


@tool
async def get_financial_calendar(target_date: date) -> str:
    """Legacy function - use get_comprehensive_financial_calendar instead."""
    return get_comprehensive_financial_calendar(target_date)


# Updated tools list
financial_calendar_tools = [
    fetch_earnings_calendar,
    fetch_economic_calendar,
    get_comprehensive_financial_calendar,
]
