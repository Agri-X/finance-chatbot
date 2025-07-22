import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from requests.cookies import RequestsCookieJar
from langchain.tools import tool
from tabulate import tabulate


@tool
async def scrape_trading_economics_calendar():
    """
    Scrapes economic calendar data from Trading Economics.

    Args:
        url (str): The URL of the Trading Economics calendar page.

    Returns:
        str: strung the scraped calendar data.
    """
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    }

    try:
        jar = RequestsCookieJar()
        jar.set(
            "calendar-countries",
            "usa",
            domain="tradingeconomics.com",
            path="/calendar",
        )

        response = requests.get(
            "https://tradingeconomics.com/calendar",
            headers=headers,
            cookies=jar,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL : {e}")
        return None

    soup = BeautifulSoup(response.content, "html.parser")

    calendar_table = soup.find("table", {"id": "calendar"})
    if not calendar_table:
        print("Error: Calendar table with ID 'calendar' not found.")
        return None

    data = []
    current_date = None

    for element in calendar_table.find_all(["thead", "tr"]):

        if element.name == "thead" and "table-header" in element.get("class", []):
            date_text = element.get_text(strip=True)

            date_match = re.search(r"(\w+\s+\w+\s+\d+\s+\d+)", date_text)
            if date_match:
                current_date = date_match.group(1)

        elif element.name == "tr" and element.get("data-url"):
            row_data = {}
            cells = element.find_all("td")

            if len(cells) >= 7:

                row_data["Date"] = current_date

                time_span = cells[0].find("span")
                row_data["Time"] = time_span.get_text(strip=True) if time_span else ""

                country_iso_span = cells[1].find("span", class_="calendar-iso")
                if country_iso_span:
                    row_data["Country"] = country_iso_span.get_text(strip=True)
                else:

                    row_data["Country"] = cells[1].get_text(strip=True)

                event_link = cells[4].find("a", class_="calendar-event")
                if event_link:
                    row_data["Event"] = event_link.get_text(strip=True)
                else:

                    row_data["Event"] = cells[2].get_text(strip=True).strip()

                actual_span = cells[5].find("span", {"id": "actual"})
                row_data["Actual"] = (
                    actual_span.get_text(strip=True) if actual_span else ""
                )

                prev_span = cells[6].find("span", {"id": "previous"})
                row_data["Previous"] = (
                    prev_span.get_text(strip=True) if prev_span else ""
                )

                consensus_elem = cells[7].find(["span", "a"], {"id": "consensus"})
                row_data["Consensus"] = (
                    consensus_elem.get_text(strip=True) if consensus_elem else ""
                )

                forecast_elem = element.find(["span", "a"], {"id": "forecast"})
                row_data["Forecast"] = (
                    forecast_elem.get_text(strip=True) if forecast_elem else ""
                )

                row_data["Category"] = element.get("data-category", "")
                row_data["Symbol"] = element.get("data-symbol", "")

                data.append(row_data)

    df = pd.DataFrame(data)
    data = f"""
{tabulate(df, headers="keys", tablefmt="github", showindex=False)}
"""
    return data
