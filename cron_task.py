import asyncio
import logging
import os
from itertools import groupby
from operator import attrgetter
from langsmith import Client as LangSmithClient

from models.response import TechnicalAnalysisReport
from dotenv import load_dotenv

from tools.email import send_markdown_email_legacy
from utils.markdown_report import create_markdown_report_from_pydantic
from utils.query_watchlist import query_watchlists
from utils.stock_analyzer import StockAnalyzer

load_dotenv()

queries = asyncio.run(query_watchlists())

unique_tickers = set()

for query in queries:
    unique_tickers.add(query.get("stock_symbol"))

data = StockAnalyzer(list(unique_tickers), period="6mo")

data.run_full_analysis()

queries = [
    {
        **query,
        "data_table": data.get_table_for_ticker(query.get("stock_symbol")),
        "data_summary": data.get_summary_for_ticker(query.get("stock_symbol")),
    }
    for query in queries
]


langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
client = LangSmithClient(api_key=langsmith_api_key)

pulled_object = client.pull_prompt("finance-cron-structured", include_model=True)
prompt = pulled_object.first
llm = pulled_object.last

if not llm:
    raise ValueError(
        "The specified prompt 'finance-cron-structured' does not have a model attached."
    )

runnable = prompt | llm.with_structured_output(TechnicalAnalysisReport)

result = runnable.batch(queries)

result.sort(key=attrgetter("email"))

grouped_by_email = {}

for email, group in groupby(result, key=attrgetter("email")):
    grouped_by_email[email] = list(group)

for email in grouped_by_email.keys():
    a = []

    for x in grouped_by_email[email]:
        try:
            a.append(create_markdown_report_from_pydantic(x))
        except Exception as e:
            logging.error("unable to generate report")
            logging.error(e)

    content = "\n---\n".join(a)

    logging.info(f"Sending email to: {email}")

    send_markdown_email_legacy(
        markdown_content=content, recipient_email=email, subject="Daily stock report"
    )
