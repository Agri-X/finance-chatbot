import asyncio
import json
import logging
import os

from datetime import datetime
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langsmith import Client as LangSmithClient

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from tools.calendar import financial_calendar_tools
from utils.prompt import system_template

import asyncpg

load_dotenv()

logger = logging.getLogger(__name__)


DEFAULT_RECURSION_LIMIT = 100
BASE_PATH = Path(__file__).resolve().parent


async def query_watchlists():
    """
    Connects to a PostgreSQL database using asyncpg and executes a query
    to retrieve user watchlists with stock information aggregated as JSON objects.
    This function is designed to be run asynchronously.
    """
    connection = None
    try:
        connection = await asyncpg.connect(os.environ.get("DATABASE_URL"))
        logger.info("Asynchronous database connection established successfully.")

        sql_query = """
        SELECT
            u.identifier,
            json_agg(
                json_build_object(
                    'stock_symbol', w.stock_symbol,
                    'bought_price', w.bought_price,
                    'quantity', w.quantity
                )
            ) AS stocks
        FROM
            "User" u
        JOIN
            watchlists w ON w.user_id = u.identifier
        GROUP BY
            u.identifier;
        """
        results = await connection.fetch(sql_query)

        res = []
        for row in results:
            user_id = row["identifier"]
            stocks_json = row["stocks"]
            res.append(
                {
                    "user_id": user_id,
                    "stocks_json": stocks_json,
                }
            )
        return res

    except Exception as error:
        logger.info(f"Error while connecting to or querying PostgreSQL: {error}")
        raise error

    finally:
        if connection:
            await connection.close()
            logger.info("Asynchronous PostgreSQL connection is closed.")


async def load_system_prompt():
    """Load system prompt from LangSmith or use default."""
    try:
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        if not langsmith_api_key:
            raise ValueError("LANGSMITH_API_KEY not set")

        client = LangSmithClient(api_key=langsmith_api_key)
        prompt_template = client.pull_prompt("finance-chatbot-cron", include_model=True)

        prompt_str = prompt_template.steps[0].format(
            date=datetime.now().strftime("%Y-%m-%d")
        )

        param = prompt_template.steps[1].bound._get_ls_params()
        model_name = prompt_template.steps[1].bound.model.split("/")[-1]

        model_provider = param["ls_provider"]

        logger.info(f"Loaded system prompt from LangSmith with model: {model_name}")
        return SystemMessage(content=prompt_str), model_name, model_provider

    except Exception as e:
        logger.warning(f"Could not pull from LangSmith: {e}. Using default prompt.")

        rendered = system_template.format(date=datetime.now().strftime("%Y-%m-%d"))
        return SystemMessage(content=rendered), None, None


async def on_message(msgs: List[str]):
    prompt, session_model_name, session_model_provider = await load_system_prompt()

    model_name = os.getenv("MODEL_NAME", session_model_name)
    model_provider = os.getenv("MODEL_PROVIDER", session_model_provider)

    mcp_client = MultiServerMCPClient(
        {
            "yf_server": {
                "command": "python",
                "args": [str(BASE_PATH / "mcp-server/yf_server/server.py")],
                "env": {"PYTHONPATH": str(BASE_PATH)},
                "transport": "stdio",
            },
            "finance": {
                "command": "python",
                "args": [str(BASE_PATH / "mcp-server/investor_agent/server.py")],
                "env": {"PYTHONPATH": str(BASE_PATH)},
                "transport": "stdio",
            },
            "news_agent": {
                "command": "python",
                "args": [str(BASE_PATH / "mcp-server/news_agent/server.py")],
                "env": {
                    "PYTHONPATH": str(BASE_PATH),
                    "NEWS_API_KEY": os.environ.get("NEWS_API_KEY", ""),
                },
                "transport": "stdio",
            },
        }
    )
    mcp_tools = await mcp_client.get_tools()
    all_tools = mcp_tools + financial_calendar_tools

    logger.info(f"Populated {len(all_tools)} tools.")

    model = init_chat_model(f"{model_provider}:{model_name}")

    main_model = create_react_agent(
        model,
        all_tools,
        prompt=prompt,
    )

    final_answer = await main_model.abatch(
        inputs=[{"messages": [HumanMessage(content=msg)]} for msg in msgs],
    )

    return final_answer


from os import getenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from markdown import markdown
from markdown.extensions.tables import TableExtension


async def send_markdown_email(
    recipient_email: str,
    subject: str,
    markdown_content: str,
) -> bool:
    """
    Sends an email with content formatted from Markdown.

    This function converts the provided Markdown content into HTML
    and sends it as a rich-text email.

    Args:
        recipient_email (str): The email address of the recipient.
        subject (str): The subject line of the email.
        markdown_content (str): The content of the email in Markdown format.

    Returns:
        bool: True if the email was sent successfully, False otherwise.
    """
    try:
        mail_from = getenv("MAIL_FROM")
        mail_host = getenv("MAIL_HOST")
        mail_port = getenv("MAIL_PORT")
        mail_username = getenv("MAIL_USERNAME")
        mail_password = getenv("MAIL_PASSWORD")

        if (
            not mail_host
            or not mail_port
            or not mail_username
            or not mail_password
            or not mail_from
        ):
            raise Exception("Config invalid")

        html_content = markdown(markdown_content, extensions=[TableExtension()])

        msg = MIMEMultipart("alternative")
        mail_to = recipient_email
        mail_subject = subject

        msg["From"] = mail_from
        msg["To"] = mail_to
        msg["Subject"] = mail_subject

        part1 = MIMEText(markdown_content, "plain")
        part2 = MIMEText(html_content, "html")

        msg.attach(part1)
        msg.attach(part2)

        with smtplib.SMTP(mail_host, int(mail_port)) as server:
            server.starttls()
            server.login(mail_username, mail_password)
            server.sendmail(mail_username, mail_to, msg.as_string())

        logger.info(f"Email sent successfully to {recipient_email}")
        return True

    except Exception as e:
        logger.info(f"Failed to send email: {e}")
        return False


async def main():
    prompts_to_send = []
    users = await query_watchlists()

    # Step 1: Prepare prompts for all users
    for user in users:
        watchlists = json.loads(user["stocks_json"])

        prompt_template = """Act like a professional technical analyst specializing in stock price charting and trading strategies. 

Objective: I want a detailed technical analysis of all these stocks I currently hold, including support and resistance levels, trading signals, and exit strategy advice.

Here is my current portfolio:\n"""

        for stock in watchlists:
            prompt_template += f"- {stock['stock_symbol']}"

            # Use 'get' to safely access keys and avoid KeyError
            bought_price = stock.get("bought_price")
            quantity = stock.get("quantity")

            parts = []
            if bought_price is not None and bought_price != 0:
                parts.append(f"entry price = {bought_price}")

            if quantity is not None and quantity != 0:
                parts.append(f"quantity = {quantity}")

            if parts:
                prompt_template += " | " + ", ".join(parts)

            prompt_template += "\n"

        prompts_to_send.append({"email": user["user_id"], "prompt": prompt_template})

    ai_responses = await on_message([p["prompt"] for p in prompts_to_send])

    for user_prompt, ai_response in zip(prompts_to_send, ai_responses):
        email_address = user_prompt["email"]

        markdown_content = ai_response.get("messages", [{}])[-1].content

        await send_markdown_email(
            markdown_content=markdown_content,
            recipient_email=email_address,
            subject="Daily report",
        )

    logger.info("All emails sent successfully.")


if __name__ == "__main__":

    asyncio.run(main())
