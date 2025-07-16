from os import getenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
import markdown
from langchain.tools import tool


@tool
def send_markdown_email(
    recipient_email: str,
    subject: str,
    markdown_content: str,
) -> bool:
    """
    Sends an email with content formatted from Markdown.

    This function converts the provided Markdown content into HTML
    and sends it as a rich-text email. It requires sender credentials
    and SMTP server details.

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

        html_content = markdown.markdown(markdown_content)

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

        print(f"Email sent successfully to {recipient_email}")
        return True

    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


email_tools = [send_markdown_email]
