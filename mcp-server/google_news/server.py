import logging
from GoogleNews import GoogleNews
from typing import List, Dict, Any
import newspaper

from mcp.server.fastmcp import FastMCP

# --- Basic Setup ---
logger = logging.getLogger(__name__)

# --- MCP Agent Definition ---
mcp = FastMCP(
    dependencies=["GoogleNews"],
    name="google-news-agent",
    prompt="""
You are a specialized news assistant powered by Google News search. Use the following tools to fetch headlines or archived news:

1. `search_news`: Keyword-based search across Google News via news.google.com.
2. `set_topic`: Limit searches to a Google News topic ID.
3. `set_section`: Further narrow by section under a selected topic. q1
4. `paginate`: Retrieve specific pages of results.
5. `get_results`: Return structured article data.
6. `clear_results`: Reset cached search results.

Always return the raw output of the tools; downstream logic will handle formatting and reasoning.
""",
)


# --- Internal helper to manage client state ---
def _get_client() -> GoogleNews:
    return GoogleNews()


# --- MCP Tools ---
@mcp.tool()
def search_news(query: str) -> List[Dict[str, Any]]:
    """
    Perform a keyword-based Google News search.

    Args:
        query (str): Your search query text. Supports plain keywords and
            advanced operators like `"exact phrase"`, +mustInclude, -mustExclude.

    Returns:
        List[Dict[str, Any]]: A list of articles, each with:
            - title: Headline text
            - media: Source name (e.g., CNN, BBC)
            - date: Publication date as string
            - datetime: ISO‑formatted datetime
            - desc: Snippet description
            - link: URL to full article
            - img: URL to thumbnail image (if any)

    Example:
        results = search_news("climate change")
    """
    client = _get_client()
    try:
        client.get_news(query)
        results = client.results()
        logger.info(f"search_news: fetched {len(results)} items")
        # for item in results:
        #     item["context"] = convert_news_to_markdown(item["link"])
        return results
    except Exception as e:
        logger.error(f"search_news error: {e}", exc_info=True)
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def set_topic(topic_id: str) -> str:
    """
    Limit subsequent searches to a specific Google News topic.

    Args:
        topic_id (str): Google News topic token (e.g.
            "CAAqKggKIiRDQkFTR...") from the URL under news.google.com/topics/

    Returns:
        str: Confirmation message.

    Example:
        # Set to "Sports"
        set_topic("CAAqKggKIiRDQkFTR...")
    """
    client = _get_client()
    client.set_topic(topic_id)
    return f"topic set to {topic_id}"


@mcp.tool()
def set_section(section_id: str) -> str:
    """
    Further narrow searches within a previously set topic.

    Args:
        section_id (str): Section token (e.g.
            "CAQiS0NCQVNNZ29JTDIwdk1E...") found in the URL under topics.

    Returns:
        str: Confirmation message.

    Example:
        set_section("CAQiS0NCQVNNZ29JTDIwdk1E...")  # e.g., “Football” under Sports
    """
    client = _get_client()
    client.set_section(section_id)
    return f"section set to {section_id}"


@mcp.tool()
def paginate(page: int) -> List[Dict[str, Any]]:
    """
    Retrieve a specific page of the current search results.

    Args:
        page (int): The page number (1-based).

    Returns:
        List[Dict[str, Any]]: Articles on the requested page.

    Example:
        paginate(2)  # Get the second page of results
    """
    client = _get_client()
    client.get_page(page)
    results = client.results()
    logger.info(f"paginate to page {page}: {len(results)} items")
    return results


@mcp.tool()
def get_results(sort: bool = False) -> List[Dict[str, Any]]:
    """
    Return the list of current results, optionally sorted chronologically.

    Args:
        sort (bool): If True, sort results by descending date.

    Returns:
        List[Dict[str, Any]]: Current batch of articles.

    Example:
        get_results(sort=True)
    """
    client = _get_client()
    return client.results(sort=sort)


@mcp.tool()
def clear_results() -> str:
    """
    Clear previously fetched search data to prepare for a fresh query.

    Returns:
        str: Confirmation message.

    Example:
        clear_results()
    """
    client = _get_client()
    client.clear()
    return "results cleared"


# --- Internal helper functions ---
def _process_article(url: str) -> newspaper.Article:
    """
    Internal helper to download and parse an article.

    Args:
        url (str): The URL of the news article

    Returns:
        newspaper.Article: Processed article object

    Raises:
        Exception: If article cannot be processed
    """
    article = newspaper.Article(url)
    article.download()
    article.parse()
    article.nlp()  # Run NLP for summary and keywords
    return article


def convert_news_to_markdown(url: str) -> Dict[str, Any]:
    """
    Fetches a news article from the given URL and converts its content
    into a Markdown formatted string.

    Args:
        url (str): The URL of the news article to convert

    Returns:
        Dict[str, Any]: A dictionary containing:
            - status: "success" or "error"
            - markdown: The Markdown formatted string (if successful)
            - message: Error message (if failed)
            - url: The original URL

    Example:
        result = convert_news_to_markdown("https://example.com/news-article")
    """
    try:
        article = _process_article(url)

        markdown_output = []

        # Add Title (H1)
        if article.title:
            markdown_output.append(f"# {article.title}\n")
        else:
            markdown_output.append("# Untitled Article\n")

        # Add Author(s)
        if article.authors:
            authors_str = ", ".join(article.authors)
            markdown_output.append(f"**Authors:** {authors_str}\n")

        # Add Publish Date
        if article.publish_date:
            formatted_date = article.publish_date.strftime("%B %d, %Y")
            markdown_output.append(f"**Published:** {formatted_date}\n")

        # Add Top Image (if available) as Markdown image link
        if article.top_image:
            markdown_output.append(
                f"![{article.title or 'Article Image'}]({article.top_image})\n"
            )

        markdown_output.append("---\n")  # Separator

        # Add Article Text
        if article.text:
            clean_text = "\n\n".join(
                [p.strip() for p in article.text.split("\n") if p.strip()]
            )
            markdown_output.append(clean_text)
        else:
            markdown_output.append(
                "*(No main content could be extracted for this article.)*\n"
            )

        # Add keywords if available
        if article.keywords:
            markdown_output.append("\n\n---\n")
            markdown_output.append("## Keywords\n")
            markdown_output.append(
                ", ".join([f"`{keyword}`" for keyword in article.keywords])
            )
            markdown_output.append("\n")

        # Add summary if available
        if article.summary:
            markdown_output.append("\n\n---\n")
            markdown_output.append("## Summary\n")
            markdown_output.append(f"> {article.summary}\n")

        markdown_result = "\n".join(markdown_output)

        logger.info(f"convert_news_to_markdown: successfully processed {url}")
        return {"status": "success", "markdown": markdown_result, "url": url}

    except Exception as e:
        error_msg = f"Error processing URL '{url}': {e}"
        logger.error(f"convert_news_to_markdown error: {error_msg}", exc_info=True)
        return {"status": "error", "message": error_msg, "url": url}


if __name__ == "__main__":
    print("Starting Investor Agent Server...")
    mcp.run()
