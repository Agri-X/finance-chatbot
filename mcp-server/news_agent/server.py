import logging
import os
from newsapi import NewsApiClient
from typing import Optional, Dict, Any
from newspaper import Article

from mcp.server.fastmcp import FastMCP

# --- Basic Setup ---
logger = logging.getLogger(__name__)

# --- MCP Agent Definition ---
mcp = FastMCP(
    dependencies=["newsapi-python"],
    name="financial-news-agent",
    prompt="""
You are a highly specialized financial news assistant. Your purpose is to find relevant financial news articles about companies, markets, and economic topics. You can use advanced search operators like `+`, `-`, `AND`, `OR`, and `"` in your queries for more precision.

You have three tools at your disposal:

1.  `get_top_headlines`: Use this to fetch live, top, and breaking financial news. This is best for general market updates or top business news from a specific country.
2.  `get_all_news`: Use this for in-depth historical searches on specific financial topics or companies. It supports advanced search queries and allows you to specify which fields to search in (title, description, content).
3.  `get_sources`: Use this to discover which financial news sources are available.

When a user asks for information, use these tools to find the most relevant financial news. For example:
- "What's the latest news on Apple?" -> `get_top_headlines(query='Apple')`
- "Find articles about Tesla but not Elon Musk." -> `get_all_news(query='Tesla -Musk')`
- "Search for 'initial public offering' in article titles." -> `get_all_news(query='"initial public offering"', search_in='title')`

Always return the direct, unmodified output from the tools. Your internal mechanisms will handle focusing the queries on financial data.
""",
)


# --- Helper Function ---
def _get_news_client() -> NewsApiClient | Dict[str, str]:
    """
    Initializes and returns a NewsApiClient instance or an error dictionary.
    """
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        error_msg = (
            "Error: NEWS_API_KEY environment variable not set. Cannot fetch news."
        )
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}
    return NewsApiClient(api_key=api_key)


# --- MCP Tools ---
@mcp.tool()
def get_all_news(
    query: str,
    sources: Optional[str] = None,
    domains: Optional[str] = None,
    from_param: Optional[str] = None,
    to: Optional[str] = None,
    language: str = "en",
    sort_by: str = "publishedAt",
    page_size: int = 100,
    page: int = 1,
):
    """
    Search through millions of articles. This tool is automatically tailored for financial news.

    Args:
        query (str): Keywords or phrases. Advanced search is supported:
            - Surround phrases with quotes (") for exact match.
            - Prepend words with + to require them (e.g., +bitcoin).
            - Prepend words with - to exclude them (e.g., -bitcoin).
            - Use AND / OR / NOT for boolean logic (e.g., crypto AND (ethereum OR litecoin)).
        search_in (str, optional): The fields to restrict your search to. Multiple options can be
            comma-separated (e.g., 'title,content'). Possible options: 'title', 'description', 'content'.
            Default is all fields.
        sources (str, optional): A comma-separated string of news source identifiers.
        domains (str, optional): A comma-separated string of domains to restrict the search to.
        from_param (str, optional): Start date (YYYY-MM-DD).
        to (str, optional): End date (YYYY-MM-DD).
        language (str, optional): The 2-letter ISO-639-1 code of the language. Defaults to 'en'.
        sort_by (str, optional): Order to sort articles in. 'relevancy', 'popularity', 'publishedAt'.
        page_size (int, optional): The number of results to return per page. Max 100.
        page (int, optional): Use this to page through the results.
    """
    client = _get_news_client()
    if isinstance(client, dict):
        return client

    financial_query = f"({query})"
    logger.info(f"Original query: '{query}', Enhanced query: '{financial_query}'")

    try:
        response = client.get_everything(
            q=financial_query,
            sources=sources,
            domains=domains,
            from_param=from_param,
            to=to,
            language=language,
            sort_by=sort_by,
            page_size=page_size,
            page=page,
        )
        logger.info(f"get_all_news called. Status: {response.get('status')}")
        return to_markdown(response.get("articles", []))
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in get_all_news: {e}", exc_info=True
        )
        return f"An unexpected error occurred: {e}"


@mcp.tool()
def get_top_headlines(
    query: Optional[str] = None,
    sources: Optional[str] = None,
    category: str = "business",
    language: str = "en",
    country: Optional[str] = None,
    page_size: int = 100,
    page: int = 1,
) -> Dict[str, Any]:
    """
    Provides live top headlines, defaulting to the business category.

    Args:
        query (str, optional): Keywords or phrases. Advanced search is supported:
            - Surround phrases with quotes (") for exact match.
            - Prepend words with + to require them (e.g., +bitcoin).
            - Prepend words with - to exclude them (e.g., -bitcoin).
            - Use AND / OR / NOT for boolean logic (e.g., crypto AND (ethereum OR litecoin)).
        sources (str, optional): A comma-separated string of news source identifiers.
        category (str, optional): The category to get headlines for. Defaults to 'business'.
        language (str, optional): The 2-letter ISO-639-1 code of the language. Defaults to 'en'.
        country (str, optional): The 2-letter ISO-3166-1 code of the country.
        page_size (int, optional): The number of results per page. Max 100.
        page (int, optional): Use this to page through the results.
    """
    client = _get_news_client()
    if isinstance(client, dict):
        return client

    final_query = query
    if query:
        final_query = f"({query})"
        logger.info(f"Original query: '{query}', Enhanced query: '{final_query}'")

    try:
        response = client.get_top_headlines(
            q=final_query,
            sources=sources,
            category=category,
            language=language,
            country=country,
            page_size=page_size,
            page=page,
        )
        logger.info(f"get_top_headlines called. Status: {response.get('status')}")
        return response
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in get_top_headlines: {e}", exc_info=True
        )
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}


@mcp.tool()
def get_sources(
    category: str = "business", language: str = "en", country: Optional[str] = None
) -> Dict[str, Any]:
    """
    Returns news publishers, defaulting to the business category.
    """
    client = _get_news_client()
    if isinstance(client, dict):
        return client

    try:
        response = client.get_sources(
            category=category, language=language, country=country
        )
        logger.info(f"get_sources called. Status: {response.get('status')}")
        return response
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_sources: {e}", exc_info=True)
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}


@mcp.tool()
def fetch_article_content(url: str) -> str:
    """
    Fetches the full content of an article from its URL.

    Args:
        url (str): The URL of the article to fetch.

    Returns:
        str: The full content of the article.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logger.error(f"Error fetching article content: {e}", exc_info=True)
        return f"Error fetching article content: {e}"


def to_markdown(articles):
    """
    Converts a list of article dictionaries into a Markdown formatted string.

    Args:
      articles: A list of dictionaries, where each dictionary represents an article.

    Returns:
      A string in Markdown format.
    """
    markdown_output = ""
    for article in articles:
        # Extract the relevant fields from the article dictionary
        title = article.get("title", "No Title")
        published_at = article.get("publishedAt", "Unknown Publication Date")
        description = article.get("description", "No Description Available.")
        content = article.get("content", "No Content Available.")

        logger.info(content)

        # Format the extracted information into Markdown
        markdown_output += f"### {title}\n\n"

        if description:
            markdown_output += f"{description}\n\n"

        if content:
            markdown_output += f"**Content:** {content}\n\n"

        markdown_output += f"**Published on:** {published_at.split('T')[0]}\n"
        markdown_output += "\n---\n\n"

    return markdown_output


# --- Main Execution Block ---
if __name__ == "__main__":
    # Before running, ensure you have set the API key:
    # export NEWS_API_KEY='your_api_key_here'
    print("Starting Financial News Agent Server...")
    mcp.run()
