# tools.py
import os
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from ddgs import DDGS  # pip install ddgs  (duckduckgo-search was renamed to ddgs)

MAX_SEARCH_RESULTS = 5
MAX_PAGE_CHARS = 3000
REQUEST_TIMEOUT = 8


@tool
def web_search(query: str) -> str:
    """Search the web for recent, specific info about a company — funding,
    launches, hiring, pricing changes, LinkedIn activity, news. Pass a
    focused query like '<company name> funding 2026' or '<company name>
    LinkedIn hiring'. Returns a short list of result titles + snippets."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_SEARCH_RESULTS))
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    formatted = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        formatted.append(f"- {title}: {body} ({href})")
    return "\n".join(formatted)


@tool
def fetch_company_website(url: str) -> str:
    """Fetch and extract readable text from a company's website (e.g. their
    homepage or about page). Use this to pull positioning, product, or
    'about us' language directly from the source when web_search results
    are thin. Pass a full URL including https://."""
    if not url or not url.startswith(("http://", "https://")):
        return f"Invalid URL: '{url}'"

    try:
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SalesResearchBot/1.0)"},
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Failed to fetch {url}: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    if not text:
        return f"No readable text found at {url}"

    return text[:MAX_PAGE_CHARS] + ("...[truncated]" if len(text) > MAX_PAGE_CHARS else "")


def make_agent_tools():
    """Tools available to the research agent's ReAct loop."""
    return [web_search, fetch_company_website]