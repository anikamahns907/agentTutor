import re
import xml.etree.ElementTree as ET
from html import unescape
from typing import Dict, List, Tuple
from urllib.parse import quote_plus

import requests

from utils.text_extraction import clean_extracted_text


REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
}

PUBLIC_HEALTH_FEEDS: Tuple[Tuple[str, str], ...] = (
    ("CDC MMWR", "https://www.cdc.gov/mmwr/rss/mmwr.xml"),
    ("WHO Disease Outbreak News", "https://www.who.int/feeds/entity/csr/don/en/rss.xml"),
    ("Nature Public Health", "https://www.nature.com/subjects/public-health.rss"),
)


def get_bruknow_search_url(search_query: str) -> str:
    """Generate a BruKnow search URL for a keyword query."""
    encoded_query = quote_plus(search_query)
    return (
        "https://bruknow.library.brown.edu/discovery/search"
        f"?query=any,contains,{encoded_query}"
        "&tab=Everything&search_scope=MyInstitution"
        "&vid=01BU_INST:BROWN&offset=0"
    )


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks suitable for embedding."""
    cleaned = clean_extracted_text(text)
    if not cleaned:
        return []

    words = cleaned.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def prepare_article_documents(
    text: str,
    article_title: str,
    source_label: str,
    source_url: str | None = None,
) -> Tuple[List[str], List[Dict]]:
    """Return chunked text + metadata tuples ready for embedding."""
    chunks = chunk_text(text)
    metadata = []
    for idx, chunk in enumerate(chunks):
        metadata.append(
            {
                "source": source_label or "Uploaded Article",
                "source_type": "session_article",
                "article_title": article_title or "Uploaded Article",
                "chunk_index": idx,
                "url": source_url,
            }
        )
    return chunks, metadata


def search_bruknow_articles(query: str, max_results: int = 5) -> List[Dict]:
    """Use CrossRef metadata as a proxy to suggest BruKnow search targets."""
    normalized = query.strip()
    if not normalized:
        return []

    results: List[Dict] = []
    try:
        resp = requests.get(
            "https://api.crossref.org/works",
            params={"query": normalized, "rows": max_results, "sort": "relevance"},
            headers=REQUEST_HEADERS,
            timeout=12,
        )
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
        for item in items:
            title_list = item.get("title") or []
            title = title_list[0] if title_list else "Untitled article"
            journal = ""
            container = item.get("container-title") or []
            if container:
                journal = container[0]
            doi_url = item.get("URL")
            bruknow_link = get_bruknow_search_url(title)
            snippet = item.get("abstract") or item.get("subtitle") or ""
            snippet = clean_extracted_text(re.sub("<.*?>", " ", snippet or ""))

            results.append(
                {
                    "title": title,
                    "url": doi_url or bruknow_link,
                    "bruknow_url": bruknow_link,
                    "source": journal or "CrossRef",
                    "snippet": snippet or "Open this link in BruKnow to access the full text.",
                }
            )

            if len(results) >= max_results:
                break
    except requests.RequestException:
        results = []

    if not results:
        results.append(
            {
                "title": f"BruKnow search results for '{normalized}'",
                "url": get_bruknow_search_url(normalized),
                "bruknow_url": get_bruknow_search_url(normalized),
                "source": "BruKnow",
                "snippet": "Open this link in BruKnow to explore library access options.",
            }
        )

    return results[:max_results]


def search_public_health_articles(query: str, max_results: int = 5) -> List[Dict]:
    """Search public health RSS feeds for matching articles."""
    normalized = query.strip().lower()
    if not normalized:
        return []

    results: List[Dict] = []
    for feed_name, feed_url in PUBLIC_HEALTH_FEEDS:
        try:
            feed_resp = requests.get(feed_url, headers=REQUEST_HEADERS, timeout=12)
            feed_resp.raise_for_status()
            items = _parse_rss_items(feed_resp.content)
        except requests.RequestException:
            continue
        except ET.ParseError:
            continue

        for item in items:
            search_blob = f"{item['title']} {item['summary']}".lower()
            if normalized in search_blob:
                results.append(
                    {
                        "title": item["title"],
                        "url": item["link"],
                        "source": feed_name,
                        "snippet": item["summary"],
                    }
                )
                if len(results) >= max_results:
                    return results

    if not results:
        for feed_name, _ in PUBLIC_HEALTH_FEEDS:
            results.append(
                {
                    "title": f"{feed_name} search for '{query}'",
                    "url": f"https://www.google.com/search?q={quote_plus(feed_name + ' ' + query)}",
                    "source": feed_name,
                    "snippet": "Open this search to explore recent public health reporting.",
                }
            )
            if len(results) >= max_results:
                break

    return results[:max_results]


def _parse_rss_items(xml_bytes: bytes) -> List[Dict]:
    """Parse RSS/Atom feeds to a basic list of dicts."""
    entries: List[Dict] = []
    root = ET.fromstring(xml_bytes)

    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = (item.findtext("description") or "").strip()
        entries.append(
            {
                "title": unescape(title),
                "link": link,
                "summary": _strip_tags(description),
            }
        )

    if entries:
        return entries

    atom_ns = "{http://www.w3.org/2005/Atom}"
    for entry in root.findall(f".//{atom_ns}entry"):
        title = (entry.findtext(f"{atom_ns}title") or "").strip()
        link_el = entry.find(f"{atom_ns}link")
        link_href = link_el.get("href") if link_el is not None else ""
        summary = (
            entry.findtext(f"{atom_ns}summary") or entry.findtext(f"{atom_ns}content") or ""
        ).strip()
        entries.append(
            {
                "title": unescape(title),
                "link": link_href,
                "summary": _strip_tags(summary),
            }
        )

    return entries


def _strip_tags(text: str) -> str:
    """Remove HTML tags from an RSS snippet."""
    no_tags = re.sub(r"<[^>]+>", " ", text or "")
    return clean_extracted_text(unescape(no_tags))


__all__ = [
    "chunk_text",
    "get_bruknow_search_url",
    "prepare_article_documents",
    "search_bruknow_articles",
    "search_public_health_articles",
]

