from typing import List, Dict, Optional, Iterable, Set

# Prefer the new package name `ddgs`, fall back to legacy `duckduckgo_search`
try:
    from ddgs import DDGS  # type: ignore
except Exception:
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Install the search client with: pip install ddgs (preferred) or pip install duckduckgo-search"
        ) from exc


def ddgs_search(
    query: str,
    max_results: int = 10,
    safesearch: str = "moderate",
    region: str = "it-it",
    timelimit: Optional[str] = None,
    timeout: int = 10,
    exclude_domains: Optional[Iterable[str]] = None,
    exclude_tlds: Optional[Iterable[str]] = None,
    min_latin_ratio: float = 0.5,
) -> List[Dict[str, str]]:
    """Perform a DuckDuckGo web search using DDGS with SSL verify disabled.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        safesearch: One of "on", "moderate", "off".
        region: Region code, e.g. "wt-wt" (worldwide).
        timelimit: Optional time range filter, e.g. "d", "w", "m", "y".
        timeout: Per-request timeout in seconds.

    Returns:
        A list of result dicts with keys: "title", "href", "body".
    """
    if not isinstance(query, str) or not query.strip():
        return []

    # verify=False disables SSL verification as requested
    with DDGS(verify=False, timeout=timeout) as ddgs:
        # In the current API, the first argument is called `keywords`
        results_iter = ddgs.text(
            keywords=query.strip(),
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            max_results=max_results,
        )
        # Convert generator to list to return concrete results
        raw_results: List[Dict[str, str]] = list(results_iter)

        # Post-filters to reduce Chinese-language/China-domain results
        from urllib.parse import urlparse

        excluded_domains: Set[str] = set(
            d.lower() for d in (exclude_domains or [
                "baidu.com",
                "zhidao.baidu.com",
                "tieba.baidu.com",
                "weibo.com",
                "qq.com",
            ])
        )
        excluded_tlds: Set[str] = set(t.lower() for t in (exclude_tlds or [".cn"]))

        def is_latin_heavy(text: str, threshold: float) -> bool:
            if not text:
                return True
            latin = sum("a" <= ch.lower() <= "z" for ch in text)
            total = sum(ch.isalpha() for ch in text)
            if total == 0:
                return True
            return (latin / total) >= threshold

        filtered: List[Dict[str, str]] = []
        for item in raw_results:
            url = item.get("href", "")
            netloc = urlparse(url).netloc.lower()
            # Domain/TLD filters
            if any(domain in netloc for domain in excluded_domains):
                continue
            if any(netloc.endswith(tld) for tld in excluded_tlds):
                continue
            # Language heuristic on title+body
            text_blob = f"{item.get('title', '')} {item.get('body', '')}"
            if not is_latin_heavy(text_blob, min_latin_ratio):
                continue
            filtered.append(item)

        return filtered


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "duckduckgo search"
    for idx, item in enumerate(ddgs_search(q, max_results=10), start=1):
        print(f"[{idx}] {item.get('title')}")
        print(item.get("href"))
        print(item.get("body"))
        print("-" * 80)

# write a function that can research on interent using ddgs with verify=false
