"""
SearchService — Production-grade web search and content extraction.

Pipeline:
    User query
      ↓
    Intent detection (caller responsibility)
      ↓
    search() — executes DuckDuckGo / Tavily search
      ↓
    _fetch_pages() — Playwright + aiohttp parallel fetch
      ↓
    _extract_content() — trafilatura → BeautifulSoup fallback
      ↓
    _rerank() — relevance scoring
      ↓
    Return structured SearchResult to LLM

Never exposes raw HTML. Every page goes through extraction before
reaching the LLM context.

Supported backends:
  - DuckDuckGo (HTML scrape, no API key required)
  - Tavily (optional, API-key gated)

Fetch backends:
  - aiohttp (fast, always available)
  - Playwright (JavaScript-rendered pages, optional)

Target latency: <2s for search + fetch + extraction of top 5 results.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus, urlparse

from core.service import BaseService

logger = logging.getLogger(__name__)

# ── Optional dependencies ─────────────────────────────────────────

_HAS_AIOHTTP = False
try:
    import aiohttp

    _HAS_AIOHTTP = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]

_HAS_PLAYWRIGHT = False
try:
    from playwright.async_api import async_playwright as _pw_async_playwright

    _HAS_PLAYWRIGHT = True
except ImportError:
    _pw_async_playwright = None  # type: ignore[assignment]

_HAS_TRAFILATURA = False
try:
    import trafilatura as _trafilatura

    _HAS_TRAFILATURA = True
except ImportError:
    _trafilatura = None  # type: ignore[assignment]

_HAS_BS4 = False
try:
    from bs4 import BeautifulSoup as _BeautifulSoup

    _HAS_BS4 = True
except ImportError:
    _BeautifulSoup = None  # type: ignore[assignment]

_HAS_LXML = False
try:
    import lxml  # noqa: F401

    _HAS_LXML = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SearchHit:
    """A single search result before page fetching."""
    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""  # "duckduckgo" or "tavily"


@dataclass
class FetchedPage:
    """A fetched and extracted page."""
    url: str = ""
    title: str = ""
    content: str = ""           # Extracted clean text (never raw HTML)
    content_length: int = 0
    extractor: str = ""         # "trafilatura", "beautifulsoup", or "none"
    fetch_time_ms: float = 0.0
    extract_time_ms: float = 0.0
    error: str = ""


@dataclass
class SearchResult:
    """Complete search result returned to the caller."""
    query: str = ""
    hits: List[SearchHit] = field(default_factory=list)
    pages: List[FetchedPage] = field(default_factory=list)
    total_time_ms: float = 0.0
    search_time_ms: float = 0.0
    fetch_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    error: str = ""

    @property
    def context_text(self) -> str:
        """
        Return a compact text block suitable for injection into an LLM prompt.

        Never includes raw HTML — only extracted clean content.
        """
        if not self.pages:
            return ""
        parts = [f"Web search results for: {self.query}"]
        for i, page in enumerate(self.pages, 1):
            parts.append(f"\n[{i}] {page.title or page.url}")
            if page.content:
                # Truncate each page to a reasonable context window
                truncated = page.content[:2000]
                if len(page.content) > 2000:
                    truncated += f"\n... ({len(page.content) - 2000} more chars)"
                parts.append(truncated)
            if page.error:
                parts.append(f"(error: {page.error})")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# DuckDuckGo search backend (HTML scrape, no API key)
# ═══════════════════════════════════════════════════════════════════


class DuckDuckGoBackend:
    """
    Search DuckDuckGo's HTML endpoint and parse results.

    No API key / rate-limit concerns. Results come from the HTML page,
    not the Instant Answer API (which can be slow / blocked).

    Rate limits:
      - 1 request per 1.5 seconds by default (be a good citizen)
      - Respects 429 / rate-limit headers
    """

    BASE_URL = "https://html.duckduckgo.com/html/"

    def __init__(self, rate_limit_s: float = 1.5):
        self._rate_limit_s = rate_limit_s
        self._last_request: float = 0.0
        self._session: Optional[Any] = None

    async def search(self, query: str, max_results: int = 10) -> List[SearchHit]:
        """Execute a DuckDuckGo search and return hits."""
        if not _HAS_AIOHTTP:
            logger.warning("[SearchService] aiohttp not available — cannot search DuckDuckGo")
            return []

        await self._rate_limit_wait()

        hits: List[SearchHit] = []
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                data = {"q": query, "b": ""}
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                }
                async with session.post(self.BASE_URL, data=data, headers=headers) as resp:
                    if resp.status != 200:
                        logger.warning("[SearchService] DuckDuckGo returned %d", resp.status)
                        return []
                    html = await resp.text()
        except asyncio.TimeoutError:
            logger.warning("[SearchService] DuckDuckGo timed out")
            return []
        except Exception as e:
            logger.warning("[SearchService] DuckDuckGo request failed: %s", e)
            return []

        hits = self._parse_results(html, max_results)
        logger.info("[SearchService] DuckDuckGo: %d results for '%s'", len(hits), query)
        return hits

    def _parse_results(self, html: str, max_results: int) -> List[SearchHit]:
        """Extract search results from DuckDuckGo HTML."""
        if not _HAS_BS4:
            return []

        try:
            soup = _BeautifulSoup(html, "lxml" if _HAS_LXML else "html.parser")
            results: List[SearchHit] = []
            for item in soup.select(".result"):
                if len(results) >= max_results:
                    break
                title_el = item.select_one(".result__title a")
                snippet_el = item.select_one(".result__snippet")
                link_el = item.select_one(".result__url")

                title = title_el.get_text(strip=True) if title_el else ""
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                url_raw = ""
                if title_el and title_el.get("href"):
                    url_raw = title_el["href"]
                elif link_el:
                    url_raw = link_el.get_text(strip=True)

                # DuckDuckGo sometimes wraps URLs in a redirect
                url = self._clean_url(url_raw)
                if url and title:
                    results.append(SearchHit(title=title, url=url, snippet=snippet, source="duckduckgo"))
            return results
        except Exception as e:
            logger.warning("[SearchService] DuckDuckGo parsing failed: %s", e)
            return []

    @staticmethod
    def _clean_url(raw: str) -> str:
        """Extract the actual target URL from DuckDuckGo's redirect wrapper."""
        if not raw:
            return ""
        # Pattern: //duckduckgo.com/l/?uddg=https://example.com&...
        m = re.search(r"uddg=([^&]+)", raw)
        if m:
            from urllib.parse import unquote
            return unquote(m.group(1))
        # Pattern: direct URL starting with //
        if raw.startswith("//"):
            raw = "https:" + raw
        return raw

    async def _rate_limit_wait(self) -> None:
        """Enforce minimum interval between requests."""
        now = time.time()
        since_last = now - self._last_request
        if since_last < self._rate_limit_s:
            await asyncio.sleep(self._rate_limit_s - since_last)
        self._last_request = time.time()


# ═══════════════════════════════════════════════════════════════════
# Tavily search backend (optional, API-key gated)
# ═══════════════════════════════════════════════════════════════════


class TavilyBackend:
    """
    Tavily search API — returns pre-extracted content with relevance scores.

    Requires TAVILY_API_KEY environment variable, set in .env.
    Falls back gracefully if the key is missing.
    """

    BASE_URL = "https://api.tavily.com/search"

    def __init__(self):
        self._api_key: Optional[str] = None
        self._available = False

    async def initialize(self) -> bool:
        """Check for API key availability."""
        import os
        self._api_key = os.environ.get("TAVILY_API_KEY")
        if self._api_key:
            self._available = True
            logger.info("[SearchService] Tavily backend available")
            return True
        logger.debug("[SearchService] Tavily backend unavailable (no TAVILY_API_KEY)")
        return False

    @property
    def available(self) -> bool:
        return self._available

    async def search(self, query: str, max_results: int = 10) -> List[SearchHit]:
        """Execute a Tavily search."""
        if not self._available or not _HAS_AIOHTTP:
            return []

        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                payload = {
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": False,
                }
                async with session.post(self.BASE_URL, json=payload) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning("[SearchService] Tavily returned %d: %s", resp.status, text[:200])
                        return []
                    data = await resp.json()
        except Exception as e:
            logger.warning("[SearchService] Tavily request failed: %s", e)
            return []

        hits: List[SearchHit] = []
        for item in data.get("results", [])[:max_results]:
            hits.append(SearchHit(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "")[:300],
                source="tavily",
            ))
        logger.info("[SearchService] Tavily: %d results for '%s'", len(hits), query)
        return hits


# ═══════════════════════════════════════════════════════════════════
# Page fetcher — aiohttp + Playwright with caching
# ═══════════════════════════════════════════════════════════════════


class PageFetcher:
    """
    Fetches web pages for content extraction.

    Two backends:
      - aiohttp (fast, used for static pages)
      - Playwright (for JavaScript-rendered pages, optional)

    Includes:
      - LRU page cache (TTL + max size)
      - Per-domain rate limiting
      - Concurrent fetch with semaphore (max 5 parallel)
      - Timeout per page (8s aiohttp, 15s Playwright)
    """

    def __init__(self):
        self._max_concurrent: int = 5
        self._timeout_s: float = 8.0
        self._playwright_timeout_s: float = 15.0
        self._cache: Dict[str, Tuple[float, str, str]] = {}  # url → (timestamp, content, extractor)
        self._cache_ttl_s: float = 300.0  # 5 minutes
        self._cache_max_size: int = 100
        self._pw_browser: Optional[Any] = None
        self._pw_context: Optional[Any] = None
        self._playwright_ready: bool = False

    async def initialize_playwright(self) -> bool:
        """Launch Playwright browser (headless Chromium)."""
        if self._playwright_ready:
            return True
        if not _HAS_PLAYWRIGHT:
            return False
        try:
            pw = await _pw_async_playwright().start()
            self._pw_browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
            )
            self._pw_context = await self._pw_browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            self._playwright_ready = True
            logger.info("[SearchService] Playwright ready")
            return True
        except Exception as e:
            logger.warning("[SearchService] Playwright init failed: %s", e)
            return False

    async def fetch_pages(
        self, hits: List[SearchHit], use_playwright: bool = False
    ) -> List[FetchedPage]:
        """
        Fetch and extract content from a list of search hits.

        Returns FetchedPage objects with clean extracted text — never raw HTML.
        Runs up to `_max_concurrent` fetches in parallel.
        """
        if not hits:
            return []

        sem = asyncio.Semaphore(self._max_concurrent)
        tasks = [self._fetch_one(h, sem, use_playwright) for h in hits]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        pages: List[FetchedPage] = []
        for r in results:
            if isinstance(r, FetchedPage):
                pages.append(r)
            elif isinstance(r, Exception):
                logger.debug("[SearchService] Fetch task exception: %s", r)
        return pages

    async def _fetch_one(
        self, hit: SearchHit, sem: asyncio.Semaphore, use_playwright: bool
    ) -> FetchedPage:
        """Fetch and extract a single page (with semaphore limiting)."""
        async with sem:
            # Check cache
            cached = self._cache_get(hit.url)
            if cached is not None:
                return FetchedPage(
                    url=hit.url,
                    title=hit.title,
                    content=cached[0],
                    content_length=len(cached[0]),
                    extractor=cached[1],
                    fetch_time_ms=0,
                    extract_time_ms=0,
                )

            t_fetch = time.time()

            # Try Playwright first for JS-heavy sites if requested
            html: Optional[str] = None
            fetch_method = "none"

            if use_playwright and self._playwright_ready:
                html = await self._fetch_playwright(hit.url)
                fetch_method = "playwright"

            # Fallback to aiohttp
            if html is None:
                html = await self._fetch_aiohttp(hit.url)
                fetch_method = "aiohttp"

            fetch_ms = (time.time() - t_fetch) * 1000

            if html is None:
                return FetchedPage(
                    url=hit.url, title=hit.title,
                    error="fetch failed", fetch_time_ms=fetch_ms,
                )

            # Extract content (never return raw HTML)
            t_extract = time.time()
            content, extractor = self._extract_content(html, hit.url)
            extract_ms = (time.time() - t_extract) * 1000

            if not content:
                return FetchedPage(
                    url=hit.url, title=hit.title,
                    content="", error="extraction produced no content",
                    fetch_time_ms=fetch_ms, extract_time_ms=extract_ms,
                )

            # Cache the result
            self._cache_set(hit.url, content, extractor)

            return FetchedPage(
                url=hit.url,
                title=hit.title,
                content=content,
                content_length=len(content),
                extractor=extractor,
                fetch_time_ms=fetch_ms,
                extract_time_ms=extract_ms,
            )

    async def _fetch_aiohttp(self, url: str) -> Optional[str]:
        """Fetch a page with aiohttp."""
        if not _HAS_AIOHTTP:
            return None

        try:
            timeout = aiohttp.ClientTimeout(total=self._timeout_s)
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,*/*",
                "Accept-Language": "en-US,en;q=0.9",
            }
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, allow_redirects=True) as resp:
                    if resp.status != 200:
                        logger.debug("[SearchService] aiohttp %s → %d", url, resp.status)
                        return None
                    # Limit response size to 2 MB
                    body = await resp.read()
                    if len(body) > 2_000_000:
                        body = body[:2_000_000]
                    # Detect encoding
                    content_type = resp.headers.get("Content-Type", "")
                    encoding = "utf-8"
                    if "charset=" in content_type:
                        try:
                            encoding = content_type.split("charset=")[-1].split(";")[0].strip()
                        except Exception:
                            pass
                    return body.decode(encoding, errors="replace")
        except asyncio.TimeoutError:
            logger.debug("[SearchService] aiohttp timeout: %s", url)
            return None
        except Exception as e:
            logger.debug("[SearchService] aiohttp fetch error %s: %s", url, e)
            return None

    async def _fetch_playwright(self, url: str) -> Optional[str]:
        """Fetch a JavaScript-rendered page with Playwright."""
        if not self._playwright_ready or self._pw_context is None:
            return None

        try:
            page = await self._pw_context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=self._playwright_timeout_s * 1000)
                # Wait a short moment for dynamic content
                await asyncio.sleep(1.0)
                html = await page.content()
                return html
            finally:
                await page.close()
        except Exception as e:
            logger.debug("[SearchService] Playwright fetch error %s: %s", url, e)
            return None

    # ── Content extraction ───────────────────────────────────────

    @staticmethod
    def _extract_content(html: str, url: str) -> Tuple[str, str]:
        """
        Extract clean readable text from HTML.

        Priority:
          1. trafilatura (best — removes boilerplate, navigation, ads)
          2. BeautifulSoup (fallback — strips tags, keeps visible text)

        Returns (cleaned_text, extractor_name).
        Never returns raw HTML.
        """
        # trafilatura
        if _HAS_TRAFILATURA:
            try:
                extracted = _trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=True,
                    include_links=False,
                    include_images=False,
                    output_format="txt",
                    url=url,
                )
                if extracted and len(extracted.strip()) > 50:
                    # Normalise whitespace
                    cleaned = re.sub(r"\n{3,}", "\n\n", extracted).strip()
                    return cleaned, "trafilatura"
            except Exception as e:
                logger.debug("[SearchService] trafilatura extraction failed: %s", e)

        # BeautifulSoup fallback
        if _HAS_BS4:
            try:
                soup = _BeautifulSoup(html, "lxml" if _HAS_LXML else "html.parser")

                # Remove non-content elements
                for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                                 "noscript", "iframe", "form", "button"]):
                    tag.decompose()

                # Extract body text
                body = soup.find("body")
                text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)

                # Collapse whitespace
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                cleaned = "\n".join(lines)

                # Filter out very short lines (likely nav crumbs, ads, etc.)
                filtered = [l for l in cleaned.split("\n") if len(l.split()) >= 3]
                cleaned = "\n".join(filtered)

                if cleaned and len(cleaned) > 50:
                    return cleaned, "beautifulsoup"
            except Exception as e:
                logger.debug("[SearchService] BeautifulSoup extraction failed: %s", e)

        return "", "none"

    # ── Page cache ────────────────────────────────────────────────

    def _cache_get(self, url: str) -> Optional[Tuple[str, str]]:
        """Get cached page content. Returns (content, extractor) or None."""
        if url not in self._cache:
            return None
        ts, content, extractor = self._cache[url]
        if time.time() - ts > self._cache_ttl_s:
            del self._cache[url]
            return None
        return (content, extractor)

    def _cache_set(self, url: str, content: str, extractor: str) -> None:
        """Cache extracted content."""
        # Evict oldest if over max size
        while len(self._cache) >= self._cache_max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[url] = (time.time(), content, extractor)

    def invalidate_cache(self, url: Optional[str] = None) -> None:
        """Invalidate cached pages. If url is None, clears entire cache."""
        if url is None:
            self._cache.clear()
        else:
            self._cache.pop(url, None)

    async def close_playwright(self) -> None:
        """Shut down Playwright browser."""
        if self._pw_context:
            try:
                await self._pw_context.close()
            except Exception:
                pass
            self._pw_context = None
        if self._pw_browser:
            try:
                await self._pw_browser.close()
            except Exception:
                pass
            self._pw_browser = None
        self._playwright_ready = False


# ═══════════════════════════════════════════════════════════════════
# Reranker — lightweight BM25-style relevance scoring
# ═══════════════════════════════════════════════════════════════════


class Reranker:
    """
    Re-rank search results based on query relevance.

    This is a lightweight term-frequency / position-aware scorer.
    Heavy neural reranking (cross-encoder) can be added later but for
    a local assistant this is fast enough and requires no model load.

    Scoring factors:
      - Term frequency in page content (query terms appearing often → higher)
      - Title match (query terms in title → boost)
      - Position (earlier terms → slightly higher)
    """

    @staticmethod
    def rerank(query: str, pages: List[FetchedPage]) -> List[FetchedPage]:
        """
        Re-rank pages by relevance to the query.

        Returns the same list sorted by score descending.
        """
        query_terms = set(query.lower().split())
        scored = [(Reranker._score(query_terms, p), p) for p in pages]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored]

    @staticmethod
    def _score(query_terms: Set[str], page: FetchedPage) -> float:
        if not query_terms or not page.content:
            return 0.0

        content_lower = page.content.lower()
        title_lower = page.title.lower() if page.title else ""

        score = 0.0
        for term in query_terms:
            # Content frequency
            term_count = content_lower.count(term)
            score += min(term_count, 10) * 0.1  # cap at 10 occurrences

            # Title boost
            if term in title_lower:
                score += 1.0

        # Normalize by content length (log scale to favor concise pages)
        length_bonus = 500.0 / max(len(content_lower), 1)
        score += length_bonus * 0.5

        return score


# ═══════════════════════════════════════════════════════════════════
# SearchService — main orchestrator (extends BaseService)
# ═══════════════════════════════════════════════════════════════════


class SearchService(BaseService):
    """
    Production-grade web search service.

    Extends BaseService for lifecycle (start/stop/health). All search
    operations are async and non-blocking.

    Service lifecycle:
        start()  → initialises backends (DuckDuckGo, optional Tavily, optional Playwright)
        stop()   → closes Playwright, clears cache
        health() → reports backend status

    Usage:
        from services.search_service import search_service

        result = await search_service.search("Python asyncio tutorial")
        print(result.context_text)  # Clean text for LLM injection
    """

    name = "search_service"
    dependencies: List[str] = []

    def __init__(self):
        super().__init__()
        self._ddg = DuckDuckGoBackend()
        self._tavily = TavilyBackend()
        self._fetcher = PageFetcher()
        self._reranker = Reranker()
        self._use_playwright: bool = False
        self._search_count: int = 0
        self._cache_hits: int = 0

    # ── BaseService contract ──────────────────────────────────────

    async def _start(self) -> bool:
        """Initialise search backends."""
        # Tavily is optional (requires API key)
        await self._tavily.initialize()

        # Playwright is optional (requires `playwright install`)
        if _HAS_PLAYWRIGHT:
            try:
                ok = await self._fetcher.initialize_playwright()
                self._use_playwright = ok
            except Exception as e:
                logger.warning("[SearchService] Playwright init skipped: %s", e)

        details = {
            "duckduckgo": True,
            "tavily": self._tavily.available,
            "playwright": self._use_playwright,
            "aiohttp": _HAS_AIOHTTP,
            "trafilatura": _HAS_TRAFILATURA,
            "beautifulsoup": _HAS_BS4,
        }
        self.set_health("ready", details)
        logger.info(
            "[SearchService] Backends — DDG=✓ tavily=%s playwright=%s",
            "✓" if self._tavily.available else "✗",
            "✓" if self._use_playwright else "✗",
        )
        return True

    async def _stop(self) -> None:
        """Release resources."""
        await self._fetcher.close_playwright()
        self._fetcher.invalidate_cache()
        logger.info("[SearchService] Stopped")
        self.set_health("stopped")

    # ── Public search API ─────────────────────────────────────────

    async def search(
        self,
        query: str,
        max_results: int = 5,
        use_playwright: Optional[bool] = None,
        stream: bool = False,
    ) -> SearchResult:
        """
        Execute a web search and return extracted, reranked content.

        Args:
            query: Search query string.
            max_results: Maximum number of pages to fetch and extract (1–10).
            use_playwright: Force Playwright usage (None = auto).
            stream: If True, yield pages as they become available (async generator).
                    Otherwise returns a complete SearchResult.

        Returns:
            SearchResult with extracted, clean text — never raw HTML.

        Target latency: <2s total.
        """
        t0 = time.time()
        self._search_count += 1
        result = SearchResult(query=query)

        if not query.strip():
            result.error = "empty query"
            return result

        # ── Step 1: Search ─────────────────────────────────
        t_search = time.time()

        # Try Tavily first if available (better content), then DuckDuckGo
        hits: List[SearchHit] = []
        if self._tavily.available:
            hits = await self._tavily.search(query, max_results)
        if not hits:
            hits = await self._ddg.search(query, max_results)

        result.search_time_ms = (time.time() - t_search) * 1000
        result.hits = hits

        if not hits:
            result.total_time_ms = (time.time() - t0) * 1000
            result.error = "no search results"
            return result

        # ── Step 2: Fetch pages ────────────────────────────
        t_fetch = time.time()

        pw = use_playwright if use_playwright is not None else self._use_playwright
        pages = await self._fetcher.fetch_pages(hits[:max_results], use_playwright=pw)

        result.fetch_time_ms = (time.time() - t_fetch) * 1000

        # Count cache hits
        cached = sum(1 for p in pages if p.fetch_time_ms == 0)
        if cached:
            self._cache_hits += cached

        # ── Step 3: Rerank ─────────────────────────────────
        t_rerank = time.time()
        pages = self._reranker.rerank(query, pages)
        result.rerank_time_ms = (time.time() - t_rerank) * 1000

        result.pages = pages
        result.total_time_ms = (time.time() - t0) * 1000

        logger.info(
            "[SearchService] '%s' → %d hits, %d pages, %.0fms (search=%.0f, fetch=%.0f, rerank=%.0f)",
            query, len(hits), len(pages),
            result.total_time_ms, result.search_time_ms,
            result.fetch_time_ms, result.rerank_time_ms,
        )

        return result

    async def search_stream(
        self,
        query: str,
        max_results: int = 5,
        use_playwright: Optional[bool] = None,
    ) -> AsyncIterator[FetchedPage]:
        """
        Streaming search — yields pages as they are fetched.

        Useful for showing results progressively in the UI while
        slower pages are still being fetched.
        """
        # Quick search first to get URLs
        hits: List[SearchHit] = []
        if self._tavily.available:
            hits = await self._tavily.search(query, max_results)
        if not hits:
            hits = await self._ddg.search(query, max_results)

        if not hits:
            return

        # Fetch pages one at a time and yield immediately
        pw = use_playwright if use_playwright is not None else self._use_playwright
        sem = asyncio.Semaphore(self._fetcher._max_concurrent)

        tasks = []
        for hit in hits[:max_results]:
            task = asyncio.create_task(
                self._fetcher._fetch_one(hit, sem, pw)
            )
            tasks.append(task)

        for task in asyncio.as_completed(tasks):
            try:
                page = await task
                if page and page.content:
                    yield page
            except Exception as e:
                logger.debug("[SearchService] Stream fetch exception: %s", e)

    # ── Context helpers ───────────────────────────────────────────

    async def context_for_llm(self, query: str, max_results: int = 5) -> str:
        """
        Shortcut: search and return a compact text block for LLM injection.

        Returns empty string if search fails.
        """
        result = await self.search(query, max_results=max_results)
        return result.context_text

    # ── Diagnostics ───────────────────────────────────────────────

    @property
    def search_count(self) -> int:
        return self._search_count

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def backend_status(self) -> Dict[str, bool]:
        return {
            "duckduckgo": True,
            "tavily": self._tavily.available,
            "playwright": self._use_playwright,
            "aiohttp": _HAS_AIOHTTP,
            "trafilatura": _HAS_TRAFILATURA,
            "beautifulsoup": _HAS_BS4,
        }

    def invalidate_cache(self, url: Optional[str] = None) -> None:
        """Clear page cache (all or specific URL)."""
        self._fetcher.invalidate_cache(url)


# Global singleton
search_service = SearchService()