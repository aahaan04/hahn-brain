"""Scrape Hahn Agency pages with crawl4ai and save to hahn_content.json."""

import asyncio
import json

from crawl4ai import AsyncWebCrawler

BASE = "https://hahn.agency"

PATHS = [
    "/",
    "/services/",
    "/services/creative-brand/",
    "/services/marketing-and-media/",
    "/services/strategy-and-communications/",
    "/services/digital-experience/",
    "/services/ai-and-analytics/",
    "/industry-expertise/energy-and-utilities/",
    "/industry-expertise/health-and-nutrition/",
    "/industry-expertise/food-and-beverage/",
    "/our-work/",
    "/about-us/",
]

URLS = [BASE + path for path in PATHS]


async def main():
    results = []
    async with AsyncWebCrawler() as crawler:
        for url in URLS:
            try:
                result = await crawler.arun(url=url)
                if not result.success:
                    print(f"SKIP (crawl failed): {url} -> {result.error_message}")
                    continue
                content = (result.markdown or "").strip()
                if not content:
                    print(f"SKIP (empty content): {url}")
                    continue
                results.append({"url": url, "content": content})
                print(f"OK: {url} ({len(content)} chars)")
            except Exception as exc:  # skip and continue on any failure
                print(f"SKIP (error): {url} -> {exc}")
                continue

    with open("hahn_content.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} pages to hahn_content.json")


if __name__ == "__main__":
    asyncio.run(main())
