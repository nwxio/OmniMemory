from __future__ import annotations

from datetime import date
from pathlib import Path
import os


DEFAULT_SITE_URL = "https://nwxio.github.io/OmniMemory"

INDEXABLE_PATHS = [
    "/",
    "/index.html",
    "/README.md",
    "/INSTALL.md",
    "/ENV_CONFIGS.md",
    "/CONTRIBUTING.md",
    "/SECURITY.md",
    "/RELEASE_CHECKLIST.md",
    "/CODE_OF_CONDUCT.md",
    "/docker/README.md",
    "/docs/architecture.md",
]


def build_robots(site_url: str) -> str:
    return f"User-agent: *\nAllow: /\n\nSitemap: {site_url}/sitemap.xml\n"


def build_sitemap(site_url: str, lastmod: str) -> str:
    urls = []
    for path in INDEXABLE_PATHS:
        urls.append(
            "  <url>\n"
            f"    <loc>{site_url}{path}</loc>\n"
            f"    <lastmod>{lastmod}</lastmod>\n"
            "    <changefreq>weekly</changefreq>\n"
            "    <priority>0.8</priority>\n"
            "  </url>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(urls)
        + "\n</urlset>\n"
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    site_url = os.getenv("SEO_SITE_URL", DEFAULT_SITE_URL).rstrip("/")
    lastmod = date.today().isoformat()

    robots_txt = build_robots(site_url)
    sitemap_xml = build_sitemap(site_url, lastmod)

    (repo_root / "robots.txt").write_text(robots_txt, encoding="utf-8")
    (repo_root / "sitemap.xml").write_text(sitemap_xml, encoding="utf-8")

    print("Generated:")
    print("- robots.txt")
    print("- sitemap.xml")
    print(f"Site URL: {site_url}")


if __name__ == "__main__":
    main()
