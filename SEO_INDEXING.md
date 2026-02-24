# Google indexing checklist

This document covers everything repository-side needed for Google indexing.

## Included in this repository

- `index.html` with:
  - title and description metadata
  - robots directives
  - canonical URL
  - Open Graph and Twitter metadata
  - JSON-LD structured data (`SoftwareApplication`)
- `robots.txt` with a sitemap reference
- `sitemap.xml` with indexable pages
- `scripts/generate_seo_assets.py` to regenerate SEO assets

## 1) Set your final site URL

By default the generator uses:

`https://nwxio.github.io/OmniMemory`

If you use another domain, regenerate SEO files with:

```bash
SEO_SITE_URL="https://your-domain.example" python3 scripts/generate_seo_assets.py
```

## 2) Deploy and verify files are public

After deployment, these URLs must return `200 OK`:

- `/`
- `/robots.txt`
- `/sitemap.xml`

## 3) Submit to Google Search Console

1. Open Google Search Console.
2. Add property for your final domain.
3. Verify domain ownership.
4. Submit sitemap URL: `https://your-domain.example/sitemap.xml`.

## 4) Request indexing for key pages

Use URL Inspection in Search Console and request indexing for:

- home page (`/`)
- `README.md`
- `INSTALL.md`
- `ENV_CONFIGS.md`

## 5) Keep content index-friendly

- Avoid duplicate content with conflicting canonicals.
- Keep internal links between major docs.
- Update sitemap when adding/removing indexable docs.
- Keep pages crawlable (no accidental `noindex` headers/tags).

## 6) Optional but recommended

- Add a custom 404 page.
- Add performance monitoring (Core Web Vitals).
- Add external backlinks to improve discovery speed.
