"""Quick test: can Playwright fetch the GIGABYTE spec page?"""

from playwright.sync_api import sync_playwright

URL = "https://www.gigabyte.com/Laptop/AORUS-MASTER-16-AM6H/sp"

with sync_playwright() as p:
    # headless=False to avoid Akamai WAF detection
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1920, "height": 1080},
    )
    page = context.new_page()
    print(f"Fetching {URL} ...")
    page.goto(URL, wait_until="networkidle", timeout=30000)
    title = page.title()
    html = page.content()
    browser.close()

print(f"Title: {title}")
print(f"HTML length: {len(html)} chars")
print(f"First 500 chars:\n{html[:500]}")
