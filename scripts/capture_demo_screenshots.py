"""Capture Streamlit demo screenshots after the app is running.

Usage:
  1. Start: streamlit run main.py
  2. Run:  python scripts/capture_demo_screenshots.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCREENSHOTS_DIR = ROOT / "reports" / "screenshots"
BASE_URL = "http://localhost:8501"
READY_TEXT = "AI Assistant Comparison Lab"
MAX_READY_WAIT_SEC = 120


def wait_for_app(url: str, timeout_sec: int = 180) -> None:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"Streamlit did not become ready at {url} within {timeout_sec}s")


def wait_for_ui(page) -> None:
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=120_000)
    page.get_by_text(READY_TEXT, exact=False).wait_for(state="visible", timeout=MAX_READY_WAIT_SEC * 1000)
    # Allow widgets, fonts, and session state to finish rendering
    page.wait_for_timeout(12_000)
    page.wait_for_load_state("networkidle", timeout=60_000)
    page.wait_for_timeout(3000)


def capture() -> None:
    from playwright.sync_api import sync_playwright

    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    wait_for_app(BASE_URL)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        wait_for_ui(page)
        page.screenshot(path=str(SCREENSHOTS_DIR / "chat_home.png"), full_page=True)

        eval_tab = page.get_by_role("tab", name="Evaluation Dashboard")
        eval_tab.click()
        page.wait_for_timeout(5000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "evaluation_dashboard.png"), full_page=True)

        chat_tab = page.get_by_role("tab", name="Chat")
        chat_tab.click()
        page.wait_for_timeout(3000)

        # Sidebar: global memory section
        page.screenshot(path=str(SCREENSHOTS_DIR / "memory_demo.png"), full_page=True)

        # Switch to Frontier Assistant in sidebar selectbox
        try:
            selects = page.locator('[data-testid="stSelectbox"]')
            # Assistant model is typically the second selectbox (after Chats)
            if selects.count() >= 2:
                selects.nth(1).click()
                page.wait_for_timeout(800)
                page.get_by_text("Frontier Assistant", exact=True).click()
                page.wait_for_timeout(4000)
                page.screenshot(path=str(SCREENSHOTS_DIR / "chat_frontier.png"), full_page=True)

                selects.nth(1).click()
                page.wait_for_timeout(800)
                page.get_by_text("OSS Assistant", exact=True).click()
                page.wait_for_timeout(4000)
                page.screenshot(path=str(SCREENSHOTS_DIR / "chat_oss.png"), full_page=True)
        except Exception:
            page.screenshot(path=str(SCREENSHOTS_DIR / "chat_oss.png"), full_page=True)

        browser.close()

    # Legacy numbered names for README compatibility
    mapping = {
        "chat_home.png": "01_home_chat.png",
        "evaluation_dashboard.png": "02_evaluation_dashboard.png",
        "chat_frontier.png": "03_chat_frontier.png",
        "chat_oss.png": "04_chat_oss.png",
        "memory_demo.png": "05_sidebar_memory.png",
    }
    for src, dst in mapping.items():
        src_path = SCREENSHOTS_DIR / src
        if src_path.exists():
            (SCREENSHOTS_DIR / dst).write_bytes(src_path.read_bytes())

    print(f"Screenshots saved to {SCREENSHOTS_DIR}")


if __name__ == "__main__":
    capture()
