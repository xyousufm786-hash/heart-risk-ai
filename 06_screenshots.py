"""Capture real screenshots of the running Streamlit app (system-demo figures for the report)."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
URL = "http://localhost:8501"
PRESETS = [("Example A - younger, asymptomatic-free", "case_A_low"),
           ("Example B - middle-aged, mixed signals", "case_B_mixed"),
           ("Example C - older, multiple risk markers", "case_C_high")]

with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium" if Path("/opt/pw-browsers/chromium").exists() else None)
    pg = b.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1.5)
    pg.goto(URL, wait_until="networkidle"); pg.wait_for_selector("text=Risk estimate", timeout=60000); pg.wait_for_timeout(2500)
    pg.screenshot(path=str(OUT / "00_default.png"))
    for label, fname in PRESETS:
        pg.get_by_label("Load an example").click(); pg.wait_for_timeout(600)
        pg.get_by_role("option", name=label).click(); pg.wait_for_timeout(3500)
        pg.screenshot(path=str(OUT / f"{fname}.png"))
    pg.get_by_text("Model card").click(); pg.wait_for_timeout(800)
    pg.screenshot(path=str(OUT / "model_card.png"), full_page=True)
    b.close()
print("ok")
