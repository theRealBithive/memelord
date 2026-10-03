"""
Measure the rate-and-advance latency of the review page as a phone sees it.

    PERF_USER=... PERF_PASS=... uv run python tools/measure_review_latency.py \
        https://host 20 [none|wifi|lte] [readonly]

For each cycle: press a score key (or, with `readonly`, advance to the next
card the way the next arrow does but without its `?left=` stamp, so a live
library is not written to), wait for htmx to swap the card, wait for the new
card's image to finish loading. Two numbers per cycle, measured in the page
with performance.now():
  swap  = key press → htmx:afterSwap (server round trip + DOM swap)
  paint = key press → the new image's load event (what the user waits for)
Network timings per request come from Playwright's request.timing. The
throttle profiles use Chromium's network emulation; the browser is Playwright's
headless Chromium with the Pixel 7 device profile (`uv run playwright install
chromium-headless-shell` once if it is missing). Credentials come from the
environment only. RATE_ONE=1 adds one real rating (score 3) at the end and
prints the image it rated.
"""

import os
import statistics
import sys

from playwright.sync_api import sync_playwright

BASE = sys.argv[1]
CYCLES = int(sys.argv[2])
THROTTLE = sys.argv[3] if len(sys.argv) > 3 else "none"   # none | wifi | lte
# readonly: advance to the next card with the next-arrow's GET minus its ?left=
# stamp, so nothing is written (for a live library); default: press score keys.
READONLY = len(sys.argv) > 4 and sys.argv[4] == "readonly"
USERNAME = os.environ["PERF_USER"]
PASSWORD = os.environ["PERF_PASS"]

PROFILES = {
    "wifi": {"latency": 20, "downloadThroughput": 40 * 1024 * 1024 // 8, "uploadThroughput": 10 * 1024 * 1024 // 8},
    "lte": {"latency": 60, "downloadThroughput": 12 * 1024 * 1024 // 8, "uploadThroughput": 4 * 1024 * 1024 // 8},
}

INSTRUMENT = """
window.__m = {t0: null, result: null};
document.addEventListener('keydown', (e) => {
  if ('0123456'.includes(e.key)) window.__m.t0 = performance.now();
}, true);
document.body.addEventListener('htmx:afterSwap', (e) => {
  if (!e.detail.target || e.detail.target.id !== 'review-card') return;
  const tSwap = performance.now();
  const img = document.querySelector('#review-card .image-wrap img');
  const finish = (tImg) => { window.__m.result = {tSwap, tImg, src: img ? img.src : null}; };
  if (!img) { finish(tSwap); return; }
  if (img.complete && img.naturalWidth > 0) { finish(performance.now()); return; }
  img.addEventListener('load', () => finish(performance.now()), {once: true});
  img.addEventListener('error', () => finish(-1), {once: true});
});
"""


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p))]


def summarize(name, values):
    if not values:
        print(f"  {name}: no samples")
        return
    print(f"  {name:<14} n={len(values):<3} median={statistics.median(values):7.0f} ms  p90={pct(values, 0.9):7.0f} ms  max={max(values):7.0f} ms")


def main():
    net = {"score_post": [], "media_ttfb": [], "media_download": [], "media_bytes": []}

    def on_finished(request):
        timing = request.timing
        if timing["responseStart"] < 0:
            return
        ttfb = timing["responseStart"] - timing["requestStart"]
        download = timing["responseEnd"] - timing["responseStart"]
        if request.method == "POST" and "/score/" in request.url:
            net["score_post"].append(ttfb + download)
        elif "/media/" in request.url:
            net["media_ttfb"].append(ttfb)
            net["media_download"].append(download)
            response = request.response()
            length = response.headers.get("content-length") if response else None
            if length:
                net["media_bytes"].append(int(length))

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(**p.devices["Pixel 7"])
        page = context.new_page()
        if THROTTLE in PROFILES:
            cdp = context.new_cdp_session(page)
            cdp.send("Network.emulateNetworkConditions", {"offline": False, **PROFILES[THROTTLE]})
        page.on("requestfinished", on_finished)
        page.on("response", lambda r: print("  status", r.status, r.url[:90]) if r.status >= 400 else None)

        page.goto(f"{BASE}/login/")
        page.fill("input[name=username]", USERNAME)
        page.fill("input[name=password]", PASSWORD)
        page.click("button[type=submit], input[type=submit]")
        page.wait_for_url(f"{BASE}/review/**")
        page.on("response", lambda r: print("  media", r.status, r.url.rsplit("/", 1)[-1][:40], r.headers.get("content-type")) if "/media/" in r.url and r.status != 200 else None)
        page.wait_for_selector("#review-card .image-wrap img", state="attached")
        page.wait_for_function("(() => { const i = document.querySelector('#review-card .image-wrap img'); return i && i.complete; })()")
        print("  first image natural width:", page.evaluate("document.querySelector('#review-card .image-wrap img').naturalWidth"))
        page.evaluate(INSTRUMENT)
        version = page.evaluate("(document.querySelector('.nav-more-version') || {textContent: 'unknown'}).textContent.trim()")
        print("  server version:", version)
        nav_job = page.evaluate("document.querySelector('#nav-job') ? document.querySelector('#nav-job').textContent.trim() : ''")
        badges = page.evaluate("[...document.querySelectorAll('.badge')].map(b => b.id + '=' + b.textContent.trim()).join(' ')")
        print("  nav job:", repr(nav_job), "| badges:", badges)
        page.wait_for_timeout(1500)

        swap_ms, paint_ms = [], []
        for i in range(CYCLES):
            page.evaluate("window.__m.result = null; window.__m.t0 = null;")
            if READONLY:
                page.evaluate("""() => {
                    const btn = document.querySelector("[data-action='next']");
                    const url = btn.getAttribute('hx-get').split('?')[0];
                    window.__m.t0 = performance.now();
                    htmx.ajax('GET', url, {target: '#review-card', swap: 'outerHTML'});
                }""")
            else:
                page.keyboard.press(str(1 + i % 6))
            page.wait_for_function("window.__m.result !== null", timeout=60000)
            m = page.evaluate("({t0: window.__m.t0, ...window.__m.result})")
            if m["tImg"] == -1 or m["t0"] is None:
                print("  cycle", i, "image error or no key timestamp", m)
                continue
            swap_ms.append(m["tSwap"] - m["t0"])
            paint_ms.append(m["tImg"] - m["t0"])
            # Let the browser settle like a user would between two ratings.
            page.wait_for_timeout(700)
        if os.environ.get("RATE_ONE") == "1":
            target = page.evaluate("document.querySelector('[data-score=\"3\"]').getAttribute('hx-post')")
            page.evaluate("window.__m.result = null; window.__m.t0 = null;")
            page.keyboard.press("3")
            page.wait_for_function("window.__m.result !== null", timeout=60000)
            m = page.evaluate("({t0: window.__m.t0, ...window.__m.result})")
            print(f"  one real rating (score 3) via {target}: swap {m['tSwap']-m['t0']:.0f} ms, paint {m['tImg']-m['t0']:.0f} ms")
        browser.close()

    print(f"throttle={THROTTLE} cycles={CYCLES} mode={'readonly GET' if READONLY else 'score POST'}")
    summarize("swap", swap_ms)
    summarize("paint", paint_ms)
    summarize("POST total", net["score_post"])
    summarize("media TTFB", net["media_ttfb"])
    summarize("media download", net["media_download"])
    if net["media_bytes"]:
        print(f"  media bytes     median={statistics.median(net['media_bytes'])/1024:7.0f} KB  p90={pct(net['media_bytes'], 0.9)/1024:7.0f} KB  max={max(net['media_bytes'])/1024:7.0f} KB")


if __name__ == "__main__":
    main()
