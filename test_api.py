
import argparse
import csv
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()
BASE = os.getenv("API_URL", "http://127.0.0.1:8000")
OUT = "observation_table.csv"
SECRET = os.getenv("CLOUD_API_KEY", "").strip()

SUPPORTED_Q = "How many titles can I borrow at a time with Kindle Unlimited?"
UNSUPPORTED_Q = "Do you sell home insurance?"


def no_secrets(text: str) -> bool:
    return "Traceback" not in text and (not SECRET or SECRET not in text)


def post(question):
    return requests.post(f"{BASE}/ask", json={"question": question}, timeout=180)


def run_main():
    rows = []

    r = requests.get(f"{BASE}/health", timeout=10)
    rows.append(("Health endpoint", "API reports running",
                 f"HTTP {r.status_code}: {r.json()}", r.status_code == 200 and r.json().get("api") == "running"))

    r = post(SUPPORTED_Q)
    j = r.json()
    rows.append(("Supported question", "Grounded answer is returned",
                 f"HTTP {r.status_code}: {j.get('answer', j)!s:.160} | sources={j.get('sources', [])[:1]}",
                 r.status_code == 200 and not j.get("used_fallback")))
    supported = (r, j)

    r = post(UNSUPPORTED_Q)
    j = r.json()
    rows.append(("Unsupported question", "Insufficient information/fallback",
                 f"HTTP {r.status_code}: {j.get('answer', j)!s:.120} | note={j.get('review_note')}",
                 r.status_code == 200 and j.get("used_fallback") is True))

    r = post("   ")
    rows.append(("Empty question", "Validation error",
                 f"HTTP {r.status_code}: {r.json().get('detail')}", r.status_code == 422))

    r = post("a" * 1500)
    rows.append(("Very long question (1500 chars)", "Validation error",
                 f"HTTP {r.status_code}: {r.json().get('detail')}", r.status_code == 422))

    r, j = supported
    hdr = r.headers.get("X-Response-Time-ms")
    rows.append(("Response-time measurement", "Latency is recorded",
                 f"latency_ms={j.get('latency_ms')} | X-Response-Time-ms={hdr} (also in logs/app.log)",
                 bool(hdr) and (j.get("latency_ms") or 0) > 0))
    return rows


def run_backend_down():
    r = post(SUPPORTED_Q)
    ok = r.status_code == 503 and no_secrets(r.text)
    return [("Backend unavailable", "Controlled error; no secret details",
             f"HTTP {r.status_code}: {r.json()}", ok)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-down", action="store_true")
    args = ap.parse_args()
    try:
        rows = run_backend_down() if args.backend_down else run_main()
    except requests.ConnectionError:
        sys.exit(f"Cannot reach {BASE} - start the API first: uvicorn app:app --port 8000")

    mode = "a" if args.backend_down and os.path.exists(OUT) else "w"
    with open(OUT, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["Test case", "Expected result", "Actual result", "Pass/Fail"])
        for name, exp, act, ok in rows:
            w.writerow([name, exp, act, "Pass" if ok else "Fail"])

    for name, exp, act, ok in rows:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:32s} | expected: {exp}\n        actual: {act}\n")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
