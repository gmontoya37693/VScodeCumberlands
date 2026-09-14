#!/usr/bin/env python3
"""
Smoke test for the ALC API.

Hits health, read-only, and one non-mutating run endpoint (daily-preview) and
checks status codes and response shape. Does NOT call invoice/bank-payable/
month-end/commit endpoints, since those post real financial history.

Usage:
    ALC_API_KEY=... python scripts/smoke_test.py --base-url https://alc-api.example.com
    python scripts/smoke_test.py --base-url http://127.0.0.1:8000 --api-key testkey
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import date, timedelta


def _request(
    method: str,
    url: str,
    api_key: str | None = None,
    body: dict | None = None,
) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _check(label: str, condition: bool, results: list[tuple[str, bool]]) -> None:
    results.append((label, condition))
    print(f"{'PASS' if condition else 'FAIL'}: {label}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ALC API smoke test")
    parser.add_argument("--base-url", required=True, help="e.g. https://alc-api.example.com")
    parser.add_argument("--api-key", default=os.environ.get("ALC_API_KEY"))
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    results: list[tuple[str, bool]] = []

    status, body = _request("GET", f"{base}/health")
    _check("health returns 200 without a key", status == 200 and body.get("status") == "ok", results)

    status, _ = _request("GET", f"{base}/api/v1/assets")
    _check("assets without a key returns 401", status == 401, results)

    status, _ = _request("GET", f"{base}/api/v1/outputs/download?blob_name=outputs/dummy.txt")
    _check("outputs/download without a key returns 401", status == 401, results)

    status, _ = _request("GET", f"{base}/api/v1/inputs/download?name=assets.csv")
    _check("inputs/download without a key returns 401", status == 401, results)

    if args.api_key:
        status, body = _request("GET", f"{base}/api/v1/assets", api_key=args.api_key)
        _check("assets with a key returns 200 with request_id", status == 200 and "request_id" in body, results)

        status, body = _request("GET", f"{base}/api/v1/rates", api_key=args.api_key)
        _check("rates with a key returns 200 with request_id", status == 200 and "request_id" in body, results)

        status, body = _request(
            "GET", f"{base}/api/v1/state/posted-invoices", api_key=args.api_key
        )
        _check("posted-invoices returns 200", status == 200, results)

        status, body = _request("GET", f"{base}/api/v1/state/bank-payable", api_key=args.api_key)
        _check("bank-payable returns 200", status == 200, results)

        status, body = _request("GET", f"{base}/api/v1/state/closed-periods", api_key=args.api_key)
        _check("closed-periods returns 200", status == 200, results)

        status, body = _request("GET", f"{base}/api/v1/outputs", api_key=args.api_key)
        _check(
            "outputs list returns 200 (blob mode) or 400 (local mode)",
            status in (200, 400),
            results,
        )

        status, body = _request(
            "GET",
            f"{base}/api/v1/outputs/download?blob_name=outputs/does-not-exist.csv",
            api_key=args.api_key,
        )
        _check(
            "outputs/download with a key returns request_id (400 for missing/wrong mode)",
            status == 400 and "request_id" in body,
            results,
        )

        status, body = _request(
            "GET", f"{base}/api/v1/inputs/download?name=assets.csv", api_key=args.api_key
        )
        _check(
            "inputs/download with a key returns 200 or 400 with request_id",
            status in (200, 400) and "request_id" in body,
            results,
        )

        as_of = (date.today() - timedelta(days=1)).isoformat()
        status, body = _request(
            "POST",
            f"{base}/api/v1/runs/daily-preview",
            api_key=args.api_key,
            body={"operator": "smoke-test", "as_of": as_of, "billing_day": 22},
        )
        _check(
            "daily-preview returns 200 with request_id and summary",
            status == 200 and "request_id" in body and "summary" in body,
            results,
        )
    else:
        print("SKIP: authenticated checks (no --api-key / ALC_API_KEY provided)")

    failed = [label for label, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("Failed checks:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
