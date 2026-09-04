"""Local, reproducible API overhead benchmark; never a market-latency claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "services" / "backend" / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from nifty_terminal.api.app import create_app  # noqa: E402
from nifty_terminal.settings import Settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--assert-p95-ms", type=float)
    arguments = parser.parse_args()
    if not 20 <= arguments.iterations <= 10_000:
        parser.error("--iterations must be between 20 and 10000")

    settings = Settings(
        app_name="NIFTY benchmark",
        environment="test",
        log_level="WARNING",
        market_data_mode="replay",
        market_data_provider=None,
        live_signal_kill_switch=True,
    )
    with TestClient(create_app(settings=settings)) as client:
        for _ in range(10):
            client.get("/api/v1/live")
        results = {
            "liveness": measure(client, "/api/v1/live", arguments.iterations),
            "readiness_fail_closed": measure(client, "/api/v1/ready", arguments.iterations),
        }
    report = {
        "schema_version": 1,
        "scope": "IN_PROCESS_LOCAL_API_OVERHEAD_ONLY",
        "iterations": arguments.iterations,
        "not_measured": [
            "provider_latency",
            "internet_latency",
            "candle_finalization_latency",
            "model_inference_latency",
            "browser_render_latency",
        ],
        "results_ms": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if arguments.assert_p95_ms is not None:
        worst = max(item["p95"] for item in results.values())
        if worst > arguments.assert_p95_ms:
            return 1
    return 0


def measure(client: TestClient, path: str, iterations: int) -> dict[str, float]:
    samples: list[float] = []
    for _ in range(iterations):
        started = perf_counter_ns()
        client.get(path)
        samples.append((perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(samples)
    return {
        "median": round(statistics.median(ordered), 3),
        "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
        "maximum": round(max(ordered), 3),
    }


if __name__ == "__main__":
    raise SystemExit(main())
