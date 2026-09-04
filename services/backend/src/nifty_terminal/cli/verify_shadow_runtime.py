"""Verify Step 17 shadow artifacts and ledger without starting market data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from nifty_terminal.settings import Settings
from nifty_terminal.shadow.runtime import build_shadow_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    arguments = parser.parse_args()
    if not arguments.env_file.is_file():
        raise SystemExit(f"Environment file does not exist: {arguments.env_file}")
    load_dotenv(arguments.env_file, override=True)
    settings = Settings.from_environment()
    if not settings.shadow_mode_enabled:
        raise SystemExit("SHADOW_MODE_ENABLED must be true for verification")
    if not settings.live_signal_kill_switch:
        raise SystemExit("LIVE_SIGNAL_KILL_SWITCH must remain true")
    assert settings.shadow_runtime_manifest_path is not None
    runtime = build_shadow_runtime(
        manifest_path=settings.shadow_runtime_manifest_path,
        ledger_path=settings.shadow_ledger_path,
    )
    print("Step 17 shadow runtime verifier")
    print(json.dumps(runtime.status.to_contract(), indent=2, sort_keys=True))
    print("RESULT: SHADOW RUNTIME VERIFICATION PASSED")
    print("Official signals, precise probability display and trading remain disabled.")


if __name__ == "__main__":
    main()
