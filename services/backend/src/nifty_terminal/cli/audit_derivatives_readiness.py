"""Report whether forward derivatives context is ready for model research."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nifty_terminal.derivatives.features import (
    DERIVATIVES_FEATURE_SET_HASH,
    DERIVATIVES_FEATURE_VERSION,
    evaluate_derivatives_readiness,
)
from nifty_terminal.derivatives.forward import SQLiteDerivativesLedger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=Path("data/derivatives-forward.sqlite3")
    )
    args = parser.parse_args()
    contracts = SQLiteDerivativesLedger(args.database).load_contracts()
    report = evaluate_derivatives_readiness(contracts)
    print(
        json.dumps(
            {
                "feature_version": DERIVATIVES_FEATURE_VERSION,
                "feature_set_hash": DERIVATIVES_FEATURE_SET_HASH,
                **report.to_contract(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
