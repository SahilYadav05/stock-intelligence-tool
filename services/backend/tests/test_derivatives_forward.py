from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import sqlite3
import tempfile
from unittest import TestCase

from nifty_terminal.derivatives.forward import (
    SQLiteDerivativesLedger,
    build_derivatives_snapshot,
    resolve_nifty_contracts,
)


class DerivativesForwardTests(TestCase):
    def test_resolves_nearest_unexpired_contracts(self) -> None:
        master = [
            {"name": "NIFTY", "exch_seg": "NFO", "instrumenttype": "FUTIDX", "expiry": "24SEP2026", "symbol": "NIFTY24SEP26FUT", "token": "100"},
            {"name": "NIFTY", "exch_seg": "NFO", "instrumenttype": "FUTIDX", "expiry": "29OCT2026", "symbol": "NIFTY29OCT26FUT", "token": "101"},
            {"name": "NIFTY", "exch_seg": "NFO", "instrumenttype": "OPTIDX", "expiry": "10SEP2026", "symbol": "NIFTY10SEP2625000CE", "token": "200"},
        ]

        future, option = resolve_nifty_contracts(master, as_of=date(2026, 9, 4))

        self.assertEqual(future["token"], "100")
        self.assertEqual(option["token"], "200")

    def test_snapshot_derives_oi_book_and_option_surface_without_orders(self) -> None:
        future = {"symbol": "NIFTY24SEP26FUT", "token": "100", "expiry": "24SEP2026"}
        option = {"symbol": "NIFTY10SEP2625000CE", "token": "200", "expiry": "10SEP2026"}
        snapshot = build_derivatives_snapshot(
            observed_at=datetime(2026, 9, 4, 4, 0, tzinfo=timezone.utc),
            future_contract=future,
            option_contract=option,
            market_response={"status": True, "data": {"fetched": [
                {"symbolToken": "100", "ltp": 25000, "tradeVolume": 1000, "opnInterest": 2000, "totBuyQuan": 600, "totSellQuan": 400},
                {"symbolToken": "99926000", "ltp": 24950},
            ]}},
            pcr_response={"status": True, "data": [{"tradingSymbol": "NIFTY24SEP26FUT", "pcr": 1.1}]},
            greeks_response={"status": True, "data": [
                {"optionType": "CE", "delta": 0.50, "impliedVolatility": 12, "tradeVolume": 100},
                {"optionType": "PE", "delta": -0.50, "impliedVolatility": 14, "tradeVolume": 120},
                {"optionType": "CE", "delta": 0.25, "impliedVolatility": 13, "tradeVolume": 50},
                {"optionType": "PE", "delta": -0.25, "impliedVolatility": 16, "tradeVolume": 80},
            ]},
            spot_token="99926000",
        )

        self.assertAlmostEqual(snapshot.futures_book_imbalance, 0.2)
        self.assertEqual(snapshot.atm_implied_volatility, 13)
        self.assertEqual(snapshot.delta25_put_call_iv_skew, 3)
        self.assertAlmostEqual(snapshot.option_volume_put_call_ratio, 4 / 3)
        self.assertEqual(snapshot.spot_ltp, 24950)
        self.assertEqual(snapshot.futures_basis_points, 50)
        self.assertAlmostEqual(snapshot.futures_basis_bps, 50 / 24950 * 10_000)
        self.assertEqual(snapshot.unavailable_sources, ())
        self.assertTrue(snapshot.research_only)

    def test_ledger_is_append_only_and_idempotent(self) -> None:
        future = {"symbol": "NIFTY24SEP26FUT", "token": "100", "expiry": "24SEP2026"}
        option = {"symbol": "NIFTY10SEP2625000CE", "token": "200", "expiry": "10SEP2026"}
        snapshot = build_derivatives_snapshot(
            observed_at=datetime(2026, 9, 4, 4, 0, tzinfo=timezone.utc),
            future_contract=future,
            option_contract=option,
            market_response={"status": True, "data": {"fetched": [
                {"symbolToken": "100", "ltp": 25000, "tradeVolume": 1000, "opnInterest": 2000, "totBuyQuan": 600, "totSellQuan": 400},
                {"symbolToken": "99926000", "ltp": 24950},
            ]}},
            pcr_response={"status": True, "data": []},
            greeks_response={"status": True, "data": []},
            spot_token="99926000",
        )
        self.assertEqual(snapshot.unavailable_sources, ("PUT_CALL_RATIO", "OPTION_GREEKS"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite3"
            ledger = SQLiteDerivativesLedger(path)
            self.assertTrue(ledger.append(snapshot))
            self.assertFalse(ledger.append(snapshot))
            connection = sqlite3.connect(path)
            try:
                count = connection.execute("SELECT COUNT(*) FROM derivatives_snapshots").fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(count, 1)
            contracts = ledger.load_contracts()
            self.assertEqual(len(contracts), 1)
            self.assertEqual(contracts[0]["futures_basis_points"], 50)
