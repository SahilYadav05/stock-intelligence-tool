from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase


class ContractTests(TestCase):
    def test_market_event_schema_is_versioned_and_parseable(self) -> None:
        root = Path(__file__).parents[3]
        schema_path = root / "contracts" / "market-event.v1.schema.json"

        with schema_path.open("r", encoding="utf-8") as schema_file:
            schema = json.load(schema_file)

        self.assertEqual(schema["$id"], "market-event.v1")
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        self.assertIn("event_id", schema["required"])
        self.assertIn("deduplication_key", schema["required"])

    def test_step_three_contracts_are_versioned_and_parseable(self) -> None:
        root = Path(__file__).parents[3]
        for filename, required_field in (
            ("candle.v1.schema.json", "revision"),
            ("market-state-snapshot.v1.schema.json", "candle_revision_checksum"),
        ):
            with (root / "contracts" / filename).open("r", encoding="utf-8") as file:
                schema = json.load(file)
            self.assertTrue(schema["$id"].endswith(".v1"))
            self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
            self.assertIn(required_field, schema["required"])

    def test_step_four_delivery_contracts_are_versioned(self) -> None:
        root = Path(__file__).parents[3]
        for filename in (
            "market-state-view.v1.schema.json",
            "websocket-message.v1.schema.json",
        ):
            with (root / "contracts" / filename).open("r", encoding="utf-8") as file:
                schema = json.load(file)
            self.assertTrue(schema["$id"].endswith(".v1"))
            self.assertEqual(schema["properties"]["schema_version"]["const"], 1)

    def test_step_five_research_contracts_are_versioned(self) -> None:
        root = Path(__file__).parents[3]
        for filename in (
            "historical-dataset.v1.schema.json",
            "feature-snapshot.v1.schema.json",
        ):
            with (root / "contracts" / filename).open("r", encoding="utf-8") as file:
                schema = json.load(file)
            self.assertTrue(schema["$id"].endswith(".v1"))

    def test_step_six_ml_and_replay_contracts_are_versioned(self) -> None:
        root = Path(__file__).parents[3]
        for filename in (
            "first-touch-label.v1.schema.json",
            "ml-training-run.v1.schema.json",
            "replay-prediction.v1.schema.json",
            "replay-assessment.v1.schema.json",
        ):
            with (root / "contracts" / filename).open("r", encoding="utf-8") as file:
                schema = json.load(file)
            self.assertTrue(schema["$id"].endswith(".v1"))

    def test_step_eight_dashboard_contracts_are_versioned(self) -> None:
        root = Path(__file__).parents[3]
        for filename in (
            "analysis-view.v1.schema.json",
            "analysis-availability.v1.schema.json",
        ):
            with (root / "contracts" / filename).open("r", encoding="utf-8") as file:
                schema = json.load(file)
            self.assertTrue(schema["$id"].endswith(".v1"))
            self.assertEqual(schema["properties"]["schema_version"]["const"], 1)

    def test_step_nine_tracking_contracts_are_versioned(self) -> None:
        root = Path(__file__).parents[3]
        for filename in (
            "tracked-prediction.v1.schema.json",
            "prediction-assessment.v1.schema.json",
            "paper-trade.v1.schema.json",
            "paper-trade-event.v1.schema.json",
            "monitoring-view.v1.schema.json",
            "tracking-overview.v1.schema.json",
        ):
            with (root / "contracts" / filename).open("r", encoding="utf-8") as file:
                schema = json.load(file)
            self.assertTrue(schema["$id"].endswith(".v1"))
            self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
