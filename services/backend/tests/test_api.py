from __future__ import annotations

from unittest import TestCase

from fastapi.testclient import TestClient

from market_state_fixture import build_market_state_view
from dashboard_fixture import build_analysis_view
from nifty_terminal.dashboard.read_model import InMemoryAnalysisReadModel
from nifty_terminal.api.app import create_app
from nifty_terminal.delivery.read_model import InMemoryMarketStateReadModel
from nifty_terminal.delivery.service import MarketStateDeliveryService
from nifty_terminal.settings import Settings


def test_settings() -> Settings:
    return Settings(
        app_name="NIFTY Intelligence Terminal Test",
        environment="test",
        log_level="WARNING",
        market_data_mode="replay",
        market_data_provider=None,
    )


class ApiTests(TestCase):
    def test_empty_service_fails_closed_for_http_and_websocket(self) -> None:
        application = create_app(settings=test_settings())
        with TestClient(application) as client:
            health = client.get("/api/v1/health")
            state = client.get("/api/v1/market-state/NIFTY50_SPOT?timeframe=5m")
            with client.websocket_connect(
                "/ws/v1/market-state?instrument_id=NIFTY50_SPOT&timeframe=5m"
            ) as socket:
                message = socket.receive_json()

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["data_status"], "DISCONNECTED")
        self.assertFalse(health.json()["live_analysis_available"])
        self.assertEqual(state.status_code, 503)
        self.assertEqual(message["message_type"], "STATUS")
        self.assertEqual(message["payload"]["reason"], "LIVE_PROVIDER_NOT_CONFIGURED")

    def test_atomic_view_is_identical_over_http_and_initial_websocket(self) -> None:
        view = build_market_state_view()
        read_model = InMemoryMarketStateReadModel()
        read_model.put(view)
        delivery = MarketStateDeliveryService(read_model=read_model)
        application = create_app(settings=test_settings(), delivery=delivery)

        with TestClient(application) as client:
            response = client.get("/api/v1/market-state/NIFTY50_SPOT?timeframe=5m")
            instruments = client.get("/api/v1/instruments")
            with client.websocket_connect(
                "/ws/v1/market-state?instrument_id=NIFTY50_SPOT&timeframe=5m"
            ) as socket:
                message = socket.receive_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), message["payload"])
        self.assertEqual(response.json()["snapshot"]["snapshot_id"], view.snapshot.snapshot_id)
        self.assertFalse(instruments.json()["instruments"][0]["volume_supported"])

    def test_unsupported_instrument_is_rejected(self) -> None:
        application = create_app(settings=test_settings())
        with TestClient(application) as client:
            response = client.get("/api/v1/market-state/FAKE?timeframe=5m")
        self.assertEqual(response.status_code, 404)

    def test_research_status_does_not_claim_data_model_or_volume(self) -> None:
        application = create_app(settings=test_settings())
        with TestClient(application) as client:
            response = client.get("/api/v1/research/status")

        payload = response.json()
        self.assertEqual(payload["feature_version"], "price_features.v1")
        self.assertFalse(payload["historical_provider_configured"])
        self.assertFalse(payload["nifty_spot_volume_features_enabled"])
        self.assertFalse(payload["model_available"])
        self.assertEqual(payload["label_version"], "nifty_5m_atr_first_touch.v1")
        self.assertEqual(payload["prediction_horizon_minutes"], 60)
        self.assertEqual(payload["up_atr_multiplier"], "1.0")
        self.assertEqual(payload["down_atr_multiplier"], "1.0")
        self.assertTrue(payload["chronological_walk_forward_supported"])
        self.assertFalse(payload["raw_probability_display_allowed"])
        self.assertEqual(payload["calibration_method"], "MULTICLASS_TEMPERATURE_SCALING")
        self.assertTrue(payload["calibration_requires_disjoint_chronological_evaluation"])
        self.assertEqual(payload["calibration_maximum_ece"], 0.05)
        self.assertTrue(payload["wait_is_hard_gate_default"])
        self.assertFalse(payload["automatic_trading_enabled"])
        self.assertFalse(payload["calibration_available"])
        self.assertFalse(payload["signal_available"])
        self.assertTrue(payload["snapshot_synchronization_required"])
        self.assertTrue(payload["chart_overlay_support"])
        self.assertTrue(payload["paper_trading_supported"])
        self.assertFalse(payload["paper_trading_executes_orders"])
        self.assertEqual(payload["minimum_analytics_sample"], 30)

    def test_tracking_endpoint_is_truthful_when_no_evidence_exists(self) -> None:
        application = create_app(settings=test_settings())
        with TestClient(application) as client:
            response = client.get("/api/v1/tracking/NIFTY50_SPOT?timeframe=5m")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["analytics"]["tracked_predictions"], 0)
        self.assertIsNone(payload["analytics"]["accuracy"])
        self.assertEqual(payload["analytics"]["metrics_status"], "UNAVAILABLE")
        self.assertEqual(payload["monitoring"]["overall_status"], "CRITICAL")
        self.assertTrue(payload["paper_only"])
        self.assertFalse(payload["automatic_execution"])

    def test_analysis_endpoint_suppresses_until_exact_snapshot_analysis_exists(self) -> None:
        market = build_market_state_view()
        read_model = InMemoryMarketStateReadModel()
        read_model.put(market)
        delivery = MarketStateDeliveryService(read_model=read_model)
        application = create_app(settings=test_settings(), delivery=delivery)

        with TestClient(application) as client:
            response = client.get(
                f"/api/v1/analysis/NIFTY50_SPOT?snapshot_id={market.snapshot.snapshot_id}"
            )

        payload = response.json()
        self.assertEqual(payload["sync_state"], "SYNCING_ANALYSIS")
        self.assertTrue(payload["signal_suppressed"])
        self.assertIsNone(payload["analysis"])
        self.assertEqual(payload["reason"], "NO_APPROVED_ANALYSIS_FOR_SNAPSHOT")

    def test_analysis_endpoint_returns_only_matching_analysis_revision(self) -> None:
        market = build_market_state_view()
        market_store = InMemoryMarketStateReadModel()
        market_store.put(market)
        analysis_store = InMemoryAnalysisReadModel()
        analysis_store.put(build_analysis_view(market))
        application = create_app(
            settings=test_settings(),
            delivery=MarketStateDeliveryService(read_model=market_store),
            analysis=analysis_store,
        )

        with TestClient(application) as client:
            response = client.get(
                f"/api/v1/analysis/NIFTY50_SPOT?snapshot_id={market.snapshot.snapshot_id}"
            )

        payload = response.json()
        self.assertEqual(payload["sync_state"], "SYNCED")
        self.assertEqual(payload["analysis"]["snapshot_id"], market.snapshot.snapshot_id)
        self.assertTrue(payload["signal_suppressed"])
