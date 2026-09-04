"""Live delivery, synchronized analysis, and hardened operational APIs."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from nifty_terminal.calibration.definitions import CALIBRATION_VERSION, RELEASE_GATE_VERSION
from nifty_terminal.api.messages import (
    current_view_message,
    market_state_message,
    status_message,
)
from nifty_terminal.dashboard.read_model import InMemoryAnalysisReadModel
from nifty_terminal.delivery.service import MarketStateDeliveryService
from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.domain.instruments import build_mvp_instrument_registry
from nifty_terminal.features.definitions import FEATURE_SET_HASH, FEATURE_VERSION, MINIMUM_HISTORY
from nifty_terminal.hardening.models import DriftEvidence, ReleaseManifest
from nifty_terminal.hardening.release import evaluate_release, load_release_manifest
from nifty_terminal.hardening.security import SecurityMiddleware, valid_websocket_request
from nifty_terminal.ml.definitions import (
    DOWN_ATR_MULTIPLIER,
    HORIZON_BARS,
    HORIZON_MINUTES,
    LABEL_DEFINITION_HASH,
    LABEL_VERSION,
    RESEARCH_VERSION,
    UP_ATR_MULTIPLIER,
)
from nifty_terminal.price_action.engine import PRICE_ACTION_VERSION, PriceActionEngine
from nifty_terminal.signals.definitions import (
    LIFECYCLE_POLICY_VERSION,
    RISK_POLICY_VERSION,
    SIGNAL_POLICY_VERSION,
)
from nifty_terminal.signals.models import PolicyConfig, SignalDirection
from nifty_terminal.settings import Settings
from nifty_terminal.tracking.service import TrackingService
from nifty_terminal.runtime.live_market import LiveMarketRuntime, build_angelone_live_runtime


MVP_INSTRUMENT_ID = "NIFTY50_SPOT"
MVP_TIMEFRAME = "5m"


def create_app(
    *,
    settings: Settings | None = None,
    delivery: MarketStateDeliveryService | None = None,
    analysis: InMemoryAnalysisReadModel | None = None,
    tracking: TrackingService | None = None,
    release_manifest: ReleaseManifest | None = None,
    drift_evidence: DriftEvidence | None = None,
    live_runtime: LiveMarketRuntime | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_environment()
    resolved_delivery = delivery or MarketStateDeliveryService()
    resolved_analysis = analysis or InMemoryAnalysisReadModel()
    resolved_tracking = tracking or TrackingService(drift_evidence=drift_evidence)
    price_action_engine = PriceActionEngine()
    resolved_manifest = (
        release_manifest
        if release_manifest is not None
        else load_release_manifest(resolved_settings.release_manifest_path)
    )
    registry = build_mvp_instrument_registry()
    resolved_runtime = live_runtime
    if (
        resolved_runtime is None
        and resolved_settings.market_data_mode == "live"
        and resolved_settings.market_data_provider == "angelone"
    ):
        resolved_runtime = build_angelone_live_runtime(
            settings=resolved_settings,
            delivery=resolved_delivery,
        )
    websocket_clients: set[int] = set()
    websocket_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if resolved_runtime is not None:
            await resolved_runtime.start()
        try:
            yield
        finally:
            if resolved_runtime is not None:
                await resolved_runtime.stop()

    application = FastAPI(
        title="NIFTY Intelligence Market API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.delivery = resolved_delivery
    application.state.analysis = resolved_analysis
    application.state.tracking = resolved_tracking
    application.state.release_manifest = resolved_manifest
    application.state.drift_evidence = drift_evidence
    application.state.live_runtime = resolved_runtime
    application.add_middleware(SecurityMiddleware, settings=resolved_settings)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.api_allowed_origins),
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @application.get("/api/v1/live")
    async def live() -> dict[str, object]:
        return {"schema_version": 1, "service": "market-delivery-api", "status": "ALIVE"}

    @application.get("/api/v1/ready")
    async def ready() -> dict[str, object]:
        readiness = _current_readiness(
            settings=resolved_settings,
            delivery=resolved_delivery,
            analysis=resolved_analysis,
            manifest=resolved_manifest,
            drift=drift_evidence,
        )
        if not readiness.signal_allowed:
            raise HTTPException(status_code=503, detail=readiness.to_contract())
        return readiness.to_contract()

    @application.get("/api/v1/security/status")
    async def security_status() -> dict[str, object]:
        return {
            "schema_version": 1,
            "environment": resolved_settings.environment,
            "authentication": resolved_settings.api_auth_mode.upper(),
            "cors_wildcard_allowed": False,
            "request_body_limit_bytes": resolved_settings.request_body_limit_bytes,
            "requests_per_minute": resolved_settings.requests_per_minute,
            "websocket_connection_limit": resolved_settings.websocket_connection_limit,
            "provider_credentials_browser_exposed": False,
            "automatic_execution_enabled": False,
        }

    @application.get("/api/v1/health")
    async def health() -> dict[str, object]:
        view = resolved_delivery.read_model.get(MVP_INSTRUMENT_ID)
        runtime_health = resolved_runtime.health if resolved_runtime is not None else None
        return {
            "schema_version": 1,
            "service": "market-delivery-api",
            "status": "READY" if view is not None else "DEGRADED",
            "data_status": (
                runtime_health.data_status.value
                if runtime_health is not None
                else view.snapshot.data_status.value
                if view
                else ConnectionState.DISCONNECTED.value
            ),
            "live_provider_configured": resolved_runtime is not None,
            "live_analysis_available": _current_readiness(
                settings=resolved_settings,
                delivery=resolved_delivery,
                analysis=resolved_analysis,
                manifest=resolved_manifest,
                drift=drift_evidence,
            ).signal_allowed,
            "reason": (
                runtime_health.reason
                if runtime_health is not None
                else None
                if view
                else "LIVE_PROVIDER_NOT_CONFIGURED"
            ),
        }

    @application.get("/api/v1/provider/health")
    async def provider_health() -> dict[str, object]:
        if resolved_runtime is None:
            return {
                "schema_version": 1,
                "running": False,
                "data_status": ConnectionState.DISCONNECTED.value,
                "reason": "LIVE_PROVIDER_NOT_CONFIGURED",
                "provider": resolved_settings.market_data_provider,
                "credentials_exposed": False,
                "automatic_trading_enabled": False,
                "nifty_spot_volume_enabled": False,
            }
        return resolved_runtime.health.to_contract()

    @application.get("/api/v1/shadow/status")
    async def shadow_status() -> dict[str, object]:
        if resolved_runtime is None or resolved_runtime.shadow_status is None:
            return {
                "schema_version": 1,
                "enabled": False,
                "runtime_mode": "DISABLED",
                "healthy": True,
                "reason": "SHADOW_MODE_NOT_CONFIGURED",
                "shadow_only": True,
                "precise_probability_display_allowed": False,
                "official_signal_available": False,
                "automatic_trading_enabled": False,
            }
        return resolved_runtime.shadow_status

    @application.get("/api/v1/instruments")
    async def instruments() -> dict[str, object]:
        instrument = registry.get(MVP_INSTRUMENT_ID)
        return {
            "schema_version": 1,
            "instruments": [
                {
                    "instrument_id": instrument.instrument_id,
                    "display_name": instrument.display_name,
                    "venue": instrument.venue,
                    "timezone": instrument.timezone,
                    "currency": instrument.currency,
                    "volume_supported": instrument.volume_supported,
                    "primary_timeframe": MVP_TIMEFRAME,
                    "context_timeframes": ["15m", "1h"],
                }
            ],
        }

    @application.get("/api/v1/research/status")
    async def research_status() -> dict[str, object]:
        return {
            "schema_version": 1,
            "historical_store": "SQLITE_LOCAL_APPEND_ONLY",
            "historical_provider_configured": False,
            "feature_version": FEATURE_VERSION,
            "feature_set_hash": FEATURE_SET_HASH,
            "minimum_candles_per_timeframe": MINIMUM_HISTORY,
            "timeframes": ["5m", "15m", "1h"],
            "nifty_spot_volume_features_enabled": False,
            "label_version": LABEL_VERSION,
            "label_definition_hash": LABEL_DEFINITION_HASH,
            "prediction_horizon_minutes": HORIZON_MINUTES,
            "prediction_horizon_bars": HORIZON_BARS,
            "up_atr_multiplier": format(UP_ATR_MULTIPLIER, "f"),
            "down_atr_multiplier": format(DOWN_ATR_MULTIPLIER, "f"),
            "research_version": RESEARCH_VERSION,
            "chronological_walk_forward_supported": True,
            "purge_and_embargo_required": True,
            "historical_simulated_live_replay_supported": True,
            "raw_probability_display_allowed": False,
            "calibration_method": "MULTICLASS_TEMPERATURE_SCALING",
            "calibration_version": CALIBRATION_VERSION,
            "calibration_release_gate_version": RELEASE_GATE_VERSION,
            "calibration_requires_disjoint_chronological_evaluation": True,
            "calibration_maximum_ece": 0.05,
            "calibration_requires_positive_brier_skill": True,
            "signal_policy_version": SIGNAL_POLICY_VERSION,
            "risk_policy_version": RISK_POLICY_VERSION,
            "lifecycle_policy_version": LIFECYCLE_POLICY_VERSION,
            "signal_policy_defaults": PolicyConfig().to_contract(),
            "wait_is_hard_gate_default": True,
            "automatic_trading_enabled": False,
            "model_available": False,
            "calibration_available": False,
            "signal_available": False,
            "dashboard_analysis_contract": "analysis-view.v1",
            "snapshot_synchronization_required": True,
            "chart_overlay_support": True,
            "causal_price_action_version": PRICE_ACTION_VERSION,
            "price_action_decision_support_available": True,
            "price_action_is_official_signal": False,
            "price_action_uses_finalized_candles_only": True,
            "prediction_outcome_tracking_supported": True,
            "paper_trading_supported": True,
            "paper_trading_unit": "NIFTY_INDEX_POINTS",
            "paper_trading_executes_orders": False,
            "minimum_analytics_sample": 30,
            "precise_tracking_metrics_require_sample_gate": True,
            "monitoring_contract": "monitoring-view.v1",
            "tracking_overview_contract": "tracking-overview.v1",
            "release_readiness_contract": "release-readiness.v1",
            "artifact_manifest_contract": "artifact-manifest.v1",
            "drift_requires_explicit_reference_distribution": True,
            "live_signal_kill_switch_active": resolved_settings.live_signal_kill_switch,
        }

    @application.get("/api/v1/tracking/{instrument_id}")
    async def tracking_overview(
        instrument_id: str,
        timeframe: str = MVP_TIMEFRAME,
    ) -> dict[str, object]:
        _validate_subscription(instrument_id, timeframe)
        market_view = resolved_delivery.read_model.get(instrument_id)
        current_analysis = None
        if market_view is not None:
            current_analysis = resolved_analysis.get(market_view.snapshot.snapshot_id)
            if current_analysis is not None and (
                current_analysis.candle_revision_checksum
                != market_view.snapshot.candle_revision_checksum
            ):
                current_analysis = None
        return resolved_tracking.overview(
            instrument_id,
            generated_at=datetime.now(timezone.utc),
            market_view=market_view,
            analysis=current_analysis,
        ).to_contract()

    @application.get("/api/v1/market-state/{instrument_id}")
    async def market_state(instrument_id: str, timeframe: str = MVP_TIMEFRAME) -> dict[str, object]:
        _validate_subscription(instrument_id, timeframe)
        view = resolved_delivery.read_model.get(instrument_id)
        if view is None:
            runtime_health = resolved_runtime.health if resolved_runtime is not None else None
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "LIVE_ANALYSIS_UNAVAILABLE",
                    "data_status": (
                        runtime_health.data_status.value
                        if runtime_health is not None
                        else ConnectionState.DISCONNECTED.value
                    ),
                    "reason": (
                        runtime_health.reason
                        if runtime_health is not None
                        else "LIVE_PROVIDER_NOT_CONFIGURED"
                    ),
                },
            )
        return view.to_contract()

    @application.get("/api/v1/chart-history/{instrument_id}")
    async def chart_history(
        instrument_id: str,
        decision_time: datetime,
        candle_revision_checksum: str,
        timeframe: str = MVP_TIMEFRAME,
    ) -> dict[str, object]:
        _validate_subscription(instrument_id, timeframe)
        if resolved_runtime is None:
            raise HTTPException(
                status_code=503,
                detail={"code": "LIVE_CHART_HISTORY_UNAVAILABLE"},
            )
        view = resolved_delivery.read_model.get(instrument_id)
        if view is None:
            raise HTTPException(
                status_code=503,
                detail={"code": "CANONICAL_MARKET_STATE_UNAVAILABLE"},
            )
        if decision_time.tzinfo is None or decision_time.utcoffset() is None:
            raise HTTPException(status_code=422, detail="decision_time must include an offset")
        requested_time = decision_time.astimezone(timezone.utc)
        snapshot = view.snapshot
        if (
            requested_time != snapshot.decision_time
            or candle_revision_checksum != snapshot.candle_revision_checksum
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "CHART_HISTORY_SNAPSHOT_MISMATCH",
                    "current_decision_time": snapshot.decision_time.isoformat(),
                    "current_candle_revision_checksum": snapshot.candle_revision_checksum,
                },
            )
        candles = resolved_runtime.chart_history(decision_time=snapshot.decision_time)
        return {
            "schema_version": 1,
            "instrument_id": instrument_id,
            "exchange_timezone": "Asia/Kolkata",
            "decision_time": snapshot.decision_time.isoformat().replace("+00:00", "Z"),
            "data_as_of": snapshot.data_as_of.isoformat().replace("+00:00", "Z"),
            "candle_revision_checksum": snapshot.candle_revision_checksum,
            "candles": [candle.to_contract() for candle in candles],
        }

    @application.get("/api/v1/analysis/{instrument_id}")
    async def analysis_view(
        instrument_id: str,
        snapshot_id: str,
        timeframe: str = MVP_TIMEFRAME,
    ) -> dict[str, object]:
        _validate_subscription(instrument_id, timeframe)
        market_view = resolved_delivery.read_model.get(instrument_id)
        if market_view is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "LIVE_ANALYSIS_UNAVAILABLE",
                    "data_status": ConnectionState.DISCONNECTED.value,
                    "reason": "LIVE_PROVIDER_NOT_CONFIGURED",
                },
            )
        snapshot = market_view.snapshot
        if snapshot.snapshot_id != snapshot_id:
            return _analysis_availability(
                snapshot_id=snapshot.snapshot_id,
                candle_revision_checksum=snapshot.candle_revision_checksum,
                analysis=None,
                reason="MARKET_SNAPSHOT_ADVANCED",
            )
        current = resolved_analysis.get(snapshot_id)
        if current is None:
            return _analysis_availability(
                snapshot_id=snapshot.snapshot_id,
                candle_revision_checksum=snapshot.candle_revision_checksum,
                analysis=None,
                reason="NO_APPROVED_ANALYSIS_FOR_SNAPSHOT",
            )
        if current.candle_revision_checksum != snapshot.candle_revision_checksum:
            return _analysis_availability(
                snapshot_id=snapshot.snapshot_id,
                candle_revision_checksum=snapshot.candle_revision_checksum,
                analysis=None,
                reason="ANALYSIS_REVISION_MISMATCH",
            )
        return {
            "schema_version": 1,
            "sync_state": "SYNCED",
            "snapshot_id": snapshot.snapshot_id,
            "candle_revision_checksum": snapshot.candle_revision_checksum,
            "analysis": current.to_contract(),
            "signal_suppressed": current.signal.direction is SignalDirection.WAIT,
            "reason": None,
        }

    @application.get("/api/v1/price-action/{instrument_id}")
    async def price_action_view(
        instrument_id: str,
        snapshot_id: str,
        timeframe: str = MVP_TIMEFRAME,
    ) -> dict[str, object]:
        """Return causal technical decision support for one exact snapshot.

        This endpoint is deliberately independent of the official model release
        path.  A price-action setup is conditional research output, never a
        calibrated probability or an executable order instruction.
        """
        _validate_subscription(instrument_id, timeframe)
        market_view = resolved_delivery.read_model.get(instrument_id)
        if market_view is None:
            raise HTTPException(
                status_code=503,
                detail={"code": "PRICE_ACTION_UNAVAILABLE_NO_MARKET_SNAPSHOT"},
            )
        snapshot = market_view.snapshot
        if snapshot.snapshot_id != snapshot_id:
            return {
                "schema_version": 1,
                "sync_state": "SYNCING",
                "snapshot_id": snapshot.snapshot_id,
                "candle_revision_checksum": snapshot.candle_revision_checksum,
                "analysis": None,
                "reason": "MARKET_SNAPSHOT_ADVANCED",
            }
        analysis_market_view = market_view
        if resolved_runtime is not None:
            history = resolved_runtime.chart_history(decision_time=snapshot.decision_time)
            by_id = {item.candle_id: item for item in (*market_view.finalized_candles, *history)}
            analysis_market_view = replace(
                market_view,
                finalized_candles=tuple(
                    sorted(by_id.values(), key=lambda item: (item.timeframe.value, item.opens_at))
                ),
            )
        result = price_action_engine.analyze(
            analysis_market_view,
            generated_at=datetime.now(timezone.utc),
        )
        return {
            "schema_version": 1,
            "sync_state": "SYNCED",
            "snapshot_id": snapshot.snapshot_id,
            "candle_revision_checksum": snapshot.candle_revision_checksum,
            "analysis": result.to_contract(),
            "reason": result.blockers[0] if result.blockers else None,
        }

    @application.websocket("/ws/v1/market-state")
    async def market_state_socket(websocket: WebSocket) -> None:
        instrument_id = websocket.query_params.get("instrument_id", MVP_INSTRUMENT_ID)
        timeframe = websocket.query_params.get("timeframe", MVP_TIMEFRAME)
        if not valid_websocket_request(websocket, resolved_settings):
            await websocket.close(code=1008, reason="Authentication or origin rejected")
            return
        client_identity = id(websocket)
        async with websocket_lock:
            if len(websocket_clients) >= resolved_settings.websocket_connection_limit:
                await websocket.close(code=1013, reason="Connection limit reached")
                return
            websocket_clients.add(client_identity)
        try:
            _validate_subscription(instrument_id, timeframe)
        except HTTPException:
            async with websocket_lock:
                websocket_clients.discard(client_identity)
            await websocket.close(code=1008, reason="Unsupported instrument or timeframe")
            return

        await websocket.accept()
        try:
            async with resolved_delivery.hub.subscribe(instrument_id) as queue:
                current = resolved_delivery.read_model.get(instrument_id)
                if current is None:
                    runtime_health = (
                        resolved_runtime.health if resolved_runtime is not None else None
                    )
                    await websocket.send_json(
                        status_message(
                            instrument_id=instrument_id,
                            data_status=(
                                runtime_health.data_status
                                if runtime_health is not None
                                else ConnectionState.DISCONNECTED
                            ),
                            reason=(
                                runtime_health.reason
                                if runtime_health is not None
                                else "LIVE_PROVIDER_NOT_CONFIGURED"
                            ),
                        )
                    )
                else:
                    await websocket.send_json(current_view_message(current, instrument_id))

                while True:
                    queue_task = asyncio.create_task(queue.get())
                    receive_task = asyncio.create_task(websocket.receive())
                    done, pending = await asyncio.wait(
                        {queue_task, receive_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    if receive_task in done:
                        incoming = receive_task.result()
                        if incoming["type"] == "websocket.disconnect":
                            return
                        continue
                    await websocket.send_json(market_state_message(queue_task.result()))
        except WebSocketDisconnect:
            return
        finally:
            async with websocket_lock:
                websocket_clients.discard(client_identity)

    return application


def _validate_subscription(instrument_id: str, timeframe: str) -> None:
    if instrument_id != MVP_INSTRUMENT_ID:
        raise HTTPException(status_code=404, detail="Unsupported instrument")
    if timeframe != MVP_TIMEFRAME:
        raise HTTPException(status_code=422, detail="Official MVP transport uses the 5m primary snapshot")


app = create_app()


def _current_readiness(
    *,
    settings: Settings,
    delivery: MarketStateDeliveryService,
    analysis: InMemoryAnalysisReadModel,
    manifest: ReleaseManifest | None,
    drift: DriftEvidence | None,
):
    market_view = delivery.read_model.get(MVP_INSTRUMENT_ID)
    current_analysis = analysis.get(market_view.snapshot.snapshot_id) if market_view else None
    return evaluate_release(
        evaluated_at=datetime.now(timezone.utc),
        settings=settings,
        manifest=manifest,
        market_view=market_view,
        analysis=current_analysis,
        drift=drift,
    )


def _analysis_availability(
    *,
    snapshot_id: str,
    candle_revision_checksum: str,
    analysis: None,
    reason: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "sync_state": "SYNCING_ANALYSIS",
        "snapshot_id": snapshot_id,
        "candle_revision_checksum": candle_revision_checksum,
        "analysis": analysis,
        "signal_suppressed": True,
        "reason": reason,
    }
