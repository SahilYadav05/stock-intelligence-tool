"""Step 7 orchestration without changing immutable Step 6 predictions."""

from __future__ import annotations

from datetime import datetime, timezone

from nifty_terminal.calibration.models import CalibrationConfig, CalibrationObservation
from nifty_terminal.calibration.pipeline import CalibrationPipeline
from nifty_terminal.signals.models import PolicyConfig
from nifty_terminal.signals.replay import (
    SignalReplayInput,
    Step7ResearchReport,
    replay_signal_policy,
)


class Step7ResearchPipeline:
    def run(
        self,
        *,
        observations: tuple[CalibrationObservation, ...],
        replay_inputs: tuple[SignalReplayInput, ...],
        calibration_config: CalibrationConfig | None = None,
        policy_config: PolicyConfig | None = None,
        created_at: datetime | None = None,
    ) -> Step7ResearchReport:
        timestamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        calibration = CalibrationPipeline().run(
            observations=observations,
            config=calibration_config,
            created_at=timestamp,
        )
        resolved_policy = policy_config or PolicyConfig()
        decisions = replay_signal_policy(
            calibration=calibration,
            inputs=replay_inputs,
            config=resolved_policy,
        )
        return Step7ResearchReport(
            schema_version=1,
            created_at=timestamp,
            calibration=calibration,
            policy_config=resolved_policy,
            decisions=decisions,
        )
