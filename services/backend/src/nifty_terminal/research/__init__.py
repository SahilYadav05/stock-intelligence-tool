"""Real-data research orchestration that cannot enable live signals."""

from nifty_terminal.research.real_data import (
    RealDataResearchGate,
    RealDataResearchResult,
    evaluate_real_data_research,
)
from nifty_terminal.research.v2 import ResearchV2Report, screen_target
from nifty_terminal.research.step16 import run_locked_research
from nifty_terminal.research.step17 import run_policy_research
from nifty_terminal.research.step18 import run_model_v2_research
from nifty_terminal.research.step18b import run_trade_aligned_research
from nifty_terminal.research.step19 import run_price_action_research
from nifty_terminal.research.step20 import run_pooled_directional_research
from nifty_terminal.research.step21 import run_event_price_action_research
from nifty_terminal.research.step23 import run_conditional_direction_research
from nifty_terminal.research.step25 import run_compact_feature_audit

__all__ = [
    "RealDataResearchGate",
    "RealDataResearchResult",
    "evaluate_real_data_research",
    "ResearchV2Report",
    "screen_target",
    "run_locked_research",
    "run_policy_research",
    "run_model_v2_research",
    "run_trade_aligned_research",
    "run_price_action_research",
    "run_pooled_directional_research",
    "run_event_price_action_research",
    "run_conditional_direction_research",
    "run_compact_feature_audit",
]
