"""Market-data provider adapter interfaces and safe test implementations."""

from nifty_terminal.providers.angelone import (
    AngelOneConfig,
    AngelOneCredentials,
    AngelOneProviderAdapter,
)
from nifty_terminal.providers.base import FinalizedMinuteProvider, ProviderAdapter, ProviderHealth
from nifty_terminal.providers.replay import ReplayProviderAdapter

__all__ = [
    "FinalizedMinuteProvider",
    "AngelOneConfig",
    "AngelOneCredentials",
    "AngelOneProviderAdapter",
    "ProviderAdapter",
    "ProviderHealth",
    "ReplayProviderAdapter",
]
