"""Synchronized dashboard analysis views."""

from nifty_terminal.dashboard.models import AnalysisView
from nifty_terminal.dashboard.read_model import InMemoryAnalysisReadModel

__all__ = ["AnalysisView", "InMemoryAnalysisReadModel"]
