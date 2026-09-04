from __future__ import annotations

import os
from unittest import TestCase
from unittest.mock import patch

from nifty_terminal.settings import Settings


class SettingsTests(TestCase):
    def test_defaults_are_safe_for_local_development(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_environment()

        self.assertEqual(settings.environment, "development")
        self.assertEqual(settings.market_data_mode, "replay")
        self.assertIsNone(settings.market_data_provider)
        self.assertFalse(settings.live_analysis_available)
        self.assertNotIn("*", settings.api_allowed_origins)

    def test_invalid_environment_is_rejected(self) -> None:
        with patch.dict(os.environ, {"APP_ENV": "unknown"}, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_environment()

    def test_live_mode_requires_an_explicit_provider(self) -> None:
        with patch.dict(os.environ, {"MARKET_DATA_MODE": "live"}, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_environment()

    def test_wildcard_cors_origin_is_rejected(self) -> None:
        with patch.dict(os.environ, {"API_ALLOWED_ORIGINS": "*"}, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_environment()
