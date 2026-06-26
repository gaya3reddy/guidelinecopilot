"""
Unit tests for apps/api/config.py

Settings.load() reads from environment variables.
We use monkeypatch to control the env — no .env file, no real API key needed.
Each test is hermetic: env vars are restored automatically after each test.
"""

from __future__ import annotations

import sys

import pytest


def _reload_settings(monkeypatch, env: dict[str, str]):
    """
    Set env vars via monkeypatch, then reload the config module so that
    Settings.load() sees the fresh values.
    Returns the reloaded `settings` singleton.
    """
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    # Force reload so module-level `settings = Settings.load()` re-runs
    if "apps.api.config" in sys.modules:
        del sys.modules["apps.api.config"]

    import apps.api.config as cfg

    return cfg.settings


class TestSettingsLoad:
    def test_defaults_are_applied(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        s = _reload_settings(monkeypatch, {})
        assert s.app_name == "guidelinecopilot-api"
        assert s.api_port == 8000
        assert s.openai_chat_model == "gpt-4o-mini"
        assert s.openai_embed_model == "text-embedding-3-small"

    def test_custom_app_name_respected(self, monkeypatch, tmp_path):
        s = _reload_settings(
            monkeypatch,
            {
                "OPENAI_API_KEY": "sk-test",
                "DATA_DIR": str(tmp_path),
                "APP_NAME": "my-custom-app",
            },
        )
        assert s.app_name == "my-custom-app"

    def test_custom_port_parsed_as_int(self, monkeypatch, tmp_path):
        s = _reload_settings(
            monkeypatch,
            {
                "OPENAI_API_KEY": "sk-test",
                "DATA_DIR": str(tmp_path),
                "API_PORT": "9090",
            },
        )
        assert s.api_port == 9090
        assert isinstance(s.api_port, int)

    def test_openai_key_present(self, monkeypatch, tmp_path):
        s = _reload_settings(
            monkeypatch,
            {
                "OPENAI_API_KEY": "sk-real-key",
                "DATA_DIR": str(tmp_path),
            },
        )
        assert s.openai_api_key == "sk-real-key"

    # def test_missing_openai_key_sets_none(self, monkeypatch, tmp_path):
    #     monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    #     s = _reload_settings(monkeypatch, {
    #         "DATA_DIR": str(tmp_path),
    #     })
    #     assert s.openai_api_key is None
    def test_missing_openai_key_sets_none(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Prevent load_dotenv from re-loading the key from .env
        monkeypatch.setattr("dotenv.load_dotenv", lambda **kwargs: None)
        s = _reload_settings(
            monkeypatch,
            {
                "DATA_DIR": str(tmp_path),
            },
        )
        assert s.openai_api_key is None

    def test_data_dirs_are_created(self, monkeypatch, tmp_path):
        s = _reload_settings(
            monkeypatch,
            {
                "OPENAI_API_KEY": "sk-test",
                "DATA_DIR": str(tmp_path / "data"),
            },
        )
        assert s.raw_dir.exists()
        assert s.processed_dir.exists()

    def test_max_upload_mb_default(self, monkeypatch, tmp_path):
        s = _reload_settings(
            monkeypatch,
            {
                "OPENAI_API_KEY": "sk-test",
                "DATA_DIR": str(tmp_path),
            },
        )
        assert s.max_upload_mb == 30

    def test_max_upload_mb_override(self, monkeypatch, tmp_path):
        s = _reload_settings(
            monkeypatch,
            {
                "OPENAI_API_KEY": "sk-test",
                "DATA_DIR": str(tmp_path),
                "MAX_UPLOAD_MB": "50",
            },
        )
        assert s.max_upload_mb == 50

    def test_settings_is_frozen(self, monkeypatch, tmp_path):
        """Settings is a frozen dataclass — mutation must raise."""
        s = _reload_settings(
            monkeypatch,
            {
                "OPENAI_API_KEY": "sk-test",
                "DATA_DIR": str(tmp_path),
            },
        )
        with pytest.raises((AttributeError, TypeError)):
            s.api_port = 1234  # type: ignore[misc]
