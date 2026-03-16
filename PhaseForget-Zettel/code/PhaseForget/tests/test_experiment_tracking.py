"""Test experiment_id namespacing functionality."""

import os
from pathlib import Path

from phaseforget.config.settings import get_settings


def test_experiment_id_default():
    """Test default experiment uses default paths."""
    settings = get_settings(experiment_id="default")
    assert settings.experiment_id == "default"
    assert settings.chroma_persist_dir == "./data/chroma_db"
    assert settings.sqlite_db_path == "./data/phaseforget.db"
    assert settings.log_file == "./data/phaseforget.log"


def test_experiment_id_custom():
    """Test custom experiment_id namespaces paths."""
    settings = get_settings(experiment_id="qwen2.5-exp1")
    assert settings.experiment_id == "qwen2.5-exp1"
    assert settings.chroma_persist_dir == "./data/qwen2.5-exp1/chroma_db"
    assert settings.sqlite_db_path == "./data/qwen2.5-exp1/phaseforget.db"
    assert settings.log_file == "./data/qwen2.5-exp1/phaseforget.log"


def test_experiment_id_multiple():
    """Test multiple experiments get different paths."""
    exp1 = get_settings(experiment_id="gpt4o-exp1")
    exp2 = get_settings(experiment_id="claude-exp1")

    assert exp1.chroma_persist_dir == "./data/gpt4o-exp1/chroma_db"
    assert exp2.chroma_persist_dir == "./data/claude-exp1/chroma_db"
    assert exp1.chroma_persist_dir != exp2.chroma_persist_dir


def test_experiment_id_from_env():
    """Test experiment_id can come from environment variable."""
    os.environ["PHASEFORGET_EXPERIMENT_ID"] = "env-exp"
    try:
        settings = get_settings()  # Should pick up from env
        assert settings.experiment_id == "env-exp"
        assert settings.chroma_persist_dir == "./data/env-exp/chroma_db"
    finally:
        del os.environ["PHASEFORGET_EXPERIMENT_ID"]


def test_experiment_id_parameter_overrides_env():
    """Test parameter takes precedence over environment variable."""
    os.environ["PHASEFORGET_EXPERIMENT_ID"] = "env-exp"
    try:
        settings = get_settings(experiment_id="param-exp")
        assert settings.experiment_id == "param-exp"
        assert settings.chroma_persist_dir == "./data/param-exp/chroma_db"
    finally:
        del os.environ["PHASEFORGET_EXPERIMENT_ID"]


if __name__ == "__main__":
    test_experiment_id_default()
    print("[PASS] test_experiment_id_default")

    test_experiment_id_custom()
    print("[PASS] test_experiment_id_custom")

    test_experiment_id_multiple()
    print("[PASS] test_experiment_id_multiple")

    test_experiment_id_from_env()
    print("[PASS] test_experiment_id_from_env")

    test_experiment_id_parameter_overrides_env()
    print("[PASS] test_experiment_id_parameter_overrides_env")

    print("\nAll tests passed!")
