# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from pathlib import Path

import pytest

from vllm_omni.entrypoints.k8s.stage_config_loader import (
    get_stage_config_path,
    get_stage_id,
    load_stage_config,
)

# Sample config for testing
SAMPLE_CONFIG = """
stage_args:
  - stage_id: 0
    stage_type: llm
    runtime:
      process: true
      devices: "0"
    engine_args:
      model_stage: thinker
      gpu_memory_utilization: 0.8

  - stage_id: 1
    stage_type: diffusion
    runtime:
      process: true
      devices: "1"
    engine_args:
      model_stage: talker
      gpu_memory_utilization: 0.9

runtime:
  enabled: true
  defaults:
    window_size: -1
"""


@pytest.fixture
def config_file():
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(SAMPLE_CONFIG)
        config_path = f.name

    yield config_path

    # Cleanup
    Path(config_path).unlink()


class TestStageConfigLoader:
    """Tests for stage configuration loader."""

    def test_load_stage_config_stage_0(self, config_file):
        """Test loading config for stage 0."""
        result = load_stage_config(config_file, stage_id=0)

        assert result["stage_id"] == 0
        assert "stage_config" in result
        assert "runtime_config" in result

        stage_config = result["stage_config"]
        assert stage_config["stage_id"] == 0
        assert stage_config["stage_type"] == "llm"
        assert stage_config["engine_args"]["model_stage"] == "thinker"

        runtime_config = result["runtime_config"]
        assert runtime_config["enabled"] is True

    def test_load_stage_config_stage_1(self, config_file):
        """Test loading config for stage 1."""
        result = load_stage_config(config_file, stage_id=1)

        assert result["stage_id"] == 1
        stage_config = result["stage_config"]
        assert stage_config["stage_id"] == 1
        assert stage_config["stage_type"] == "diffusion"
        assert stage_config["engine_args"]["model_stage"] == "talker"

    def test_load_stage_config_invalid_stage_id(self, config_file):
        """Test error when loading non-existent stage ID."""
        with pytest.raises(ValueError, match="Stage ID 99 not found"):
            load_stage_config(config_file, stage_id=99)

    def test_load_stage_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_stage_config("/nonexistent/config.yaml", stage_id=0)

    def test_load_stage_config_invalid_yaml(self):
        """Test error when YAML is invalid."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [[[")
            invalid_path = f.name

        try:
            with pytest.raises(Exception):  # yaml.YAMLError
                load_stage_config(invalid_path, stage_id=0)
        finally:
            Path(invalid_path).unlink()

    def test_get_stage_config_path(self, monkeypatch):
        """Test getting config path from environment."""
        monkeypatch.setenv("STAGE_CONFIG_PATH", "/path/to/config.yaml")
        assert get_stage_config_path() == "/path/to/config.yaml"

    def test_get_stage_config_path_not_set(self, monkeypatch):
        """Test error when STAGE_CONFIG_PATH not set."""
        monkeypatch.delenv("STAGE_CONFIG_PATH", raising=False)
        with pytest.raises(ValueError, match="STAGE_CONFIG_PATH"):
            get_stage_config_path()

    def test_get_stage_id(self, monkeypatch):
        """Test getting stage ID from environment."""
        monkeypatch.setenv("STAGE_ID", "2")
        assert get_stage_id() == 2

    def test_get_stage_id_not_set(self, monkeypatch):
        """Test error when STAGE_ID not set."""
        monkeypatch.delenv("STAGE_ID", raising=False)
        with pytest.raises(ValueError, match="STAGE_ID"):
            get_stage_id()

    def test_get_stage_id_invalid(self, monkeypatch):
        """Test error when STAGE_ID is not a valid integer."""
        monkeypatch.setenv("STAGE_ID", "invalid")
        with pytest.raises(ValueError, match="Invalid STAGE_ID"):
            get_stage_id()
