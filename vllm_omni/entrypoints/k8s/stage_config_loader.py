# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage configuration loader for K8s deployment."""

import os
from pathlib import Path
from typing import Any

import yaml
from vllm.logger import init_logger

logger = init_logger(__name__)


def load_stage_config(config_path: str | Path, stage_id: int) -> dict[str, Any]:
    """Load stage configuration from YAML file.

    Args:
        config_path: Path to stage configuration YAML file
        stage_id: Stage ID to extract from config

    Returns:
        Dictionary containing stage configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If stage_id not found in config
        yaml.YAMLError: If config file is not valid YAML
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Stage config file not found: {config_path}")

    logger.info(f"Loading stage config from: {config_path}")

    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    if not isinstance(full_config, dict):
        raise ValueError(f"Invalid config format: expected dict, got {type(full_config)}")

    # Extract stage_args list
    stage_args = full_config.get("stage_args", [])
    if not isinstance(stage_args, list):
        raise ValueError(f"Invalid stage_args format: expected list, got {type(stage_args)}")

    # Find stage config for this stage_id
    stage_config = None
    for stage in stage_args:
        if stage.get("stage_id") == stage_id:
            stage_config = stage
            break

    if stage_config is None:
        raise ValueError(f"Stage ID {stage_id} not found in config. Available stages: "
                         f"{[s.get('stage_id') for s in stage_args]}")

    # Merge with runtime config if present
    runtime_config = full_config.get("runtime", {})

    result = {
        "stage_id": stage_id,
        "stage_config": stage_config,
        "runtime_config": runtime_config,
    }

    logger.info(f"Loaded config for stage {stage_id}: type={stage_config.get('stage_type')}, "
                f"model_stage={stage_config.get('engine_args', {}).get('model_stage')}")

    return result


def get_stage_config_path() -> str:
    """Get stage config path from environment or default location.

    Returns:
        Path to stage config file

    Raises:
        ValueError: If STAGE_CONFIG_PATH not set
    """
    config_path = os.getenv("STAGE_CONFIG_PATH")
    if not config_path:
        raise ValueError("STAGE_CONFIG_PATH environment variable not set")

    return config_path


def get_stage_id() -> int:
    """Get stage ID from environment.

    Returns:
        Stage ID

    Raises:
        ValueError: If STAGE_ID not set or invalid
    """
    stage_id_str = os.getenv("STAGE_ID")
    if not stage_id_str:
        raise ValueError("STAGE_ID environment variable not set")

    try:
        stage_id = int(stage_id_str)
    except ValueError as e:
        raise ValueError(f"Invalid STAGE_ID: {stage_id_str}") from e

    return stage_id
