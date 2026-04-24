# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus import (
    get_qwen_image_edit_plus_pre_process_func,
)
from vllm_omni.exceptions import OmniInputValidationError

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _make_od_config(tmp_path: Path, *, max_multimodal_image_inputs=None):
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir(exist_ok=True)
    (vae_dir / "config.json").write_text(json.dumps({"z_dim": 16}))
    return SimpleNamespace(model=str(tmp_path), max_multimodal_image_inputs=max_multimodal_image_inputs)


def test_qwen_image_edit_plus_rejects_too_many_input_images(tmp_path: Path):
    od_config = _make_od_config(tmp_path)
    pre_process = get_qwen_image_edit_plus_pre_process_func(od_config)
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    request = SimpleNamespace(
        prompts=[
            {
                "prompt": "combine",
                "multi_modal_data": {"image": [image, image, image, image, image]},
            }
        ],
        sampling_params=SimpleNamespace(height=None, width=None),
    )

    with pytest.raises(OmniInputValidationError, match=r"At most 4 images are supported"):
        pre_process(request)


def test_qwen_image_edit_plus_rejects_images_exceeding_config_limit(tmp_path: Path):
    od_config = _make_od_config(tmp_path, max_multimodal_image_inputs=2)
    pre_process = get_qwen_image_edit_plus_pre_process_func(od_config)
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    request = SimpleNamespace(
        prompts=[
            {
                "prompt": "combine",
                "multi_modal_data": {"image": [image, image, image]},
            }
        ],
        sampling_params=SimpleNamespace(height=None, width=None),
    )

    with pytest.raises(OmniInputValidationError, match=r"At most 2 images are supported"):
        pre_process(request)
