# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.stage_input_processors.lingbot_video import (
    _has_any_hint,
    _prune_negative,
    rewriter_to_dit,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _map_output(text: str):
    return SimpleNamespace(
        outputs=[SimpleNamespace(text=text)],
    )


SAMPLE_CAPTION = json.dumps(
    {
        "comprehensive_description": {
            "scene_content_description": "A dark moody alley at night.",
            "camera_movement_description": "",
        },
        "prominent_elements": [],
        "camera_info": {
            "color": "Cool",
            "frame_size": "Wide",
            "shot_type_angle": "Eye level",
            "lens_size": "Wide",
            "composition": "Center",
            "lighting": "Low contrast",
            "lighting_type": "Artificial light",
        },
    }
)


class TestHasAnyHint:
    def test_match(self):
        assert _has_any_hint("a dark moody scene", ("dark", "dim"))

    def test_no_match(self):
        assert not _has_any_hint("a bright sunny day", ("dark", "dim"))

    def test_case_insensitive(self):
        assert _has_any_hint("DARK alley", ("dark",))

    def test_multi_word_hint(self):
        assert _has_any_hint("a scene with motion blur effect", ("motion blur",))


class TestPruneNegative:
    def test_dark_scene_removes_underexposed(self):
        neg = json.loads(_prune_negative("a dark moody alley at night"))
        vq = neg["universal_negative"]["visual_quality"]
        assert "underexposed" not in vq
        assert "subject hidden in darkness" not in vq
        assert "crushed blacks" not in vq
        assert "low quality" in vq

    def test_motion_blur_removes_motion_blur(self):
        neg = json.loads(_prune_negative("a car with motion blur"))
        tms = neg["universal_negative"]["temporal_and_motion_stability"]
        assert "motion blur" not in tms
        assert "flickering" in tms

    def test_fantasy_removes_physical_plausibility(self):
        neg = json.loads(_prune_negative("a magical fantasy world"))
        assert "physical_plausibility" not in neg["universal_negative"]

    def test_cartoon_removes_artistic_style(self):
        neg = json.loads(_prune_negative("a cartoon character walking"))
        assert "artistic_style" not in neg["universal_negative"]

    def test_neutral_prompt_keeps_all(self):
        neg = json.loads(_prune_negative("a person walking on a sidewalk"))
        cats = neg["universal_negative"]
        assert "physical_plausibility" in cats
        assert "artistic_style" in cats
        assert "underexposed" in cats["visual_quality"]
        assert "motion blur" in cats["temporal_and_motion_stability"]


class TestRewriterToDit:
    def test_empty_outputs(self):
        assert rewriter_to_dit([]) is None

    def test_empty_text(self):
        assert rewriter_to_dit([_map_output("")]) is None
        assert rewriter_to_dit([_map_output("   ")]) is None

    def test_valid_json_caption(self):
        result = rewriter_to_dit([_map_output(SAMPLE_CAPTION)])
        assert result is not None
        assert "prompt" in result
        assert "negative_prompt" in result
        parsed = json.loads(result["prompt"])
        assert "comprehensive_description" in parsed

    def test_json_repair_fixes_trailing_comma(self):
        broken = '{"key": "value",}'
        result = rewriter_to_dit([_map_output(broken)])
        assert result is not None
        parsed = json.loads(result["prompt"])
        assert parsed == {"key": "value"}

    def test_negative_prompt_pruned_for_dark(self):
        result = rewriter_to_dit([_map_output(SAMPLE_CAPTION)])
        assert result is not None
        neg = json.loads(result["negative_prompt"])
        vq = neg["universal_negative"]["visual_quality"]
        assert "underexposed" not in vq
        assert "crushed blacks" not in vq

    def test_raw_text_fallback(self):
        result = rewriter_to_dit([_map_output("not json at all")])
        assert result is not None
        assert result["prompt"] == "not json at all"
        assert "negative_prompt" in result
