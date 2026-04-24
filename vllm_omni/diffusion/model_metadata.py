# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionModelMetadata:
    # Keep serving-facing capability metadata in a lightweight shared module so
    # config/model plumbing can read it without importing concrete pipelines.
    supports_multimodal_inputs: bool = False
    max_multimodal_image_inputs: int | None = None
    max_multimodal_text_tokens: int | None = None


QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES = 4


_DIFFUSION_MODEL_METADATA: dict[str, DiffusionModelMetadata] = {
    "Flux2Pipeline": DiffusionModelMetadata(
        max_multimodal_text_tokens=512,
    ),
    "QwenImagePipeline": DiffusionModelMetadata(
        max_multimodal_text_tokens=1024,
    ),
    "QwenImageLayeredPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=1,
        max_multimodal_text_tokens=1024,
    ),
    "QwenImageEditPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=1,
        max_multimodal_text_tokens=1024,
    ),
    "QwenImageEditPlusPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES,
        max_multimodal_text_tokens=1024,
    ),
}


def get_diffusion_model_metadata(model_class_name: str | None) -> DiffusionModelMetadata:
    # Unknown models fall back to "no special multimodal capabilities" so new
    # pipelines do not accidentally inherit limits meant for other models.
    if model_class_name is None:
        return DiffusionModelMetadata()
    return _DIFFUSION_MODEL_METADATA.get(model_class_name, DiffusionModelMetadata())
