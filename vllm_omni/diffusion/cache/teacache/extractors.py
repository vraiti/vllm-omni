# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Model-specific extractors for TeaCache.

This module provides a registry of extractor functions that know how to extract
modulated inputs from different transformer architectures. Adding support for
a new model requires only adding a new extractor function to the registry.

With Option B enhancement, extractors now return a CacheContext object containing
all model-specific information needed for generic caching, including preprocessing,
transformer execution, and postprocessing logic.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class CacheContext:
    """
    Context object containing all model-specific information for caching.

    This allows the TeaCacheHook to remain completely generic - all model-specific
    logic is encapsulated in the extractor that returns this context.

    Attributes:
        modulated_input: Tensor used for cache decision (similarity comparison).
            Must be a torch.Tensor extracted from the first transformer block,
            typically after applying normalization and modulation.

        hidden_states: Current hidden states (will be modified by caching).
            Must be a torch.Tensor representing the main image/latent states
            after preprocessing but before transformer blocks.

        encoder_hidden_states: Optional encoder states (for dual-stream models).
            Set to None for single-stream models (e.g., Flux).
            For dual-stream models (e.g., Qwen), contains text encoder outputs.

        temb: Timestep embedding tensor.
            Must be a torch.Tensor containing the timestep conditioning.

        run_transformer_blocks: Callable that executes model-specific transformer blocks.
            Signature: () -> tuple[torch.Tensor, ...]

            Returns:
                tuple containing:
                - [0]: processed hidden_states (required)
                - [1]: processed encoder_hidden_states (optional, only for dual-stream)

            Example for single-stream:
                def run_blocks():
                    h = hidden_states
                    for block in module.transformer_blocks:
                        h = block(h, temb=temb)
                    return (h,)

            Example for dual-stream:
                def run_blocks():
                    h, e = hidden_states, encoder_hidden_states
                    for block in module.transformer_blocks:
                        e, h = block(h, e, temb=temb)
                    return (h, e)

        postprocess: Callable that does model-specific output postprocessing.
            Signature: (torch.Tensor) -> Union[torch.Tensor, Transformer2DModelOutput, tuple]

            Takes the processed hidden_states and applies final transformations
            (normalization, projection) to produce the model output.

            Example:
                def postprocess(h):
                    h = module.norm_out(h, temb)
                    output = module.proj_out(h)
                    return Transformer2DModelOutput(sample=output)

        extra_states: Optional dict for additional model-specific state.
            Use this for models that need to pass additional context beyond
            the standard fields.
    """

    modulated_input: torch.Tensor
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor | None
    temb: torch.Tensor | None
    run_transformer_blocks: Callable[[], tuple[torch.Tensor, ...]]
    postprocess: Callable[[torch.Tensor], Any]
    extra_states: dict[str, Any] | None = None

    def validate(self) -> None:
        """
        Validate that the CacheContext contains valid data.

        Raises:
            TypeError: If fields have wrong types
            ValueError: If tensors have invalid properties
            RuntimeError: If callables fail basic invocation tests

        This method should be called after creating a CacheContext to catch
        common developer errors early with clear error messages.
        """
        # Validate tensor fields
        if not isinstance(self.modulated_input, torch.Tensor):
            raise TypeError(f"modulated_input must be torch.Tensor, got {type(self.modulated_input)}")

        if not isinstance(self.hidden_states, torch.Tensor):
            raise TypeError(f"hidden_states must be torch.Tensor, got {type(self.hidden_states)}")

        if self.encoder_hidden_states is not None and not isinstance(self.encoder_hidden_states, torch.Tensor):
            raise TypeError(
                f"encoder_hidden_states must be torch.Tensor or None, got {type(self.encoder_hidden_states)}"
            )

        if self.temb is not None and not isinstance(self.temb, torch.Tensor):
            raise TypeError(f"temb must be torch.Tensor or None, got {type(self.temb)}")

        # Validate callables
        if not callable(self.run_transformer_blocks):
            raise TypeError(f"run_transformer_blocks must be callable, got {type(self.run_transformer_blocks)}")

        if not callable(self.postprocess):
            raise TypeError(f"postprocess must be callable, got {type(self.postprocess)}")

        # Validate tensor shapes are compatible
        if self.modulated_input.shape[0] != self.hidden_states.shape[0]:
            raise ValueError(
                f"Batch size mismatch: modulated_input has batch size "
                f"{self.modulated_input.shape[0]}, but hidden_states has "
                f"{self.hidden_states.shape[0]}"
            )

        # Validate devices match
        if self.modulated_input.device != self.hidden_states.device:
            raise ValueError(
                f"Device mismatch: modulated_input on {self.modulated_input.device}, "
                f"hidden_states on {self.hidden_states.device}"
            )


def extract_qwen_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    timestep: torch.Tensor | float | int,
    img_shapes: torch.Tensor,
    txt_seq_lens: torch.Tensor,
    guidance: torch.Tensor | None = None,
    additional_t_cond: torch.Tensor | None = None,
    attention_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for QwenImageTransformer2DModel.

    Delegates preprocessing, block execution, and postprocessing to the model's
    own private methods (_preprocess, _run_blocks, _postprocess) so that logic
    is not duplicated here. The only TeaCache-specific logic is the extraction
    of the modulated input from the first transformer block.

    Args:
        module: QwenImageTransformer2DModel instance
        hidden_states: Input hidden states tensor
        encoder_hidden_states: Text encoder outputs
        encoder_hidden_states_mask: Mask for text encoder
        timestep: Current diffusion timestep
        img_shapes: Image shapes for position embedding
        txt_seq_lens: Text sequence lengths
        guidance: Optional guidance scale for CFG
        additional_t_cond: Optional additional timestep conditioning
        attention_kwargs: Additional attention arguments
        **kwargs: Additional keyword arguments ignored by this extractor

    Returns:
        CacheContext with all information needed for generic caching
    """
    if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
        raise ValueError("Module must have transformer_blocks")

    return_dict = kwargs.get("return_dict", True)

    # Delegate all preprocessing to the model
    state = module._preprocess(
        hidden_states, encoder_hidden_states, encoder_hidden_states_mask,
        timestep, img_shapes, txt_seq_lens, guidance, additional_t_cond,
        attention_kwargs, return_dict,
    )

    # Extract modulated input from the first transformer block (TeaCache-specific)
    block = module.transformer_blocks[0]
    img_mod_params = block.img_mod(state.temb)
    img_mod1, _ = img_mod_params.chunk(2, dim=-1)
    img_modulated, _ = block.img_norm1(state.hidden_states, img_mod1, state.modulate_index)

    def run_transformer_blocks():
        h, e = module._run_blocks(state)
        return (h, e)

    def postprocess(h):
        return module._postprocess(h, state.temb, state.return_dict)

    return CacheContext(
        modulated_input=img_modulated,
        hidden_states=state.hidden_states,
        encoder_hidden_states=state.encoder_hidden_states,
        temb=state.temb,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


def extract_bagel_context(
    module: nn.Module,
    x_t: torch.Tensor,
    timestep: torch.Tensor | float | int,
    packed_vae_token_indexes: torch.LongTensor,
    packed_vae_position_ids: torch.LongTensor,
    packed_text_ids: torch.LongTensor,
    packed_text_indexes: torch.LongTensor,
    packed_indexes: torch.LongTensor,
    packed_position_ids: torch.LongTensor,
    packed_seqlens: torch.IntTensor,
    key_values_lens: torch.IntTensor,
    past_key_values: Any,
    packed_key_value_indexes: torch.LongTensor,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for Bagel model.

    Delegates preprocessing to the model's _preprocess method. Uses _run_blocks
    for the single non-CFG forward path and _postprocess for output projection.

    Args:
        module: Bagel instance
        x_t: Latent image input
        timestep: Current timestep
        packed_vae_token_indexes: Indexes for VAE tokens in packed sequence
        packed_vae_position_ids: Position IDs for VAE tokens
        packed_text_ids: Text token IDs
        packed_text_indexes: Indexes for text tokens in packed sequence
        packed_indexes: Global indexes
        packed_position_ids: Global position IDs
        packed_seqlens: Sequence lengths
        key_values_lens: KV cache lengths
        past_key_values: KV cache
        packed_key_value_indexes: KV cache indexes
        **kwargs: Additional keyword arguments

    Returns:
        CacheContext with all information needed for generic caching
    """
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor([timestep], device=x_t.device)
    if timestep.dim() == 0:
        timestep = timestep.unsqueeze(0)

    state = module._preprocess(
        x_t, timestep, packed_vae_token_indexes, packed_vae_position_ids,
        packed_text_ids, packed_text_indexes, packed_seqlens,
    )

    def run_transformer_blocks():
        h = module._run_blocks(
            state, packed_seqlens, packed_position_ids, packed_indexes,
            past_key_values, key_values_lens, packed_key_value_indexes,
            packed_vae_token_indexes, packed_text_indexes,
        )
        return (h,)

    def postprocess(h):
        return module._postprocess(h, packed_vae_token_indexes)

    return CacheContext(
        modulated_input=state.packed_sequence,
        hidden_states=state.packed_sequence,
        encoder_hidden_states=None,
        temb=state.packed_timestep_embeds,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


def extract_zimage_context(
    module: nn.Module,
    x: list[torch.Tensor],
    t: torch.Tensor,
    cap_feats: list[torch.Tensor],
    patch_size: int = 2,
    f_patch_size: int = 1,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for ZImageTransformer2DModel.

    Delegates preprocessing to the model's _preprocess method. The only
    TeaCache-specific logic is the modulated input extraction from the first
    main transformer block.

    Args:
        module: ZImageTransformer2DModel instance
        x: List of image tensors per batch item
        t: Timestep tensor
        cap_feats: List of caption feature tensors per batch item
        patch_size: Patch size for patchification (default: 2)
        f_patch_size: Frame patch size (default: 1)
        **kwargs: Additional keyword arguments ignored by this extractor

    Returns:
        CacheContext with all information needed for generic caching
    """
    if not hasattr(module, "layers") or len(module.layers) == 0:
        raise ValueError("Module must have main transformer layers")

    state = module._preprocess(x, t, cap_feats, patch_size, f_patch_size)

    # Extract modulated input from the first main transformer block (TeaCache-specific)
    block = module.layers[0]
    mod_params = block.adaLN_modulation(state.adaln_input).unsqueeze(1).chunk(4, dim=2)
    scale_msa = 1.0 + mod_params[0]
    modulated_input = block.attention_norm1(state.unified) * scale_msa

    def run_transformer_blocks():
        h = module._run_blocks(state)
        return (h,)

    def postprocess(h):
        return module._postprocess(h, state)

    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=state.unified,
        encoder_hidden_states=None,
        temb=state.adaln_input,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


def extract_flux2_klein_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for Flux2Klein model.

    Delegates preprocessing to the model's _preprocess method. Uses _run_blocks
    for the full dual+single stream execution path.

    Args:
        module: Flux2Transformer2DModel instance (Klein variant)
        hidden_states: Input image hidden states tensor
        encoder_hidden_states: Input text hidden states tensor
        timestep: Current diffusion timestep
        img_ids: Image position IDs for RoPE
        txt_ids: Text position IDs for RoPE
        guidance: Optional guidance scale for CFG
        joint_attention_kwargs: Additional attention kwargs

    Returns:
        CacheContext with all information needed for generic caching
    """
    if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
        raise ValueError("Module must have transformer_blocks")

    return_dict = kwargs.get("return_dict", True)

    state = module._preprocess(
        hidden_states, encoder_hidden_states, timestep,
        img_ids, txt_ids, guidance, joint_attention_kwargs, return_dict,
    )

    # Extract modulated input from the first dual-stream block (TeaCache-specific)
    block = module.transformer_blocks[0]
    norm_hidden_states = block.norm1(state.hidden_states)
    modulated_input = (1 + state.double_stream_mod_img[0][1]) * norm_hidden_states + state.double_stream_mod_img[0][0]

    def run_flux2_full_transformer_with_single(ori_h, ori_c):
        h = module._run_blocks(state)
        return h, ori_c

    def postprocess(h):
        return module._postprocess(h, state.temb, state.return_dict)

    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=state.hidden_states,
        encoder_hidden_states=state.encoder_hidden_states,
        temb=state.temb,
        run_transformer_blocks=lambda: None,
        postprocess=postprocess,
        extra_states={
            "run_flux2_full_transformer_with_single": run_flux2_full_transformer_with_single,
        },
    )


def extract_stable_audio_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    global_hidden_states: torch.Tensor | None = None,
    rotary_embedding: tuple[torch.Tensor, torch.Tensor] | None = None,
    return_dict: bool = True,
    attention_mask: torch.Tensor | None = None,
    encoder_attention_mask: torch.Tensor | None = None,
    **kwargs: Any,
) -> CacheContext:
    # Cast inputs to match model weights dtype
    hidden_states = hidden_states.to(module.dtype)
    encoder_hidden_states = encoder_hidden_states.to(module.dtype)
    if global_hidden_states is not None:
        global_hidden_states = global_hidden_states.to(module.dtype)

    state = module._preprocess(
        hidden_states, timestep, encoder_hidden_states,
        global_hidden_states, rotary_embedding, return_dict,
        attention_mask, encoder_attention_mask,
    )

    # Stable Audio prepends the combined global+time embedding to the sequence.
    # The standard LayerNorm captures the timestep signal within the first token,
    # giving the cache discriminator the information it needs.
    first_block = module.transformer_blocks[0]
    modulated_input = first_block.norm1(state.hidden_states)

    def run_transformer_blocks() -> tuple[torch.Tensor]:
        h = module._run_blocks(state)
        return (h,)

    def postprocess(h: torch.Tensor) -> Any:
        return module._postprocess(h, state.return_dict)

    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=state.hidden_states,
        encoder_hidden_states=None,
        temb=None,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


def extract_flux2_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
    **kwargs: Any,
) -> CacheContext:
    state = module._preprocess(
        hidden_states, encoder_hidden_states, timestep,
        img_ids, txt_ids, guidance, joint_attention_kwargs, return_dict,
    )

    block = module.transformer_blocks[0]
    (shift_msa, scale_msa, gate_msa), _ = state.double_stream_mod_img
    modulated_input = block.norm1(state.hidden_states)
    modulated_input = (1 + scale_msa) * modulated_input + shift_msa

    def run_transformer_blocks() -> tuple[torch.Tensor]:
        h = module._run_blocks(state)
        return (h,)

    def postprocess(h: torch.Tensor) -> Any:
        return module._postprocess(h, state.temb, state.return_dict)

    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=state.hidden_states,
        encoder_hidden_states=state.encoder_hidden_states,
        temb=state.temb,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


# Registry for model-specific extractors
# Key: Transformer class name
# Value: extractor function with signature (module, *args, **kwargs) -> CacheContext
#
# Note: Use the transformer class name as specified in pipelines as TeaCache hooks operate
# on the transformer module and multiple pipelines can share the same transformer.
EXTRACTOR_REGISTRY: dict[str, Callable] = {
    "QwenImageTransformer2DModel": extract_qwen_context,
    "Bagel": extract_bagel_context,
    "ZImageTransformer2DModel": extract_zimage_context,
    "Flux2Klein": extract_flux2_klein_context,
    "StableAudioDiTModel": extract_stable_audio_context,
    "Flux2Transformer2DModel": extract_flux2_context,
    # Future models:
    # "FluxTransformer2DModel": extract_flux_context,
    # "CogVideoXTransformer3DModel": extract_cogvideox_context,
}


def register_extractor(transformer_cls_name: str, extractor_fn: Callable) -> None:
    """
    Register a new extractor function for a model type.

    This allows extending TeaCache support to new models without modifying
    the core TeaCache code.

    Args:
        transformer_cls_name: Transformer model type identifier (class name or type string)
        extractor_fn: Function with signature (module, *args, **kwargs) -> CacheContext

    Example:
        >>> def extract_flux_context(module, hidden_states, timestep, guidance=None, **kwargs):
        ...     # Preprocessing
        ...     temb = module.time_text_embed(timestep, guidance)
        ...     # Extract modulated input
        ...     modulated = module.transformer_blocks[0].norm1(hidden_states, emb=temb)
        ...     # Define execution
        ...     def run_blocks():
        ...         h = hidden_states
        ...         for block in module.transformer_blocks:
        ...             h = block(h, temb=temb)
        ...         return (h,)
        ...     # Define postprocessing
        ...     def postprocess(h):
        ...         return module.proj_out(module.norm_out(h, temb))
        ...     # Return context
        ...     return CacheContext(modulated, hidden_states, None, temb, run_blocks, postprocess)
        >>> register_extractor("FluxTransformer2DModel", extract_flux_context)
    """
    EXTRACTOR_REGISTRY[transformer_cls_name] = extractor_fn


def get_extractor(transformer_cls_name: str) -> Callable:
    """
    Get extractor function for given transformer class.

    This function looks up the extractor based on the exact transformer_cls_name string,
    which should match the transformer type in the pipeline (i.e., pipeline.transformer.__class__.__name__).

    Args:
        transformer_cls_name: Transformer class name (e.g., "QwenImageTransformer2DModel")
                                Must exactly match a key in EXTRACTOR_REGISTRY.

    Returns:
        Extractor function with signature (module, *args, **kwargs) -> CacheContext

    Raises:
        ValueError: If model type not found in registry

    Example:
        >>> # Get extractor for QwenImageTransformer2DModel
        >>> extractor = get_extractor("QwenImageTransformer2DModel")
        >>> ctx = extractor(transformer, hidden_states, encoder_hidden_states, timestep, ...)
    """
    # Direct lookup - no substring matching
    if transformer_cls_name in EXTRACTOR_REGISTRY:
        return EXTRACTOR_REGISTRY[transformer_cls_name]

    # No match found
    available_types = list(EXTRACTOR_REGISTRY.keys())
    raise ValueError(
        f"Unknown model type: '{transformer_cls_name}'. "
        f"Available types: {available_types}\n"
        f"To add support for a new model, use register_extractor() or add to EXTRACTOR_REGISTRY."
    )
