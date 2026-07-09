# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from LingBot-Video (https://github.com/Robbyant/lingbot-video).

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.lingbot_video.pipeline_lingbot_video import (
    IMG_PROMPT_TEMPLATE,
    LingBotVideoPipeline,
    _batch_cfg_prompt_inputs,
    _compute_refiner_sigmas,
    _extract_prompt,
    _group_global_rank,
    _module_device,
    _module_dtype,
    _transformer_autocast,
    _transformer_timestep,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    max_pixels = max_pixels if max_pixels is not None else IMAGE_MAX_TOKEN_NUM * factor**2
    min_pixels = min_pixels if min_pixels is not None else IMAGE_MIN_TOKEN_NUM * factor**2
    if max_pixels < min_pixels:
        raise ValueError("max_pixels must be greater than or equal to min_pixels.")
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}.")

    resized_height = max(factor, _round_by_factor(height, factor))
    resized_width = max(factor, _round_by_factor(width, factor))
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = _floor_by_factor(height / beta, factor)
        resized_width = _floor_by_factor(width / beta, factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * beta, factor)
        resized_width = _ceil_by_factor(width * beta, factor)
    return resized_height, resized_width


def _pixel_tensor_to_pil(pixel: torch.Tensor) -> PIL.Image.Image:
    frame = pixel[0, :, 0].detach().cpu().clamp(0, 1)
    array = frame.permute(1, 2, 0).mul(255).byte().numpy()
    return PIL.Image.fromarray(array, mode="RGB")


def get_lingbot_video_i2v_pre_process_func(od_config: OmniDiffusionConfig):
    del od_config

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
        raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
        if raw_image is None:
            raise ValueError(
                "No image provided. This model requires an image. "
                'Set "multi_modal_data": {"image": <image>} in the prompt.'
            )
        if not isinstance(raw_image, (str, PIL.Image.Image)):
            raise TypeError(f"Unsupported image format {type(raw_image)}.")
        image = PIL.Image.open(raw_image).convert("RGB") if isinstance(raw_image, str) else raw_image

        if request.sampling_params.height is None or request.sampling_params.width is None:
            max_area = 480 * 832
            aspect_ratio = image.height / image.width
            mod_value = 16
            height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
            if request.sampling_params.height is None:
                request.sampling_params.height = height
            if request.sampling_params.width is None:
                request.sampling_params.width = width

        image = image.resize(
            (request.sampling_params.width, request.sampling_params.height),
            PIL.Image.Resampling.LANCZOS,
        )
        multi_modal_data["image"] = image
        return request

    return pre_process_func


def get_lingbot_video_i2v_post_process_func(od_config: OmniDiffusionConfig):
    del od_config

    def post_process_func(frames: torch.Tensor, sampling_params=None):
        output_type = getattr(sampling_params, "output_type", None) or "pt"
        if output_type == "np" and isinstance(frames, torch.Tensor):
            return frames.float().cpu().numpy()
        return frames

    return post_process_func


class LingBotVideoI2VPipeline(LingBotVideoPipeline, SupportImageInput):
    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.img_prompt_template = IMG_PROMPT_TEMPLATE

    @staticmethod
    def _apply_inpainting(latents: torch.Tensor, cond_latent: torch.Tensor) -> torch.Tensor:
        cond_t = cond_latent.shape[2]
        latents[:, :, :cond_t] = cond_latent.float()
        return latents

    def preprocess_image(self, image: PIL.Image.Image, height: int, width: int) -> torch.Tensor:
        raw = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).contiguous()
        old_h, old_w = raw.shape[-2:]
        scale = max(height / old_h, width / old_w)
        new_h = max(math.ceil(old_h * scale), height)
        new_w = max(math.ceil(old_w * scale), width)
        resized = F.interpolate(raw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        top = int(round((new_h - height) / 2.0))
        left = int(round((new_w - width) / 2.0))
        cropped = resized[:, :, top : top + height, left : left + width].float() / 255.0
        return cropped.unsqueeze(2)

    def _vision_patch_size(self) -> int:
        for obj in (
            getattr(getattr(self.text_encoder, "config", None), "vision_config", None),
            getattr(getattr(self.processor, "image_processor", None), "config", None),
            getattr(self.processor, "image_processor", None),
        ):
            patch = getattr(obj, "patch_size", None)
            if patch is not None:
                return int(patch)
        return 16

    def _vlm_image(self, pixel: torch.Tensor) -> PIL.Image.Image:
        image = _pixel_tensor_to_pil(pixel)
        patch_factor = self._vision_patch_size() * SPATIAL_MERGE_SIZE
        width, height = image.size
        resized_height, resized_width = smart_resize(height, width, factor=patch_factor)
        return image.resize((resized_width, resized_height))

    def _build_prompt_inputs_with_image(self, prompt: str | list[str], images: list | None = None):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        visual_template = self.img_prompt_template if images is not None else ""
        texts = [self.apply_text_to_template(visual_template + text, self.prompt_template) for text in prompts]
        return self.processor(
            text=texts,
            images=images,
            videos=None,
            do_resize=False,
            truncation=True,
            max_length=self.token_length,
            padding="longest",
            return_tensors="pt",
        )

    @torch.no_grad()
    def encode_prompt_with_image(
        self,
        prompt: str | list[str],
        *,
        images: list | None = None,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device) if device is not None else self.device
        inputs = self._build_prompt_inputs_with_image(prompt, images=images).to(device)
        outputs = self.text_encoder(
            **inputs,
            output_hidden_states=self.hidden_state_skip_layer is not None,
        )
        if self.hidden_state_skip_layer is not None:
            prompt_embeds = outputs.hidden_states[-(self.hidden_state_skip_layer + 1)]
        else:
            prompt_embeds = outputs.last_hidden_state

        prompt_mask = inputs["attention_mask"]
        crop_start = self._compute_crop_start()
        if crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_mask = prompt_mask[:, crop_start:]

        if prompt_embeds.shape[0] == 1:
            true_len = int(prompt_mask[0].sum().item())
            prompt_embeds = prompt_embeds[:, :true_len]
            prompt_mask = prompt_mask[:, :true_len]
        return prompt_embeds, prompt_mask

    @torch.no_grad()
    def encode_image_latent(
        self,
        pixel: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = _module_device(self.vae)
        pixel = pixel.to(device=device, dtype=torch.float32)
        norm_pixel = (pixel - 0.5) / 0.5
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            latents = self.vae.encode(norm_pixel).latent_dist.sample(generator)

        mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=torch.float32)
        std_inv = 1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=torch.float32)
        mean = mean.view(1, -1, 1, 1, 1)
        std_inv = std_inv.view(1, -1, 1, 1, 1)
        return (latents.float() - mean) * std_inv

    @torch.no_grad()
    def _generate(
        self,
        *,
        prompt: str,
        image: PIL.Image.Image,
        negative_prompt: str,
        height: int = 480,
        width: int = 480,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 6.0,
        shift: float = 3.0,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pt",
        cfg_parallel_group: Any | None = None,
        batch_cfg: bool = False,
        null_cond_clone_zero: bool = False,
        t_thresh: float | None = None,
        refiner_sigma_tail_steps: int = 2,
        offload_vae_during_denoise: bool = False,
        **extra_args,
    ) -> torch.Tensor:
        del extra_args
        self.check_inputs(height, width, num_frames)
        device = self.device
        do_cfg = guidance_scale > 1.0
        effective_batch_cfg = bool(batch_cfg)
        cfg_parallel = cfg_parallel_group is not None
        cfg_parallel_rank = 0
        if cfg_parallel:
            import torch.distributed as dist

            if not dist.is_available() or not dist.is_initialized():
                raise ValueError("`cfg_parallel_group` requires an initialized process group.")
            if effective_batch_cfg:
                raise ValueError("`cfg_parallel_group` and `batch_cfg` are mutually exclusive.")
            if not do_cfg:
                raise ValueError("CFG parallel requires `guidance_scale > 1.0`.")
            cfg_parallel_rank = dist.get_rank(cfg_parallel_group)
            cfg_parallel_world_size = dist.get_world_size(cfg_parallel_group)
            if cfg_parallel_world_size != 2:
                raise ValueError(f"CFG parallel currently requires exactly 2 ranks, got {cfg_parallel_world_size}.")

        pixel = self.preprocess_image(image, height, width).to(device=device, dtype=torch.float32)
        vlm_image = self._vlm_image(pixel)

        negative_embeds = None
        negative_mask = None
        if cfg_parallel and cfg_parallel_rank == 1:
            negative_embeds, negative_mask = self.encode_prompt_with_image(
                negative_prompt, images=[vlm_image], device=device
            )
            prompt_embeds = prompt_mask = None
        else:
            prompt_embeds, prompt_mask = self.encode_prompt_with_image(prompt, images=[vlm_image], device=device)
            if do_cfg and not cfg_parallel:
                if null_cond_clone_zero:
                    negative_embeds = torch.zeros_like(prompt_embeds)
                    negative_mask = prompt_mask.clone()
                else:
                    negative_embeds, negative_mask = self.encode_prompt_with_image(
                        negative_prompt, images=[vlm_image], device=device
                    )

        cond_latent = self.encode_image_latent(pixel, generator=generator)
        cond_latent = cond_latent.to(device=device, dtype=torch.float32)

        latents = self.prepare_latents(num_frames, height, width, generator, latents, device)
        latents = self._apply_inpainting(latents, cond_latent)

        sigmas = _compute_refiner_sigmas(
            sigma_max=float(self.scheduler.sigma_max),
            sigma_min=float(self.scheduler.sigma_min),
            num_inference_steps=num_inference_steps,
            shift=shift,
            t_thresh=t_thresh,
            tail_steps=refiner_sigma_tail_steps,
        )
        if sigmas is None:
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        else:
            self.scheduler.set_timesteps(int(sigmas.shape[0]), device=device, sigmas=sigmas, shift=1.0)

        transformer_dtype = _module_dtype(self.transformer)
        vae_restore_device: torch.device | None = None
        vae_offloaded = False
        if offload_vae_during_denoise and output_type != "latent":
            vae_device = _module_device(self.vae)
            if vae_device.type == "cuda":
                self.vae.to("cpu")
                torch.accelerator.empty_cache()
                vae_restore_device = vae_device
                vae_offloaded = True

        cfg_latent_src = _group_global_rank(cfg_parallel_group, 0)
        cfg_uncond_src = _group_global_rank(cfg_parallel_group, 1)
        for timestep in self.progress_bar(self.scheduler.timesteps):
            if cfg_parallel:
                import torch.distributed as dist

                dist.broadcast(latents, src=cfg_latent_src, group=cfg_parallel_group)
            timestep_batch = _transformer_timestep(timestep, transformer_dtype).expand(1).to(device)
            latent_model_input = latents
            if cfg_parallel:
                import torch.distributed as dist

                if cfg_parallel_rank == 0:
                    branch_embeds = prompt_embeds
                    branch_mask = prompt_mask
                else:
                    branch_embeds = negative_embeds
                    branch_mask = negative_mask
                if branch_embeds is None:
                    raise RuntimeError("CFG branch embeddings were not initialized.")
                with _transformer_autocast(device, transformer_dtype):
                    branch_noise_pred = self.transformer(
                        latent_model_input,
                        timestep_batch,
                        branch_embeds.to(transformer_dtype),
                        encoder_attention_mask=branch_mask,
                        return_dict=False,
                    )[0].float()
                if cfg_parallel_rank == 0:
                    noise_pred = branch_noise_pred
                    noise_pred_uncond = torch.empty_like(noise_pred)
                else:
                    noise_pred_uncond = branch_noise_pred
                dist.broadcast(noise_pred_uncond, src=cfg_uncond_src, group=cfg_parallel_group)
                if cfg_parallel_rank != 0:
                    continue
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            else:
                if prompt_embeds is None:
                    raise RuntimeError("Prompt embeddings were not initialized.")
                prompt_model_input = prompt_embeds.to(transformer_dtype)
                if do_cfg and effective_batch_cfg:
                    if negative_embeds is None or negative_mask is None:
                        raise RuntimeError("Negative embeddings were not initialized for CFG.")
                    cfg_embeds, cfg_mask = _batch_cfg_prompt_inputs(
                        prompt_model_input,
                        prompt_mask,
                        negative_embeds.to(transformer_dtype),
                        negative_mask,
                        null_cond_clone_zero=False,
                    )
                    cfg_latents = torch.cat([latent_model_input, latent_model_input], dim=0)
                    cfg_timesteps = torch.cat([timestep_batch, timestep_batch], dim=0)
                    with _transformer_autocast(device, transformer_dtype):
                        noise_batched = self.transformer(
                            cfg_latents,
                            cfg_timesteps,
                            cfg_embeds,
                            encoder_attention_mask=cfg_mask,
                            return_dict=False,
                        )[0].float()
                    noise_pred, noise_pred_uncond = noise_batched.chunk(2, dim=0)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
                else:
                    with _transformer_autocast(device, transformer_dtype):
                        noise_pred = self.transformer(
                            latent_model_input,
                            timestep_batch,
                            prompt_model_input,
                            encoder_attention_mask=prompt_mask,
                            return_dict=False,
                        )[0].float()

                if do_cfg and not effective_batch_cfg:
                    if negative_embeds is None or negative_mask is None:
                        raise RuntimeError("Negative embeddings were not initialized for CFG.")
                    with _transformer_autocast(device, transformer_dtype):
                        noise_pred_uncond = self.transformer(
                            latent_model_input,
                            timestep_batch,
                            negative_embeds.to(transformer_dtype),
                            encoder_attention_mask=negative_mask,
                            return_dict=False,
                        )[0].float()
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False, generator=generator)[0]
            latents = self._apply_inpainting(latents, cond_latent)

        if cfg_parallel:
            import torch.distributed as dist

            dist.barrier(group=cfg_parallel_group)
            if cfg_parallel_rank != 0:
                return latents if output_type == "latent" else []

        if output_type == "latent":
            return latents
        if output_type in {"pt", "np"}:
            if vae_offloaded and vae_restore_device is not None:
                self.vae.to(device=vae_restore_device)
                torch.accelerator.empty_cache()
            return self._decode_latents(latents)
        raise ValueError(f"Unsupported output_type: {output_type}")

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if req.num_reqs != 1:
            raise ValueError(f"LingBotVideoI2VPipeline only supports one request per batch, got {req.num_reqs}.")
        request = req.requests[0]
        prompt, prompt_negative = _extract_prompt(request)
        sampling = request.sampling_params
        extra_args = dict(sampling.extra_args or {})

        prompt_obj = request.prompt
        multi_modal_data = prompt_obj.get("multi_modal_data", {}) if not isinstance(prompt_obj, str) else None
        raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
        if raw_image is None:
            raise ValueError("Image is required for I2V generation.")
        if isinstance(raw_image, list):
            raw_image = raw_image[0]
        if isinstance(raw_image, str):
            image = PIL.Image.open(raw_image).convert("RGB")
        elif isinstance(raw_image, PIL.Image.Image):
            image = raw_image
        else:
            raise TypeError(f"Unsupported image type: {type(raw_image)}.")

        generator = sampling.generator
        if isinstance(generator, list):
            generator = generator[0] if generator else None
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(sampling.seed))

        height = sampling.height if sampling.height is not None else extra_args.pop("height", 480)
        width = sampling.width if sampling.width is not None else extra_args.pop("width", 480)
        num_frames = sampling.num_frames or extra_args.pop("num_frames", 81)
        num_inference_steps = (
            sampling.num_inference_steps
            if sampling.num_inference_steps is not None
            else extra_args.pop("num_inference_steps", 40)
        )
        guidance_scale = (
            sampling.guidance_scale
            if sampling.guidance_scale_provided or sampling.guidance_scale > 0
            else extra_args.pop("guidance_scale", 6.0)
        )
        shift = extra_args.pop(
            "shift",
            extra_args.pop("flow_shift", getattr(self.od_config, "flow_shift", None) or 3.0),
        )
        negative_prompt = extra_args.pop("negative_prompt", prompt_negative or self.default_negative_prompt)
        output_type = (
            sampling.output_type or getattr(self.od_config, "output_type", None) or extra_args.pop("output_type", "pt")
        )
        if output_type not in {"pt", "np", "latent"}:
            output_type = "pt"
        sampling.output_type = output_type

        frames = self._generate(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            generator=generator,
            latents=sampling.latents,
            output_type=output_type,
            batch_cfg=bool(extra_args.pop("batch_cfg", False)),
            null_cond_clone_zero=bool(extra_args.pop("null_cond_clone_zero", False)),
            offload_vae_during_denoise=bool(extra_args.pop("offload_vae_during_denoise", False)),
            t_thresh=extra_args.pop("t_thresh", None),
            refiner_sigma_tail_steps=int(extra_args.pop("refiner_sigma_tail_steps", 2)),
        )
        return DiffusionOutput(output=frames)
