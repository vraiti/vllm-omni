"""Input preprocessor for MistralDiffusionEncoder.

Applies the Mistral chat template with a fixed system message before
tokenization, matching the reference diffusers pipeline's text encoding
(Flux2Pipeline._get_mistral_3_small_prompt_embeds).

No padding is applied here. vLLM uses causal attention without attention
masks, so padding tokens would contaminate real token hidden states.
The downstream stage input processor (encoder2diffusion) left-pads the
output *embeddings* with zeros to match the DiT's expected sequence
length and positional layout.
"""

from typing import Any

from vllm_omni.inputs.data import OmniTextPrompt, OmniTokenInputs, token_inputs_omni
from vllm_omni.inputs.preprocess import OmniInputPreprocessor

SYSTEM_MESSAGE = (
    "You are an AI that reasons about image descriptions. You give structured "
    "responses focusing on object relationships, object attribution and actions "
    "without speculation."
)

SYSTEM_MESSAGE_UPSAMPLING_T2I = (
    "You are an expert prompt engineer for FLUX.2 by Black Forest Labs. "
    "Rewrite user prompts to be more descriptive while strictly preserving "
    "their core subject and intent.\n\n"
    "Guidelines:\n"
    "1. Structure: Keep structured inputs structured (enhance within fields). "
    "Convert natural language to detailed paragraphs.\n"
    "2. Details: Add concrete visual specifics - form, scale, textures, "
    "materials, lighting (quality, direction, color), shadows, spatial "
    "relationships, and environmental context.\n"
    "3. Text in Images: Put ALL text in quotation marks, matching the "
    "prompt's language. Always provide explicit quoted text for objects that "
    "would contain text in reality (signs, labels, screens, etc.) - without "
    "it, the model generates gibberish.\n\n"
    "Output only the revised prompt and nothing else."
)

MAX_SEQUENCE_LENGTH = 512


class MistralDiffusionPreprocessor(OmniInputPreprocessor):

    def _tokenize_prompt(
        self,
        prompt: str,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        system_message: str | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        tokenizer = self.get_tokenizer()
        sys_msg = system_message if system_message is not None else SYSTEM_MESSAGE
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_msg}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)

        if len(token_ids) > MAX_SEQUENCE_LENGTH:
            token_ids = token_ids[-MAX_SEQUENCE_LENGTH:]

        return token_ids

    def _process_text(
        self,
        parsed_content: OmniTextPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> OmniTokenInputs:
        mm_kwargs = parsed_content.get("mm_processor_kwargs") or {}
        system_message = mm_kwargs.get("system_message")

        if system_message is None:
            return super()._process_text(parsed_content, tokenization_kwargs)

        add_generation_prompt = mm_kwargs.get("add_generation_prompt", False)
        prompt_text = parsed_content["prompt"]
        prompt_token_ids = self._tokenize_prompt(
            prompt_text,
            tokenization_kwargs=tokenization_kwargs,
            system_message=system_message,
            add_generation_prompt=add_generation_prompt,
        )
        inputs = token_inputs_omni(prompt_token_ids)
        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt
        return inputs
