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

from vllm_omni.inputs.preprocess import OmniInputPreprocessor

SYSTEM_MESSAGE = (
    "You are an AI that reasons about image descriptions. You give structured "
    "responses focusing on object relationships, object attribution and actions "
    "without speculation."
)

MAX_SEQUENCE_LENGTH = 512


class MistralDiffusionPreprocessor(OmniInputPreprocessor):

    def _tokenize_prompt(
        self,
        prompt: str,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        tokenizer = self.get_tokenizer()
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)

        if len(token_ids) > MAX_SEQUENCE_LENGTH:
            token_ids = token_ids[-MAX_SEQUENCE_LENGTH:]

        return token_ids
