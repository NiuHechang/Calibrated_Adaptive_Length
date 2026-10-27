"""CAL score-only and official decoding adapter for LaViDa."""

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


DEFAULT_INSTRUCTION = "Provide a one-sentence caption for the provided image."

VISION_KWARGS = {
    "mm_vision_tower": "google/siglip-so400m-patch14-384",
    "mm_resampler_type": None,
    "mm_projector_type": "mlp2x_gelu",
    "mm_hidden_size": 1152,
    "use_mm_proj": True,
}


@dataclass
class LaViDaContext:
    input_ids: torch.Tensor
    image_tensor: list[torch.Tensor]
    image_size: tuple[int, int]
    context_embeddings: torch.Tensor
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class CanvasScore:
    canvas_ids: torch.Tensor
    mask_logits: torch.Tensor
    top1_token_ids: torch.Tensor
    token_confidences: torch.Tensor
    confidence: float


class LaViDaCALScorer:
    def __init__(self, checkpoint, device="cuda:0", vision_checkpoint="google/siglip-so400m-patch14-384"):
        self.device = torch.device(device)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            str(checkpoint),
            None,
            "llava_llada",
            device_map=device,
            vision_kwargs={**VISION_KWARGS, "mm_vision_tower": str(vision_checkpoint)},
            torch_dtype="bfloat16",
        )
        self.model.eval()
        self.model.tie_weights()
        self.model.to(torch.bfloat16)
        self.core_model = self.model.get_model()
        self.mask_id = self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>")

    def build_prompt(self, instruction):
        question = DEFAULT_IMAGE_TOKEN + "\n" + instruction
        conversation = copy.deepcopy(conv_templates["llada"])
        conversation.append_message(conversation.roles[0], question)
        conversation.append_message(conversation.roles[1], None)
        return conversation.get_prompt()

    @torch.inference_mode()
    def encode_context(
        self,
        image_path,
        instruction=DEFAULT_INSTRUCTION,
        resize_to=None,
    ):
        image = Image.open(image_path).convert("RGB")
        if resize_to is not None:
            image = image.resize(resize_to, Image.Resampling.BICUBIC)
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [
            tensor.to(dtype=torch.bfloat16, device=self.device)
            for tensor in image_tensor
        ]

        prompt = self.build_prompt(instruction)
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)

        _, _, _, _, context_embeddings, _ = (
            self.model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                None,
                None,
                None,
                image_tensor,
                ["image"],
                [image.size],
            )
        )
        past_key_values = self.core_model(
            None,
            input_embeddings=context_embeddings,
            use_cache=True,
        ).attn_key_values

        return LaViDaContext(
            input_ids=input_ids,
            image_tensor=image_tensor,
            image_size=image.size,
            context_embeddings=context_embeddings,
            past_key_values=past_key_values,
        )

    def build_canvas(self, prefix_ids, suffix_ids, length):
        canvas_ids = torch.tensor(
            prefix_ids + [self.mask_id] * length + suffix_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        mask_start = len(prefix_ids)
        mask_end = mask_start + length
        return canvas_ids, mask_start, mask_end

    @torch.inference_mode()
    def score_length(self, context, prefix_ids, suffix_ids, length):
        canvas_ids, mask_start, mask_end = self.build_canvas(
            prefix_ids,
            suffix_ids,
            length,
        )
        canvas_embeddings = self.core_model.transformer.wte(canvas_ids)
        logits = self.core_model(
            None,
            input_embeddings=canvas_embeddings,
            past_key_values=context.past_key_values,
        ).logits
        mask_logits = logits[:, mask_start:mask_end]
        top1_token_ids = mask_logits.argmax(dim=-1)
        probabilities = F.softmax(mask_logits.to(torch.float64), dim=-1)
        token_confidences = probabilities.gather(
            dim=-1,
            index=top1_token_ids.unsqueeze(-1),
        ).squeeze(-1)

        return CanvasScore(
            canvas_ids=canvas_ids,
            mask_logits=mask_logits,
            top1_token_ids=top1_token_ids,
            token_confidences=token_confidences,
            confidence=token_confidences.mean().item(),
        )

    @torch.inference_mode()
    def decode_infilling(
        self,
        context,
        prefix_ids,
        suffix_ids,
        length,
        decode_steps,
    ):
        canvas_ids, mask_start, mask_end = self.build_canvas(
            prefix_ids,
            suffix_ids,
            length,
        )
        canvas_length = canvas_ids.shape[1]
        generated_ids = self.model.generate(
            context.input_ids,
            images=context.image_tensor,
            image_sizes=[context.image_size],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=canvas_length,
            block_length=canvas_length,
            step_per_block=decode_steps,
            tokenizer=self.tokenizer,
            prefix_lm=True,
            draft_tokens=canvas_ids,
            verbose=False,
        )
        return generated_ids[:, mask_start:mask_end]
