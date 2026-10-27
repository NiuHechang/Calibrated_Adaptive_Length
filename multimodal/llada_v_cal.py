"""CAL score-only and official decoding adapter for LLaDA-V."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image


from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


DEFAULT_INSTRUCTION = "Provide a one-sentence caption for the provided image."
DEFAULT_SYSTEM = (
    "You are a helpful language and vision assistant. You are able to understand "
    "the visual content that the user provides, and assist the user with a variety "
    "of tasks using natural language."
)


@dataclass
class LLaDAVContext:
    input_ids: torch.Tensor
    image_tensor: list[torch.Tensor]
    image_size: tuple[int, int]
    context_embeddings: torch.Tensor


@dataclass
class CanvasScore:
    canvas_ids: torch.Tensor
    mask_logits: torch.Tensor
    top1_token_ids: torch.Tensor
    token_confidences: torch.Tensor
    confidence: float


class LLaDAVCALScorer:
    def __init__(self, checkpoint, device="cuda:0", vision_checkpoint="google/siglip2-so400m-patch14-384"):
        self.device = torch.device(device)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            str(checkpoint),
            None,
            "llava_llada",
            attn_implementation="sdpa",
            device_map=device,
            torch_dtype="float16",
            overwrite_config={"mm_vision_tower": str(vision_checkpoint)},
        )
        self.model.eval()
        self.core_model = self.model.get_model()
        self.mask_id = self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>")

    def build_prompt(self, instruction):
        question = DEFAULT_IMAGE_TOKEN + "\n" + instruction
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

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
            tensor.to(dtype=torch.float16, device=self.device)
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

        return LLaDAVContext(
            input_ids=input_ids,
            image_tensor=image_tensor,
            image_size=image.size,
            context_embeddings=context_embeddings,
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
        canvas_embeddings = self.core_model.embed_tokens(canvas_ids)
        full_embeddings = torch.cat(
            [context.context_embeddings, canvas_embeddings],
            dim=1,
        )
        context_length = context.context_embeddings.shape[1]
        with torch.cuda.amp.autocast(enabled=True):
            hidden_states = self.core_model(inputs_embeds=full_embeddings)[0]
            mask_hidden_states = hidden_states[
                :,
                context_length + mask_start : context_length + mask_end,
            ]
            mask_logits = self.model.lm_head(mask_hidden_states).float()
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
        prefix_ids = torch.tensor(
            prefix_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        prefix_embeddings = self.core_model.embed_tokens(prefix_ids)
        inputs_embeds = torch.cat(
            [context.context_embeddings, prefix_embeddings],
            dim=1,
        )
        suffix_text = self.tokenizer.decode(
            suffix_ids,
            skip_special_tokens=False,
        )
        generated_ids = self.model.generate_with_embeds(
            inputs_embeds=inputs_embeds,
            steps=min(decode_steps, length),
            gen_length=length,
            block_length=length,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=self.mask_id,
            tokenizer=self.tokenizer,
            generation_suffix=suffix_text,
        )
        return generated_ids[:, :length]
