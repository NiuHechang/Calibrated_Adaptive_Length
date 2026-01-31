import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from generation_utils import DreamGenerationConfig

"""
CAL (Calibrated Adaptive Length) framework adapted for DreamCoder.
This script performs training-free infilling length discovery before formal decoding.
See Section 4: Method in the paper.
"""

def get_masked_avg_conf(x0_p, mask_index, prefix_len, curr_gen_length):
    """Computes the average confidence Phi(L) for the masked region."""
    mask_region_mask = mask_index[:, prefix_len : prefix_len + curr_gen_length]
    mask_region_conf = x0_p[:, prefix_len : prefix_len + curr_gen_length]
    
    sum_conf = (mask_region_conf * mask_region_mask.float()).sum(dim=-1)
    count_mask = mask_region_mask.sum(dim=-1).float().clamp(min=1)
    return sum_conf / count_mask

def get_bias(l):
    """Length Bias function B(L) for calibration. See Appendix A."""
    # Parameters: a=1.00, b=1.77, c=0.56, d=0.06, e=0.24
    a, b, c, d, e = 1.00, 1.77, 0.56, 0.06, 0.24
    return a * np.exp(-b * l) + c * np.exp(-d * l) + e

@ torch.no_grad()
def generate_fim(model, tokenizer, prefix, suffix, gen_length=8, steps=None, temperature=0.0, alg="entropy",
                 span=1, max_gen_length=64, dstep=4, use_bias=True):
    """
    FIM generation for DreamCoder with calibrated confidence search (CAL).
    """
    device = model.device
    mask_token_id = getattr(model.config, "mask_token_id", 151666)
    
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    prefix_len = len(prefix_ids)
    suffix_len = len(suffix_ids)

    # --- Stage 1: Length Discovery (Probing) ---
    best_len = gen_length
    search_steps = 0  
    
    if dstep >= 0:
        def get_conf_at_len(l):
            nonlocal search_steps
            search_steps += 1
            if l < 1: return -1.0
            
            p_tensor = torch.tensor([prefix_ids], device=device, dtype=torch.long)
            s_tensor = torch.tensor([suffix_ids], device=device, dtype=torch.long)
            m_tensor = torch.full((1, l), mask_token_id, device=device, dtype=torch.long)
            x_t = torch.cat([p_tensor, m_tensor, s_tensor], dim=1)
            
            att_mask = torch.ones_like(x_t)
            tok_idx = att_mask.long().cumsum(-1) - 1
            att_mask_4d = torch.logical_and(
                att_mask.unsqueeze(1).unsqueeze(-2),
                att_mask.unsqueeze(1).unsqueeze(-1),
            )

            # First-step forward pass
            logits = model(x_t, attention_mask=att_mask_4d, position_ids=tok_idx).logits
            # Alignment correction for DreamCoder
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            p = F.softmax(logits, dim=-1)
            x0_p_t, _ = p.max(dim=-1)
            
            m_idx = (x_t == mask_token_id)
            curr_conf = get_masked_avg_conf(x0_p_t, m_idx, prefix_len, l).mean().item()
            
            # Calibration: Phi_norm(L) = Phi(L) / B(L)
            if use_bias:
                curr_conf /= get_bias(l)
                
            return curr_conf

        # Bidirectional Hill-Climbing Search (See Algorithm 1)
        best_conf = get_conf_at_len(gen_length)
        best_len = gen_length
        
        # Search upwards
        curr_l = gen_length + span
        consecutive_decreases = 0
        while curr_l <= max_gen_length:
            curr_conf = get_conf_at_len(curr_l)
            if curr_conf > best_conf:
                best_conf, best_len = curr_conf, curr_l
                consecutive_decreases = 0
            else:
                consecutive_decreases += 1
                if consecutive_decreases >= dstep: break
            curr_l += span

        # Search downwards
        curr_l = gen_length - span
        consecutive_decreases = 0
        while curr_l >= 1:
            curr_conf = get_conf_at_len(curr_l)
            if curr_conf > best_conf:
                best_conf, best_len = curr_conf, curr_l
                consecutive_decreases = 0
            else:
                consecutive_decreases += 1
                if consecutive_decreases >= dstep: break
            curr_l -= span

    # --- Stage 2: Formal Decoding ---
    curr_gen_length = best_len
    prefix_tensor = torch.tensor([prefix_ids], device=device, dtype=torch.long)
    suffix_tensor = torch.tensor([suffix_ids], device=device, dtype=torch.long)
    mask_tensor = torch.full((1, curr_gen_length), mask_token_id, device=device, dtype=torch.long)
    input_ids = torch.cat([prefix_tensor, mask_tensor, suffix_tensor], dim=1)
    
    gen_config = DreamGenerationConfig(
        max_length=input_ids.shape[1],
        steps=steps if steps is not None else curr_gen_length,
        temperature=temperature,
        alg=alg,
        mask_token_id=mask_token_id,
        return_dict_in_generate=True,
        output_history=False,
        top_p=0.95
    )
    
    output = model._sample(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        generation_config=gen_config,
        generation_tokens_hook_func=lambda step, x, logits: x,
        generation_logits_hook_func=lambda step, x, logits: logits
    )
    
    gen_ids = output.sequences[0, prefix_len : prefix_len + curr_gen_length]
    return tokenizer.decode(gen_ids, skip_special_tokens=True), curr_gen_length, search_steps

if __name__ == "__main__":
    # --- Execution Logic ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "Dream-org/Dream-Coder-v0-Base-7B"
    
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device).eval()

    prefix_code = "def multiply(a, b):\n    \"\"\"Return the product of a and b.\"\"\"\n"
    suffix_code = "    return a * b"
    
    # Execute generation with CAL
    completion, final_len, s_steps = generate_fim(model, tokenizer, prefix=prefix_code, suffix=suffix_code, 
        gen_length=8, steps=None, alg="entropy", span=1, max_gen_length=64, dstep=4, use_bias=True)

    print(f"\n--- Result (Length {final_len}, Search Steps {s_steps}) ---")
    print(f"{prefix_code}{completion}{suffix_code}")