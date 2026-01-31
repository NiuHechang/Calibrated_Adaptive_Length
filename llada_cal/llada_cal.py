import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

"""
CAL (Calibrated Adaptive Length) framework adapted for LLaDA.
This script performs training-free infilling length discovery before formal decoding.
See Section 4: Method in the paper.
"""

def add_gumbel_noise(logits, temperature):
    """
    Gumbel-max sampling for categorical distributions.
    High-precision float64 is used to maintain generation quality.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    """
    Computes the number of tokens to transition at each denoising step.
    LLaDA uses a linear noise schedule.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

def get_masked_avg_conf(x0_p, mask_index, prefix_len, curr_gen_length):
    """Computes the average confidence Phi(L) for the masked positions."""
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
def generate(model, prefix_ids, suffix_ids=None, attention_mask=None, suffix_attention_mask=None, steps=None, gen_length=8, block_length=None, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False,
             span=1, max_gen_length=64, dstep=4, use_bias=True):
    """
    Adaptive length generation for LLaDA using CAL.
    Stage 1: Discover optimal infilling length via calibrated confidence.
    Stage 2: Standard iterative denoising with the discovered length.
    """
    prefix_len = prefix_ids.shape[1]
    suffix_len = suffix_ids.shape[1] if suffix_ids is not None else 0
    
    # --- Stage 1: Length Discovery (Probing) ---
    best_len = gen_length
    search_steps = 0
    
    if dstep >= 0:
        def get_conf_at_len(l):
            nonlocal search_steps
            search_steps += 1
            if l < 1: return -1.0
            
            # Construct probing sequence: [P; [MASK]^L; S]
            x_t = torch.full((prefix_ids.shape[0], prefix_len + l + suffix_len), mask_id, dtype=torch.long).to(model.device)
            x_t[:, :prefix_len] = prefix_ids.clone()
            if suffix_ids is not None:
                x_t[:, prefix_len + l:] = suffix_ids.clone()

            if attention_mask is not None:
                m_mask = torch.ones((prefix_ids.shape[0], l), dtype=attention_mask.dtype, device=model.device)
                if suffix_ids is not None:
                    s_mask = suffix_attention_mask if suffix_attention_mask is not None else torch.ones((prefix_ids.shape[0], suffix_len), dtype=attention_mask.dtype, device=model.device)
                    curr_att = torch.cat([attention_mask, m_mask, s_mask], dim=-1)
                else:
                    curr_att = torch.cat([attention_mask, m_mask], dim=-1)
            else:
                curr_att = None

            p_idx = (x_t != mask_id)
            m_idx = (x_t == mask_id)

            if cfg_scale > 0.:
                un_x = x_t.clone(); un_x[p_idx] = mask_id
                x_ = torch.cat([x_t, un_x], dim=0)
                a_mask_ = torch.cat([curr_att, curr_att], dim=0) if curr_att is not None else None
                logits = model(x_, attention_mask=a_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x_t, attention_mask=curr_att).logits

            if logits_eos_inf: logits[:, :, 126081] = -torch.inf
            p = F.softmax(logits, dim=-1)
            x0_t = torch.argmax(logits, dim=-1)
            x0_p_t = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0_t, -1)), -1)
            
            # Raw Phi(L)
            curr_conf = get_masked_avg_conf(x0_p_t, m_idx, prefix_len, l).mean().item()
            
            # Calibration: Phi_norm(L) = Phi(L) / B(L)
            if use_bias:
                curr_conf /= get_bias(l)
                
            return curr_conf

        # Bidirectional Hill-Climbing Search
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
    curr_steps = steps if steps is not None else curr_gen_length
    curr_block_length = block_length if block_length is not None else curr_gen_length
    
    x = torch.full((prefix_ids.shape[0], prefix_len + curr_gen_length + suffix_len), mask_id, dtype=torch.long).to(model.device)
    x[:, :prefix_len] = prefix_ids.clone()
    if suffix_ids is not None:
        x[:, prefix_len + curr_gen_length:] = suffix_ids.clone()

    if attention_mask is not None:
        middle_mask = torch.ones((prefix_ids.shape[0], curr_gen_length), dtype=attention_mask.dtype, device=model.device)
        if suffix_ids is not None:
            s_mask = suffix_attention_mask if suffix_attention_mask is not None else torch.ones((prefix_ids.shape[0], suffix_len), dtype=attention_mask.dtype, device=model.device)
            curr_attention_mask = torch.cat([attention_mask, middle_mask, s_mask], dim=-1)
        else:
            curr_attention_mask = torch.cat([attention_mask, middle_mask], dim=-1)
    else:
        curr_attention_mask = None

    prompt_index = (x != mask_id)
    num_blocks = curr_gen_length // curr_block_length
    steps_per_block = curr_steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prefix_len + num_block * curr_block_length
        block_end = prefix_len + (num_block + 1) * curr_block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone(); un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                a_mask_ = torch.cat([curr_attention_mask, curr_attention_mask], dim=0) if curr_attention_mask is not None else None
                logits = model(x_, attention_mask=a_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=curr_attention_mask).logits

            if logits_eos_inf: logits[:, :, 126081] = -torch.inf
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            if confidence_eos_eot_inf: logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x, search_steps

if __name__ == '__main__':
    # --- Execution Logic ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'GSAI-ML/LLaDA-8B-Instruct'
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    prefix_code = "def add_elements(a, b):\n    \"\"\"Sum lists a and b.\"\"\"\n"
    suffix_code = "    return [x + y for x, y in zip(a, b)]"

    encoded_prefix = tokenizer([prefix_code], add_special_tokens=False, padding=True, return_tensors="pt")
    encoded_suffix = tokenizer([suffix_code], add_special_tokens=False, padding=True, return_tensors="pt")
    
    prefix_ids = encoded_prefix['input_ids'].to(device)
    prefix_mask = encoded_prefix['attention_mask'].to(device)
    suffix_ids = encoded_suffix['input_ids'].to(device)
    suffix_mask = encoded_suffix['attention_mask'].to(device)

    # Execute generation with CAL
    out, s_steps = generate(model, prefix_ids=prefix_ids, suffix_ids=suffix_ids, attention_mask=prefix_mask, suffix_attention_mask=suffix_mask, 
                   steps=8, gen_length=8, block_length=8, temperature=0., cfg_scale=0., remasking='low_confidence',
                   span=1, max_gen_length=64, dstep=4, use_bias=True)

    print(f"\n--- Result (Search Steps {s_steps}) ---")
    print(tokenizer.decode(out[0], skip_special_tokens=True))
