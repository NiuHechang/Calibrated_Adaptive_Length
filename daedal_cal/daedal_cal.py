import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

"""
Implementation of the DAEDAL baseline for Diffusion Language Model infilling.
DAEDAL is a confidence-guided adaptive length method originally proposed for 
open-ended generation. We adapt it for infilling tasks as a comparison baseline.
See Section 5.2: Code Infilling Results in the paper.
"""

def add_gumbel_noise(logits, temperature):
    """
    Gumbel-max sampling for categorical distributions.
    """
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

@torch.no_grad()
def generate(model, tokenizer, prefix_ids, suffix_ids=None, enable_stage1=False, enable_stage2=True, initial_gen_length=8, max_gen_length=64, 
            block_length=32, temperature=0.0, cfg_scale=0.0, high_conf_threshold=0.90, low_conf_threshold=0.30, expansion_factor=2,
            mask_id=126336, eos_token_id=126081, eos_confidence_threshold=0.5, expand_eos_confidence_threshold=0.9, eos_check_tokens=8,
):
    """
    Generate tokens using the DAEDAL heuristic.
    Stage 1: Initial length adjustment based on <eos> confidence.
    Stage 2: Runtime expansion based on low-confidence tokens during denoising.
    """
    def _calculate_eos_confidence(logits, gen_lengths, prefix_len, eos_check_tokens):
        if eos_token_id is None:
            return torch.zeros(logits.shape[0], device=logits.device)
        confidences = F.softmax(logits, dim=-1)
        predicted_tokens = torch.argmax(logits, dim=-1)
        batch_eos_confidences = []
        for i in range(logits.shape[0]):
            eos_confs_for_avg = []
            # Detection position is at the junction of middle content and suffix
            start_scan_pos = prefix_len + gen_lengths[i].item() - 1
            end_scan_pos = prefix_len - 1
            for pos in range(start_scan_pos, end_scan_pos, -1):
                if len(eos_confs_for_avg) >= eos_check_tokens:
                    break
                if predicted_tokens[i, pos] == eos_token_id:
                    eos_confs_for_avg.append(confidences[i, pos, eos_token_id].item())
            avg_conf = sum(eos_confs_for_avg) / eos_check_tokens if eos_confs_for_avg else 0.0
            batch_eos_confidences.append(avg_conf)
        return torch.tensor(batch_eos_confidences, device=logits.device)

    with torch.autocast(device_type="cuda"):
        batch_size = prefix_ids.shape[0]
        device = prefix_ids.device
        prefix_len = prefix_ids.shape[1]
        suffix_len = suffix_ids.shape[1] if suffix_ids is not None else 0
        
        gen_lengths = torch.full((batch_size,), initial_gen_length, dtype=torch.long, device=device)
        # Initialize sequence: [prefix, masks, suffix]
        x = torch.full(
            (batch_size, prefix_len + initial_gen_length + suffix_len),
            mask_id,
            dtype=torch.long,
            device=device,
        )
        x[:, :prefix_len] = prefix_ids.clone()
        if suffix_ids is not None:
            x[:, prefix_len + initial_gen_length:] = suffix_ids.clone()
        
        prompt_index = x != mask_id

        reached_max_stage1 = torch.zeros(batch_size, dtype=torch.bool, device=device)
        reached_max_stage2 = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # --- Stage 1: Initial Length Adjustment (based on EOS signal) ---
        if enable_stage1:
            while True:
                total_len = x.shape[1]
                attention_mask = torch.ones((batch_size, total_len), device=device, dtype=torch.long)
                
                logits_pre = model(x, attention_mask=attention_mask).logits
                batch_eos_confidences = _calculate_eos_confidence(logits_pre, gen_lengths, prefix_len, eos_check_tokens)
                
                reached_max_stage1 = reached_max_stage1 | ((gen_lengths >= max_gen_length) & (batch_eos_confidences < eos_confidence_threshold))

                sequences_to_expand = (batch_eos_confidences < eos_confidence_threshold) & (gen_lengths < max_gen_length)
                
                new_gen_lengths = gen_lengths.clone()
                new_gen_lengths[sequences_to_expand] = torch.clamp(gen_lengths[sequences_to_expand] + expansion_factor, max=max_gen_length)
                
                if (new_gen_lengths == gen_lengths).all():
                    break
                
                max_new_total_len = prefix_len + new_gen_lengths.max() + suffix_len
                new_x_tensor = torch.full((batch_size, max_new_total_len), mask_id, dtype=torch.long, device=device)
                
                for i in range(batch_size):
                    new_x_tensor[i, :prefix_len] = prefix_ids[i]
                    new_x_tensor[i, prefix_len : prefix_len + gen_lengths[i]] = x[i, prefix_len : prefix_len + gen_lengths[i]]
                    if suffix_ids is not None:
                        new_x_tensor[i, prefix_len + new_gen_lengths[i] : prefix_len + new_gen_lengths[i] + suffix_len] = suffix_ids[i]
                
                x = new_x_tensor
                gen_lengths = new_gen_lengths
                prompt_index = x != mask_id

        # Prepare for Stage 2
        max_new_total_len = prefix_len + gen_lengths.max() + suffix_len
        intermediate_x_tensor = torch.full((batch_size, max_new_total_len), mask_id, dtype=torch.long, device=device)
        for i in range(batch_size):
            intermediate_x_tensor[i, :prefix_len] = prefix_ids[i]
            if suffix_ids is not None:
                intermediate_x_tensor[i, prefix_len + gen_lengths[i] : prefix_len + gen_lengths[i] + suffix_len] = suffix_ids[i]
        x = intermediate_x_tensor
        prompt_index = x != mask_id

        # --- Stage 2: Iterative Denoising and Runtime Expansion ---
        current_pos = torch.full((batch_size,), prefix_len, dtype=torch.long, device=device)
        denoise_only_mode = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        while (current_pos < prefix_len + gen_lengths).any():
            x_before_step = x.clone()
            total_len = x.shape[1]
            
            for i in range(batch_size):
                if gen_lengths[i] >= max_gen_length and not denoise_only_mode[i]:
                    if current_pos[i] < prefix_len + gen_lengths[i]:
                        reached_max_stage2[i] = True
                        denoise_only_mode[i] = True

            attention_mask = torch.ones((batch_size, total_len), device=device, dtype=torch.long)
            
            if cfg_scale > 0.0:
                un_x = x.clone(); un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits, un_logits = torch.chunk(model(x_, attention_mask=attention_mask_).logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            predicted_tokens = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
            confidences = F.softmax(logits, dim=-1)
            predicted_confidences = torch.gather(confidences, dim=-1, index=predicted_tokens.unsqueeze(-1)).squeeze(-1)
            batch_eos_confidences = _calculate_eos_confidence(logits, gen_lengths, prefix_len, eos_check_tokens)

            block_mask = torch.zeros_like(x, dtype=torch.bool, device=device)
            for i in range(batch_size):
                middle_end = prefix_len + gen_lengths[i]
                if current_pos[i] >= middle_end: continue
                block_mask[i, current_pos[i]:min(current_pos[i] + block_length, middle_end.item())] = True
            
            currently_masked = (x == mask_id)
            high_conf_indices = (predicted_confidences > high_conf_threshold) & block_mask & currently_masked & (predicted_tokens != mask_id)

            for i in range(batch_size):
                middle_end = prefix_len + gen_lengths[i]
                if current_pos[i] >= middle_end: continue
                start_idx, end_idx = current_pos[i], min(current_pos[i] + block_length, middle_end.item())
                if not high_conf_indices[i, start_idx:end_idx].any():
                    valid_fallback_mask = block_mask[i] & currently_masked[i]
                    if not valid_fallback_mask.any(): continue
                    candidate_indices = torch.where(valid_fallback_mask)[0]
                    candidate_confs = predicted_confidences[i, candidate_indices]
                    candidate_tokens = predicted_tokens[i, candidate_indices]
                    sorted_confs, sort_indices = torch.sort(candidate_confs, descending=True)
                    best_idx_to_fill = -1
                    for sorted_idx in sort_indices:
                        if candidate_tokens[sorted_idx] != mask_id:
                            best_idx_to_fill = candidate_indices[sorted_idx]; break
                    if best_idx_to_fill != -1:
                        high_conf_indices[i, best_idx_to_fill] = True
                    else:
                        stuck_logits = logits[i, candidate_indices]
                        stuck_logits[:, mask_id] = -torch.inf
                        new_confidences = F.softmax(stuck_logits, dim=-1)
                        new_best_confs, new_best_tokens = torch.max(new_confidences, dim=-1)
                        best_of_the_best_local_idx = torch.argmax(new_best_confs)
                        pos_to_fill = candidate_indices[best_of_the_best_local_idx]
                        token_to_fill = new_best_tokens[best_of_the_best_local_idx]
                        predicted_tokens[i, pos_to_fill] = token_to_fill
                        high_conf_indices[i, pos_to_fill] = True

            potential_expand_mask = (predicted_confidences < low_conf_threshold) & block_mask & currently_masked & (~high_conf_indices)
            expand_indices = torch.zeros_like(x, dtype=torch.bool, device=device)
            
            if enable_stage2:
                for i in range(batch_size):
                    if batch_eos_confidences[i] >= expand_eos_confidence_threshold or gen_lengths[i] >= max_gen_length: 
                        if gen_lengths[i] >= max_gen_length and batch_eos_confidences[i] < expand_eos_confidence_threshold:
                            reached_max_stage2[i] = True
                        continue
                    if denoise_only_mode[i] or current_pos[i] >= prefix_len + gen_lengths[i]: continue
                    masked_candidates = torch.where(potential_expand_mask[i])[0]
                    if len(masked_candidates) > 0:
                        candidate_confs = predicted_confidences[i, masked_candidates]
                        num_to_expand = min(1, len(masked_candidates))
                        if num_to_expand > 0:
                            _, lowest_conf_local_indices = torch.topk(candidate_confs, num_to_expand, largest=False)
                            indices_to_expand_global = masked_candidates[lowest_conf_local_indices]
                            expand_indices[i, indices_to_expand_global] = True
            
            fill_mask = high_conf_indices
            if not expand_indices.any():
                x[fill_mask] = predicted_tokens[fill_mask]
            else:
                x[fill_mask] = predicted_tokens[fill_mask]
                temp_new_gen_lengths = gen_lengths.clone()
                for i in range(batch_size):
                    expansion_count = expand_indices[i].sum().item()
                    if expansion_count > 0:
                        new_len = gen_lengths[i].item() + expansion_count * (expansion_factor - 1)
                        temp_new_gen_lengths[i] = min(new_len, max_gen_length)
                
                max_new_total_len = prefix_len + temp_new_gen_lengths.max() + suffix_len
                new_x_tensor = torch.full((batch_size, max_new_total_len), mask_id, device=device, dtype=torch.long)
                new_gen_lengths = torch.zeros_like(gen_lengths)

                for i in range(batch_size):
                    new_x_tensor[i, :prefix_len] = prefix_ids[i]
                    write_ptr = prefix_len
                    for j in range(prefix_len, prefix_len + gen_lengths[i].item()):
                        if write_ptr >= max_new_total_len - suffix_len: break
                        if expand_indices[i, j]:
                            end_write = min(write_ptr + expansion_factor, max_new_total_len - suffix_len)
                            new_x_tensor[i, write_ptr:end_write] = mask_id
                            write_ptr = end_write
                        else:
                            new_x_tensor[i, write_ptr] = x[i, j]
                            write_ptr += 1
                    new_gen_lengths[i] = write_ptr - prefix_len
                    if suffix_ids is not None:
                        new_x_tensor[i, write_ptr : write_ptr + suffix_len] = suffix_ids[i]
                x = new_x_tensor
                gen_lengths = new_gen_lengths
                prompt_index = x != mask_id

            for i in range(batch_size):
                middle_end = prefix_len + gen_lengths[i]
                while current_pos[i] < middle_end:
                    start_check = current_pos[i]
                    end_check = min(start_check + block_length, middle_end.item())
                    if start_check == end_check: break
                    if not (x[i, start_check:end_check] == mask_id).any():
                        current_pos[i] = start_check + block_length
                    else:
                        break
            if torch.equal(x, x_before_step):
                break
            
        final_outputs = []
        for i in range(batch_size):
            total_len_i = prefix_len + gen_lengths[i] + suffix_len
            final_outputs.append(x[i, :total_len_i])
        return final_outputs, reached_max_stage1, reached_max_stage2

if __name__ == '__main__':
    # --- Execution Logic ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'GSAI-ML/LLaDA-8B-Instruct'
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.padding_side != 'left': tokenizer.padding_side = 'left'

    prefix_code = "def add5(aa): return 5 + a"
    suffix_code = "" 

    encoded_prefixes = tokenizer([prefix_code], add_special_tokens=False, padding=True, return_tensors="pt")
    encoded_suffixes = tokenizer([suffix_code], add_special_tokens=False, padding=True, return_tensors="pt")
    
    prefix_ids = encoded_prefixes['input_ids'].to(device)
    suffix_ids = encoded_suffixes['input_ids'].to(device)

    # Execute generation with DAEDAL
    out_list, s1_max, s2_max = generate(model, tokenizer, prefix_ids=prefix_ids, suffix_ids=suffix_ids, 
                        enable_stage1=False, initial_gen_length=8, max_gen_length=64)

    for i, o in enumerate(out_list):
        print(f"\n--- Result {i+1} ---")
        p_len = prefix_ids.shape[1]
        s_len = suffix_ids.shape[1]
        middle_part = o[p_len : len(o) - s_len]
        
        completion = tokenizer.decode(middle_part, skip_special_tokens=True)
        print(f"Generated Completion:\n{completion}")
        
        full_text = tokenizer.decode(o, skip_special_tokens=True)
        print(f"Full Text Output:\n{full_text}")
        print('-' * 50)
