import logging
from copy import deepcopy
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
logger = logging.getLogger(__name__)

class ParallelDecoding(object):

    def __init__(self, model: str, tokenizer: str, device: torch.device, using_norm=False, using_entropy=False):
        self.model = model
        self.tokenizer = tokenizer
        

        self.device = device
        self.using_norm = using_norm
        self.using_entropy = using_entropy
    

    def cut_off_expert_logits(self, alpha: float, source_logit: torch.FloatTensor):
        if alpha <= 0.:
            return source_logit
        cutoff_val = torch.log(torch.tensor(alpha + 1e-40)).to(self.device) + source_logit.max(dim=-1, keepdim=True).values 
        target_logit = source_logit.masked_fill(source_logit < cutoff_val, -float("inf")) 
        return target_logit
            
    def cal_kl_div(self, p:torch.FloatTensor, q:torch.FloatTensor):
        # d_kl(p||q); shape of logits_p is: (doc_size, vocab_size), logits_qq is: (doc_size, vocab_size)
        # p = torch.nn.functional.softmax(logits_p, dim=-1)
        # q = torch.nn.functional.softmax(logits_q, dim=-1)
        log_p = torch.log(p + 1e-40)
        log_q = torch.log(q + 1e-40)
        kl_value = (p * (log_p - log_q)).sum(dim = -1) # (doc_size)
        return kl_value
    
    def cal_jsd(self, p, q):
        m = 0.5 * (p + q)
        jsd = 0.5 * self.cal_kl_div(p, m) + 0.5 * self.cal_kl_div(q, m) #(doc_size)
        return jsd
    
    def get_max_jsd_dist(self, expert_logits, candidate_logits):
        expand_expert_logits = expert_logits.repeat(len(candidate_logits), 1)
        expand_expert_probs = torch.nn.functional.softmax(expand_expert_logits, dim=-1)
        candidate_probs = torch.nn.functional.softmax(candidate_logits, dim=-1)
        candidate_jsd = self.cal_jsd(expand_expert_probs, candidate_probs)
        # print(len(candidate_jsd), candidate_jsd)
        max_jsd_idx = candidate_jsd.argmax()
        return candidate_logits[max_jsd_idx]

    def get_ensemble_layer_logits(self, hidden_states, layer_indices: List[int]):
        lm_head = self.model.lm_head 
        if max(layer_indices) >= len(hidden_states):
            raise IndexError(f"Layer index {max(layer_indices)} is out of range for the model with {len(hidden_states)} layers.")
        last_token_states = torch.stack([hidden_states[idx][0, -1, :] for idx in layer_indices])
        last_token_states = last_token_states.mean(dim=0,keepdim=True) # (1, shape)
        logits = lm_head(last_token_states).squeeze() # (vocab_size,)
        return logits

    def calcuate_entropy(self, candidate_logits):
        candidate_probs = torch.nn.functional.softmax(candidate_logits, dim=-1)
        log_probs = torch.log(candidate_probs + 1e-40)
        # log_probs = candidate_logits - torch.logsumexp(candidate_logits,dim=-1,keepdim=True)
        entropy = (- candidate_probs * log_probs).sum(dim=-1)
        return entropy

    
    def get_specific_layer_logits(self, hidden_states, layer_indices: List[int]):

        lm_head = self.model.lm_head 

        if max(layer_indices) >= len(hidden_states):
            raise IndexError(f"Layer index {max(layer_indices)} is out of range for the model with {len(hidden_states)} layers.")

        last_token_states = torch.stack([hidden_states[idx][0, -1, :] for idx in layer_indices])
        if self.using_norm:
            last_token_states = self.model.model.norm(last_token_states)
        logits = lm_head(last_token_states) 
        return logits


    @torch.inference_mode()
    def clehe_using_logits(self, prompt_template: str, prompt_template_wo_results: str, question: str, document_texts: List[str], max_tokens=100, beta=0., temp_cpmi = 0.1, metric_criterion = "entropy", temperature = 0.5, sampling_method = "greedy", alpha=0.0, candidate_layers=[16], reweight_logit=False):

        batch_doc = [prompt_template.format(search_results=document, question=question) for document in document_texts]
        first_ele = prompt_template_wo_results.format(question=question)
        batch = [first_ele] + batch_doc
        print(max_tokens)
        # print(metric_criterion)
        # print(alpha,beta)
        inputs = self.tokenizer(batch, padding=True,return_tensors='pt').to(self.device)
        input_ids = inputs.input_ids
        n = input_ids.shape[0]
        attention_mask = inputs.attention_mask
        past_key_values = None 

        generate_token_list = [] 
        token_document_idx = []

        top1_idx_consistent_ratio_list = [] # some self-defiend logging metirc, will not influence the main logic

        for i in range(max_tokens):
            
            outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                            use_cache=True,
                            output_hidden_states=True,
                            past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
            candidate_query_logits = self.get_specific_layer_logits(outputs.hidden_states, candidate_layers)
            logits = outputs.logits[:, -1, :] # (bsz, vocab_size)

            if metric_criterion == "weighted_entropy":
                temp = temp_cpmi
                probs = torch.nn.functional.softmax(logits, dim =-1) # (bsz, vocab_size)
                entropy = - (probs * torch.log(probs + 1e-16)).sum(dim=-1) # (bsz, )
                entropy_weight = torch.nn.functional.softmax((-entropy[1:]) / temp, dim=-1).unsqueeze(1) # (bsz-1,)
                logits_max = (logits[1:] * entropy_weight).sum(dim=0)
                if self.using_entropy:
                    candidate_entropy = self.calcuate_entropy(candidate_query_logits)
                    max_entropy_idx = candidate_entropy.argmax()
                    logits_uncond = candidate_query_logits[max_entropy_idx]
                else:
                    logits_uncond = self.get_max_jsd_dist(logits_max, candidate_query_logits)
                # print(self.cal_jsd(torch.nn.functional.softmax(logits_max), torch.nn.functional.softmax(logits_uncond)))

                if reweight_logit:
                    l2_norm_p = torch.norm(logits_max, p=2)
                    l2_norm_q = torch.norm(logits_uncond, p=2)
                    logits_uncond = logits_uncond / l2_norm_q * l2_norm_p
                logits_max = self.cut_off_expert_logits(alpha, logits_max)
                logits_merged = (1 + beta) * logits_max - beta * logits_uncond

                before_contrast_top1_token_idx = logits_max.argmax().item()
                after_contrast_top1_token_idx = logits_merged.argmax().item()
                if before_contrast_top1_token_idx == after_contrast_top1_token_idx:
                    top1_idx_consistent_ratio_list.append(1.)
                    # print(before_contrast_top1_token_idx, after_contrast_top1_token_idx, True)
                else:
                    top1_idx_consistent_ratio_list.append(0.)
                    # print(before_contrast_top1_token_idx, after_contrast_top1_token_idx, False)
                k = entropy[1:].argmin() + 1
            else:
                raise NotImplementedError

            probs = torch.nn.functional.softmax(logits_merged / temperature, dim = -1) # still (vocab_size, )

            if sampling_method == "greedy":
                next_token_idx = torch.argmax(probs)
            else:
                raise NotImplementedError
            
            generate_token_list.append(next_token_idx.item())
            if metric_criterion == "entropy_top2":
                token_document_idx.append(k[0].item() - 1)
            else:
                token_document_idx.append(k.item() - 1)
            if next_token_idx == self.tokenizer.eos_token_id:
                break

            input_ids = next_token_idx.tile(n, 1)
            attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=self.device)], dim=-1)
        response = self.tokenizer.decode(generate_token_list, skip_special_tokens=True)
        # print(top1_idx_consistent_ratio_list)
        top1_idx_consistent_ratio = sum(top1_idx_consistent_ratio_list) / len(top1_idx_consistent_ratio_list)

        return token_document_idx, generate_token_list, response, top1_idx_consistent_ratio

