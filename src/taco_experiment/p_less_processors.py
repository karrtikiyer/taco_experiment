"""HuggingFace LogitsProcessor wrappers for p-less and p-less-norm decoding.

Threshold math mirrors the authors' implementation in
src/p-less-sampling/p_less_samplers.py (https://arxiv.org/abs/2509.23234).
"""

import torch
from transformers import LogitsProcessor


class PLessLogitsProcessor(LogitsProcessor):
    """p-less truncation: threshold = sum(probs^2).

    Mirrors p_less_samplers.p_less_decode (line 25).
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.softmax(scores, dim=-1)
        threshold = probs.square().sum(dim=-1, keepdim=True)
        scores[probs < threshold] = float("-inf")
        return scores


class PLessNormLogitsProcessor(LogitsProcessor):
    """p-less-norm truncation: threshold = (V * sum(probs^2) - 1) / (V - 1).

    Mirrors p_less_samplers.p_less_norm_decode (lines 55-56).
    Relaxed variant that favors diversity over coherence.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.softmax(scores, dim=-1)
        v = probs.size(-1)
        threshold = (v * probs.square().sum(dim=-1, keepdim=True) - 1.0) / (v - 1.0)
        scores[probs < threshold] = float("-inf")
        return scores
