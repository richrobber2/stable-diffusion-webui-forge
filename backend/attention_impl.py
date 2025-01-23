from typing import Optional
import torch
from torch import Tensor

from .attention import (
    AttentionBase,
    attention_xformers,
    attention_pytorch,
    attention_split,
    attention_sub_quad
)

class XFormersAttention(AttentionBase):
    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                heads: int, mask: Optional[Tensor] = None) -> Tensor:
        return attention_xformers(q, k, v, heads, mask, self.config.attn_precision)

class PyTorchAttention(AttentionBase):
    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                heads: int, mask: Optional[Tensor] = None) -> Tensor:
        return attention_pytorch(q, k, v, heads, mask, self.config.attn_precision)

class SplitAttention(AttentionBase):
    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                heads: int, mask: Optional[Tensor] = None) -> Tensor:
        return attention_split(q, k, v, heads, mask, self.config.attn_precision)

class SubQuadraticAttention(AttentionBase):
    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                heads: int, mask: Optional[Tensor] = None) -> Tensor:
        return attention_sub_quad(q, k, v, heads, mask, self.config.attn_precision)
