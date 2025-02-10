import torch
from typing import List, Dict, Union, Optional
from math import gcd, ceil


def repeat_to_batch_size(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    return tensor.repeat([ceil(batch_size / tensor.shape[0])] + [1] * (len(tensor.shape) - 1))[:batch_size]


def lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b)


class Condition:
    def __init__(self, cond: torch.Tensor):
        self.cond = cond

    def _copy_with(self, cond: torch.Tensor) -> 'Condition':
        return self.__class__(cond)

    def process_cond(self, batch_size: int, device: torch.device, **kwargs) -> 'Condition':
        return self._copy_with(repeat_to_batch_size(self.cond, batch_size).to(device))

    def can_concat(self, other: 'Condition') -> bool:
        return self.cond.shape == other.cond.shape

    def concat(self, others: List['Condition']) -> torch.Tensor:
        return torch.cat([self.cond] + [x.cond for x in others])


class ConditionNoiseShape(Condition):
    def process_cond(self, batch_size: int, device: torch.device, area: tuple, **kwargs) -> 'ConditionNoiseShape':
        data = self.cond[:, :, area[2]:area[0] + area[2], area[3]:area[1] + area[3]]
        return self._copy_with(repeat_to_batch_size(data, batch_size).to(device))


class ConditionCrossAttn(Condition):
    def can_concat(self, other: 'ConditionCrossAttn') -> bool:
        s1, s2 = self.cond.shape, other.cond.shape
        if s1 == s2:
            return True
        if s1[0] != s2[0] or s1[2] != s2[2]:
            return False
        return lcm(s1[1], s2[1]) // min(s1[1], s2[1]) <= 4

    def concat(self, others: List['ConditionCrossAttn']) -> torch.Tensor:
        conds = [self.cond]
        max_len = self.cond.shape[1]
        
        for x in others:
            conds.append(x.cond)
            max_len = lcm(max_len, x.cond.shape[1])
        
        return torch.cat([
            c.repeat(1, max_len // c.shape[1], 1) if c.shape[1] < max_len else c
            for c in conds
        ])


class ConditionConstant(Condition):
    def process_cond(self, batch_size: int, device: torch.device, **kwargs) -> 'ConditionConstant':
        return self._copy_with(self.cond)

    def can_concat(self, other: 'ConditionConstant') -> bool:
        return torch.equal(self.cond, other.cond)


def compile_conditions(cond: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Optional[List[Dict[str, Union[torch.Tensor, Dict[str, Condition]]]]]:
    if cond is None:
        return None

    if isinstance(cond, torch.Tensor):
        return [dict(
            cross_attn=cond,
            model_conds=dict(
                c_crossattn=ConditionCrossAttn(cond),
            )
        )]

    cross_attn = cond['crossattn']
    pooled_output = cond['vector']

    result = dict(
        cross_attn=cross_attn,
        pooled_output=pooled_output,
        model_conds=dict(
            c_crossattn=ConditionCrossAttn(cross_attn),
            y=Condition(pooled_output)
        )
    )

    if 'guidance' in cond:
        result['model_conds']['guidance'] = Condition(cond['guidance'])

    return [result]


def safe_tensor_indexing(tensor, indices):
    """Handle indexing for Float8 tensors by temporarily converting to float16"""
    original_dtype = tensor.dtype
    if str(original_dtype) == 'Float8_e5m2':
        tensor = tensor.to(dtype=torch.float16)
        result = tensor[indices]
        return result.to(dtype=original_dtype)
    return tensor[indices]

def compile_weighted_conditions(cond: Union[torch.Tensor, Dict[str, torch.Tensor]], weights: List[List[Union[int, float]]]) -> List[Dict[str, Union[torch.Tensor, Dict[str, Condition]]]]:
    transposed = list(map(list, zip(*weights)))
    results = []

    for cond_pre in transposed:
        current_indices = []
        current_weight = 0
        for i, w in cond_pre:
            current_indices.append(i)
            current_weight = w

        if hasattr(cond, 'advanced_indexing'):
            feed = cond.advanced_indexing(current_indices)
        else:
            feed = safe_tensor_indexing(cond, current_indices)

        h = compile_conditions(feed)
        h[0]['strength'] = current_weight
        results += h

    return results
