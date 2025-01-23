import itertools
import torch
import numpy as np
from typing import Tuple, Optional, Dict

class WeightCompressionError(Exception):
    """Custom exception for weight compression errors."""
    pass

def compute_optimal_rank(S: torch.Tensor, energy_threshold: float = 0.9) -> int:
    """
    Determine optimal rank based on singular value energy retention.
    """
    total_energy = (S ** 2).sum()
    cumulative_energy = torch.cumsum(S ** 2, dim=0)
    retained_energy = cumulative_energy / total_energy
    return torch.where(retained_energy >= energy_threshold)[0][0].item() + 1

def svd_compress_weight(
    weight: torch.Tensor,
    rank: Optional[int] = None,
    energy_threshold: float = 0.9,
    dtype: Optional[torch.dtype] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compress a weight matrix using truncated SVD.
    """
    if weight.dim() != 2:
        raise WeightCompressionError("SVD compression only supports 2D weight matrices")
    
    dtype = dtype or weight.dtype
    with torch.no_grad():
        try:
            U, S, Vt = torch.linalg.svd(weight.float(), full_matrices=False)
            if rank is None:
                rank = compute_optimal_rank(S, energy_threshold)
            
            U = U[:, :rank].to(dtype)
            S = S[:rank].to(dtype)
            Vt = Vt[:rank, :].to(dtype)
            
            return U, S, Vt
        except Exception as e:
            raise WeightCompressionError(f"SVD compression failed: {str(e)}") from e

def blockwise_svd_compress(
    weight: torch.Tensor,
    block_size: int = 1024,
    rank: Optional[int] = None,
    energy_threshold: float = 0.9
) -> Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Perform blockwise SVD compression for large matrices.
    """
    if weight.dim() != 2:
        raise WeightCompressionError("Blockwise SVD only supports 2D matrices")

    blocks = {}
    rows, cols = weight.shape

    for i, j in itertools.product(range(0, rows, block_size), range(0, cols, block_size)):
        r_end = min(i + block_size, rows)
        c_end = min(j + block_size, cols)
        block = weight[i:r_end, j:c_end]

        if block.numel() > 0:
            U, S, Vt = svd_compress_weight(block, rank, energy_threshold)
            blocks[(i, j)] = (U, S, Vt)

    return blocks

def reconstruct_from_factors(U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct weight matrix from its SVD factors.
    """
    return U @ (S.unsqueeze(1) * Vt)

class CompressedLinear(torch.nn.Module):
    """
    Linear layer using compressed weights via SVD.
    """
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                 rank: Optional[int] = None, energy_threshold: float = 0.9):
        super().__init__()
        U, S, Vt = svd_compress_weight(weight, rank, energy_threshold)
        self.register_parameter('U', torch.nn.Parameter(U, requires_grad=False))
        self.register_parameter('S', torch.nn.Parameter(S, requires_grad=False))
        self.register_parameter('Vt', torch.nn.Parameter(Vt, requires_grad=False))
        if bias is not None:
            self.register_parameter('bias', torch.nn.Parameter(bias, requires_grad=False))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Efficient matrix multiplication order: (x @ Vt.T) * S) @ U.T
        out = x @ self.Vt.T
        out = out * self.S
        out = out @ self.U.T
        if self.bias is not None:
            out = out + self.bias
        return out

def compress_model_weights(
    model: torch.nn.Module,
    min_weight_size: int = 1024,
    rank: Optional[int] = None,
    energy_threshold: float = 0.9,
    block_size: Optional[int] = None
) -> torch.nn.Module:
    """
    Compress large weight matrices in a model using SVD.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.weight.size(0) >= min_weight_size:
            weight = module.weight.data
            bias = module.bias.data if module.bias is not None else None
            
            if block_size and min(weight.shape) > block_size:
                # Use blockwise SVD for very large matrices
                blocks = blockwise_svd_compress(weight, block_size, rank, energy_threshold)
                # Store block information for custom forward pass
                setattr(module, 'compressed_blocks', blocks)
                setattr(module, 'original_forward', module.forward)
                module.forward = lambda x, m=module: blockwise_forward(x, m)
            else:
                # Replace with compressed linear layer
                compressed = CompressedLinear(weight, bias, rank, energy_threshold)
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, compressed)
                else:
                    setattr(model, child_name, compressed)
    
    return model

def blockwise_forward(x: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
    """
    Custom forward pass for blockwise compressed layers.
    """
    result = torch.zeros(x.size(0), module.weight.size(0), 
                        device=x.device, dtype=x.dtype)
    
    for (i, j), (U, S, Vt) in module.compressed_blocks.items():
        block_result = x @ Vt.T
        block_result = block_result * S
        block_result = block_result @ U.T
        result[:, i:i+U.size(0)] += block_result
    
    if module.bias is not None:
        result += module.bias
    
    return result
