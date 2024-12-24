import torch
import bitsandbytes as bnb

from typing import Optional, Dict, Any
from backend import utils, memory_management
from bitsandbytes.nn.modules import Params4bit, QuantState
from bitsandbytes.functional import dequantize_4bit


def functional_linear_4bits(x: torch.Tensor, weight: Params4bit, bias: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Perform a 4-bit quantized linear transformation using bitsandbytes.

    Args:
        x (torch.Tensor): Input tensor of shape (N, D_in).
        weight (Params4bit): Quantized weight parameters.
        bias (Optional[torch.Tensor]): Optional bias of shape (D_out,).

    Returns:
        torch.Tensor: The transformed output of shape (N, D_out).
    """
    out = bnb.matmul_4bit(
        x, weight.t(), bias=bias, quant_state=weight.quant_state
    )
    return out.to(x.device)


def functional_dequantize_4bit(weight: Params4bit) -> torch.Tensor:
    """
    Dequantize a 4-bit quantized parameter tensor.

    Args:
        weight (Params4bit): The quantized weight.

    Returns:
        torch.Tensor: The dequantized weight tensor.
    """
    if not weight.bnb_quantized:
        return weight

    original_device = weight.device
    if original_device.type != 'cuda':
        weight = weight.to('cuda')

    weight = dequantize_4bit(
        weight,
        quant_state=weight.quant_state,
        blocksize=weight.blocksize,
        quant_type=weight.quant_type
    )

    if original_device.type != 'cuda':
        weight = weight.to(device=original_device)

    return weight


def copy_quant_state(state: Optional[QuantState], device: Optional[torch.device] = None) -> Optional[QuantState]:
    """
    Create a copy of a given QuantState object on the specified device.

    Args:
        state (Optional[QuantState]): The original quantization state object.
        device (Optional[torch.device]): Target device for the state copy.

    Returns:
        Optional[QuantState]: A copy of the quantization state on the specified device,
                              or None if the input state is None.
    """
    if state is None:
        return None

    device = device or state.absmax.device

    # Only copy state2 if nested and state2 is present.
    state2 = None
    if state.nested and state.state2 is not None:
        state2 = QuantState(
            absmax=state.state2.absmax.to(device),
            shape=state.state2.shape,
            code=state.state2.code.to(device),
            blocksize=state.state2.blocksize,
            quant_type=state.state2.quant_type,
            dtype=state.state2.dtype,
        )

    offset = state.offset.to(device) if (state.nested and state.offset is not None) else None

    return QuantState(
        absmax=state.absmax.to(device),
        shape=state.shape,
        code=state.code.to(device),
        blocksize=state.blocksize,
        quant_type=state.quant_type,
        dtype=state.dtype,
        offset=offset,
        state2=state2,
    )


def to_device_and_dtype(
    tensor: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    non_blocking: bool = False
) -> torch.Tensor:
    """
    Safely move a tensor to a specified device and/or dtype.

    Args:
        tensor (torch.Tensor): The tensor to move.
        device (Optional[torch.device]): The target device.
        dtype (Optional[torch.dtype]): The target dtype.
        non_blocking (bool): Whether the copy should be non-blocking.

    Returns:
        torch.Tensor: The tensor on the specified device and dtype.
    """
    if device is not None or dtype is not None:
        tensor = tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)
    return tensor


class ForgeParams4bit(Params4bit):
    """
    A subclass of Params4bit for handling on-the-fly quantization of parameters
    when moved to a CUDA device.
    """

    def _quantize(self, device: torch.device) -> 'ForgeParams4bit':
        """
        Override the quantization step to ensure memory management hooks are triggered.
        """
        memory_management.signal_empty_cache = True
        return super()._quantize(device)

    def to(self, *args, **kwargs) -> 'ForgeParams4bit':
        """
        Move the parameter to a specified device and/or dtype. If moved to CUDA
        and the parameter is not already quantized, quantize it.
        """
        # Parse the device, dtype, etc. from *args, **kwargs
        # Using a known stable approach rather than internal PyTorch APIs:
        device = kwargs.get('device', None)
        dtype = kwargs.get('dtype', None)
        non_blocking = kwargs.get('non_blocking', False)

        if device is not None and device.type == "cuda" and not self.bnb_quantized:
            # Quantize if moved to CUDA and not yet quantized
            return self._quantize(device)
        else:
            # Move or cast parameters normally, preserving quantization state
            moved_tensor = to_device_and_dtype(self, device, dtype, non_blocking)
            return ForgeParams4bit(
                moved_tensor,
                requires_grad=self.requires_grad,
                quant_state=copy_quant_state(self.quant_state, device),
                blocksize=self.blocksize,
                compress_statistics=self.compress_statistics,
                quant_type=self.quant_type,
                quant_storage=self.quant_storage,
                bnb_quantized=self.bnb_quantized,
            )

    def pin_memory(self, device: Optional[torch.device] = None) -> 'ForgeParams4bit':
        """
        Pin the parameter's memory, potentially for faster CPU-to-GPU transfers.
        """
        pinned_tensor = torch.Tensor.pin_memory(self, device=device)
        return ForgeParams4bit(
            pinned_tensor,
            requires_grad=self.requires_grad,
            quant_state=self.quant_state,
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_type=self.quant_type,
            quant_storage=self.quant_storage,
            bnb_quantized=self.bnb_quantized,
        )


class ForgeLoader4Bit(torch.nn.Module):
    """
    A module that can load and handle 4-bit quantized parameters, providing utilities
    to handle both pre-quantized and non-quantized states.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        quant_type: str,
        **kwargs: Any
    ):
        super().__init__()
        # A placeholder parameter to record the device and dtype before actual weights are loaded.
        self.placeholder_param = torch.nn.Parameter(torch.empty(1, device=device, dtype=dtype))
        self.weight: Optional[ForgeParams4bit] = None
        self.bias: Optional[torch.nn.Parameter] = None
        self.quant_type = quant_type

    def _apply(self, fn, recurse=True):
        """
        Override _apply to ensure parameters remain parameters after applying fn.
        """
        for k, p in self.named_parameters(recurse=False, remove_duplicate=True):
            if p is not None:
                setattr(self, k, utils.tensor2parameter(fn(p)))
        return self

    def _save_to_state_dict(self, destination: Dict[str, torch.Tensor], prefix: str, keep_vars: bool):
        """
        Save the current state to a state_dict, including quantization metadata if available.
        """
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.weight is not None and hasattr(self.weight, "quant_state") and self.weight.quant_state is not None:
            quant_state_dict = self.weight.quant_state.as_dict(packed=True)
            for k, v in quant_state_dict.items():
                destination[f"{prefix}weight.{k}"] = v if keep_vars else v.detach()

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list
    ):
        """
        Load parameters from a state dict. If pre-quantized parameters are found, 
        reconstruct them; otherwise, load them normally.
        """
        quant_state_keys = {
            k[len(f"{prefix}weight."):]
            for k in state_dict.keys()
            if k.startswith(f"{prefix}weight.")
        }

        # If bitsandbytes quantization data is present, it's a pre-quantized model.
        if any('bitsandbytes' in k for k in quant_state_keys):
            quant_state_dict = {k: state_dict[f"{prefix}weight.{k}"] for k in quant_state_keys}

            if f'{prefix}weight' not in state_dict:
                error_msgs.append(f"Missing key '{prefix}weight' in state_dict for pre-quantized loading.")
                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
                return

            self.weight = ForgeParams4bit.from_prequantized(
                data=state_dict[f'{prefix}weight'],
                quantized_stats=quant_state_dict,
                requires_grad=False,
                device=self.placeholder_param.device,
            )

            if f'{prefix}bias' in state_dict:
                self.bias = torch.nn.Parameter(state_dict[f'{prefix}bias'].to(self.placeholder_param))

            # After loading the actual weight and bias, placeholder is no longer needed.
            del self.placeholder_param

        else:
            # If no quantization keys are found, load as a non-quantized parameter.
            # This might be a scenario where weights are loaded first time in full precision.
            if f'{prefix}weight' in state_dict:
                w = state_dict[f'{prefix}weight'].to(self.placeholder_param)
                self.weight = ForgeParams4bit(
                    w,
                    requires_grad=False,
                    compress_statistics=False,
                    blocksize=64,
                    quant_type=self.quant_type,
                    quant_storage=torch.uint8,
                )

            if f'{prefix}bias' in state_dict:
                self.bias = torch.nn.Parameter(state_dict[f'{prefix}bias'].to(self.placeholder_param))

            if hasattr(self, 'placeholder_param'):
                del self.placeholder_param

            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def reload_weight(self, weight: torch.Tensor) -> 'ForgeLoader4Bit':
        """
        Reload the weight by creating a new ForgeParams4bit and quantizing it if necessary.

        Args:
            weight (torch.Tensor): The new weight tensor to load.

        Returns:
            ForgeLoader4Bit: The module with updated weight.
        """
        original_device = weight.device
        new_weight = ForgeParams4bit(
            weight,
            requires_grad=False,
            compress_statistics=self.weight.compress_statistics if self.weight else False,
            blocksize=self.weight.blocksize if self.weight else 64,
            quant_type=self.weight.quant_type if self.weight else self.quant_type,
            quant_storage=self.weight.quant_storage if self.weight else torch.uint8,
            bnb_quantized=False
        )

        # Move the new weight to the original device, ensuring correct quantization if needed.
        new_weight = new_weight.to(original_device)
        self.weight = new_weight
        return self
