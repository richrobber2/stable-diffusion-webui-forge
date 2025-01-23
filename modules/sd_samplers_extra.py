import torch
import tqdm
from typing import Optional, Dict, Any, Callable
from k_diffusion.sampling import to_d, get_sigmas_karras

@torch.no_grad()
def restart_sampler(
    model: Callable,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    s_noise: float = 1.,
    restart_list: Optional[Dict[float, list]] = None
) -> torch.Tensor:
    """Implements restart sampling following "Restart Sampling for Improving Generative Processes" (2023)
    
    Args:
        model: The denoising model
        x: Input tensor to denoise
        sigmas: Noise schedule
        extra_args: Additional arguments for the model
        callback: Optional callback function for progress updates
        disable: Whether to disable progress bar
        s_noise: Noise scale factor
        restart_list: Dictionary of {min_sigma: [restart_steps, restart_times, max_sigma]}
    
    Returns:
        Denoised tensor
    """
    extra_args = {} if extra_args is None else extra_args
    
    def heun_step(x: torch.Tensor, old_sigma: float, new_sigma: float, step_id: int) -> tuple[torch.Tensor, int]:
        denoised = model(x, old_sigma, **extra_args)
        d = to_d(x, old_sigma, denoised)
        if callback is not None:
            callback({'x': x, 'i': step_id, 'sigma': new_sigma, 'sigma_hat': old_sigma, 'denoised': denoised})
        
        dt = new_sigma - old_sigma
        if new_sigma == 0:
            x.add_(d * dt)  # Euler step for final iteration
        else:
            # Heun's method
            x_2 = x.addcmul(d, dt)
            denoised_2 = model(x_2, new_sigma, **extra_args)
            d_2 = to_d(x_2, new_sigma, denoised_2)
            d_prime = (d + d_2) / 2
            x.add_(d_prime * dt)
        return x, step_id + 1

    # Determine restart schedule
    steps = sigmas.shape[0] - 1
    if restart_list is None:
        restart_list = {}
        if steps >= 36:
            restart_list = {0.1: [steps // 4 + 1, 2, 2]}
        elif steps >= 20:
            restart_list = {0.1: [9 + 1, 1, 2]}

    # Convert sigma values to indices
    restart_indices = {
        int(torch.argmin(abs(sigmas - key), dim=0)): value 
        for key, value in restart_list.items()
    }

    # Generate step schedule with restarts
    step_list = []
    for i in range(len(sigmas) - 1):
        step_list.append((sigmas[i], sigmas[i + 1]))
        if i + 1 in restart_indices:
            restart_steps, restart_times, restart_max = restart_indices[i + 1]
            min_idx, max_idx = i + 1, int(torch.argmin(abs(sigmas - restart_max), dim=0))
            if max_idx < min_idx:
                sigma_restart = get_sigmas_karras(
                    restart_steps,
                    sigmas[min_idx].item(),
                    sigmas[max_idx].item(),
                    device=sigmas.device
                )[:-1]
                for _ in range(restart_times):
                    step_list.extend(zip(sigma_restart[:-1], sigma_restart[1:]))

    # Execute sampling steps
    step_id = 0
    last_sigma = None
    for old_sigma, new_sigma in tqdm.tqdm(step_list, disable=disable):
        if last_sigma is not None and last_sigma < old_sigma:
            noise_scale = s_noise * (old_sigma ** 2 - last_sigma ** 2) ** 0.5
            x.add_(torch.randn_like(x) * noise_scale)
        x, step_id = heun_step(x, old_sigma, new_sigma, step_id)
        last_sigma = new_sigma

    return x
