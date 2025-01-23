import dataclasses
import torch
import k_diffusion
import numpy as np
from scipy import stats

from modules import shared

def denoiser_to_karras_derivative(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised).div_(sigma)

k_diffusion.sampling.to_d = denoiser_to_karras_derivative

def loglinear_interpolation(values, num_steps):
    """Performs log-linear interpolation of decreasing numbers."""
    xs = np.linspace(0, 1, len(values))
    ys = np.log(values[::-1])
    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)
    return np.exp(new_ys)[::-1].copy()

# Predefined sigma sequences
SIGMA_SEQUENCES = {
    'sdxl': {
        '11_steps': [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029],
        '32_steps': [14.615, 11.149, 8.505, 6.488, 5.437, 4.604, 3.899, 3.274, 2.744, 2.300, 1.954,
                    1.671, 1.429, 1.232, 1.068, 0.926, 0.803, 0.697, 0.604, 0.529, 0.468, 0.414,
                    0.363, 0.310, 0.265, 0.223, 0.177, 0.140, 0.106, 0.055, 0.029, 0.015],
    },
    'sd15': {
        '11_steps': [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029],
        '32_steps': [14.615, 11.240, 8.644, 6.647, 5.573, 4.716, 3.992, 3.520, 3.135, 2.792, 2.488,
                    2.217, 1.975, 1.779, 1.615, 1.465, 1.315, 1.166, 1.035, 0.916, 0.807, 0.712,
                    0.622, 0.531, 0.453, 0.375, 0.275, 0.201, 0.141, 0.067, 0.032, 0.015],
    }
}

def get_align_your_steps_sigmas(n, sigma_min, sigma_max, device, sequence='11_steps'):
    """Generate sigmas using Align Your Steps methodology."""
    model_type = 'sdxl' if shared.sd_model.is_sdxl else 'sd15'
    sigmas = SIGMA_SEQUENCES[model_type][sequence]
    
    if n != len(sigmas):
        sigmas = np.append(loglinear_interpolation(sigmas, n), 0.0)
    else:
        sigmas = np.append(sigmas, 0.0)
    
    return torch.from_numpy(sigmas).to(device)

def uniform(n, sigma_min, sigma_max, inner_model, device):
    return inner_model.get_sigmas(n).to(device)

def sgm_uniform(n, sigma_min, sigma_max, inner_model, device):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max, device=device))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min, device=device))
    timesteps = torch.linspace(start, end, n + 1, device=device)[:-1]
    sigs = inner_model.t_to_sigma(timesteps)
    sigs = torch.cat((sigs, sigs.new_tensor([0.0])))
    return sigs

def kl_optimal(n, sigma_min, sigma_max, device):
    alpha_min = torch.arctan(torch.tensor(sigma_min, device=device))
    alpha_max = torch.arctan(torch.tensor(sigma_max, device=device))
    step_indices = torch.arange(n + 1, device=device, dtype=torch.float32)
    step_indices.div_(n)  # In-place division
    return torch.tan(
        step_indices * alpha_min
        + (1.0 - step_indices).mul_(alpha_max - alpha_min).add_(alpha_min)
    )

def simple_scheduler(n, sigma_min, sigma_max, inner_model, device):
    ss = len(inner_model.sigmas) / n
    sigs = [float(inner_model.sigmas[-(1 + int(x * ss))]) for x in range(n)]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)

def normal_scheduler(n, sigma_min, sigma_max, inner_model, device, sgm=False, floor=False):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max, device=device))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min, device=device))

    if sgm:
        timesteps = torch.linspace(start, end, n + 1, device=device)
        timesteps = timesteps[:-1]
    else:
        timesteps = torch.linspace(start, end, n, device=device)

    sigs = inner_model.t_to_sigma(timesteps)
    sigs = torch.cat((sigs, sigs.new_zeros(1)))  # Use in-place new_zeros
    return sigs

def ddim_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = max(len(inner_model.sigmas) // n, 1)
    x = 1
    while x < len(inner_model.sigmas):
        sigs += [float(inner_model.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)

def beta_scheduler(n, sigma_min, sigma_max, inner_model, device):
    # From "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024) """
    alpha = shared.opts.beta_dist_alpha
    beta = shared.opts.beta_dist_beta
    timesteps = 1 - np.linspace(0, 1, n)
    timesteps = stats.beta.ppf(timesteps, alpha, beta)
    sigmas = sigma_min + (timesteps * (sigma_max - sigma_min))
    sigmas = np.append(sigmas, 0.0)
    sigmas = torch.from_numpy(sigmas).to(device)
    return sigmas

def turbo_scheduler(n, sigma_min, sigma_max, inner_model, device):
    unet = inner_model.inner_model.forge_objects.unet
    timesteps = torch.flip(torch.arange(1, n + 1) * float(1000.0 / n) - 1, (0,)).round().long().clip(0, 999)
    sigmas = unet.model.predictor.sigma(timesteps)
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    return sigmas.to(device)

@dataclasses.dataclass
class Scheduler:
    name: str
    label: str
    function: any
    default_rho: float = -1
    need_inner_model: bool = False

schedulers = [
    Scheduler('automatic', 'Automatic', None),
    Scheduler('uniform', 'Uniform', uniform, need_inner_model=True),
    Scheduler('karras', 'Karras', k_diffusion.sampling.get_sigmas_karras, default_rho=7.0),
    Scheduler('exponential', 'Exponential', k_diffusion.sampling.get_sigmas_exponential),
    Scheduler('polyexponential', 'Polyexponential', k_diffusion.sampling.get_sigmas_polyexponential, default_rho=1.0),
    Scheduler('sgm_uniform', 'SGM Uniform', sgm_uniform, need_inner_model=True),
    Scheduler('kl_optimal', 'KL Optimal', kl_optimal),
    Scheduler('align_your_steps', 'Align Your Steps', lambda n, s_min, s_max, d: get_align_your_steps_sigmas(n, s_min, s_max, d, '11_steps')),
    Scheduler('simple', 'Simple', simple_scheduler, need_inner_model=True),
    Scheduler('normal', 'Normal', normal_scheduler, need_inner_model=True),
    Scheduler('ddim', 'DDIM', ddim_scheduler, need_inner_model=True),
    Scheduler('beta', 'Beta', beta_scheduler, need_inner_model=True),
    Scheduler('turbo', 'Turbo', turbo_scheduler, need_inner_model=True),
    Scheduler('align_your_steps_32', 'Align Your Steps 32', lambda n, s_min, s_max, d: get_align_your_steps_sigmas(n, s_min, s_max, d, '32_steps')),
]

schedulers_map = {**{x.name: x for x in schedulers}, **{x.label: x for x in schedulers}}
