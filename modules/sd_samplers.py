from __future__ import annotations

import functools
import logging
from modules import sd_samplers_kdiffusion, sd_samplers_timesteps, sd_samplers_lcm, shared, sd_samplers_common, sd_schedulers

# imports for functions that previously were here and are used by other modules
from modules.sd_samplers_common import samples_to_image_grid, sample_to_image  # noqa: F401
from modules_forge import alter_samplers

all_samplers = [
    *sd_samplers_kdiffusion.samplers_data_k_diffusion,
    *sd_samplers_timesteps.samplers_data_timesteps,
    *sd_samplers_lcm.samplers_data_lcm,
    *alter_samplers.samplers_data_alter
]
all_samplers_map = {x.name: x for x in all_samplers}

samplers: list[sd_samplers_common.SamplerData] = []
samplers_for_img2img: list[sd_samplers_common.SamplerData] = []
samplers_map = {}
samplers_hidden = {}

def find_sampler_config(name):
    return (
        all_samplers_map.get(name)
        if name is not None
        else all_samplers[0]
    )

def create_sampler(name, model):
    config = find_sampler_config(name)

    if config is None:
        raise ValueError(f'Bad sampler name: {name}')

    if model.is_sdxl and config.options.get("no_sdxl", False):
        raise ValueError(f"Sampler {config.name} is not supported for SDXL")

    sampler = config.constructor(model)
    sampler.config = config

    return sampler

def set_samplers():
    global samplers, samplers_for_img2img, samplers_hidden

    samplers_hidden = set(shared.opts.hide_samplers)
    samplers = all_samplers
    samplers_for_img2img = all_samplers

    samplers_map.clear()
    for sampler in all_samplers:
        samplers_map[sampler.name.lower()] = sampler.name
        for alias in sampler.aliases:
            samplers_map[alias.lower()] = sampler.name

    return

def add_sampler(sampler):
    global all_samplers, all_samplers_map
    if sampler.name not in all_samplers_map:
        all_samplers.append(sampler)
        all_samplers_map[sampler.name] = sampler
        set_samplers()
    return

def visible_sampler_names():
    return [x.name for x in samplers if x.name not in samplers_hidden]

def visible_samplers():
    return [x for x in samplers if x.name not in samplers_hidden]

def get_sampler_from_infotext(d: dict):
    return get_sampler_and_scheduler(d.get("Sampler"), d.get("Schedule type"))[0]

def get_scheduler_from_infotext(d: dict):
    return get_sampler_and_scheduler(d.get("Sampler"), d.get("Schedule type"))[1]

def get_hr_sampler_and_scheduler(d: dict):
    hr_sampler = d.get("Hires sampler", "Use same sampler")
    sampler = d.get("Sampler") if hr_sampler == "Use same sampler" else hr_sampler

    hr_scheduler = d.get("Hires schedule type", "Use same scheduler")
    scheduler = d.get("Schedule type") if hr_scheduler == "Use same scheduler" else hr_scheduler

    sampler, scheduler = get_sampler_and_scheduler(sampler, scheduler)

    sampler = sampler if sampler != d.get("Sampler") else "Use same sampler"
    scheduler = scheduler if scheduler != d.get("Schedule type") else "Use same scheduler"

    return sampler, scheduler

def get_hr_sampler_from_infotext(d: dict):
    return get_hr_sampler_and_scheduler(d)[0]

def get_hr_scheduler_from_infotext(d: dict):
    return get_hr_sampler_and_scheduler(d)[1]

@functools.cache
def get_sampler_and_scheduler(sampler_name, scheduler_name, *, convert_automatic=True):
    """Gets sampler and scheduler from their combined name or separately specified names."""

    # Legacy combined names support
    if scheduler_name is None and sampler_name is not None and '+' in sampler_name:
        sampler_name, scheduler_name = sampler_name.split('+', maxsplit=1)

    available_samplers = all_samplers_map
    sampler = available_samplers.get(sampler_name, None)
    
    # Scheduler lookup without using aliases
    scheduler = None
    if scheduler_name:
        scheduler = sd_schedulers.schedulers_map.get(scheduler_name)
    
    if scheduler is None and convert_automatic:
        scheduler = sd_schedulers.schedulers_map.get('automatic')
        
    # If we still don't have a valid scheduler, get the default one
    if scheduler is None:
        scheduler = sd_schedulers.schedulers[0]

    return sampler.name if sampler else None, scheduler.name if scheduler else None

def fix_p_invalid_sampler_and_scheduler(p):
    i_sampler_name, i_scheduler = p.sampler_name, p.scheduler
    p.sampler_name, p.scheduler = get_sampler_and_scheduler(p.sampler_name, p.scheduler, convert_automatic=False)
    if p.sampler_name != i_sampler_name or i_scheduler != p.scheduler:
        logging.warning(f'Sampler Scheduler autocorrection: "{i_sampler_name}" -> "{p.sampler_name}", "{i_scheduler}" -> "{p.scheduler}"')

set_samplers()
