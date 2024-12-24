from __future__ import annotations

import dataclasses
import inspect
import os
from typing import Optional, Any

from fastapi import FastAPI
from gradio import Blocks

from modules import errors, timer, extensions, shared, util


def report_exception(c, job):
    errors.report(f"Error executing callback {job} for {c.script}", exc_info=True)


class ImageSaveParams:
    def __init__(self, image, p, filename, pnginfo):
        self.image = image
        """the PIL image itself"""

        self.p = p
        """p object with processing parameters; either StableDiffusionProcessing or an object with same fields"""

        self.filename = filename
        """name of file that the image would be saved to"""

        self.pnginfo = pnginfo
        """dictionary with parameters for image's PNG info data; infotext will have the key 'parameters'"""


class ExtraNoiseParams:
    def __init__(self, noise, x, xi):
        self.noise = noise
        """Random noise generated by the seed"""

        self.x = x
        """Latent representation of the image"""

        self.xi = xi
        """Noisy latent representation of the image"""


class CFGDenoiserParams:
    def __init__(self, x, image_cond, sigma, sampling_step, total_sampling_steps, text_cond, text_uncond, denoiser=None):
        self.x = x
        """Latent image representation in the process of being denoised"""

        self.image_cond = image_cond
        """Conditioning image"""

        self.sigma = sigma
        """Current sigma noise step value"""

        self.sampling_step = sampling_step
        """Current Sampling step number"""

        self.total_sampling_steps = total_sampling_steps
        """Total number of sampling steps planned"""

        self.text_cond = text_cond
        """ Encoder hidden states of text conditioning from prompt"""

        self.text_uncond = text_uncond
        """ Encoder hidden states of text conditioning from negative prompt"""

        self.denoiser = denoiser
        """Current CFGDenoiser object with processing parameters"""


class CFGDenoisedParams:
    def __init__(self, x, sampling_step, total_sampling_steps, inner_model):
        self.x = x
        """Latent image representation in the process of being denoised"""

        self.sampling_step = sampling_step
        """Current Sampling step number"""

        self.total_sampling_steps = total_sampling_steps
        """Total number of sampling steps planned"""

        self.inner_model = inner_model
        """Inner model reference used for denoising"""


class AfterCFGCallbackParams:
    def __init__(self, x, sampling_step, total_sampling_steps):
        self.x = x
        """Latent image representation in the process of being denoised"""

        self.sampling_step = sampling_step
        """Current Sampling step number"""

        self.total_sampling_steps = total_sampling_steps
        """Total number of sampling steps planned"""


class UiTrainTabParams:
    def __init__(self, txt2img_preview_params):
        self.txt2img_preview_params = txt2img_preview_params


class ImageGridLoopParams:
    def __init__(self, imgs, cols, rows):
        self.imgs = imgs
        self.cols = cols
        self.rows = rows


@dataclasses.dataclass
class BeforeTokenCounterParams:
    prompt: str
    steps: int
    styles: list

    is_positive: bool = True


@dataclasses.dataclass
class ScriptCallback:
    script: str
    callback: any
    name: str = "unnamed"


def add_callback(callbacks, fun, *, name=None, category='unknown', filename=None):
    if filename is None:
        stack = [x for x in inspect.stack() if x.filename != __file__]
        filename = stack[0].filename if stack else 'unknown file'

    extension = extensions.find_extension(filename)
    extension_name = extension.canonical_name if extension else 'base'

    callback_name = f"{extension_name}/{os.path.basename(filename)}/{category}"
    if name is not None:
        callback_name += f'/{name}'

    unique_callback_name = callback_name
    for index in range(1000):
        existing = any(x.name == unique_callback_name for x in callbacks)
        if not existing:
            break

        unique_callback_name = f'{callback_name}-{index+1}'

    callbacks.append(ScriptCallback(filename, fun, unique_callback_name))


def sort_callbacks(category, unordered_callbacks, *, enable_user_sort=True):
    callbacks = unordered_callbacks.copy()
    callback_lookup = {x.name: x for x in callbacks}
    dependencies = {}

    order_instructions = {}
    for extension in extensions.extensions:
        for order_instruction in extension.metadata.list_callback_order_instructions():
            if order_instruction.name in callback_lookup:
                if order_instruction.name not in order_instructions:
                    order_instructions[order_instruction.name] = []

                order_instructions[order_instruction.name].append(order_instruction)

    if order_instructions:
        for callback in callbacks:
            dependencies[callback.name] = []

        for callback in callbacks:
            for order_instruction in order_instructions.get(callback.name, []):
                for after in order_instruction.after:
                    if after not in callback_lookup:
                        continue

                    dependencies[callback.name].append(after)

                for before in order_instruction.before:
                    if before not in callback_lookup:
                        continue

                    dependencies[before].append(callback.name)

        sorted_names = util.topological_sort(dependencies)
        callbacks = [callback_lookup[x] for x in sorted_names]

    if enable_user_sort:
        for name in reversed(getattr(shared.opts, f'prioritized_callbacks_{category}', [])):
            index = next((i for i, callback in enumerate(callbacks) if callback.name == name), None)
            if index is not None:
                callbacks.insert(0, callbacks.pop(index))

    return callbacks


def ordered_callbacks(category, unordered_callbacks=None, *, enable_user_sort=True):
    if unordered_callbacks is None:
        unordered_callbacks = callback_map.get(f'callbacks_{category}', [])

    if not enable_user_sort:
        return sort_callbacks(category, unordered_callbacks, enable_user_sort=False)

    callbacks = ordered_callbacks_map.get(category)
    if callbacks is not None and len(callbacks) == len(unordered_callbacks):
        return callbacks

    callbacks = sort_callbacks(category, unordered_callbacks)

    ordered_callbacks_map[category] = callbacks
    return callbacks


def enumerate_callbacks():
    for category, callbacks in callback_map.items():
        if category.startswith('callbacks_'):
            category = category[10:]

        yield category, callbacks


callback_map = dict(
    callbacks_app_started=[],
    callbacks_model_loaded=[],
    callbacks_ui_tabs=[],
    callbacks_ui_train_tabs=[],
    callbacks_ui_settings=[],
    callbacks_before_image_saved=[],
    callbacks_image_saved=[],
    callbacks_extra_noise=[],
    callbacks_cfg_denoiser=[],
    callbacks_cfg_denoised=[],
    callbacks_cfg_after_cfg=[],
    callbacks_before_component=[],
    callbacks_after_component=[],
    callbacks_image_grid=[],
    callbacks_infotext_pasted=[],
    callbacks_script_unloaded=[],
    callbacks_before_ui=[],
    callbacks_on_reload=[],
    callbacks_list_optimizers=[],
    callbacks_list_unets=[],
    callbacks_before_token_counter=[],
)

ordered_callbacks_map = {}


def clear_callbacks():
    for callback_list in callback_map.values():
        callback_list.clear()

    ordered_callbacks_map.clear()


def call_callbacks(category, *args, **kwargs):
    """Helper to call callbacks for a given category."""
    for c in ordered_callbacks(category):
        try:
            c.callback(*args, **kwargs)
        except Exception:
            report_exception(c, f'{category}_callback')


def app_started_callback(demo: Optional[Blocks], app: FastAPI):
    call_callbacks('app_started', demo, app)


def app_reload_callback():
    call_callbacks('on_reload')


def model_loaded_callback(sd_model):
    call_callbacks('model_loaded', sd_model)


def ui_tabs_callback():
    res = []
    def wrapper():
        for c in ordered_callbacks('ui_tabs'):
            try:
                r = c.callback() or []
                res.extend(r)
            except Exception:
                report_exception(c, 'ui_tabs_callback')
    wrapper()
    return res


def ui_train_tabs_callback(params: UiTrainTabParams):
    call_callbacks('ui_train_tabs', params=params)


def ui_settings_callback():
    call_callbacks('ui_settings')


def before_image_saved_callback(params: ImageSaveParams):
    call_callbacks('before_image_saved', params=params)


def image_saved_callback(params: ImageSaveParams):
    call_callbacks('image_saved', params=params)


def extra_noise_callback(params: ExtraNoiseParams):
    call_callbacks('extra_noise', params=params)


def cfg_denoiser_callback(params: CFGDenoiserParams):
    call_callbacks('cfg_denoiser', params=params)


def cfg_denoised_callback(params: CFGDenoisedParams):
    call_callbacks('cfg_denoised', params=params)


def cfg_after_cfg_callback(params: AfterCFGCallbackParams):
    call_callbacks('cfg_after_cfg', params=params)


def before_component_callback(component, **kwargs):
    call_callbacks('before_component', component, **kwargs)


def after_component_callback(component, **kwargs):
    call_callbacks('after_component', component, **kwargs)


def image_grid_callback(params: ImageGridLoopParams):
    call_callbacks('image_grid', params=params)


def infotext_pasted_callback(infotext: str, params: dict[str, Any]):
    call_callbacks('infotext_pasted', infotext, params)


def script_unloaded_callback():
    for c in reversed(ordered_callbacks('script_unloaded')):
        try:
            c.callback()
        except Exception:
            report_exception(c, 'script_unloaded')


def before_ui_callback():
    for c in reversed(ordered_callbacks('before_ui')):
        try:
            c.callback()
        except Exception:
            report_exception(c, 'before_ui')


def list_optimizers_callback():
    res = []
    for c in ordered_callbacks('list_optimizers'):
        try:
            c.callback(res)
        except Exception:
            report_exception(c, 'list_optimizers')

    return res


def list_unets_callback():
    res = []
    for c in ordered_callbacks('list_unets'):
        try:
            c.callback(res)
        except Exception:
            report_exception(c, 'list_unets')

    return res


def before_token_counter_callback(params: BeforeTokenCounterParams):
    call_callbacks('before_token_counter', params=params)


def remove_current_script_callbacks():
    stack = [x for x in inspect.stack() if x.filename != __file__]
    filename = stack[0].filename if stack else 'unknown file'
    if filename == 'unknown file':
        return
    for callback_list in callback_map.values():
        for callback_to_remove in [cb for cb in callback_list if cb.script == filename]:
            callback_list.remove(callback_to_remove)
    for ordered_callbacks_list in ordered_callbacks_map.values():
        for callback_to_remove in [cb for cb in ordered_callbacks_list if cb.script == filename]:
            ordered_callbacks_list.remove(callback_to_remove)


def remove_callbacks_for_function(callback_func):
    for callback_list in callback_map.values():
        for callback_to_remove in [cb for cb in callback_list if cb.callback == callback_func]:
            callback_list.remove(callback_to_remove)
    for ordered_callback_list in ordered_callbacks_map.values():
        for callback_to_remove in [cb for cb in ordered_callback_list if cb.callback == callback_func]:
            ordered_callback_list.remove(callback_to_remove)


def on_app_started(callback, *, name=None):
    """register a function to be called when the webui started, the gradio `Block` component and
    fastapi `FastAPI` object are passed as the arguments"""
    add_callback(callback_map['callbacks_app_started'], callback, name=name, category='app_started')


def on_before_reload(callback, *, name=None):
    """register a function to be called just before the server reloads."""
    add_callback(callback_map['callbacks_on_reload'], callback, name=name, category='on_reload')


def on_model_loaded(callback, *, name=None):
    """register a function to be called when the stable diffusion model is created; the model is
    passed as an argument; this function is also called when the script is reloaded. """
    add_callback(callback_map['callbacks_model_loaded'], callback, name=name, category='model_loaded')


def on_ui_tabs(callback, *, name=None):
    """register a function to be called when the UI is creating new tabs.
    The function must either return a None, which means no new tabs to be added, or a list, where
    each element is a tuple:
        (gradio_component, title, elem_id)

    gradio_component is a gradio component to be used for contents of the tab (usually gr.Blocks)
    title is tab text displayed to user in the UI
    elem_id is HTML id for the tab
    """
    add_callback(callback_map['callbacks_ui_tabs'], callback, name=name, category='ui_tabs')


def on_ui_train_tabs(callback, *, name=None):
    """register a function to be called when the UI is creating new tabs for the train tab.
    Create your new tabs with gr.Tab.
    """
    add_callback(callback_map['callbacks_ui_train_tabs'], callback, name=name, category='ui_train_tabs')


def on_ui_settings(callback, *, name=None):
    """register a function to be called before UI settings are populated; add your settings
    by using shared.opts.add_option(shared.OptionInfo(...)) """
    add_callback(callback_map['callbacks_ui_settings'], callback, name=name, category='ui_settings')


def on_before_image_saved(callback, *, name=None):
    """register a function to be called before an image is saved to a file.
    The callback is called with one argument:
        - params: ImageSaveParams - parameters the image is to be saved with. You can change fields in this object.
    """
    add_callback(callback_map['callbacks_before_image_saved'], callback, name=name, category='before_image_saved')


def on_image_saved(callback, *, name=None):
    """register a function to be called after an image is saved to a file.
    The callback is called with one argument:
        - params: ImageSaveParams - parameters the image was saved with. Changing fields in this object does nothing.
    """
    add_callback(callback_map['callbacks_image_saved'], callback, name=name, category='image_saved')


def on_extra_noise(callback, *, name=None):
    """register a function to be called before adding extra noise in img2img or hires fix;
    The callback is called with one argument:
        - params: ExtraNoiseParams - contains noise determined by seed and latent representation of image
    """
    add_callback(callback_map['callbacks_extra_noise'], callback, name=name, category='extra_noise')


def on_cfg_denoiser(callback, *, name=None):
    """register a function to be called in the kdiffussion cfg_denoiser method after building the inner model inputs.
    The callback is called with one argument:
        - params: CFGDenoiserParams - parameters to be passed to the inner model and sampling state details.
    """
    add_callback(callback_map['callbacks_cfg_denoiser'], callback, name=name, category='cfg_denoiser')


def on_cfg_denoised(callback, *, name=None):
    """register a function to be called in the kdiffussion cfg_denoiser method after building the inner model inputs.
    The callback is called with one argument:
        - params: CFGDenoisedParams - parameters to be passed to the inner model and sampling state details.
    """
    add_callback(callback_map['callbacks_cfg_denoised'], callback, name=name, category='cfg_denoised')


def on_cfg_after_cfg(callback, *, name=None):
    """register a function to be called in the kdiffussion cfg_denoiser method after cfg calculations are completed.
    The callback is called with one argument:
        - params: AfterCFGCallbackParams - parameters to be passed to the script for post-processing after cfg calculation.
    """
    add_callback(callback_map['callbacks_cfg_after_cfg'], callback, name=name, category='cfg_after_cfg')


def on_before_component(callback, *, name=None):
    """register a function to be called before a component is created.
    The callback is called with arguments:
        - component - gradio component that is about to be created.
        - **kwargs - args to gradio.components.IOComponent.__init__ function

    Use elem_id/label fields of kwargs to figure out which component it is.
    This can be useful to inject your own components somewhere in the middle of vanilla UI.
    """
    add_callback(callback_map['callbacks_before_component'], callback, name=name, category='before_component')


def on_after_component(callback, *, name=None):
    """register a function to be called after a component is created. See on_before_component for more."""
    add_callback(callback_map['callbacks_after_component'], callback, name=name, category='after_component')


def on_image_grid(callback, *, name=None):
    """register a function to be called before making an image grid.
    The callback is called with one argument:
       - params: ImageGridLoopParams - parameters to be used for grid creation. Can be modified.
    """
    add_callback(callback_map['callbacks_image_grid'], callback, name=name, category='image_grid')


def on_infotext_pasted(callback, *, name=None):
    """register a function to be called before applying an infotext.
    The callback is called with two arguments:
       - infotext: str - raw infotext.
       - result: dict[str, any] - parsed infotext parameters.
    """
    add_callback(callback_map['callbacks_infotext_pasted'], callback, name=name, category='infotext_pasted')


def on_script_unloaded(callback, *, name=None):
    """register a function to be called before the script is unloaded. Any hooks/hijacks/monkeying about that
    the script did should be reverted here"""

    add_callback(callback_map['callbacks_script_unloaded'], callback, name=name, category='script_unloaded')


def on_before_ui(callback, *, name=None):
    """register a function to be called before the UI is created."""

    add_callback(callback_map['callbacks_before_ui'], callback, name=name, category='before_ui')


def on_list_optimizers(callback, *, name=None):
    """register a function to be called when UI is making a list of cross attention optimization options.
    The function will be called with one argument, a list, and shall add objects of type modules.sd_hijack_optimizations.SdOptimization
    to it."""

    add_callback(callback_map['callbacks_list_optimizers'], callback, name=name, category='list_optimizers')


def on_list_unets(callback, *, name=None):
    """register a function to be called when UI is making a list of alternative options for unet.
    The function will be called with one argument, a list, and shall add objects of type modules.sd_unet.SdUnetOption to it."""

    add_callback(callback_map['callbacks_list_unets'], callback, name=name, category='list_unets')


def on_before_token_counter(callback, *, name=None):
    """register a function to be called when UI is counting tokens for a prompt.
    The function will be called with one argument of type BeforeTokenCounterParams, and should modify its fields if necessary."""

    add_callback(callback_map['callbacks_before_token_counter'], callback, name=name, category='before_token_counter')
