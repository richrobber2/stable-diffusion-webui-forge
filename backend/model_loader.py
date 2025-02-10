from backend.superperm_hooks import apply_superperm_to_model
from modules.sd_models import load_model as _original_load_model

def load_model(model_path, use_superperm=False, **kwargs):
    model = _original_load_model(model_path, **kwargs)
    if use_superperm:
        model = apply_superperm_to_model(model)
    return model
