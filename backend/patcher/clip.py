from backend import memory_management
from backend.patcher.base import ModelPatcher
from backend.nn.base import ModuleDict, ObjectDict


class JointTextEncoder(ModuleDict):
    pass


class CLIP:
    def __init__(self, model_dict=None, tokenizer_dict=None, no_init=False):
        if model_dict is None:
            model_dict = {}
        if tokenizer_dict is None:
            tokenizer_dict = {}
        if no_init:
            return

        load_device = memory_management.text_encoder_device()
        offload_device = memory_management.text_encoder_offload_device()

        self.cond_stage_model = JointTextEncoder(model_dict)
        self.tokenizer = ObjectDict(tokenizer_dict)
        self.patcher = ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        return n

    def add_patches(self, *arg, **kwargs):
        return self.patcher.add_patches(*arg, **kwargs)
