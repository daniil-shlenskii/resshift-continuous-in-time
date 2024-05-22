import torch
import builtins
import importlib

TARGET_KEY = "target"
PARAMS_KEY = "params"
CKPT_PATH_KEY = "ckpt_path"
USE_FP_16_KEY = "use_fp16"

def prepare_model(obj, config):
    if CKPT_PATH_KEY in config:
        sd = torch.load(config[CKPT_PATH_KEY])
        filter_prefix = config.get("filter_prefix")
        if filter_prefix:
            sd = {k[len(filter_prefix):]: v for k, v in sd.items() if k.startswith(filter_prefix)}
        obj.load_state_dict(sd)

    if USE_FP_16_KEY in config:
        half = config[USE_FP_16_KEY]
        if half:
            obj.to(torch.float16)
        else:
            obj.to(torch.float32)

    return obj

def instantiate_from_config(config):
    target = config[TARGET_KEY]
    params = config[PARAMS_KEY]
    obj =  get_obj_from_str(target)
    obj = obj(**params)
    return obj

def get_obj_from_str(dot_separated_names):
    if "." not in dot_separated_names:
        return getattr(builtins, dot_separated_names)
    module, cls = dot_separated_names.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)