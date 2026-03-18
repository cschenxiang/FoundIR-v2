import gc
import torch

from foundirv2.models.unet_2d_condition_vae_extension import Encoder

def torch_gc():
    gc.collect()
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def quantize_8bit(unet):
    if unet is None:
        return

    from peft.tuners.tuners_utils import BaseTunerLayer

    dtype = unet.dtype
    unet.to(torch.float8_e4m3fn)
    for module in unet.modules():  # revert lora modules to prevent errors with fp8
        if isinstance(module, BaseTunerLayer):
            module.to(dtype)
    for module in unet.modules():  # revert encoders to prevent errors with fp8
        if isinstance(module, Encoder):
            module.to(dtype)

    if hasattr(unet, "encoder_hid_proj"):  # revert ip adapter modules to prevent errors with fp8
        if unet.encoder_hid_proj is not None:
            for module in unet.encoder_hid_proj.modules():
                module.to(dtype)
    torch_gc()
