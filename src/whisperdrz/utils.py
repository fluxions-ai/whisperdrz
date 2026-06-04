import functools
from contextlib import nullcontext

import torch
from torch import nn


def load_what_you_can(checkpoint: dict, model: nn.Module):
    """Load as many weights from checkpoint as possible.

    Same-shape params are copied directly; mismatched shapes are partially
    copied along the overlapping region (e.g. extended vocab embeddings).
    """
    model_state_dict = model.state_dict()

    for name, param in checkpoint.items():
        if name not in model_state_dict:
            print(f"Ignoring parameter '{name}' because it is not found in the model")
            continue

        model_state = model_state_dict[name]
        mshape = model_state.shape
        pshape = param.shape

        if pshape == mshape:
            model_state.copy_(param)
            continue

        if len(pshape) != len(mshape):
            continue

        min_shape = [min(pshape[i], mshape[i]) for i in range(len(pshape))]
        print(name, "model:", mshape, "chkpt:", pshape, "loading:", min_shape)
        idxs = torch.meshgrid(*[torch.arange(s) for s in min_shape])
        model_state[tuple(idxs)].copy_(param[tuple(idxs)])

    return model.load_state_dict(model_state_dict)


@functools.lru_cache(maxsize=1)
def support_flash_attention() -> bool:
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    return capability >= (7, 5)


def flash_kernel(flash=True):
    """
    with flash_kernel():
        y = model(x)
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if flash:
        kernels = sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]
        )
    else:
        kernels = sdpa_kernel([SDPBackend.MATH])

    return kernels if support_flash_attention() else nullcontext()
