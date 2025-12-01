import torch
import torch.nn as nn


def freeze_layernorm_params(layer, weight_indices, weight_hook_handle=None):
    if weight_hook_handle is not None:
        weight_hook_handle.remove()

    if (weight_indices == [] or weight_indices is None):
        return

    if not isinstance(layer, nn.LayerNorm):
        raise ValueError("layer must be a valid LayerNorm layer")

    if max(weight_indices) >= layer.weight.shape[0]:
        raise IndexError("weight_indices must be less than the number output channels")

    def freezing_hook_weight_full(grad, weight_multiplier):
        return grad * weight_multiplier

    weight_multiplier = torch.ones_like(layer.weight).to(layer.weight.device)
    weight_multiplier[weight_indices] = 0

    freezing_hook_weight = lambda grad: freezing_hook_weight_full(grad, weight_multiplier)
    weight_hook_handle = layer.weight.register_hook(freezing_hook_weight)

    return weight_hook_handle


def freeze_bn_params(layer, weight_indices, weight_hook_handle=None):
    if weight_hook_handle is not None:
        weight_hook_handle.remove()

    if (weight_indices == [] or weight_indices is None):
        return

    if not isinstance(layer, nn.BatchNorm2d):
        raise ValueError("layer must be a valid BatchNorm layer")

    if max(weight_indices) >= layer.weight.shape[0]:
        raise IndexError("weight_indices must be less than the number output channels")

    def freezing_hook_weight_full(grad, weight_multiplier):
        return grad * weight_multiplier

    weight_multiplier = torch.ones_like(layer.weight).to(layer.weight.device)
    weight_multiplier[weight_indices] = 0

    freezing_hook_weight = lambda grad: freezing_hook_weight_full(grad, weight_multiplier)
    weight_hook_handle = layer.weight.register_hook(freezing_hook_weight)

    return weight_hook_handle


def freeze_conv2d_params(layer, weight_indices, bias_indices=None, weight_hook_handle=None, bias_hook_handle=None):
    if weight_hook_handle is not None:
        weight_hook_handle.remove()
    if bias_hook_handle is not None:
        bias_hook_handle.remove()

    if (weight_indices == [] or weight_indices is None) and (bias_indices == [] or bias_indices is None):
        return

    if bias_indices is None:
        bias_indices = weight_indices

    if not isinstance(layer, nn.Conv2d):
        raise ValueError("layer must be a valid Conv2d layer")

    if max(weight_indices) >= layer.weight.shape[0]:
        raise IndexError("weight_indices must be less than the number output channels")

    if layer.bias is not None:
        if max(bias_indices) >= layer.bias.shape[0]:
            raise IndexError("bias_indices must be less than the number output channels")

    def freezing_hook_weight_full(grad, weight_multiplier):
        return grad * weight_multiplier

    def freezing_hook_bias_full(grad, bias_multiplier):
        return grad * bias_multiplier

    weight_multiplier = torch.ones_like(layer.weight).to(layer.weight.device)
    weight_multiplier[weight_indices] = 0

    freezing_hook_weight = lambda grad: freezing_hook_weight_full(grad, weight_multiplier)
    weight_hook_handle = layer.weight.register_hook(freezing_hook_weight)

    if layer.bias is not None:
        bias_multiplier = torch.ones(layer.weight.shape[0]).to(layer.bias.device)
        bias_multiplier[bias_indices] = 0
        freezing_hook_bias = lambda grad: freezing_hook_bias_full(grad, bias_multiplier)
        bias_hook_handle = layer.bias.register_hook(freezing_hook_bias)
    else:
        bias_hook_handle = None

    return weight_hook_handle, bias_hook_handle


def freeze_linear_params(layer, weight_indices, bias_indices=None, weight_hook_handle=None, bias_hook_handle=None):
    if weight_hook_handle is not None:
        weight_hook_handle.remove()
    if bias_hook_handle is not None:
        bias_hook_handle.remove()

    if (weight_indices == [] or weight_indices is None) and (bias_indices == [] or bias_indices is None):
        return

    if bias_indices is None:
        bias_indices = weight_indices

    if not isinstance(layer, nn.Linear):
        raise ValueError("layer must be a valid Linear layer")

    if max(weight_indices['dim0']) >= layer.weight.shape[0] or max(weight_indices['dim1']) >= layer.weight.shape[1]:
        raise IndexError("weight_indices must be less than the number output channels")

    if layer.bias is not None:
        if max(bias_indices['dim0']) >= layer.bias.shape[0]:
            raise IndexError("bias_indices must be less than the number output channels")

    def freezing_hook_weight_full(grad, weight_multiplier):
        return grad * weight_multiplier

    def freezing_hook_bias_full(grad, bias_multiplier):
        return grad * bias_multiplier

    weight_multiplier = torch.ones_like(layer.weight).to(layer.weight.device)
    weight_multiplier[weight_indices['dim0'][:, None], weight_indices['dim1']] = 0

    freezing_hook_weight = lambda grad: freezing_hook_weight_full(grad, weight_multiplier)
    weight_hook_handle = layer.weight.register_hook(freezing_hook_weight)

    if layer.bias is not None:
        bias_multiplier = torch.ones(layer.weight.shape[0]).to(layer.bias.device)
        bias_multiplier[weight_indices['dim0']] = 0
        freezing_hook_bias = lambda grad: freezing_hook_bias_full(grad, bias_multiplier)
        bias_hook_handle = layer.bias.register_hook(freezing_hook_bias)
    else:
        bias_hook_handle = None

    return weight_hook_handle, bias_hook_handle


def freeze_conv2d_params_v2(layer, weight_indices, bias_indices=None, weight_hook_handle=None, bias_hook_handle=None):
    if weight_hook_handle is not None:
        weight_hook_handle.remove()
    if bias_hook_handle is not None:
        bias_hook_handle.remove()

    if (weight_indices == [] or weight_indices is None) and (bias_indices == [] or bias_indices is None):
        return

    if bias_indices is None:
        bias_indices = weight_indices

    if not isinstance(layer, nn.Conv2d):
        raise ValueError("layer must be a valid Conv2d layer")

    if weight_indices['dim0'].max() >= layer.weight.shape[0] or weight_indices['dim1'].max() >= layer.weight.shape[1]:
        raise IndexError("weight_indices must be less than the number output channels")

    if layer.bias is not None:
        if max(bias_indices) >= layer.bias.shape[0]:
            raise IndexError("bias_indices must be less than the number output channels")

    def freezing_hook_weight_full(grad, weight_multiplier):
        return grad * weight_multiplier

    def freezing_hook_bias_full(grad, bias_multiplier):
        return grad * bias_multiplier

    weight_multiplier = torch.ones_like(layer.weight).to(layer.weight.device)
    weight_multiplier[weight_indices["dim0"], weight_indices["dim1"], :, :] = 0

    freezing_hook_weight = lambda grad: freezing_hook_weight_full(grad, weight_multiplier)
    weight_hook_handle = layer.weight.register_hook(freezing_hook_weight)

    if layer.bias is not None:
        bias_multiplier = torch.ones(layer.weight.shape[0]).to(layer.bias.device)
        bias_multiplier[bias_indices] = 0
        freezing_hook_bias = lambda grad: freezing_hook_bias_full(grad, bias_multiplier)
        bias_hook_handle = layer.bias.register_hook(freezing_hook_bias)
    else:
        bias_hook_handle = None

    return weight_hook_handle, bias_hook_handle