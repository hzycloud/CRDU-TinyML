import torch
from torch import nn


def fake_quantize_per_channel(weight: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("bits must be at least 2")
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))

    original_shape = weight.shape
    flattened = weight.detach().reshape(original_shape[0], -1)
    max_abs = flattened.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scales = max_abs / qmax
    quantized = torch.round(flattened / scales).clamp(qmin, qmax)
    dequantized = quantized * scales
    return dequantized.reshape(original_shape)


def fake_quantize_activation_per_tensor(
    activation: torch.Tensor,
    bits: int = 8,
) -> torch.Tensor:
    if bits < 2:
        raise ValueError("bits must be at least 2")
    if activation.numel() == 0:
        return activation
    qmax = (1 << bits) - 1
    minimum = activation.detach().amin()
    maximum = activation.detach().amax()
    scale = (maximum - minimum) / qmax
    if scale.item() <= 1e-12:
        return activation
    quantized = torch.round((activation - minimum) / scale).clamp(0, qmax)
    return quantized * scale + minimum


def install_activation_quantization_hooks(
    model: nn.Module,
    bits: int = 8,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> list[torch.utils.hooks.RemovableHandle]:
    handles = []
    if bits < 2:
        raise ValueError("bits must be at least 2")
    if ranges is not None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if name not in ranges:
                    raise ValueError(f"missing fixed activation range for {name}")
                low, high = ranges[name]
                if not (float('-inf') < low <= 0 <= high < float('inf')) or high <= low:
                    raise ValueError(f"invalid fixed activation range for {name}")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if ranges is not None:
                low, high = ranges[name]
                qmax = (1 << bits) - 1
                scale = (high - low) / qmax
                zero_point = max(0, min(qmax, round(-low / scale)))
                def fixed_hook(_module, _inputs, output, scale=scale, zero_point=zero_point, qmax=qmax):
                    return ((output / scale).round().add(zero_point).clamp(0, qmax) - zero_point) * scale
                handles.append(module.register_forward_hook(fixed_hook))
                continue
            handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output: fake_quantize_activation_per_tensor(
                        output,
                        bits=bits,
                    )
                )
            )
    return handles


@torch.no_grad()
def calibrate_activation_ranges(model: nn.Module, loader, device: torch.device) -> dict[str, tuple[float, float]]:
    """Collect fixed Conv2d output ranges on a dedicated, non-test dataset.

    This is still fake quantization with float operators, not integer inference.
    Collect anew after each candidate's weight quantization; zero is representable.
    """
    ranges = {}
    handles = []
    was_training = model.training
    def observer(name):
        def observe(_module, _inputs, output):
            low, high = ranges.get(name, (0., 0.))
            ranges[name] = (min(low, float(output.min())), max(high, float(output.max())))
        return observe
    model.eval()
    try:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(observer(name)))
        for inputs, _ in loader:
            model(inputs.to(device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if not ranges:
        raise ValueError("range calibration requires nonempty data and Conv2d layers")
    return {name: (low, high if high > low else low + 1e-8) for name, (low, high) in ranges.items()}


def quantize_model_weights(model: nn.Module, bits: int) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.copy_(fake_quantize_per_channel(module.weight, bits))


def quantize_selected_layers(model: nn.Module, bits_by_name: dict[str, int]) -> None:
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in bits_by_name and isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.copy_(
                    fake_quantize_per_channel(module.weight, bits_by_name[name])
                )


def quantize_mixed_precision(
    model: nn.Module,
    bits_by_name: dict[str, int],
    default_bits: int = 8,
) -> None:
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                bits = bits_by_name.get(name, default_bits)
                module.weight.copy_(fake_quantize_per_channel(module.weight, bits))

