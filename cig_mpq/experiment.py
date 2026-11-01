from collections.abc import Iterable

from torch import nn

from .interaction import pairwise_interaction


def build_quantization_configs(
    layer_names: Iterable[str],
    pairs: Iterable[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    configs: dict[str, dict[str, int]] = {"base": {}}
    for name in layer_names:
        configs[f"w4:{name}"] = {name: 4}
    for first, second in pairs:
        configs[f"w4:{first}+{second}"] = {first: 4, second: 4}
    return configs


def choose_candidate_layers(model: nn.Module, max_layers: int) -> list[str]:
    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    return names[:max_layers]


def risk_delta_and_interactions(
    metrics_by_config: dict[str, dict],
    layer_names: list[str],
    pairs: Iterable[tuple[str, str]],
    risk_key: str,
) -> tuple[list[float], dict[tuple[int, int], float]]:
    base = float(metrics_by_config["base"][risk_key])
    deltas = [
        float(metrics_by_config[f"w4:{name}"][risk_key]) - base
        for name in layer_names
    ]
    interactions: dict[tuple[int, int], float] = {}
    for first, second in pairs:
        i = layer_names.index(first)
        j = layer_names.index(second)
        interactions[(i, j)] = pairwise_interaction(
            base,
            float(metrics_by_config[f"w4:{first}"][risk_key]),
            float(metrics_by_config[f"w4:{second}"][risk_key]),
            float(metrics_by_config[f"w4:{first}+{second}"][risk_key]),
        )
    return deltas, interactions

