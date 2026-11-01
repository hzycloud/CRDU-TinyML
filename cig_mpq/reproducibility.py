import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def group_role_split(source_ids: list[str], group_counts: dict[str, int], seed: int) -> dict[str, list[int]]:
    """Assign whole source groups to roles; counts refer to groups, not samples.

    For speech, use speaker/background-recording identifiers, never crop IDs.
    Official held-out partitions must be supplied separately, not repartitioned.
    """
    groups = sorted(set(source_ids))
    if any(not isinstance(n, int) or n <= 0 for n in group_counts.values()):
        raise ValueError("each role requires a positive integer group count")
    if sum(group_counts.values()) != len(groups):
        raise ValueError("role group counts must cover every unique source exactly")
    random.Random(seed).shuffle(groups)
    result = {}
    offset = 0
    for role, count in group_counts.items():
        assigned = set(groups[offset:offset + count])
        result[role] = [i for i, source in enumerate(source_ids) if source in assigned]
        offset += count
    return result

