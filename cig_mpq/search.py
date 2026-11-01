import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def solve_bitwidth_assignment(
    deltas: np.ndarray,
    interactions: dict[tuple[int, int], float],
    flash_w8: np.ndarray,
    flash_budget: float,
) -> np.ndarray:
    deltas = np.asarray(deltas, dtype=np.float64)
    flash_w8 = np.asarray(flash_w8, dtype=np.float64)
    n_layers = deltas.shape[0]
    edges = list(interactions.items())
    n_edges = len(edges)
    n_vars = n_layers + n_edges

    c = np.zeros(n_vars, dtype=np.float64)
    c[:n_layers] = deltas
    for edge_index, (_, weight) in enumerate(edges):
        c[n_layers + edge_index] = weight

    constraints = []
    lower_bounds = []
    upper_bounds = []

    required_reduction = max(0.0, flash_w8.sum() - float(flash_budget))
    savings_row = np.zeros(n_vars, dtype=np.float64)
    savings_row[:n_layers] = 0.5 * flash_w8
    constraints.append(savings_row)
    lower_bounds.append(required_reduction)
    upper_bounds.append(np.inf)

    for edge_index, ((i, j), _) in enumerate(edges):
        v_index = n_layers + edge_index

        row = np.zeros(n_vars)
        row[v_index] = 1.0
        row[i] = -1.0
        constraints.append(row)
        lower_bounds.append(-np.inf)
        upper_bounds.append(0.0)

        row = np.zeros(n_vars)
        row[v_index] = 1.0
        row[j] = -1.0
        constraints.append(row)
        lower_bounds.append(-np.inf)
        upper_bounds.append(0.0)

        row = np.zeros(n_vars)
        row[v_index] = 1.0
        row[i] = -1.0
        row[j] = -1.0
        constraints.append(row)
        lower_bounds.append(-1.0)
        upper_bounds.append(np.inf)

    if constraints:
        linear_constraint = LinearConstraint(
            np.vstack(constraints),
            np.asarray(lower_bounds),
            np.asarray(upper_bounds),
        )
    else:
        linear_constraint = None

    result = milp(
        c=c,
        integrality=np.ones(n_vars, dtype=np.uint8),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=linear_constraint,
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(f"bit-width search failed: {result.message}")
    return np.rint(result.x[:n_layers]).astype(np.int64)

