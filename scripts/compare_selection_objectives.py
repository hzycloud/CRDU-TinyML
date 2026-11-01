import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cig_mpq.data import build_imagefolder
from cig_mpq.experiment import risk_delta_and_interactions
from cig_mpq.models import DSCNN
from cig_mpq.reproducibility import seed_everything
from cig_mpq.search import solve_bitwidth_assignment
from run_cifar_pilot import evaluate_configuration, split_dataset


def main() -> None:
    result_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "cifar_pilot_det1"
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    layer_names = summary["candidate_layers"]
    pairs = [tuple(pair) for pair in summary["pairs"]]
    args = summary["args"]

    metrics_rows = list(csv.DictReader((result_dir / "config_metrics.csv").open(encoding="utf-8")))
    metrics_by_config = {
        row["config"]: {
            key: float(value)
            for key, value in row.items()
            if key != "config"
        }
        for row in metrics_rows
    }

    full_train = build_imagefolder(args["train_root"], image_size=32, normalize=True)
    test_dataset = build_imagefolder(args["test_root"], image_size=32, normalize=True)
    _, temp_subset = split_dataset(
        full_train,
        int(args["train_limit"]),
        int(args["temp_limit"]),
        int(args["seed"]),
    )
    eval_subset = Subset(test_dataset, list(range(min(int(args["eval_limit"]), len(test_dataset)))))
    temp_loader = DataLoader(temp_subset, batch_size=int(args["batch_size"]), shuffle=False, num_workers=0)
    eval_loader = DataLoader(eval_subset, batch_size=int(args["batch_size"]), shuffle=False, num_workers=0)

    checkpoint = torch.load(result_dir / "model.pt", map_location=device, weights_only=False)
    model = DSCNN(num_classes=len(checkpoint["classes"]), in_channels=3).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    flash_w8 = np.asarray(summary["flash_w8"], dtype=np.float64)
    flash_budget = float(summary["flash_budget"])
    activation_bits = int(args.get("activation_bits", 8))
    objectives = {
        "task_loss": "task_risk",
        "nll_brier": "calibration_risk",
        "brier": "brier",
        "ece": "ece",
    }
    output = {}
    for name, risk_key in objectives.items():
        deltas, interactions = risk_delta_and_interactions(
            metrics_by_config,
            layer_names,
            pairs,
            risk_key,
        )
        selected = solve_bitwidth_assignment(
            np.asarray(deltas, dtype=np.float64),
            interactions,
            flash_w8,
            flash_budget,
        )
        bits = {
            layer: 4
            for layer, flag in zip(layer_names, selected.tolist())
            if flag == 1
        }
        result = evaluate_configuration(
            model,
            bits,
            temp_loader,
            eval_loader,
            device,
            activation_bits=activation_bits,
        )
        output[name] = {
            "selected": selected.tolist(),
            "bits": bits,
            "deltas": deltas,
            "interactions": [
                {"pair": [layer_names[i], layer_names[j]], "value": interactions[(i, j)]}
                for i, j in sorted(interactions)
            ],
            "result": result,
        }

    (result_dir / "selection_objectives.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({
        name: {
            "bits": item["bits"],
            "accuracy": item["result"]["metrics"]["accuracy"],
            "ece": item["result"]["metrics"]["ece"],
            "nll": item["result"]["metrics"]["nll"],
            "brier": item["result"]["metrics"]["brier"],
            "temperature": item["result"]["temperature"],
        }
        for name, item in output.items()
    }, indent=2))


if __name__ == "__main__":
    main()
