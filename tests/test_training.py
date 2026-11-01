import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cig_mpq.training import evaluate_logits


class ConstantModel(nn.Module):
    def forward(self, inputs):
        batch = inputs.shape[0]
        logits = torch.tensor([[3.0, -3.0]], device=inputs.device)
        return logits.repeat(batch, 1)


class TrainingTest(unittest.TestCase):
    def test_evaluate_logits_returns_metrics_and_outputs(self):
        inputs = torch.randn(4, 3, 8, 8)
        targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

        metrics, logits, labels = evaluate_logits(
            ConstantModel(),
            loader,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertEqual(tuple(labels.shape), (4,))
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertIn("ece", metrics)
        self.assertIn("nll", metrics)
        self.assertIn("brier", metrics)


if __name__ == "__main__":
    unittest.main()
