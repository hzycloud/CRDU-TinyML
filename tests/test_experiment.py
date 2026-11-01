import unittest

from torch import nn

from cig_mpq.experiment import (
    build_quantization_configs,
    choose_candidate_layers,
    risk_delta_and_interactions,
)
from cig_mpq.models import DSCNN


class ExperimentTest(unittest.TestCase):
    def test_builds_base_single_and_pair_configurations(self):
        configs = build_quantization_configs(
            layer_names=["conv1", "conv2"],
            pairs=[("conv1", "conv2")],
        )

        self.assertEqual(configs["base"], {})
        self.assertEqual(configs["w4:conv1"], {"conv1": 4})
        self.assertEqual(configs["w4:conv2"], {"conv2": 4})
        self.assertEqual(configs["w4:conv1+conv2"], {"conv1": 4, "conv2": 4})

    def test_choose_candidate_layers_returns_conv_modules(self):
        model = DSCNN(num_classes=10, in_channels=3)

        names = choose_candidate_layers(model, max_layers=4)

        self.assertEqual(len(names), 4)
        modules = dict(model.named_modules())
        for name in names:
            self.assertIsInstance(modules[name], nn.Conv2d)

    def test_risk_delta_and_interactions(self):
        metrics = {
            "base": {"risk": 1.0},
            "w4:conv1": {"risk": 1.2},
            "w4:conv2": {"risk": 1.4},
            "w4:conv1+conv2": {"risk": 1.8},
        }

        deltas, interactions = risk_delta_and_interactions(
            metrics_by_config=metrics,
            layer_names=["conv1", "conv2"],
            pairs=[("conv1", "conv2")],
            risk_key="risk",
        )

        self.assertAlmostEqual(deltas[0], 0.2)
        self.assertAlmostEqual(deltas[1], 0.4)
        self.assertAlmostEqual(interactions[(0, 1)], 0.2)


if __name__ == "__main__":
    unittest.main()
