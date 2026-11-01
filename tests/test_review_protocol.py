import unittest
import numpy as np
import torch
from torch import nn
import cig_mpq.quantization as quantization
import cig_mpq.reproducibility as reproducibility
import cig_mpq.metrics as metrics


class ReviewProtocolTest(unittest.TestCase):
    def test_fixed_ranges_remove_batch_dependence_and_clip(self):
        model = nn.Sequential(nn.Conv2d(1, 1, 1, bias=False)).eval()
        model[0].weight.data.fill_(1)
        calibrate = getattr(quantization, 'calibrate_activation_ranges', None)
        self.assertIsNotNone(calibrate, 'independent range calibration is missing')
        ranges = calibrate(model, [(torch.tensor([[[[-1., 1.]]]]), torch.zeros(1))], torch.device('cpu'))
        handles = quantization.install_activation_quantization_hooks(model, bits=4, ranges=ranges)
        sample = torch.tensor([[[[0.21, 0.68]]]])
        alone = model(sample)
        batch = model(torch.cat([sample, torch.full_like(sample, 100)]))
        torch.testing.assert_close(alone, batch[:1])
        self.assertLessEqual(batch.max().item(), 1.01)
        for handle in handles:
            handle.remove()

    def test_group_split_never_shares_source(self):
        split = getattr(reproducibility, 'group_role_split', None)
        self.assertIsNotNone(split, 'source-level split manifest is missing')
        sources = ['a', 'a', 'b', 'c', 'd', 'd', 'e', 'f']
        roles = split(sources, {'train': 3, 'quantization': 1, 'temperature': 1, 'selection': 1}, 42)
        source_sets = [{sources[i] for i in indices} for indices in roles.values()]
        self.assertEqual(set(sum(roles.values(), [])), set(range(len(sources))))
        for i, first in enumerate(source_sets):
            for second in source_sets[i + 1:]:
                self.assertFalse(first & second)
        self.assertEqual(roles, split(sources, {'train': 3, 'quantization': 1, 'temperature': 1, 'selection': 1}, 42))

    def test_brier_binned_decomposition_identity(self):
        decompose = getattr(metrics, 'brier_decomposition', None)
        self.assertIsNotNone(decompose, 'binned multiclass decomposition is missing')
        p = np.array([[.9,.1],[.8,.2],[.3,.7],[.4,.6]])
        y = np.array([0,1,1,1])
        result = decompose(p, y, n_bins=2)
        score = np.mean(np.sum((p - np.eye(2)[y])**2, axis=1))
        self.assertAlmostEqual(score, result['reliability'] - result['resolution'] + result['uncertainty'] + result['binning_remainder'])
        self.assertGreaterEqual(result['reliability'], 0)

    def test_adaptive_ece_perfect_predictions(self):
        adaptive = getattr(metrics, 'adaptive_calibration_error', None)
        self.assertIsNotNone(adaptive, 'equal-count ECE is missing')
        self.assertEqual(adaptive(np.eye(2)[[0,1,0]], np.array([0,1,0]), n_bins=20), 0)


if __name__ == '__main__':
    unittest.main()
