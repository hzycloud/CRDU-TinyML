"""Analytical and protocol tests for the independently measured CLADO control."""
import copy
import importlib
import inspect
import unittest

import numpy as np
import torch
from torch import nn


def baseline():
    try:
        return importlib.import_module('cig_mpq.round2_baseline')
    except ModuleNotFoundError:
        return None


class Round2BaselineTest(unittest.TestCase):
    def implementation(self):
        module = baseline()
        self.assertIsNotNone(module, 'the published CLADO coefficient implementation is missing')
        return module

    def test_stationary_quadratic_coefficients_and_all_64_masks(self):
        module = self.implementation()
        # A known SPD Hessian with nonzero, signed cross-layer terms.
        a = np.arange(36, dtype=float).reshape(6, 6) / 40 - .4
        hessian = a.T @ a + np.eye(6)
        errors = np.array([[.3, -.02], [-.4, .01], [.2, -.03],
                           [-.1, .04], [.35, -.01], [-.25, .025]])
        directions = np.zeros((12, 6))
        for i in range(12):
            directions[i, i // 2] = errors[i // 2, i % 2]
        ref = 2.5
        loss = lambda d: ref + .5 * d @ hessian @ d
        singles = np.array([loss(d) for d in directions])
        pairs = {(i, j): loss(directions[i] + directions[j])
                 for i in range(12) for j in range(i + 1, 12) if i // 2 != j // 2}
        g = module.clado_matrix(ref, singles, pairs, 2)
        expected = directions @ hessian @ directions.T
        for i in range(6):
            expected[2*i, 2*i+1] = expected[2*i+1, 2*i] = 0
        np.testing.assert_allclose(g, expected, atol=3e-15)
        scores = module.enumerate_scores(g, 6)
        self.assertEqual(set(scores), {str(i) for i in range(64)})
        for mask in range(64):
            delta = np.array([errors[i, 0 if mask & (1 << i) else 1] for i in range(6)])
            self.assertAlmostEqual(scores[str(mask)], float(delta @ hessian @ delta), places=12)

    def test_psd_clips_only_negative_eigenvalues(self):
        module = self.implementation()
        g = np.array([[1., 2.], [2., 1.]])
        projected, eigenvalues = module.project_psd(g)
        np.testing.assert_allclose(eigenvalues, [-1., 3.])
        np.testing.assert_allclose(projected, [[1.5, 1.5], [1.5, 1.5]], atol=1e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(projected).min(), -1e-14)

    def test_metadata_extraction_ignores_all_performance_and_ranges_of_other_masks(self):
        module = self.implementation()
        names = [f'conv{i}' for i in range(6)]
        rows = [dict(mask=m, layer_names=names, activation_bits=8,
                     bits={n: 4 if m & (1 << i) else 8 for i, n in enumerate(names)},
                     activation_ranges={n: [-1., 2.] for n in names},
                     search_raw={'nll': float('nan')}, eval_raw={'accuracy': float('nan')})
                for m in range(64)]
        expected = module.candidate_spec(rows)
        changed = copy.deepcopy(rows)
        for row in changed:
            row['eval_raw'] = {'accuracy': 'TEST_LABEL_LEAK_SENTINEL'}
            row['search_raw'] = {'nll': 'SELECTION_LEAK_SENTINEL'}
            row['temperature'] = 'DO_NOT_FIT_THIS'
            if row['mask']:
                row['activation_ranges'] = {'ignored': 'MASK_SPECIFIC_RANGE_SENTINEL'}
        self.assertEqual(module.candidate_spec(changed), expected)
        changed[7]['bits'][names[0]] = 8
        with self.assertRaises(ValueError):
            module.candidate_spec(changed)

    def test_probe_selection_is_quantization_only_and_disjoint(self):
        module = self.implementation()
        manifest = {'training_namespace': 'official_train', 'test_namespace': 'official_test',
                    'roles': {'train': ['a/0.png'], 'validation': ['a/1.png'],
                              'quantization': [f'a/{i}.png' for i in range(2, 12)],
                              'temperature': ['a/12.png'], 'selection': ['a/13.png']},
                    'test': ['a/0.png']}
        first = module.select_probe_ids(manifest, 4, 42)
        self.assertEqual(first, module.select_probe_ids(manifest, 4, 42))
        self.assertEqual(len(set(first)), 4)
        self.assertTrue(set(first) <= set(manifest['roles']['quantization']))
        overlap = copy.deepcopy(manifest)
        overlap['roles']['selection'].append('a/3.png')
        with self.assertRaises(ValueError):
            module.select_probe_ids(overlap, 4, 42)

    def test_seven_layer_speech_metadata_and_exact_128_masks(self):
        module = self.implementation()
        self.assertIn('n_layers', inspect.signature(module.candidate_spec).parameters,
                      'explicit seven-layer candidate support is missing')
        names = [f'conv{i}' for i in range(7)]
        rows = [dict(mask=m, layer_names=names, activation_bits=8,
                     bits={n: 4 if m & (1 << i) else 8 for i, n in enumerate(names)},
                     activation_ranges={n: [-1., 2.] for n in names}) for m in range(128)]
        self.assertEqual(module.candidate_spec(rows, n_layers=7)[0], names)
        scores = module.enumerate_scores(np.diag(np.arange(1, 15)), 7)
        self.assertEqual(len(scores), 128)
        self.assertEqual(scores['0'], sum(range(2, 15, 2)))
        self.assertEqual(scores['127'], sum(range(1, 15, 2)))

    def test_speech_sampler_rejects_cross_role_sources_and_uses_quantization_only(self):
        module = self.implementation()
        self.assertTrue(hasattr(module, 'select_speech_probe_indices'),
                        'source-disjoint speech sampler is missing')
        roles = ['train', 'validation', 'quantization', 'temperature', 'selection', 'test']
        manifest = {'roles': {role: [dict(path=f'{role}/{i}.wav', sha256=f'{role}_sha{i}',
            kind='word', speaker_id=f'{role}_speaker{i}', background_id=None, label=i % 2)
            for i in range(10)] for role in roles}}
        indices = module.select_speech_probe_indices(manifest, 4, 42)
        self.assertEqual(indices, module.select_speech_probe_indices(manifest, 4, 42))
        self.assertEqual(len(set(indices)), 4)
        self.assertTrue(all(0 <= i < 10 for i in indices))
        overlap = copy.deepcopy(manifest)
        overlap['roles']['test'][0]['speaker_id'] = 'quantization_speaker0'
        with self.assertRaises(ValueError):
            module.select_speech_probe_indices(overlap, 4, 42)
        overlap = copy.deepcopy(manifest)
        overlap['roles']['validation'][0]['sha256'] = 'quantization_sha0'
        with self.assertRaises(ValueError):
            module.select_speech_probe_indices(overlap, 4, 42)

    def test_measurement_restores_weights_buffers_modes_and_existing_hooks(self):
        module = self.implementation()
        torch.manual_seed(13)
        model = nn.Sequential(nn.Conv2d(1, 2, 1, bias=False), nn.BatchNorm2d(2),
                              nn.Dropout(.8), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2, 2))
        model.train()
        model[1].eval()  # restoration must preserve mixed module modes too.
        state = {k: v.clone() for k, v in model.state_dict().items()}
        modes = [m.training for m in model.modules()]
        inputs = torch.randn(5, 1, 3, 3)
        labels = torch.tensor([0, 1, 0, 1, 0])
        records = module.measure_clado_losses(model, ['0'], inputs, labels, {'0': (-3., 3.)}, 2)
        self.assertEqual(set(records), {'reference', 'single:0', 'single:1'})
        self.assertTrue(all(v.shape == (5,) and np.isfinite(v).all() for v in records.values()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state[name]), name)
        self.assertEqual([m.training for m in model.modules()], modes)
        self.assertEqual(len(model[0]._forward_hooks), 0)
        again = module.measure_clado_losses(model, ['0'], inputs, labels, {'0': (-3., 3.)}, 2)
        for name in records:
            np.testing.assert_array_equal(again[name], records[name])


if __name__ == '__main__':
    unittest.main()
