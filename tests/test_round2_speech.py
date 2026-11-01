"""Checks for the independent second-review speech protocol."""
import copy
import importlib.util
import tempfile
import unittest
import wave
from unittest.mock import patch
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class Round2SpeechTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.find_spec('cig_mpq.round2_speech')
        cls.protocol = None
        if spec is not None:
            from cig_mpq import round2_speech
            cls.protocol = round2_speech

    def module(self):
        self.assertIsNotNone(self.protocol, 'source-disjoint round2 speech protocol is missing')
        return self.protocol

    def test_json_retries_transient_windows_replace_lock(self):
        p = self.module()
        original = Path.replace
        attempts = []
        def locked_once(source, target):
            attempts.append(source)
            if len(attempts) == 1:
                raise PermissionError('simulated Windows reader lock')
            return original(source, target)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'status.json'
            target.write_text('{"state":"old"}')
            with patch.object(Path, 'replace', locked_once):
                p.write_json(target, {'state': 'complete'})
            self.assertEqual(target.read_text(), '{\n  "state": "complete"\n}')
            self.assertEqual(len(attempts), 2)

    def fixture(self, root):
        from cig_mpq.speech_commands import TARGET_WORDS
        validation, testing = [], []
        for index, speaker in enumerate([f's{i:03}' for i in range(40)] + ['val', 'test']):
            for label, word in enumerate((*TARGET_WORDS, 'bird')):
                relative = f'{word}/{speaker}_nohash_0.wav'
                self.wav(root / relative, 16000, 100 + index * 20 + label)
                if speaker == 'val':
                    validation.append(relative)
                elif speaker == 'test':
                    testing.append(relative)
        for index in range(6):
            self.wav(root / f'_background_noise_/noise{index}.wav', 16000, 2000 + index, 48000)
        (root / 'validation_list.txt').write_text('\n'.join(validation))
        (root / 'testing_list.txt').write_text('\n'.join(testing))

    @staticmethod
    def wav(path, sample_rate, value, length=16000):
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), 'wb') as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(sample_rate)
            stream.writeframes(np.full(length, value, dtype='<i2').tobytes())

    def test_six_roles_preserve_official_words_and_whole_background_sources(self):
        p = self.module()
        output = ROOT / 'results/round2'
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='speech_test_', dir=output) as temporary:
            root = Path(temporary)
            self.fixture(root)
            caps = {role: 3 for role in p.ROLES}
            manifest = p.build_manifest(root, caps, split_seed=123)
            self.assertEqual(manifest, p.build_manifest(root, caps, split_seed=123))
            self.assertEqual(set(manifest['roles']), set(p.ROLES))
            self.assertTrue(manifest['audit']['passed'])
            speakers, backgrounds = [], []
            for role, rows in manifest['roles'].items():
                self.assertEqual({r['label'] for r in rows}, set(range(12)))
                speakers.append({r['speaker_id'] for r in rows if r['kind'] == 'word'})
                backgrounds.append({r['background_id'] for r in rows if r['kind'] == 'silence'})
                self.assertEqual(len(backgrounds[-1]), 1)
                for row in rows:
                    self.assertFalse(Path(row['path']).is_absolute())
                    self.assertEqual(len(row['sha256']), 64)
                    self.assertEqual(row['sample_rate'], 16000)
                    if row['kind'] == 'word' and role in ('validation', 'test'):
                        self.assertEqual(row['speaker_id'], 'val' if role == 'validation' else 'test')
            for i in range(6):
                for j in range(i):
                    self.assertFalse(speakers[i] & speakers[j])
                    self.assertFalse(backgrounds[i] & backgrounds[j])
            self.assertEqual(manifest['role_summary']['validation']['underrepresented']['down']['selected'], 1)
            broken = copy.deepcopy(manifest)
            broken['roles']['selection'][0]['speaker_id'] = manifest['roles']['train'][0]['speaker_id']
            with self.assertRaisesRegex(ValueError, 'speaker'):
                p.audit_manifest(broken)
            broken = copy.deepcopy(manifest)
            broken['roles']['selection'][0]['sha256'] = manifest['roles']['train'][0]['sha256']
            with self.assertRaisesRegex(ValueError, 'hash'):
                p.audit_manifest(broken)

    def test_actual_sample_rate_is_checked_before_mfcc(self):
        p = self.module()
        with tempfile.TemporaryDirectory(prefix='speech_test_', dir=ROOT / 'results/round2') as temporary:
            root = Path(temporary)
            self.wav(root / 'wrong.wav', 8000, 17)
            with self.assertRaisesRegex(ValueError, 'sample rate'):
                p.row_features(root, {'path': 'wrong.wav', 'kind': 'word'})
            self.wav(root / 'right.wav', 16000, 19)
            feature = p.row_features(root, {'path': 'right.wav', 'kind': 'word'})
            self.assertEqual(feature.shape, (49, 10))
            self.assertTrue(np.isfinite(feature).all())

    def test_normalization_uses_only_training(self):
        p = self.module()
        arrays = {'train': np.array([[[1.]], [[3.]]], dtype=np.float32),
                  'test': np.array([[[1001.]]], dtype=np.float32)}
        normalized, mean, std = p.normalize_features(arrays)
        np.testing.assert_allclose(mean, 2.)
        np.testing.assert_allclose(std, 1.)
        np.testing.assert_allclose(normalized['train'].mean(), 0.)
        np.testing.assert_allclose(normalized['test'], 999.)

    def test_test_gate_requires_saved_selection_outputs_and_frozen_decisions(self):
        p = self.module()
        with tempfile.TemporaryDirectory(prefix='speech_test_', dir=ROOT / 'results/round2') as temporary:
            run = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, 'frozen'):
                p.require_test_gate(run)
            from cig_mpq.models import AudioDSCNN
            from scripts.run_pareto_pilot import weight_bytes, bits_from_mask
            import torch
            model = AudioDSCNN()
            layers = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]
            self.assertEqual(len(layers), 7)
            low = bits_from_mask(127, layers)
            expected = sum((m.weight.numel() * low.get(n, 8) + 7) // 8
                           for n, m in model.named_modules()
                           if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)))
            self.assertEqual(weight_bytes(model, low), expected)
            self.assertGreater(expected, sum((m.weight.numel() * 4 + 7) // 8
                                            for m in model.modules() if isinstance(m, torch.nn.Conv2d)))

    def test_complete_freeze_receipt_rejects_tampering(self):
        p = self.module()
        with tempfile.TemporaryDirectory(prefix='speech_test_', dir=ROOT / 'results/round2') as temporary:
            run = Path(temporary)
            metrics = {'nll': .4, 'brier': .2, 'ece': .03, 'accuracy': .9}
            rows = []
            for mask in range(128):
                rows.append({'mask': mask, 'bits': {str(i): 4 if mask & (1 << i) else 8 for i in range(7)},
                    'flash_fraction': 1. - .5 * mask / 127.,
                    'search_raw': metrics, 'search_calibrated': metrics})
                np.savez(run / f'mask{mask}_selection_logits.npz',
                    selection_logits=np.zeros((2, 12)), selection_labels=np.array([0, 10]),
                    temperature_logits=np.zeros((2, 12)), temperature_labels=np.array([0, 10]),
                    selection_group_ids=np.array(['speaker:a', 'background:selection.wav']),
                    temperature_group_ids=np.array(['speaker:b', 'background:temperature.wav']))
            digest = p.freeze_selection(run, rows)
            self.assertEqual(len(digest), 64)
            self.assertEqual(digest, p.require_test_gate(run))
            (run / 'mask0_selection_logits.npz').write_bytes(b'tampered')
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                p.require_test_gate(run)


if __name__ == '__main__':
    unittest.main()
