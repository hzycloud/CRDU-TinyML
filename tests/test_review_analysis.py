import unittest
from scripts import review_reanalysis as analysis


class ReviewAnalysisTest(unittest.TestCase):
    def test_objective_split_does_not_use_test_metrics(self):
        row = {'search_raw': {'nll': 2.}, 'search_calibrated': {'nll': 1., 'brier': .4, 'ece': .1}}
        self.assertEqual(analysis.objectives(row)['ts_brier'], .4)
        self.assertAlmostEqual(analysis.objectives(row)['mix_0.5'], 1.2)
        before = analysis.objectives(row)
        row['eval_calibrated'] = {'nll': -1000}
        self.assertEqual(before, analysis.objectives(row))

    def test_additive_uses_only_reference_and_singletons(self):
        values = {0: 1., 1: 1.2, 2: 1.4, 3: 999.}
        self.assertAlmostEqual(analysis.additive_prediction(3, values, 2), 1.6)

    def test_frozen_decisions_ignore_any_test_metrics(self):
        freeze = getattr(analysis, 'freeze_decisions', None)
        self.assertIsNotNone(freeze, 'selection decisions must be frozen before test evaluation')
        rows = [dict(mask=mask, bits={'layer':8 if mask==0 else 4}, flash_fraction=1.-.5*mask,
            search_raw={'nll':1.-.1*mask,'accuracy':.9}, search_calibrated={'nll':1.-.1*mask,'brier':.2,'ece':.03})
            for mask in (0,1)]
        before = freeze(rows)
        for row in rows:
            row['eval_calibrated'] = {'accuracy':0 if row['mask']==1 else 1}
        self.assertEqual(before, freeze(rows))


if __name__ == '__main__':
    unittest.main()
