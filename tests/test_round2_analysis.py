import unittest
import numpy as np
import importlib.util


class Round2AnalysisTest(unittest.TestCase):
    def module(self):
        spec=importlib.util.find_spec('cig_mpq.round2_analysis')
        self.assertIsNotNone(spec,'round-two analysis module is missing')
        from cig_mpq import round2_analysis
        return round2_analysis

    def test_feasible_counts_distinguish_singleton_and_empty(self):
        m=self.module()
        rows=[{'mask':0,'flash_fraction':1.,'search_raw':{'accuracy':.9}},
              {'mask':1,'flash_fraction':.8,'search_raw':{'accuracy':.87}}]
        self.assertEqual(m.feasible_masks(rows,.8,.01),[])
        self.assertEqual(m.feasible_masks(rows,.8,.04),[1])

    def test_pair_surface_predicts_known_quadratic_heldout_mask(self):
        m=self.module()
        f=lambda x:1+2*(x&1>0)+3*(x&2>0)+4*(x&4>0)+5*(x&3==3)
        values={i:f(i) for i in range(8)}
        self.assertEqual(m.surface_prediction(7,values,3,True),15)
        self.assertEqual(m.surface_prediction(7,values,3,False),10)

    def test_weighted_ece_equals_explicit_resampling(self):
        m=self.module()
        p=np.array([[[.7,.3],[.2,.8],[.9,.1]]])
        y=np.array([0,0,1])
        confidence=p.max(axis=2)
        correct=p.argmax(axis=2)==y
        idx=np.array([0,1,1,2])
        got=m.ece_from_indices(confidence,correct,idx,2)[0]
        from cig_mpq.metrics import expected_calibration_error
        self.assertAlmostEqual(got,expected_calibration_error(p[0,idx],y[idx],2))


if __name__=='__main__':unittest.main()
