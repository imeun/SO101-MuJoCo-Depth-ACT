import unittest

import numpy as np
import torch

from so101_depth_act import (
    DepthACTPolicy,
    JointVelocityEstimator,
    TemporalActionEnsembler,
    decode_delta_chunk,
    depth_act_loss,
)


class DepthACTPolicyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.policy = DepthACTPolicy(
            chunk_size=5,
            d_model=64,
            nhead=4,
            encoder_layers=1,
            decoder_layers=1,
            backbone_channels=(8, 16, 32, 64),
        )

    def test_forward_contract_and_backward_are_finite(self):
        top = torch.rand(2, 1, 64, 80)
        side = torch.rand(2, 1, 64, 80)
        proprio = torch.rand(2, 12)
        prediction = self.policy(top, side, proprio)
        self.assertEqual(tuple(prediction.shape), (2, 5, 6))
        target = torch.randn_like(prediction) * 0.05
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
        loss, metrics = depth_act_loss(prediction, target, mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["delta_loss"], 0.0)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.policy.parameters()))

    def test_decode_delta_chunk_adds_current_position(self):
        current = torch.tensor([[1.0, 2.0]])
        delta = torch.tensor([[[0.1, -0.2], [0.3, 0.4]]])
        decoded = decode_delta_chunk(current, delta)
        torch.testing.assert_close(decoded, torch.tensor([[[1.1, 1.8], [1.3, 2.4]]]))

    def test_zero_delta_baseline_is_reported(self):
        target = torch.ones(1, 3, 6) * 0.2
        prediction = target.clone()
        mask = torch.ones(1, 3, dtype=torch.bool)
        loss, metrics = depth_act_loss(prediction, target, mask)
        self.assertLess(float(loss), metrics["zero_delta_baseline"])

    def test_temporal_ensemble_combines_overlapping_absolute_chunks(self):
        ensemble = TemporalActionEnsembler(chunk_size=3, decay=0.0)
        first = np.array([[1.0], [2.0], [3.0]])
        second = np.array([[4.0], [5.0], [6.0]])
        np.testing.assert_allclose(ensemble.add_and_get(first), [1.0])
        np.testing.assert_allclose(ensemble.add_and_get(second), [3.0])
        np.testing.assert_allclose(ensemble.advance_without_prediction(), [4.0])

    def test_joint_velocity_estimator_matches_collection_finite_difference(self):
        estimator = JointVelocityEstimator(control_period_s=0.034, joint_count=2)

        np.testing.assert_allclose(estimator.update(np.array([1.0, -1.0])), [0.0, 0.0])
        np.testing.assert_allclose(
            estimator.update(np.array([1.034, -0.983])),
            [1.0, 0.5],
            rtol=1e-6,
            atol=1e-6,
        )

        estimator.reset()
        np.testing.assert_allclose(estimator.update(np.array([3.0, 4.0])), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
