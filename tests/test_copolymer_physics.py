import unittest

from src.ml.copolymer_physics import (
    BinaryCopolymerPoint,
    fit_kwei_k_q,
    fox_tg_k,
    gordon_taylor_binary_tg_k,
    kwei_binary_tg_k,
)


class TestCopolymerPhysics(unittest.TestCase):
    def test_fox_returns_endpoint_for_pure_component(self):
        self.assertAlmostEqual(fox_tg_k([300.0, 500.0], [1.0, 0.0]), 300.0)

    def test_kwei_reduces_to_gordon_taylor_when_q_zero(self):
        gt = gordon_taylor_binary_tg_k(250.0, 360.0, 0.8, 1.4)
        kwei = kwei_binary_tg_k(250.0, 360.0, 0.8, 1.4, 0.0)
        self.assertAlmostEqual(kwei, gt)

    def test_fit_kwei_recovers_synthetic_interaction(self):
        points = []
        for w1 in [0.2, 0.4, 0.6, 0.8]:
            target = kwei_binary_tg_k(250.0, 360.0, w1, 1.5, 80.0)
            points.append(BinaryCopolymerPoint(250.0, 360.0, w1, target))

        k, q = fit_kwei_k_q(points)

        self.assertAlmostEqual(k, 1.5, delta=0.02)
        self.assertAlmostEqual(q, 80.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
