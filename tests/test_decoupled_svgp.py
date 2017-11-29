__author__ = 'hughsalimbeni'

import tensorflow as tf
import numpy as np
import unittest

import gpflow
from gpflow import settings
from gpflow.models.decoupled_svgp import DecoupledSVGP, SlowDecoupledSVGP
from gpflow.models.svgp import SVGP

from numpy.testing import assert_allclose

class TestDecoupled(unittest.TestCase):
    def test_fast_vs_svgp(self):
        N, D_X, D_Y, M, Ns = 10, 2, 3, 4, 10

        X = np.random.randn(N, D_X)
        Xs = np.random.randn(Ns, D_X)
        Y = np.random.randn(N, D_Y)
        Z = np.random.randn(M, D_X)
        kern = gpflow.kernels.RBF(D_X)
        lik = gpflow.likelihoods.Gaussian()

        m1 = SlowDecoupledSVGP(X, Y, kern, lik, Z, Z)
        m2 = SVGP(X, Y, kern, lik, Z, whiten=False)

        for m in m1, m2:
            m.kern.set_trainable(False)
            m.likelihood.set_trainable(False)

        m1.Z_a.set_trainable(False)
        m1.Z_B.set_trainable(False)
        m2.feature.Z.set_trainable(False)

        gpflow.train.ScipyOptimizer().minimize(m1)
        gpflow.train.ScipyOptimizer().minimize(m2)

        assert_allclose(m1.compute_log_likelihood(), m2.compute_log_likelihood(), rtol=1e-5, atol=1e-5)

        mean1, var1 = m1.predict_f(Xs)
        mean2, var2 = m1.predict_f(Xs)
        assert_allclose(mean1, mean2)
        assert_allclose(var1, var2)


    def test_fast_vs_slow(self):
        N, D_X, D_Y, M_a, M_B = 6, 2, 3, 4, 5

        X = np.random.randn(N, D_X)
        Y = np.random.randn(N, D_Y)
        Z_a = np.random.randn(M_a, D_X)
        Z_B = np.random.randn(M_B, D_X)
        kern = gpflow.kernels.RBF(D_X)
        lik = gpflow.likelihoods.Gaussian()

        q_a = np.random.randn(M_a, D_Y)
        q_B_sqrt = np.random.randn(M_B, M_B, D_Y)

        m_slow = SlowDecoupledSVGP(X, Y, kern, lik, Z_a, Z_B)
        m_fast = DecoupledSVGP(X, Y, kern, lik, Z_a, Z_B)

        for m in m_slow, m_fast:
            m.q_a = q_a
            m.q_B_sqrt = q_B_sqrt

        assert_allclose(m_slow.compute_log_likelihood(),
                        m_fast.compute_log_likelihood())


if __name__ == '__main__':
    unittest.main()