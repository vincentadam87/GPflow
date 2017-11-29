from __future__ import absolute_import
import tensorflow as tf
import numpy as np

from .. import settings
from .. import transforms
from .. import conditionals
from .. import kullback_leiblers, features

from ..params import Parameter
from ..params import Minibatch
from ..params import DataHolder

from ..decors import params_as_tensors

from ..models.model import GPModel


class DecoupledSVGP(GPModel):
    """
    SLOW IMPLEMENTATION FOR TESTING
    This is the Decoupled Sparse Variational GP (DSVGP). The key reference is

    ::

      @inproceedings{hensman2014scalable,
        title={Variational Inference for Gaussian Process Models with Linear Complexity},
        author={Ching-An Cheng, Byron Boots},
        booktitle={NIPS},
        year={2017}
      }

    """
    def __init__(self, X, Y, kern, likelihood, Z_a, Z_B,
                 mean_function=None,
                 num_latent=None,
                 minibatch_size=None,
                 num_data=None,
                 **kwargs):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        - kern, likelihood, mean_function are appropriate GPflow objects
        - Z is a matrix of pseudo inputs, size M x D
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - minibatch_size, if not None, turns on mini-batching with that size.
        - num_data is the total number of observations, default to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # sort out the X, Y into MiniBatch objects if required.
        if minibatch_size is None:
            X = DataHolder(X)
            Y = DataHolder(Y)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=1)

        # init the super class, accept args
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.num_data = num_data or X.shape[0]
        self.num_latent = num_latent or Y.shape[1]

        self.Z_a = Parameter(Z_a)
        self.Z_B = Parameter(Z_B)

        num_inducing_a = len(Z_a)
        num_inducing_B = len(Z_B)

        self.q_a = Parameter(np.zeros((num_inducing_a, self.num_latent), dtype=settings.np_float))
        q_B_sqrt = np.array([np.eye(num_inducing_B, dtype=settings.np_float)
                           for _ in range(self.num_latent)]).swapaxes(0, 2)
        self.q_B_sqrt = Parameter(q_B_sqrt, transform=transforms.LowerTriangular(num_inducing_B, self.num_latent))

    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        K_aa = tf.tile(self.kern.K(self.Z_a)[None, :, :], [self.num_latent, 1, 1])
        K_BB = tf.tile(self.kern.K(self.Z_B)[None, :, :], [self.num_latent, 1, 1])

        I = tf.eye(tf.shape(self.Z_B)[0], dtype=settings.tf_float)
        I = tf.tile(I[None, :, :], [self.num_latent, 1, 1])

        q_a = tf.transpose(self.q_a)[:, :, None]  # D_Y, M_a, 1

        q_B_sqrt = tf.transpose(self.q_B_sqrt, [2, 0, 1])  # D_Y, M_B, M_B
        q_B_sqrt = tf.matrix_band_part(q_B_sqrt, -1, 0)

        L = q_B_sqrt
        H = I + tf.matmul(L, tf.matmul(K_BB, L), transpose_a=True)

        chol_H = tf.cholesky(H)

        sqrt_LHinvLT = tf.matrix_triangular_solve(chol_H, tf.transpose(L, [0, 2, 1]), lower=True)
        LHinvLT = tf.matmul(sqrt_LHinvLT, sqrt_LHinvLT, transpose_a=True)

        KL = KL_cheng_boots(q_a, K_aa, K_BB, chol_H, LHinvLT)

        fmean, fvar = conditional_cheng_boots(self.X, self.kern,
                                              q_a, self.Z_a, self.Z_B,
                                              LHinvLT, self.mean_function, full_cov=False)

        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)
        scale = tf.cast(self.num_data, settings.tf_float) / tf.cast(tf.shape(self.X)[0], settings.tf_float)
        return tf.reduce_sum(var_exp) * scale - KL

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        K_ss = tf.tile(self.kern.K(Xnew)[None, :, :], [self.num_latent, 1, 1])
        K_sa = tf.tile(self.kern.K(Xnew, self.Z_a)[None, :, :], [self.num_latent, 1, 1])
        K_sB = tf.tile(self.kern.K(Xnew, self.Z_B)[None, :, :], [self.num_latent, 1, 1])
        K_BB = tf.tile(self.kern.K(self.Z_B)[None, :, :], [self.num_latent, 1, 1])

        q_a = tf.transpose(self.q_a)[:, :, None]  # D_Y, M_a, 1
        q_B_sqrt = tf.transpose(self.q_B_sqrt, [2, 0, 1])  # D_Y, M_B, M_B
        q_B_sqrt = tf.matrix_band_part(q_B_sqrt, -1, 0)

        mu = tf.matmul(K_sa, q_a)  # D_Y, Ns, 1

        B = tf.matmul(q_B_sqrt, q_B_sqrt, transpose_b=True)

        inv_B_inv_plus_K = tf.matrix_inverse(tf.matrix_inverse(B) + K_BB)
        var = K_ss
        var -= tf.matmul(K_sB, tf.matmul(inv_B_inv_plus_K, K_sB, transpose_b=True))

        mu = tf.transpose(mu[:, :, 0])  # take the 1 out

        if not full_cov:
            var = tf.matrix_diag_part(var)
            var = tf.transpose(var)
        else:
            var = tf.transpose(var, [1, 2, 0])

        return mu + self.mean_function(Xnew), var


def KL_cheng_boots(q_a, K_aa, K_BB, chol_H, LHinvLT):
    KL = 0.5 * tf.reduce_sum(tf.matmul(q_a, tf.matmul(K_aa, q_a), transpose_a=True))
    KL += tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_H)))
    KL -= 0.5 * tf.reduce_sum(K_BB * LHinvLT)
    return KL

def conditional_cheng_boots(Xnew, kern, q_a, Z_a, Z_B, LHinvLT, mean_function, full_cov=False):
    num_latent = tf.shape(q_a)[0]

    K_sa = tf.tile(kern.K(Xnew, Z_a)[None, :, :], [num_latent, 1, 1])
    K_sB = tf.tile(kern.K(Xnew, Z_B)[None, :, :], [num_latent, 1, 1])

    mu = tf.matmul(K_sa, q_a)  # D_Y, Ns, 1

    if full_cov:
        var = tf.tile(kern.K(Xnew)[None, :, :], [num_latent, 1, 1])
        var -= tf.matmul(K_sB, tf.matmul(LHinvLT, K_sB, transpose_b=True))
        var = tf.transpose(var, [1, 2, 0])

    else:
        var = tf.tile(kern.Kdiag(Xnew)[None, :], [num_latent, 1])
        var -= tf.reduce_sum(K_sB * tf.matmul(K_sB, LHinvLT), 2)   # D_Y, Ns
        var = tf.transpose(var)

    mu = tf.transpose(mu[:, :, 0])  # take the 1 out

    return mu + mean_function(Xnew), var


class SlowDecoupledSVGP(GPModel):
    """
    SLOW IMPLEMENTATION FOR TESTING
    This is the Decoupled Sparse Variational GP (DSVGP). The key reference is

    ::

      @inproceedings{hensman2014scalable,
        title={Variational Inference for Gaussian Process Models with Linear Complexity},
        author={Ching-An Cheng, Byron Boots},
        booktitle={NIPS},
        year={2017}
      }

    """
    def __init__(self, X, Y, kern, likelihood, Z_a, Z_B,
                 mean_function=None,
                 num_latent=None,
                 minibatch_size=None,
                 num_data=None,
                 **kwargs):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        - kern, likelihood, mean_function are appropriate GPflow objects
        - Z is a matrix of pseudo inputs, size M x D
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - minibatch_size, if not None, turns on mini-batching with that size.
        - num_data is the total number of observations, default to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # sort out the X, Y into MiniBatch objects if required.
        if minibatch_size is None:
            X = DataHolder(X)
            Y = DataHolder(Y)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=1)

        # init the super class, accept args
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.num_data = num_data or X.shape[0]
        self.num_latent = num_latent or Y.shape[1]

        self.Z_a = Parameter(Z_a)
        self.Z_B = Parameter(Z_B)

        num_inducing_a = len(Z_a)
        num_inducing_B = len(Z_B)

        self.q_a = Parameter(np.zeros((num_inducing_a, self.num_latent), dtype=settings.np_float))
        q_B_sqrt = np.array([np.eye(num_inducing_B, dtype=settings.np_float)
                           for _ in range(self.num_latent)]).swapaxes(0, 2)
        self.q_B_sqrt = Parameter(q_B_sqrt, transform=transforms.LowerTriangular(num_inducing_B, self.num_latent))

    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        K_aa = tf.tile(self.kern.K(self.Z_a)[None, :, :], [self.num_latent, 1, 1])
        K_BB = tf.tile(self.kern.K(self.Z_B)[None, :, :], [self.num_latent, 1, 1])

        I = tf.eye(tf.shape(self.Z_B)[0], dtype=settings.tf_float)
        I = tf.tile(I[None, :, :], [self.num_latent, 1, 1])

        q_a = tf.transpose(self.q_a)[:, :, None]  # D_Y, M_a, 1

        q_B_sqrt = tf.transpose(self.q_B_sqrt, [2, 0, 1])  # D_Y, M_B, M_B
        q_B_sqrt = tf.matrix_band_part(q_B_sqrt, -1, 0)

        B = tf.matmul(q_B_sqrt, q_B_sqrt, transpose_b=True)  # D_Y, M_B, M_B

        inv_B_inv_plus_K = tf.matrix_inverse(tf.matrix_inverse(B) + K_BB)

        KL = 0.5 * tf.reduce_sum(tf.matmul(q_a, tf.matmul(K_aa, q_a), transpose_a=True))
        KL += 0.5 * tf.reduce_sum(tf.log(tf.matrix_determinant(I + tf.matmul(K_BB, B))))
        KL += -0.5 * tf.reduce_sum(K_BB * inv_B_inv_plus_K)

        # Get conditionals
        fmean, fvar = self._build_predict(self.X, full_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.tf_float) / tf.cast(tf.shape(self.X)[0], settings.tf_float)

        return tf.reduce_sum(var_exp) * scale - KL

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        K_ss = tf.tile(self.kern.K(Xnew)[None, :, :], [self.num_latent, 1, 1])
        K_sa = tf.tile(self.kern.K(Xnew, self.Z_a)[None, :, :], [self.num_latent, 1, 1])
        K_sB = tf.tile(self.kern.K(Xnew, self.Z_B)[None, :, :], [self.num_latent, 1, 1])
        K_BB = tf.tile(self.kern.K(self.Z_B)[None, :, :], [self.num_latent, 1, 1])

        q_a = tf.transpose(self.q_a)[:, :, None]  # D_Y, M_a, 1
        q_B_sqrt = tf.transpose(self.q_B_sqrt, [2, 0, 1])  # D_Y, M_B, M_B
        q_B_sqrt = tf.matrix_band_part(q_B_sqrt, -1, 0)

        mu = tf.matmul(K_sa, q_a)  # D_Y, Ns, 1

        B = tf.matmul(q_B_sqrt, q_B_sqrt, transpose_b=True)

        inv_B_inv_plus_K = tf.matrix_inverse(tf.matrix_inverse(B) + K_BB)

        var = K_ss
        var -= tf.matmul(K_sB, tf.matmul(inv_B_inv_plus_K, K_sB, transpose_b=True))

        mu = tf.transpose(mu[:, :, 0])  # take the 1 out

        if not full_cov:
            var = tf.matrix_diag_part(var)
            var = tf.transpose(var)
        else:
            var = tf.transpose(var, [1, 2, 0])

        return mu + self.mean_function(Xnew), var
