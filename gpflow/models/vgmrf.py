

# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, Mark van der Wilk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

from ..models.model import Model
from ..params import Parameterized, DataHolder
from ..decors import autoflow



class VGMRF(Model):
    """
    This is the Variational GMRF : the D.U.M.B. way, not exploiting any structure

    """
    def __init__(self, X, Y, P, likelihood,
                 num_latent=None,
                 num_data=None,
                 **kwargs):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        - num_data is the total number of observations, default to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        Model.__init__(self,name='SVGMRF')

        # sort out the X, Y into MiniBatch objects if required.
        if isinstance(X, np.ndarray):
            # X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            # Y is a data matrix, rows correspond to the rows in X,
            # columns are treated independently
            Y = DataHolder(Y)
        self.X, self.Y = X, Y


        self.likelihood = likelihood

        # prior precision matrix
        self.P = tf.constant(P,dtype=settings.float_type)
        self.Lp = tf.cholesky(self.P)

        # init the super class, accept args
        self.num_data = num_data or X.shape[0]
        self.num_latent = num_latent or Y.shape[1]

        # Initialize the variational parameters
        self.q_mu = Parameter(np.zeros((self.num_data, self.num_latent), dtype=settings.float_type))
        q_sqrt_prec = np.array([np.eye(self.num_data, dtype=settings.float_type)
                           for _ in range(self.num_latent)])
        self.q_sqrt_prec = Parameter(q_sqrt_prec, transform=transforms.LowerTriangular(self.num_data, self.num_latent))

    @params_as_tensors
    def build_prior_KL(self):
        return kullback_leiblers.gauss_kl(self.q_mu,
                                          tf.matrix_inverse(self.q_sqrt_prec), K=None)

    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self._build_predict( full_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(tf.shape(self.X)[0], settings.float_type)

        return tf.reduce_sum(var_exp) * scale - KL

    @params_as_tensors
    def _build_predict(self,  full_cov=False):
        """
            Given q(x) defined using the prior as v= N(m,P= Lqv Lqv^T) and x = Lp^-T v+ mu
            q(x) is N(Lp^-T mu, P = Lqx Lqx^T ) with Lqx=LpLqv
        """

        m = tf.matrix_triangular_solve( self.Lp ,self.q_mu, adjoint=True)


        Lqx = tf.einsum('ij,rjk->rik',self.Lp,
                                    self.q_sqrt_prec)

        Px = tf.einsum('rij,rkj->rik',Lqx,Lqx) # Lqx Lqx^T
        Sx = tf.matrix_inverse(Px)
        Sx_dd = tf.einsum('rij->ir',Sx) #

        return m, Sx_dd

    @autoflow()
    def compute_build_predict(self,  full_cov=False):
        """Compute the log prior of the model."""
        return self._build_predict(full_cov=full_cov)

    @autoflow()
    def compute_log_likelihood(self):
        """Compute the log prior of the model."""
        return self._build_likelihood()




class VGMRF_banded(Model):
    """
    This is the Variational GMRF : using band diagonal precision matrices

    """
    def __init__(self, X, Y, P_band, likelihood,
                 num_latent=None,
                 num_data=None,
                 **kwargs):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        - num_data is the total number of observations, default to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        Model.__init__(self,name='SVGMRF')

        # sort out the X, Y into MiniBatch objects if required.
        if isinstance(X, np.ndarray):
            # X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            # Y is a data matrix, rows correspond to the rows in X,
            # columns are treated independently
            Y = DataHolder(Y)
        self.X, self.Y = X, Y


        self.likelihood = likelihood

        # prior precision matrix
        self.band_width = P_band.shape[0]
        self.P_band = tf.constant(P_band,dtype=settings.float_type)

        # TODO :  band -> cholesky band
        self.Lp_band = cholesky_banded(self.P_band)

        # init the super class, accept args
        self.num_data = num_data or X.shape[0]
        self.num_latent = num_latent or Y.shape[1]

        # Initialize the variational parameters
        self.q_mu = Parameter(np.zeros((self.num_data, self.num_latent), dtype=settings.float_type))

        # TODO: check dim ordering (keep as in SVGPVGMR)
        shape = (self.num_latent,self.band_width,self.num_data) # R x B x N
        q_sqrt_prec_band = np.zeros(shape, dtype=settings.float_type)
        q_sqrt_prec_band[0,:,:] = 1. # initialize as identity stored as band
        self.q_sqrt_prec_band = Parameter(q_sqrt_prec_band, transform=transforms.LowerTriangular(self.num_data, self.num_latent))

    @params_as_tensors
    def build_prior_KL(self):
        return kullback_leiblers.gauss_kl(self.q_mu,
                                          tf.matrix_inverse(self.q_sqrt_prec), K=None)

    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self._build_predict( full_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(tf.shape(self.X)[0], settings.float_type)

        return tf.reduce_sum(var_exp) * scale - KL

    @params_as_tensors
    def _build_predict(self,  full_cov=False):
        """
            Given q(x) defined using the prior as v= N(m,P= Lqv Lqv^T) and x = Lp^-T v+ mu
            q(x) is N(Lp^-T mu, P = Lqx Lqx^T ) with Lqx=LpLqv
        """

        # TODO : implement matrix_triangular_solve_banded
        m = matrix_triangular_solve_banded( self.Lp_band ,self.q_mu, adjoint=True)


        # TODO : implement band_matrices_product
        Lqx = band_matrices_product(self.Lp_band, self.q_sqrt_prec_band)

        if full_cov:
            # TODO : implement chol_band_inverse
            Sx = chol_band_inverse(Lqx)
        else:
            # TODO : implement diag_chol_band_inverse
            Sx = diag_chol_band_inverse(Lqx)

        return m, Sx

    @autoflow()
    def compute_build_predict(self,  full_cov=False):
        """Compute the log prior of the model."""
        return self._build_predict(full_cov=full_cov)

    @autoflow()
    def compute_log_likelihood(self):
        """Compute the log prior of the model."""
        return self._build_likelihood()



# TODO implement
def cholesky_banded(band,lower=False):
    """
    Computes the banded cholesky decomposition of a banded matrix
    :param band: the upper-diagonal band representation of a hermitian matrix ( K x N )
    :return: chol: the upper-diagoanl band representation of the cholesky decomposition of band ( K x N )
    """
    raise NotImplementedError

# TODO implement
def matrix_triangular_solve_banded(b, rhs,lower=True,adjoint=False):
    """
    Solves systems of linear equations with banded, upper or lower triangular matrices
    matrix * output = rhs
    adjoint(matrix) * output = rhs (if adjoint==True)
    See : https://www.tensorflow.org/api_docs/python/tf/matrix_triangular_solve
    :param b: [..., B, N]
    :param rhs: [..., N, R]
    :return: [...,N, R]
    """
    raise NotImplementedError

# TODO implement
def band_matrices_product(b1, b2, transpose_b2=False):
    """
    Computes band representation of M1 * M2 (or M1 * M2^T), which are given in band form
    :param b1: B x N
    :param b2: B x N
    :return: (2 x B - 1) x N
    """
    raise NotImplementedError

# TODO implement
def diag_chol_band_inverse(Lb):
    """
    Computes the diagonal of M^-1 for M = LL^T
    :param L: [..., B, N]
    :return: [..., B, N]
    """
    raise NotImplementedError


# TODO implement
def chol_band_inverse(Lb):
    """
    Computes the diagonal of M^-1 for M = LL^T
    :param L: [..., B, N]
    :return: [..., N]
    """
    raise NotImplementedError
