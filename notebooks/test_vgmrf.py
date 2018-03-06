import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import gpflow
import numpy as np
import matplotlib.pyplot as plt
from gpflow.models import vgmrf
from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF
N = 80
X = np.linspace(0,1,N).reshape(-1,1)
F = np.cos(10*X)
s_n = .5
Y = F + np.random.randn(N,1)*s_n

S = RBF(1,variance=1.,lengthscales=0.1).compute_K_symm(X) + np.eye(N)*1e-3
L = np.linalg.cholesky(S)

f = np.dot(L,np.random.randn(N,))

P = np.linalg.inv(S)
#a = np.ones((N,))
#P = np.diag(-1*np.ones((N-1,)), -1) +\
#    np.diag(2*np.ones((N,)), 0) +\
#    np.diag(-1*np.ones((N-1,)), 1)
#P*=5.
#P = np.eye(N)


print(P)


lik = Gaussian(var=s_n**2)
lik.variance.trainable = False
m = vgmrf.VGMRF(X, Y, P, lik)



m.compile()
opt = gpflow.train.ScipyOptimizer()
opt.minimize(m)

F_pred, V_pred = m.compute_build_predict()


print(F_pred.shape,V_pred.shape)
print(F_pred,V_pred)

plt.plot(X,F,'k-')
plt.plot(X,Y,'k.')
plt.plot(X,F_pred,'r-')
plt.fill_between(X.reshape(-1,),
                 (F_pred-np.sqrt(V_pred)).reshape(-1,),
                 (F_pred+np.sqrt(V_pred)).reshape(-1,),
                 alpha=.1,facecolor='red')
plt.show()

