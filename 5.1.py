# %%
from mxnet import autograd, nd
from mxnet.gluon import nn

# %%


def corr2d(X, k):
    h, w = k.shape
    Y = nd.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w]*k).sum()
    return Y


# %%
X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
k = nd.array([[0, 1], [2, 3]])
print(corr2d(X, k))


# %%
