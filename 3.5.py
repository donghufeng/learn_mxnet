# %%
import d2lzh as d2l
from mxnet.gluon import data as gdata
from mxnet import autograd, nd
import sys
import time


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition


def net(X):
    XX = nd.dot(X.reshape((-1, num_inputs)), W1) + b1
    return softmax(nd.dot(XX, W2)+b2)
    # return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)


def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()


def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


def evaluate_accuracy(test_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in test_iter:
        acc_sum += accuracy(net(X), y)*y.size
        n += y.size
    return acc_sum / n


# %%
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# %%
num_inputs = 784
num_l2 = 20
num_outputs = 10
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_l2))
b1 = nd.zeros(num_l2)
W2 = nd.random.normal(scale=0.01, shape=(num_l2, num_outputs))
b2 = nd.zeros(num_outputs)
W1.attach_grad()
b1.attach_grad()
W2.attach_grad()
b2.attach_grad()

# %%
num_epochs, lr = 25, 0.1


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += accuracy(y_hat, y) * y.size
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' %
              (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy,
          num_epochs, batch_size, [W1, b1, W2, b2], lr)


# %%
