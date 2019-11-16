# %%
from mxnet import nd
from mxnet.gluon import nn


# %%
X = nd.random.uniform(shape=(2, 20))

# %%


class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        x = nd.relu(nd.dot(x, self.rand_weight.data())+1)
        x = self.dense(x)
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()

class NestMLP(nn.Block):
    def __init__(self,**kwargs):
        super(NestMLP,self).__init__(**kwargs)
        self.net=nn.Sequential()
        self.net.add(nn.Dense(64,activation='relu'))
        self.net.add(nn.Dense(32,activation='relu'))
        self.dense=nn.Dense(16,activation='relu')
    def forward(self,x):
        return self.dense(self.net(x))
    
# %%
net = nn.Sequential()
net.add(NestMLP(),nn.Dense(20),FancyMLP())
net.initialize()
print(net(X))
# %%
