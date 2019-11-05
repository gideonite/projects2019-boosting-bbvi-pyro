import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam, SGD
from pyro.infer import SVI, Trace_ELBO, config_enumerate, TraceEnum_ELBO
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDelta
from pyro import poutine    
from pyro.poutine import trace, replay, block
from functools import partial
import numpy as np
import scipy.stats
from pyro.infer.autoguide import AutoDelta
from collections import defaultdict
from matplotlib import pyplot

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
n_steps = 2 if smoke_test else 2000
pyro.set_rng_seed(2)

# enable validation (e.g. validate parameters of distributions)
pyro.enable_validation(True)

# clear the param store in case we're in a REPL
pyro.clear_param_store()

data = torch.tensor([0., 1., 2., 0, 0.5, 1.5, 10., 11., 12., 10.6, 11.8, 12.2])

K = 2  # Fixed number of components.

# @config_enumerate
# def model(data):
#     weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
#     scale_1 = pyro.sample('scale_1', dist.LogNormal(0., 2.))
#     scale_2 = pyro.sample('scale_2', dist.LogNormal(0., 2.))
#     scales = [scale_1, scale_2]
#     loc_1 = pyro.sample('loc_1', dist.Normal(0., 10.))
#     loc_2 = pyro.sample('loc_2', dist.Normal(0., 10.))
#     locs = [loc_1, loc_2]

#     # Local variables.
#     with pyro.plate('data', len(data)):
#         assignment = pyro.sample('assignment', dist.Categorical(weights))
#         pyro.sample('obs', dist.Normal(locs[assignment], scales[assignment]), obs=data)

#@config_enumerate
def model(data):
    # Global variables.
    weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
    scale = pyro.sample('scale', dist.LogNormal(0., 2.))
    with pyro.plate('components', K):
        locs = pyro.sample('locs', dist.Normal(0., 10.))

    for i in pyro.plate('data', len(data)):
        # Local variables.
        assignment = pyro.sample('assignment_{}'.format(i), dist.Categorical(weights))
        pyro.sample('obs_{}'.format(i), dist.Normal(locs[assignment], scale), obs=data[i])



#@config_enumerate
def guide(data):
    weights = torch.tensor([0.5, 0.5])
    scale_1 = pyro.param('scale_q1', torch.tensor([1.0]), constraints.positive)
    scale_2 = pyro.param('scale_q2', torch.tensor([1.0]), constraints.positive)
    scales = [scale_1, scale_2]
    loc_1 = pyro.param('loc_q1', torch.tensor([5.0]))
    loc_2 = pyro.param('loc_q2', torch.tensor([5.0]))
    locs = [loc_1, loc_2]

    for i in pyro.plate('data', len(data)):
        # Local variables.
        assignment = pyro.sample('assignment_{}'.format(i), dist.Categorical(weights))
        pyro.sample('obs_{}'.format(i), dist.Normal(locs[assignment], scale_1))


@config_enumerate
def simple_model(data):
    # Global variables    
    scale = pyro.sample('scale_q', dist.LogNormal(0., 2.))
    locs = pyro.sample('locs_q', dist.Normal(0., 10.))
    for i in pyro.plate('data', len(data)):
        # Local variables.
        pyro.sample('obs_{}'.format(i), dist.Normal(locs, scale), obs=data[i])


global_guide = AutoDelta(poutine.block(simple_model, expose=['locs_q', 'scale_q']))

# setup the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=TraceEnum_ELBO())
	

def initialize(seed):
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    # Initialize weights to uniform.
    #pyro.param('auto_weights', 0.5 * torch.ones(K), constraint=constraints.simplex)
    # Assume half of the data variance is due to intra-component noise.
    pyro.param('auto_scale_q', (data.var() / 2).sqrt(), constraint=constraints.positive)
    # Initialize means from a subsample of data.
    pyro.param('auto_locs_q', data.mean());
    loss = svi.loss(model, global_guide, data)
    return loss

# Choose the best among 100 random initializations.
loss, seed = min((initialize(seed), seed) for seed in range(100))
initialize(seed)
print('seed = {}, initial_loss = {}'.format(seed, loss))

# do gradient steps
for step in range(n_steps):
    svi.step(data)
    if step % 100 == 0:
        print('.', end='')

map_estimates = global_guide(data)

#weights = map_estimates['weights']
locs = map_estimates['locs_q']
scale = map_estimates['scale_q']
#print('weights = {}'.format(weights.data.numpy()))
print('locs = {}'.format(locs.data.numpy()))
print('scale = {}'.format(scale.data.numpy()))


X = np.arange(-3,15,0.1)
Y1 = scipy.stats.norm.pdf((X - locs.item()) / scale.item())
#Y1 = weights[0].item() * scipy.stats.norm.pdf((X - locs[0].item()) / scale.item())
#Y2 = weights[1].item() * scipy.stats.norm.pdf((X - locs[1].item()) / scale.item())

pyplot.figure(figsize=(10, 4), dpi=100).set_facecolor('white')
pyplot.plot(X, Y1, 'r-')
#pyplot.plot(X, Y2, 'b-')
#pyplot.plot(X, Y1 + Y2, 'k--')
pyplot.plot(data.data.numpy(), np.zeros(len(data)), 'k*')
pyplot.title('Density of two-component mixture model')
pyplot.ylabel('probability density');
pyplot.show()
