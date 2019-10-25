import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, config_enumerate
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDelta
from pyro import poutine

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
n_steps = 2 if smoke_test else 2000
pyro.set_rng_seed(2)

# enable validation (e.g. validate parameters of distributions)
assert pyro.__version__.startswith('0.4.1')
pyro.enable_validation(True)

# clear the param store in case we're in a REPL
pyro.clear_param_store()

data = torch.tensor([0., 1., 10., 11., 12.])

K = 2  # Fixed number of components.

@config_enumerate
def model(data):
    # Global variables.
    weights = torch.tensor([0.4, 0.6])
    scale = pyro.sample('scale', dist.LogNormal(0., 2.))
    with pyro.plate('components', K):
        locs = pyro.sample('locs', dist.Normal(0., 10.))

    print("actual locs= {}".format(locs.data.numpy()))
    print("actual scale= {}".format(scale.data.numpy()))

    with pyro.plate('data', len(data)):
        # Local variables.
        assignment = pyro.sample('assignment', dist.Categorical(weights))
        pyro.sample('obs', dist.Normal(locs[assignment], scale), obs=data)

# global_guide = AutoDelta(poutine.block(model, expose=['locs', 'scale']))


def guide(data):
    scale_q = pyro.param('scale_q', torch.tensor(1.),
                         constraint=constraints.positive)

    locs_q = pyro.param('locs_q', torch.tensor(5.),
                             constraint=constraints.positive)

    pyro.sample('obs', dist.Normal(locs_q, scale_q))



# setup the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# do gradient steps
for step in range(n_steps):
    svi.step(data)
    if step % 100 == 0:
        print('.', end='')

scale = pyro.param("scale_q").item()
locs = pyro.param("locs_q").item()
print('locs = {}'.format(locs))
print('scale = {}'.format(scale))

