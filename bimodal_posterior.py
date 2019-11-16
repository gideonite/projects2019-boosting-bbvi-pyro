import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam, SGD
from pyro.infer import SVI, Trace_ELBO, config_enumerate
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDelta
from pyro import poutine    
from pyro.poutine import trace, replay, block
from functools import partial
import numpy as np
import scipy.stats
from pyro.infer.autoguide import AutoDelta
from collections import defaultdict
import matplotlib
from matplotlib import pyplot

PRINT_INTERMEDIATE_LATENT_VALUES = True
PRINT_TRACES = False

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
n_steps = 2 if smoke_test else 5000
pyro.set_rng_seed(2)

# enable validation (e.g. validate parameters of distributions)
pyro.enable_validation(True)

# clear the param store in case we're in a REPL
pyro.clear_param_store()

data = torch.tensor([4.0, 4.2, 3.9, 4.1, 3.8, 3.5, 4.3])

def guide(data, index):
    scale_q = pyro.param('scale_{}'.format(index), torch.tensor([1.0]), constraints.positive)
    loc_q = pyro.param('loc_{}'.format(index), torch.tensor([1.0]))
    pyro.sample('loc', dist.Normal(loc_q, scale_q))

@config_enumerate
def model(data):
    # Global variables.
    prior_loc = torch.tensor([0.])
    prior_scale = torch.tensor([5.])
    loc = pyro.sample('loc', dist.Normal(prior_loc, prior_scale))
    scale = torch.tensor([1.])

    with pyro.plate('data', len(data)):
        # Local variables.
        pyro.sample('obs', dist.Normal(loc**2, scale), obs=data)

@config_enumerate
def approximation(data, components, weights):
    if len(weights) > 0:
        assignment = pyro.sample('assignment', dist.Categorical(weights))
        distribution = components[assignment](data)
    return

def relbo(model, guide, *args, **kwargs):

    approximation = kwargs.pop('approximation', None)
    # Run the guide with the arguments passed to SVI.step() and trace the execution,
    # i.e. record all the calls to Pyro primitives like sample() and param().
    #print("enter relbo")
    guide_trace = trace(guide).get_trace(*args, **kwargs)
    #print(guide_trace.nodes['obs_1'])
    model_trace = trace(replay(model, guide_trace)).get_trace(*args, **kwargs)
    #print(model_trace.nodes['obs_1'])


    approximation_trace = trace(
            replay(block(approximation, expose=['loc']), guide_trace)
        ).get_trace(*args, **kwargs)
    # We will accumulate the various terms of the ELBO in `elbo`.

    # This is how we computed the ELBO before using TraceEnum_ELBO:
    elbo = model_trace.log_prob_sum() - guide_trace.log_prob_sum() - approximation_trace.log_prob_sum()

    #loss_fn = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1).differentiable_loss(model,
    #                                                           guide,
    #                                                    *args, **kwargs)
    #loss_fn = pyro.infer.Trace_ELBO().differentiable_loss(
    #        model, guide, *args, **kwargs)

    # print(loss_fn)
    # print(approximation_trace.log_prob_sum())
    #elbo = -loss_fn - approximation_trace.log_prob_sum()

    if PRINT_TRACES:
        print(
                model_trace.log_prob_sum(),
                guide_trace.log_prob_sum(),
                approximation_trace.log_prob_sum())

    # Return (-elbo) since by convention we do gradient descent on a loss and
    # the ELBO is a lower bound that needs to be maximized.
    return -elbo


def boosting_bbvi():
    n_iterations = 2

    components = []
    weights = torch.tensor([])
    wrapped_approximation = partial(approximation, components=components,
                                    weights=weights)
    locs = []
    scales = []

    gradient_norms = defaultdict(list)
    for t in range(n_iterations):
        print(weights)
        # setup the inference algorithm
        wrapped_guide = partial(guide, index=t)
        # do gradient steps
        losses = []
        # Register hooks to monitor gradient norms.
        wrapped_guide(data)
        print(pyro.get_param_store().named_parameters())

        adam_params = {"lr": 0.002, "betas": (0.90, 0.999)}
        optimizer = Adam(adam_params)
        for name, value in pyro.get_param_store().named_parameters():
            if not name in gradient_norms:
                value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

        svi = SVI(model, wrapped_guide, optimizer, loss=relbo)
        for step in range(n_steps):
            loss = svi.step(data, approximation=wrapped_approximation)
            losses.append(loss)

            if step % 100 == 0:
                if PRINT_INTERMEDIATE_LATENT_VALUES:
                    print('Loss: {}'.format(loss))
                    scale = pyro.param("scale_{}".format(t)).item()
                    loc = pyro.param("loc_{}".format(t)).item()
                    print('loc = {}'.format(loc))
                    print('scale = {}'.format(scale))
                else:
                    print('.', end=' ')

        pyplot.plot(range(len(losses)), losses)
        pyplot.xlabel('Update Steps')
        pyplot.ylabel('-ELBO')
        pyplot.title('-ELBO against time for component {}'.format(t+1));
        pyplot.show()

        components.append(wrapped_guide)
        new_weight = 2 / (t + 2)

        weights = weights * (1-new_weight)
        weights = torch.cat((weights, torch.tensor([new_weight])))

        wrapped_approximation = partial(approximation, components=components, weights=weights)

        scale = pyro.param("scale_{}".format(t)).item()
        scales.append(scale)
        loc = pyro.param("loc_{}".format(t)).item()
        locs.append(loc)
        print('loc = {}'.format(loc))
        print('scale = {}'.format(scale))

    pyplot.figure(figsize=(10, 4), dpi=100).set_facecolor('white')
    for name, grad_norms in gradient_norms.items():
        pyplot.plot(grad_norms, label=name)
        pyplot.xlabel('iters')
        pyplot.ylabel('gradient norm')
        # pyplot.yscale('log')
        pyplot.legend(loc='best')
        pyplot.title('Gradient norms during SVI');
    pyplot.show()

    print(weights)
    print(locs)
    print(scales)

    pyplot.figure(figsize=(10, 4), dpi=100).set_facecolor('white')
    X = np.arange(-10, 10, 0.1)
    Y_all = 0.0
    for t in range(n_iterations):
        Y = weights[t].item() * scipy.stats.norm.pdf((X - locs[t]) / scales[t])
        Y_all = Y_all + Y

        pyplot.plot(X, Y)
    pyplot.plot(X, Y_all, 'k--')
    pyplot.plot(data.data.numpy(), np.zeros(len(data)), 'k*')
    pyplot.title('Approximation of posterior over loc')
    pyplot.ylabel('probability density');
    pyplot.show()

if __name__ == '__main__':
  boosting_bbvi()
