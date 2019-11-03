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
from matplotlib import pyplot

PRINT_INTERMEDIATE_LATENT_VALUES = True
PRINT_TRACES = False

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
n_steps = 2 if smoke_test else 1000
pyro.set_rng_seed(2)

# enable validation (e.g. validate parameters of distributions)
pyro.enable_validation(True)

# clear the param store in case we're in a REPL
pyro.clear_param_store()

data = torch.tensor([0., 1., 2., 0, 0.5, 1.5, 10., 11., 12., 10.6, 11.8, 12.2])

# n = torch.distributions.Normal(torch.tensor([2.0]), torch.tensor([1.0]))
# m = torch.distributions.Normal(torch.tensor([10.0]), torch.tensor([1.0]))
#
# data = torch.cat((n.sample((60,)), m.sample((40,))))


K = 2  # Fixed number of components.

@config_enumerate
def model(data):

    weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
    scale = pyro.sample('scale', dist.LogNormal(0., 2.))
    with pyro.plate('components', K):
        locs = pyro.sample('locs', dist.Normal(0., 10.))

    # Local variables.
    for i in pyro.plate('data', len(data)):
        assignment = pyro.sample('assignment_{}'.format(i), dist.Categorical(weights))
        pyro.sample('obs_{}'.format(i), dist.Normal(locs[assignment], scale), obs=data[i])


def guide(data, index, sample_id=None):
    # TODO: We removed the positive constraint here to make Adam work
    scale_q = pyro.param('scale_{}'.format(index), torch.tensor(1.))

    locs_q = pyro.param('locs_{}'.format(index), torch.tensor(10.))

    pyro.sample('obs_{}'.format(sample_id), dist.Normal(locs_q, scale_q))


@config_enumerate
def approximation(data, components, weights):
    for i in pyro.plate('samples', len(data)):
        assignment = pyro.sample('assignment_{}'.format(i),
            dist.Categorical(weights))
        components[assignment](data, sample_id=i)


def dummy_approximation(data):

    scale_a = torch.tensor([1.])
    locs_a = torch.tensor([20.])
    sample = pyro.sample('obs', dist.Normal(locs_a, scale_a))


def boosting_bbvi():
    n_iterations = 2

    initial_approximation = partial(guide, index=0)
    components = [initial_approximation]
    weights = torch.tensor([1.])
    wrapped_approximation = partial(approximation, components=components,
                                    weights=weights)

    locs = [0]
    scales = [0]

    gradient_norms = defaultdict(list)
    for t in range(1, n_iterations + 1):
        # setup the inference algorithm

        block(wrapped_approximation, expose_fn=lambda site: 'obs_' not in site['name'])
        wrapped_guide = partial(guide, index=t)
        # svi = SVI(model, wrapped_guide, optimizer, loss=relbo)

        # do gradient steps
        losses = []
        # Register hooks to monitor gradient norms.
        wrapped_guide(data)
        param_name_1 = 'scale_{}'.format(t)
        param_name_2 = 'locs_{}'.format(t)
        adam_params = [pyro.param(param_name_1), pyro.param(param_name_2)]
        optimizer = torch.optim.Adam(adam_params, lr=0.001)
        pyro.get_param_store().named_parameters()
        for name, value in pyro.get_param_store().named_parameters():
            if not name in gradient_norms:
                value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))
        
        for step in range(n_steps):
            print('Step: ', step)
            approximation_log_prob = trace(wrapped_approximation).get_trace(data).log_prob_sum()
            loss_fn = pyro.infer.TraceEnum_ELBO().differentiable_loss(model,
                wrapped_guide,
                data)
            loss = loss_fn
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss)

            if PRINT_INTERMEDIATE_LATENT_VALUES:
                print('Loss: {}'.format(loss))
                scale = pyro.param("scale_{}".format(t)).item()
                loc = pyro.param("locs_{}".format(t)).item()
                print('locs = {}'.format(loc))
                print('scale = {}'.format(scale))

            if step % 100 == 0:
                print('.', end=' ')

        pyplot.plot(range(len(losses)), losses)
        pyplot.xlabel('Update Steps')
        pyplot.ylabel('-ELBO')
        pyplot.title('-ELBO against time for component {}'.format(t));
        pyplot.show()

        components.append(wrapped_guide)
        new_weight = 2 / (t + 1)

        weights = weights * (1-new_weight)
        weights = torch.cat((weights, torch.tensor([new_weight])))

        wrapped_approximation = partial(approximation, components=components, weights=weights)


        scale = pyro.param("scale_{}".format(t)).item()
        scales.append(scale)
        loc = pyro.param("locs_{}".format(t)).item()
        locs.append(loc)
        print('locs = {}'.format(loc))
        print('scale = {}'.format(scale))

    pyplot.figure(figsize=(10,4), dpi=100).set_facecolor('white')
    for name, grad_norms in gradient_norms.items():
        pyplot.plot(grad_norms, label=name)
        pyplot.xlabel('iters')
        pyplot.ylabel('gradient norm')
        # pyplot.yscale('log')
        pyplot.legend(loc='best')
        pyplot.title('Gradient norms during SVI');
    pyplot.show()  


    for t in range(1, n_iterations + 1):
        scale = pyro.param("scale_{}".format(t)).item()
        loc = pyro.param("locs_{}".format(t)).item()
        print('locs = {}'.format(loc))
        print('scale = {}'.format(scale))


    print(weights)
    print(locs)
    print(scales)

    X = np.arange(-3, 18, 0.1)
    Y1 = weights[1].item() * scipy.stats.norm.pdf((X - locs[1]) / scales[1])
    Y2 = weights[2].item() * scipy.stats.norm.pdf((X - locs[2]) / scales[2])
    #Y3 = weights[3].item() * scipy.stats.norm.pdf((X - locs[3] / scales[3]))

    pyplot.figure(figsize=(10, 4), dpi=100).set_facecolor('white')
    pyplot.plot(X, Y1, 'r-')
    pyplot.plot(X, Y2, 'b-')
    pyplot.plot(X, Y1 + Y2, 'k--')
    pyplot.plot(data.data.numpy(), np.zeros(len(data)), 'k*')
    pyplot.title('Density of two-component mixture model')
    pyplot.ylabel('probability density');
    pyplot.show()


if __name__ == '__main__':

    boosting_bbvi()

