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
import matplotlib
from matplotlib import pyplot
from bbbvi import relbo, Approximation

PRINT_INTERMEDIATE_LATENT_VALUES = False
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

model_log_prob = []
guide_log_prob = []
approximation_log_prob = []

def guide(data, index):
    variance_q = pyro.param('variance_{}'.format(index), torch.tensor([1.0]), constraints.positive)
    mu_q = pyro.param('mu_{}'.format(index), torch.tensor([1.0]))
    pyro.sample("mu", dist.Normal(mu_q, variance_q))

@config_enumerate
def model(data):
    # Global variables.
    prior_mu = torch.tensor([0.])
    prior_variance = torch.tensor([5.])
    mu = pyro.sample('mu', dist.Normal(prior_mu, prior_variance))
    variance = torch.tensor([1.])

    for i in pyro.plate('data', len(data)):
        # Local variables.
        pyro.sample('obs_{}'.format(i), dist.Normal(mu*mu, variance), obs=data[i])


def dummy_approximation(data):
    variance_q = pyro.param('variance_0', torch.tensor([1.0]), constraints.positive)
    mu_q = pyro.param('mu_0', torch.tensor([20.0]))
    pyro.sample("mu", dist.Normal(mu_q, variance_q))

def boosting_bbvi():
    n_iterations = 2
    relbo_lambda = 1
    initial_approximation = dummy_approximation
    components = [initial_approximation]
    weights = torch.tensor([1.])
    wrapped_approximation = Approximation(components, weights)

    locs = [0]
    scales = [0]

    gradient_norms = defaultdict(list)
    duality_gap = []
    entropies = []
    model_log_likelihoods = []
    for t in range(1, n_iterations + 1):
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
        
        global model_log_prob
        model_log_prob = []
        global guide_log_prob
        guide_log_prob = []
        global approximation_log_prob
        approximation_log_prob = []


        svi = SVI(model, wrapped_guide, optimizer, loss=relbo)
        for step in range(n_steps):
            loss = svi.step(data, approximation=wrapped_approximation, relbo_lambda=relbo_lambda)
            losses.append(loss)

            if PRINT_INTERMEDIATE_LATENT_VALUES:
                print('Loss: {}'.format(loss))
                variance = pyro.param("variance_{}".format(t)).item()
                mu = pyro.param("mu_{}".format(t)).item()
                print('mu = {}'.format(mu))
                print('variance = {}'.format(variance))

            if step % 100 == 0:
                print('.', end=' ')

        pyplot.plot(range(len(losses)), losses)
        pyplot.xlabel('Update Steps')
        pyplot.ylabel('-ELBO')
        pyplot.title('-ELBO against time for component {}'.format(t));
        pyplot.show()

        # pyplot.plot(range(len(guide_log_prob)), -1 * np.array(guide_log_prob), 'b-', label='- Guide log prob')
        # pyplot.plot(range(len(approximation_log_prob)), -1 * np.array(approximation_log_prob), 'r-', label='- Approximation log prob')
        # pyplot.plot(range(len(model_log_prob)), np.array(model_log_prob), 'g-', label='Model log prob')
        # pyplot.plot(range(len(model_log_prob)), np.array(model_log_prob) -1 * np.array(approximation_log_prob) -1 * np.array(guide_log_prob), label='RELBO')
        # pyplot.xlabel('Update Steps')
        # pyplot.ylabel('Log Prob')
        # pyplot.title('RELBO components throughout SVI'.format(t));
        # pyplot.legend()
        # pyplot.show()

        wrapped_approximation.components.append(wrapped_guide)
        new_weight = 2 / (t + 1)

        weights = weights * (1-new_weight)
        weights = torch.cat((weights, torch.tensor([new_weight])))

        wrapped_approximation.weights = weights

        e_log_p = 0
        n_samples = 50
        entropy = 0
        model_log_likelihood = 0
        elbo = 0
        for i in range(n_samples):
            qt_trace = trace(wrapped_approximation).get_trace(data)
            replayed_model_trace = trace(replay(model, qt_trace)).get_trace(data)
            model_log_likelihood += replayed_model_trace.log_prob_sum()
            entropy -= qt_trace.log_prob_sum()
            elbo = elbo + replayed_model_trace.log_prob_sum() - qt_trace.log_prob_sum()

        duality_gap.append(elbo/n_samples)
        model_log_likelihoods.append(model_log_likelihood/n_samples)
        entropies.append(entropy/n_samples)

        scale = pyro.param("variance_{}".format(t)).item()
        scales.append(scale)
        loc = pyro.param("mu_{}".format(t)).item()
        locs.append(loc)
        print('mu = {}'.format(loc))
        print('variance = {}'.format(scale))

    pyplot.figure(figsize=(10, 4), dpi=100).set_facecolor('white')
    for name, grad_norms in gradient_norms.items():
        pyplot.plot(grad_norms, label=name)
        pyplot.xlabel('iters')
        pyplot.ylabel('gradient norm')
        # pyplot.yscale('log')
        pyplot.legend(loc='best')
        pyplot.title('Gradient norms during SVI');
    pyplot.show()  


    pyplot.plot(range(1, len(duality_gap) + 1), duality_gap, label='ELBO')
    pyplot.plot(range(1, len(entropies) + 1), entropies, label='Entropy of q_t')
    pyplot.plot(range(1, len(model_log_likelihoods) + 1),model_log_likelihoods, label='E[logp] w.r.t. q_t')
    pyplot.title('ELBO(p, q_t)');
    pyplot.legend();
    pyplot.xlabel('Approximation components')
    pyplot.ylabel('Log probability')
    pyplot.show()
    print(weights)
    print(locs)
    print(scales)

    X = np.arange(-10, 10, 0.1)
    pyplot.figure(figsize=(10, 4), dpi=100).set_facecolor('white')
    total_approximation = np.zeros(X.shape)
    for i in range(1, n_iterations + 1):
        Y = weights[i].item() * scipy.stats.norm.pdf((X - locs[i]) / scales[i])    
        pyplot.plot(X, Y)
        total_approximation += Y
    pyplot.plot(X, total_approximation)
    pyplot.plot(data.data.numpy(), np.zeros(len(data)), 'k*')
    pyplot.title('Approximation of posterior over mu with lambda={}'.format(relbo_lambda))
    pyplot.ylabel('probability density');
    pyplot.show()

def run_standard_svi():

    adam_params = {"lr": 0.002, "betas": (0.90, 0.999)}
    optimizer = Adam(adam_params)
    gradient_norms = defaultdict(list)
    losses = []
    wrapped_guide = partial(guide, index=0)
    wrapped_guide(data)
    for name, value in pyro.get_param_store().named_parameters():
        if not name in gradient_norms:
            value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))
    

    svi = SVI(model, wrapped_guide, optimizer, loss=Trace_ELBO())
    for step in range(n_steps):
        loss = svi.step(data)
        losses.append(loss)


    pyplot.figure(figsize=(10, 4), dpi=100).set_facecolor('white')
    for name, grad_norms in gradient_norms.items():
        pyplot.plot(grad_norms, label=name)
        pyplot.xlabel('iters')
        pyplot.ylabel('gradient norm')
        # pyplot.yscale('log')
        pyplot.legend(loc='best')
        pyplot.title('Gradient norms during SVI');
    pyplot.show()  

    scale = pyro.param("variance_{}".format(0)).item()
    loc = pyro.param("mu_{}".format(0)).item()
    X = np.arange(-10, 10, 0.1)
    Y1 = scipy.stats.norm.pdf((X - loc) / scale)

    print('Resulting Mu: ', loc)
    print('Resulting Variance: ', scale)
    
    pyplot.figure(figsize=(10, 4), dpi=100).set_facecolor('white')
    pyplot.plot(X, Y1, 'r-')
    pyplot.plot(data.data.numpy(), np.zeros(len(data)), 'k*')
    pyplot.title('Standard SVI result')
    pyplot.ylabel('probability density');
    pyplot.show()


if __name__ == '__main__':
  boosting_bbvi()