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


PRINT_INTERMEDIATE_LATENT_VALUES = False
PRINT_TRACES = False

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
n_steps = 2 if smoke_test else 1000
pyro.set_rng_seed(2)

# enable validation (e.g. validate parameters of distributions)
pyro.enable_validation(True)

# clear the param store in case we're in a REPL
pyro.clear_param_store()

model_log_prob = []
guide_log_prob = []
approximation_log_prob = []

def guide(observations, input_data, index):
    variance_q = pyro.param('variance_{}'.format(index), torch.eye(input_data.shape[1]), constraints.positive)
    mu_q = pyro.param('mu_{}'.format(index), torch.zeros(input_data.shape[1]))
    pyro.sample("w", dist.MultivariateNormal(mu_q, variance_q))

def logistic_regression_model(observations, input_data):
    w = pyro.sample('w', dist.MultivariateNormal(20*torch.ones(input_data.shape[1]), torch.eye(input_data.shape[1])))
    with pyro.plate("data", input_data.shape[0]):
      sigmoid = torch.sigmoid(torch.matmul(torch.tensor(input_data).double(),torch.tensor(w).double()))
      obs = pyro.sample('obs', dist.Bernoulli(sigmoid), obs=observations)

@config_enumerate
def approximation(observations, input_data, components, weights):
    assignment = pyro.sample('assignment', dist.Categorical(weights))
    distribution = components[assignment](observations, input_data)

def dummy_approximation(observations, input_data):
    variance_q = pyro.param('variance_0', torch.eye(input_data.shape[1]))
    mu_q = pyro.param('mu_0', 20*torch.ones(input_data.shape[1]))
    pyro.sample("w", dist.MultivariateNormal(mu_q, variance_q))

def relbo(model, guide, *args, **kwargs):

    approximation = kwargs.pop('approximation', None)
    relbo_lambda = kwargs.pop('relbo_lambda', None)
    # Run the guide with the arguments passed to SVI.step() and trace the execution,
    # i.e. record all the calls to Pyro primitives like sample() and param().
    #print("enter relbo")
    guide_trace = trace(guide).get_trace(*args, **kwargs)
    #print(guide_trace.nodes['obs_1'])
    model_trace = trace(replay(model, guide_trace)).get_trace(*args, **kwargs)
    #print(model_trace.nodes['obs_1'])


    approximation_trace = trace(replay(block(approximation, expose=['w']), guide_trace)).get_trace(*args, **kwargs)
    # We will accumulate the various terms of the ELBO in `elbo`.

    guide_log_prob.append(guide_trace.log_prob_sum())
    model_log_prob.append(model_trace.log_prob_sum())
    approximation_log_prob.append(approximation_trace.log_prob_sum())

    # This is how we computed the ELBO before using TraceEnum_ELBO:
    elbo = model_trace.log_prob_sum() - relbo_lambda * guide_trace.log_prob_sum() - approximation_trace.log_prob_sum()

    # loss_fn = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1).differentiable_loss(model,
    #                                                            guide,
    #                                                     *args, **kwargs)

    # print(loss_fn)
    # print(approximation_trace.log_prob_sum())
    #elbo = -loss_fn - approximation_trace.log_prob_sum()
    # Return (-elbo) since by convention we do gradient descent on a loss and
    # the ELBO is a lower bound that needs to be maximized.

    return -elbo


def boosting_bbvi():

    npz_train_file = np.load('ds1.100_train.npz')
    npz_test_file = np.load('ds1.100_test.npz')

    X_train = torch.tensor(npz_train_file['X'])
    y_train = torch.tensor(npz_train_file['y'])
    y_train[y_train == -1] = 0
    X_test = torch.tensor(npz_test_file['X'])
    y_test = torch.tensor(npz_test_file['y'])
    y_test[y_test == -1] = 0
    n_iterations = 2

    relbo_lambda = 1
    initial_approximation = dummy_approximation
    components = [initial_approximation]
    weights = torch.tensor([1.])
    wrapped_approximation = partial(approximation, components=components,
                                    weights=weights)

    locs = [0]
    scales = [0]

    gradient_norms = defaultdict(list)
    duality_gap = []
    for t in range(1, n_iterations + 1):
        # setup the inference algorithm
        wrapped_guide = partial(guide, index=t)
        # do gradient steps
        losses = []
        # Register hooks to monitor gradient norms.
        wrapped_guide(y_train, X_train)
        print(pyro.get_param_store().named_parameters())

        adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
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


        svi = SVI(logistic_regression_model, wrapped_guide, optimizer, loss=relbo)
        for step in range(n_steps):
            loss = svi.step(y_train, X_train, approximation=wrapped_approximation, relbo_lambda=relbo_lambda)
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

        pyplot.plot(range(len(guide_log_prob)), -1 * np.array(guide_log_prob), 'b-', label='- Guide log prob')
        pyplot.plot(range(len(approximation_log_prob)), -1 * np.array(approximation_log_prob), 'r-', label='- Approximation log prob')
        pyplot.plot(range(len(model_log_prob)), np.array(model_log_prob), 'g-', label='Model log prob')
        pyplot.plot(range(len(model_log_prob)), np.array(model_log_prob) -1 * np.array(approximation_log_prob) -1 * np.array(guide_log_prob), label='RELBO')
        pyplot.xlabel('Update Steps')
        pyplot.ylabel('Log Prob')
        pyplot.title('RELBO components throughout SVI'.format(t));
        pyplot.legend()
        pyplot.show()

        components.append(wrapped_guide)
        new_weight = 2 / (t + 1)

        weights = weights * (1-new_weight)
        weights = torch.cat((weights, torch.tensor([new_weight])))

        wrapped_approximation = partial(approximation, components=components, weights=weights)

        e_log_p = 0
        for i in range(50):
            qt_trace = trace(wrapped_approximation).get_trace(y_train, X_train)
            replayed_model_trace = trace(replay(logistic_regression_model, qt_trace)).get_trace(y_train, X_train)
            e_log_p = e_log_p + replayed_model_trace.log_prob_sum()

        duality_gap.append(replayed_model_trace.log_prob_sum()/10)

        # scale = pyro.param("variance_{}".format(t)).item()
        # scales.append(scale)
        # loc = pyro.param("mu_{}".format(t)).item()
        # locs.append(loc)
        # print('mu = {}'.format(loc))
        # print('variance = {}'.format(scale))

    pyplot.figure(figsize=(10, 4), dpi=100).set_facecolor('white')
    for name, grad_norms in gradient_norms.items():
        pyplot.plot(grad_norms, label=name)
        pyplot.xlabel('iters')
        pyplot.ylabel('gradient norm')
        # pyplot.yscale('log')
        pyplot.legend(loc='best')
        pyplot.title('Gradient norms during SVI');
    pyplot.show()  


    pyplot.plot(range(1, len(duality_gap) + 1), duality_gap)
    pyplot.title('E[log p] w.r.t. q_t');
    pyplot.xlabel('Approximation components')
    pyplot.ylabel('Log probability')
    pyplot.show()
    # print(weights)
    # print(locs)
    # print(scales)

    # X = np.arange(-10, 10, 0.1)
    # pyplot.figure(figsize=(10, 4), dpi=100).set_facecolor('white')
    # total_approximation = np.zeros(X.shape)
    # for i in range(1, n_iterations + 1):
    #     Y = weights[i].item() * scipy.stats.norm.pdf((X - locs[i]) / scales[i])    
    #     pyplot.plot(X, Y)
    #     total_approximation += Y
    # pyplot.plot(X, total_approximation)
    # pyplot.plot(data.data.numpy(), np.zeros(len(data)), 'k*')
    # pyplot.title('Approximation of posterior over mu with lambda={}'.format(relbo_lambda))
    # pyplot.ylabel('probability density');
    # pyplot.show()


if __name__ == '__main__':
  boosting_bbvi()