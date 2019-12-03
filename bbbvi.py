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


    approximation_trace = trace(replay(block(approximation, expose=['mu']), guide_trace)).get_trace(*args, **kwargs)
    # We will accumulate the various terms of the ELBO in `elbo`.

    # guide_log_prob.append(guide_trace.log_prob_sum())
    # model_log_prob.append(model_trace.log_prob_sum())
    # approximation_log_prob.append(approximation_trace.log_prob_sum())

    # This is how we computed the ELBO before using TraceEnum_ELBO:
    elbo = model_trace.log_prob_sum() - relbo_lambda * guide_trace.log_prob_sum() - approximation_trace.log_prob_sum()

    loss_fn = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1).differentiable_loss(model,
                                                               guide,
                                                         *args, **kwargs)

    # print(loss_fn)
    # print(approximation_trace.log_prob_sum())
    elbo = -loss_fn - approximation_trace.log_prob_sum()
    #elbo = -loss_fn + 0.1 * pyro.infer.TraceEnum_ELBO(max_plate_nesting=1).differentiable_loss(approximation,
    #                                                           guide,
    #                                                     *args, **kwargs)
    # Return (-elbo) since by convention we do gradient descent on a loss and
    # the ELBO is a lower bound that needs to be maximized.

    return -elbo

class Approximation:

  def __init__(self, components= None, weights=None):
    if not components:
      self.components = []
    else: 
      self.components = components

    if not weights:
      self.weights = []
    else:
      self.weights = weights

  def __call__(self, *args, **kwargs):
    assignment = pyro.sample('assignment', dist.Categorical(self.weights))
    result = self.components[assignment](*args, **kwargs)
    return result

