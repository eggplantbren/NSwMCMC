# Import some packages
import copy
import numpy as np
import numpy.random as rng
from utils import randh

# How many parameters are there?
num_params = 20

# Some constants
width1 = 0.1
width2 = 0.01
widthInv1 = 1.0/width1
widthInv2 = 1.0/width2
C1 = -num_params*np.log(width1) - 0.5*num_params*np.log(2*np.pi)
C2 = -num_params*np.log(width2) - 0.5*num_params*np.log(2*np.pi)
shift = 0.031
logHalf = np.log(0.5)

def log_prior(params):
    """
    Evaluate the (log of the) prior distribution
    """
    if params[0] < 0.0 or params[0] > 1.0:
        return -np.Inf
    return 0.0

# We also need a function to generate a parameter vector
# from the prior distribution.
def from_prior():
    """
    A function to generate parameter values from the prior.
    Returns a numpy array of parameter values.
    """
    return rng.rand(num_params)

# An array of numbers, one for each parameter, that gives a rough idea of
# the width of the prior for each parameter. This is used in the proposal
# function.
jump_sizes = np.ones(num_params)

# A function that evaluates the log of the likelihood
# at the supplied value of the parameters
def log_likelihood(params):
    """
    Evaluate the (log of the) likelihood function
    """
    x = params

    # Wide 'slab'
    log_gaussian1 = -0.5*np.sum(((x - 0.5)*widthInv1)**2) + C1 

    # Narrow 'spike'
    log_gaussian2 = -0.5*np.sum(((x - 0.5 - shift)*widthInv2)**2) + C2

    biggest = np.max([log_gaussian1, log_gaussian2])
    log_gaussian1 -= biggest
    log_gaussian2 -= biggest
    logL = logHalf + np.exp(log_gaussian1) + np.exp(log_gaussian2)
    logL += biggest

    return logL

# This Nested Sampling program also uses the Metropolis algorithm to move
# around the parameter space. So we need a function that, when you pass in
# the current value of the parameters, returns a proposed new position
# that we might move to.
def proposal(params):
    """
    Generate new values for the parameters, for the Metropolis algorithm.
    """
    # Copy the parameters
    new = copy.deepcopy(params)
    # This might seem odd at first, but if you just did new=params, Python
    # wouldn't actually make a copy of the parameters! 'new' would just be
    # another name for the same thing. copy.deepcopy() actually creates a copy.

    # Choose one of the parameters to change, at random
    which = rng.randint(num_params)
    # numpy.random.randint(n) returns an integer in {0, 1, ..., n-1}, exactly
    # what we would need for an index in a numpy array (i.e. starting from
    # zero, not including the length).

    # Change the value of the chosen parameter
    new[which] += jump_sizes[which]*randh()
    # randh() is a home-made very heavy tailed distribution that I like to use.
    # More later...

    # Return the modified position
    return new

