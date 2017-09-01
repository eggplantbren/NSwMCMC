# Import some packages
import copy
import numpy as np
import numpy.random as rng
from utils import randh

# How many parameters are there?
# Originally taken as 10
num_params = 10

def log_prior(params):
    """
    Evaluate the (log of the) prior distribution
    """
    if np.any((params < -10.0) | (params > 10.0)):
        return -np.Inf
    return 0.0

# We also need a function to generate a parameter vector
# from the prior distribution.
def from_prior():
    """
    A function to generate parameter values from the prior.
    Returns a numpy array of parameter values.
    """
    return -10.0 + 20.0*rng.rand(num_params)

# An array of numbers, one for each parameter, that gives a rough idea of
# the width of the prior for each parameter. This is used in the proposal
# function.
jump_sizes = 10.0*np.ones(num_params)

# A function that evaluates the log of the likelihood
# at the supplied value of the parameters
def log_likelihood(params):
    """
    Evaluate the (log of the) likelihood function
    """
    x = params

    logl = 0.0
    logl = -100.0*(x[1:] - x[0:-1]**2)**2 - (1 - x[1:])**2
    logl = 2*logl.sum()
    return(logl)

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

