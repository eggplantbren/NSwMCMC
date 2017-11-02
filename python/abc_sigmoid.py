import copy
import numpy as np
import numpy.random as rng
from utils import randh
from numba import jit

# How many parameters are there?
num_params = 5

# Load the data
# The two columns are x and y
# and I'm fitting it with y = A + B/(1 + exp(-(x - C)/L)) + noise with standard deviation S
# with naive uniform priors for A, B, C, L, S.
data = np.loadtxt('sigmoid_data.txt')

# Some idea of how big the Metropolis proposals should be
# Base this on the prior width
jump_sizes = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

@jit
def from_prior():
  """
  A function to generate parameter values from the prior.
  Returns a numpy array of parameter values.
  """
  # I'm assuming U(0,10) priors for everything
  return 10.0*rng.rand(num_params)




@jit
def log_prior(params):
  """
  Evaluate the (log of the) prior distribution
  """
  if np.any((params < 0.0) | (params > 10.0)):
    return -np.Inf
  return 0.0



@jit
def log_likelihood(params):
  """
  Evaluate the (log of the) likelihood function
  NOTE: for ABC, this involves simulating a dataset and
  computing minus the distance function.
  """
  # Aliases for the parameters
  A, B, C, L, S = params

  # Aliases for the data
  x, y = data[:,0], data[:,1]

  # Simulate a fake dataset --- this is the bit you'll replace with your
  # simulation. In principle I could
  # be treating the noise like the other parameters, and that would be
  # more efficient.
  y_sim = A + B/(1.0 + np.exp(-(x - C)/L)) + S*rng.randn(data.shape[0])

  # Distance between real data and simulated data
  dist = np.sum((y_sim - y)**2)
  return -dist



@jit
def proposal(params):
  """
  Generate new values for the parameters, for the Metropolis algorithm.
  """
  # Copy the parameters
  new = copy.deepcopy(params)

  # Which one should we change?
  which = rng.randint(num_params)
  new[which] += jump_sizes[which]*randh()
  return new

