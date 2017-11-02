import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import copy
from numba import jit

# Set the seed
# rng.seed(0)

# Import the model
from transit_model import from_prior, log_prior, log_likelihood, proposal,\
                              num_params

# Number of particles
N = 100

# Number of NS iterations
depth = 30.0
steps = int(N*depth)

# MCMC steps per NS iteration
mcmc_steps = 1000

# Generate N particles from the prior
# and calculate their log likelihoods
particles = []
logp = np.empty(N)
logl = np.empty(N)
for i in range(0, N):
  x = from_prior()
  particles.append(x)
  logp[i] = log_prior(x)
  logl[i] = log_likelihood(x)

# Storage for results
keep = np.empty((steps, num_params + 1))

# Create figure
plt.figure(figsize=(8, 8))

# Function that does MCMC
@jit
def do_mcmc(particle, logp, logl):
  accepted = 0
  for j in range(0, mcmc_steps):
    new = proposal(particle)
    logp_new = log_prior(new)
    # Only evaluate likelihood if prior prob isn't zero

    logl_new = -np.Inf
    if logp_new != -np.Inf:
      logl_new = log_likelihood(new)
    loga = logp_new - logp

    if loga > 0.0:
      loga = 0.0

    # Accept
    if logl_new >= threshold and rng.rand() <= np.exp(loga):
      particle = new
      logp = logp_new
      logl = logl_new
      accepted += 1

  return {"particle": particle, "logp": logp, "logl": logl, "accepted": accepted}


# Main NS loop
for i in range(0, steps):
  # Clear the figure
  plt.clf()

  # Find worst particle
  worst = np.nonzero(logl == logl.min())[0][0]

  # Save its details
  keep[i, :-1] = particles[worst]
  keep[i, -1] = logl[worst]
  threshold = copy.deepcopy(logl[worst])

  # Copy survivor
  if N > 1:
    which = rng.randint(N)
    while which == worst:
      which = rng.randint(N)
    particles[worst] = copy.deepcopy(particles[which])
    logl[worst] = logl[which]
    logp[worst] = logp[which]

  # Evolve within likelihood constraint using Metropolis
  newpoint = do_mcmc(particles[worst], logp[worst], logl[worst])
  particles[worst] = newpoint["particle"]
  logp[worst] = newpoint["logp"]
  logl[worst] = newpoint["logl"]
  accepted = newpoint["accepted"]

  print("NS iteration {it}. M-H acceptance rate = {a}/{m}."
                .format(a=accepted, it=i+1, m=mcmc_steps))

  # Use the deterministic approximation
  logX = -(np.arange(0, i+1) + 1.)/N

  # Make a plot, periodically
  if (i+1) % N == 0:
    plt.subplot(2,1,1)
    plt.plot(logX, keep[0:(i+1), -1], "o-")
    # Smart ylim
    temp = keep[0:(i+1), -1].copy()
    if len(temp) >= 2:
      np.sort(temp)
      plt.ylim([temp[int(0.2*len(temp))], temp[-1]])
    plt.ylabel('$\\log(L)$')

    plt.subplot(2,1,2)
    # Rough posterior weights
    logwt = logX.copy() + keep[0:(i+1), -1]
    wt = np.exp(logwt - logwt.max())
    plt.plot(logX, wt, "o-")
    plt.ylabel('Posterior weights (relative)')
    plt.xlabel('$\\log(X)$')
    plt.savefig("progress_plot.png", bbox_inches="tight")

# Useful function
def logsumexp(values):
  biggest = np.max(values)
  x = values - biggest
  result = np.log(np.sum(np.exp(x))) + biggest
  return result

# Prior weights
logw = logX.copy()
# Normalise them
logw -= logsumexp(logw)

# Calculate marginal likelihood
logZ = logsumexp(logw + keep[:,-1])

# Normalised posterior weights
wt = wt/wt.sum()

effective_sample_size = int(np.exp(-np.sum(wt*np.log(wt + 1E-300))))

# Calculate information
H = np.sum(wt*(keep[:,-1] - logZ))

print('logZ = {logZ} +- {err}'.format(logZ=logZ, err=np.sqrt(H/N)))
print('Information = {H} nats'.format(H=H))
print('Effective Sample Size = {ess}'.format(ess=effective_sample_size))

posterior_samples = np.empty((effective_sample_size, keep.shape[1]))
k = 0
while True:
  # Choose one of the samples
  which = rng.randint(keep.shape[0])

  # Acceptance probability
  prob = wt[which]/wt.max()

  if rng.rand() <= prob:
    posterior_samples[k, :] = keep[which, :]
    k += 1

  if k >= effective_sample_size:
    break

# Save posterior samples and the rest
np.savetxt("keep.txt", keep)
np.savetxt("posterior_samples.txt", posterior_samples)

