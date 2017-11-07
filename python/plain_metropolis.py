import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import copy

# Set the seed
# rng.seed(0)

# Import the model
from transit_model import from_prior, log_prior, log_likelihood, proposal,\
                              num_params

# Number of MCMC steps to do
steps = 1000000

# Thinning
thin = 100

# Generate an initial particle
# and measure its logp and logl
particle = from_prior()
logp, logl = log_prior(particle), log_likelihood(particle)

# Storage for results (parameters and log likelihood)
keep = np.empty((steps // thin, num_params + 1))

# Keep track of acceptance count
accepted = 0

# Main loop
for i in range(0, steps):

    # Propose new position
    new = proposal(particle)
    logp_new = log_prior(new)

    # Only evaluate likelihood if prior prob isn't zero
    logl_new = -np.Inf
    if logp_new != -np.Inf:
        logl_new = log_likelihood(new)

    # Log of density ratio
    loga = (logp_new + logl_new) - (logp + logl)

    if loga > 0.0:
        loga = 0.0

    # Accept
    if rng.rand() <= np.exp(loga):
        particle = new
        logp = logp_new
        logl = logl_new
        accepted += 1

    # Store results and print a message
    if (i+1) % thin == 0:
        keep[(i-1)//thin, :] = np.hstack([particle, logl])
        print("Done {k} steps.".format(k=i+1))

print("Saving output to posterior_samples.txt.")
np.savetxt("posterior_samples.txt", keep)

# Make a trace plot of log likelihood
plt.plot(keep[:,-1])

# Set ylim smartly
temp = np.sort(keep[keep.shape[0] // 10 :,-1])
lower = temp.min() - 0.05*(temp.max() - temp.min())
upper = temp.max() + 0.05*(temp.max() - temp.min())
plt.ylim([lower, upper])

plt.xlabel("Time")
plt.ylabel("Log likelihood")
plt.title("Trace plot of log likelihood")
plt.show()

