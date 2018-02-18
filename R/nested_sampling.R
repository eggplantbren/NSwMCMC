# This is pretty much a direct translation from Python.

# Import the model
source("model.R")

# Number of particles
N = 100

# Number of NS iterations
depth = 30.0
steps = floor(N*depth)

# MCMC steps per NS iteration
mcmc_steps = 1000

# Generate N particles from the prior
# and calculate their log likelihoods
particles = list()
logps = rep(NA, N)
logls = rep(NA, N)
for(i in 1:N)
{
    particles[[i]] = from_prior()
    logps[i] = log_prior(particles[[i]])
    logls[i] = log_likelihood(particles[[i]])
}

# Storage for results
keep = array(dim=c(steps, num_params+1))

## Create figure
#plt.figure(figsize=(8, 8))

## Function that does MCMC
do_mcmc = function(particle, logp, logl)
{
    accepted = 0

    for(j in 1:mcmc_steps)
    {
        # Generate proposal
        new = proposal(particle)
        logp_new = log_prior(new)

        # Only evaluate likelihood if prior prob isn't zero
        logl_new = Inf
        if(logp_new != Inf)
            logl_new = log_likelihood(new)
        loga = logp_new - logp
        if(loga > 0)
            loga = 0


        # Accept
        if(logl_new >= threshold && runif(1) <= exp(loga))
        {
            particle = new
            logp = logp_new
            logl = logl_new
            accepted = accepted + 1
        }

    }

    return(list(particle=particle, logp=logp, logl=logl, accepted=accepted))
}





# Main NS loop
for(i in 1:steps)
{
    # Print message
    cat(paste("Iteration", i, "\n"))

    # Find worst particle
    worst = which.min(logls)

    # Save its details
    keep[i, ] = c(particles[[worst]], logls[worst])
    threshold = logls[worst]

    # Copy a survivor
    if(N > 1)
    {
        while(TRUE)
        {
            which = sample(1:N, 1)
            if(which != worst)
                break
        }
        particles[[worst]] = particles[[which]]
        logps[worst] = logps[which]
        logls[worst] = logls[which]
    }


    # Evolve within likelihood constraint using Metropolis
    newpoint = do_mcmc(particles[[worst]], logps[worst], logls[worst])
    particles[[worst]] = newpoint$particle
    logps[worst] = newpoint$logp
    logls[worst] = newpoint$logl
    accepted = newpoint$accepted

}

#for i in range(0, steps):
#  # Clear the figure
#  plt.clf()






#  print("NS iteration {it}. M-H acceptance rate = {a}/{m}."
#                .format(a=accepted, it=i+1, m=mcmc_steps))

#  # Use the deterministic approximation
#  logX = -(np.arange(0, i+1) + 1.)/N

#  # Make a plot, periodically
#  if (i+1) % N == 0:
#    plt.subplot(2,1,1)
#    plt.plot(logX, keep[0:(i+1), -1], "o-")
#    # Smart ylim
#    temp = keep[0:(i+1), -1].copy()
#    if len(temp) >= 2:
#      np.sort(temp)
#      plt.ylim([temp[int(0.2*len(temp))], temp[-1]])
#    plt.ylabel('$\\log(L)$')

#    plt.subplot(2,1,2)
#    # Rough posterior weights
#    logwt = logX.copy() + keep[0:(i+1), -1]
#    wt = np.exp(logwt - logwt.max())
#    plt.plot(logX, wt, "o-")
#    plt.ylabel('Posterior weights (relative)')
#    plt.xlabel('$\\log(X)$')
#    plt.savefig("progress_plot.png", bbox_inches="tight")

## Useful function
#def logsumexp(values):
#  biggest = np.max(values)
#  x = values - biggest
#  result = np.log(np.sum(np.exp(x))) + biggest
#  return result

## Prior weights
#logw = logX.copy()
## Normalise them
#logw -= logsumexp(logw)

## Calculate marginal likelihood
#logZ = logsumexp(logw + keep[:,-1])

## Normalised posterior weights
#wt = wt/wt.sum()

#effective_sample_size = int(np.exp(-np.sum(wt*np.log(wt + 1E-300))))

## Calculate information
#H = np.sum(wt*(keep[:,-1] - logZ))

#print('logZ = {logZ} +- {err}'.format(logZ=logZ, err=np.sqrt(H/N)))
#print('Information = {H} nats'.format(H=H))
#print('Effective Sample Size = {ess}'.format(ess=effective_sample_size))

#posterior_samples = np.empty((effective_sample_size, keep.shape[1]))
#k = 0
#while True:
#  # Choose one of the samples
#  which = rng.randint(keep.shape[0])

#  # Acceptance probability
#  prob = wt[which]/wt.max()

#  if rng.rand() <= prob:
#    posterior_samples[k, :] = keep[which, :]
#    k += 1

#  if k >= effective_sample_size:
#    break

## Save posterior samples and the rest
#np.savetxt("keep.txt", keep)
#np.savetxt("posterior_samples.txt", posterior_samples)

