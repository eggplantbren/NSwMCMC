# This is pretty much a direct translation from Python.

# Import the model
source("model.R")

# Number of particles
N = 50

# MCMC steps per NS iteration
mcmc_steps = 1000

# Depth to go to
depth = 20.0
steps = floor(N*depth)  # Number of NS iterations

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


# Useful function
logsumexp = function(xs)
{
    biggest = max(xs)
    xs = xs - biggest
    result = log(sum(exp(xs))) + biggest
    return(result)
}




# Main NS loop
for(i in 1:steps)
{
    # Print message
    cat("Iteration ", i, ". ", sep="")

    # Find worst particle
    worst = which.min(logls)

    # Save its details
    keep[i, ] = c(particles[[worst]], logls[worst])
    threshold = logls[worst]

    # Check for termination
    if(i == steps)
    {
        cat("done.\n")
        break
    }

    cat("Generating new particle...")

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

    cat("done. Accepted ", accepted, "/", mcmc_steps, " steps.\n", sep="")

    # Make a plot
    if(i %% N == 0)
    {
        logxs = -(1:i)/N
        logws = logxs + keep[1:i, dim(keep)[2]]
        logws = logws - logsumexp(logws)

        # Smart ylim
        ylim = c()
        temp = sort(keep[1:i, dim(keep)[2]])
        if(length(temp) >= 2)
        {
            ylim[1] = temp[0.1*length(temp)]
            ylim[2] = temp[length(temp)]
        }

        # Get plot window ready
        par(mfrow=c(2,1))
        plot(logxs, keep[1:i, dim(keep)[2]],
             type="b", xlab="ln(X)", ylab="ln(L)", ylim=ylim)
        plot(logxs, exp(logws),
             type="b", xlab="ln(X)", ylab="Posterior weight")
    }
}

# Prior weights
logws = -(1:steps)/N
logws = logws - logsumexp(logws)

# Calculate marginal likelihood
logZ = logsumexp(logws + keep[, dim(keep)[2]])

# Normalised posterior weights
post_weights = exp(logws + keep[, dim(keep)[2]] - logZ)

# ESS
ent = -sum(post_weights*log(post_weights + 1E-300))
ess = floor(exp(ent))

# Information
H = sum(post_weights*(keep[, dim(keep)[2]] - logZ))
err = sqrt(H/N)

# Print results
cat("\n")
cat("Marginal likelihood: logZ = ", logZ, " +- ", err, ".", sep="")
cat("\n")
cat("Information: H = ", H, " nats.", sep="")
cat("\n")
cat("Effective sample size = ", ess, ".", sep="")
cat("\n")

# Create posterior samples by resampling
posterior_samples = array(dim=c(ess, dim(keep)[2]))

# Counter
k = 1
top = max(post_weights)
while(TRUE)
{
    # Choose one of the samples
    which = sample(1:steps, 1)

    # Acceptance probability
    prob = post_weights[which]/top
    if(runif(1) <= prob)
    {
        posterior_samples[k, ] = keep[which, ]
        k = k + 1
    }

    # Check to see if we're done
    if(k == ess + 1)
        break
}

# Name the columns
colnames(keep) = c(names(particles[[1]]), "log_likelihood")
colnames(posterior_samples) = colnames(keep)

# Save the output. Just posterior
write.table(posterior_samples, file="posterior_samples.csv",
            sep=",", row.names=FALSE, col.names=TRUE)
cat("Posterior samples saved in posterior_samples.csv.\n")



