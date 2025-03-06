import numpy as np
import models
from scipy.stats import norm
from scipy.integrate import simps
from particle_algorithms import OSIWAE
from extra import Sampler
import matplotlib.pyplot as plt
import tensorflow as tf
import proposals

#%% Helper functions 
def latent_density(x_k, x_next_grid, alpha0, alpha1, alpha2, sigma_u, k):
    # Compute mean of p(x_{k+1} given x_k)
    a_k = alpha0 * x_k + (alpha1 * x_k) / (1 + x_k**2) + alpha2 * np.cos(1.2 * (k-1))
    # Return latent transition density
    return norm.pdf(x_next_grid, loc=a_k, scale=sigma_u)

def log_optimal_density(x_k, y_next, x_next_grid, alpha0, alpha1, alpha2, b, sigma_u, sigma_v, k):
    # Use latent_density function for prior
    prior = latent_density(x_k, x_next_grid, alpha0, alpha1, alpha2, sigma_u, k)
    # Compute observation likelihood
    mean_y_next = b * (x_next_grid**2)
    likelihood = norm.pdf(y_next, loc=mean_y_next, scale=sigma_v)
    # Compute unnormalized posterior
    posterior = prior * likelihood
    # Normalize posterior with numerical integrator
    posterior_normalized = posterior / simps(posterior, x_next_grid)
    # Compute log posterior, adding small value to avoid log(0)
    log_posterior = np.log(posterior_normalized + 1e-16)
    return log_posterior

# Parameters for the algorithm
max_t = 50000
N = 1000
L = 5
M = 1000


# True model parameters
alpha0 = 0.5
alpha1 = 25.
#alpha1 = 0
alpha2 = 8.
#alpha2 = 0
b = 0.05
sigma = np.sqrt(10) #Sigma_u
sigma_obs = 1. #Sigma_v
alpha_vec_true = np.array([alpha0,alpha1,alpha2])
# Initial guesses
alpha0_start = 1.2
alpha1_start = 25.
alpha2_start = 8.
b_start = 0.5
sigma_start = 1 #Sigma_u
sigma_obs_start = 1. #Sigma_v
alpha_vec_start = np.array([alpha0_start,alpha1_start,alpha2_start])

# Set up the model
model = models.GrowthModel(alpha_vec_true,b,sigma,sigma_obs)
model.generate_data(max_n=max_t)
model.set_parameters(alpha_vec_start,b_start,sigma_start,sigma_obs_start)
# Define optimizer for proposal and model parameters
optimizer_proposal = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer_model = tf.keras.optimizers.Adam(learning_rate=0.001)

# Define architecture of proposal Neural Network
mu_nodes = 12
sigma_nodes =  12
proposal = proposals.DeepGaussian_growth(model,mu_nodes,sigma_nodes)

sampler = Sampler(model, backward_sampler='full', mini_batch=100)

# Run OSIWAE algorithm
pf_osiwae = OSIWAE(optimizer_proposal, optimizer_model, model, proposal, sampler, N, L, M, smoothing=True,
                alpha=0.5, beta=0.6, n_draws=1, ancestor=True, mini_batch=100)
pf_osiwae.run(verbose_frequency=1000)


# Get parameters, states and weights for proposal and model
latent_states = pf_osiwae.state
parameters_siwae = pf_osiwae.parameters_evolution
a0_s = parameters_siwae[0]
b_s = parameters_siwae[1]
sigma_s = parameters_siwae[2]
all_weights = pf_osiwae.nn_weights

# Calculate kernel densities 
x_old = 0.1
y_next = tf.constant(6.0, dtype=tf.float64)
x_next_grid = np.linspace(-20,20,100)


#fixed kernel
log_optimal_kernel = log_optimal_density(x_old, y_next, x_next_grid, alpha0, alpha1, alpha2, b, sigma, sigma_obs, k = 1)
bootstrap_density = latent_density(x_old, x_next_grid, alpha0, alpha1, alpha2, sigma, k = 1)
log_density_bootstrap = np.log(bootstrap_density)

# Iteration number for which the densities should be calcualted 
idx = max_t-1

# current kernel with iteration idx
proposal_current = proposals.DeepGaussian_growth(model, mu_nodes,sigma_nodes)
# Set the current proposal and paramters with weights from idx iteration 
proposal_current.nn_mu.set_weights(all_weights[idx][:len(proposal_current.nn_mu.trainable_weights)])
proposal_current.nn_sigmas_diag.set_weights(all_weights[idx][len(proposal_current.nn_mu.trainable_weights):])
parameters_current = pf_osiwae.parameters_evolution[:,idx]
a0_current = parameters_current[0]
b_current = parameters_current[1]
sigma_current = parameters_current[2]

log_density_proposal = proposal_current.log_proposal_density_growth(x_old, y_next,x_next_grid,a0_current,alpha1,alpha2,k = 123)
log_density_optimal_current = log_optimal_density(x_old, y_next, x_next_grid, a0_current, alpha1, alpha2, b_current, sigma_current, sigma_obs, k = 1)


 
# Plots the kernels at step {idx}
plt.title(f'Proposals after {idx}')

plt.plot(x_next_grid,log_optimal_kernel,linestyle = '--',color = 'tab:blue',label = 'Optimal kernal')
plt.plot(x_next_grid,log_density_optimal_current,linestyle = '-',color = 'tab:blue',label = 'Optimal kernel current')
plt.plot(x_next_grid,log_density_proposal,color = 'tab:orange',label='OSIWAE')
plt.plot(x_next_grid,log_density_bootstrap,color = 'tab:green',label='Bootstrap kernel')
plt.xlabel('x_next')
plt.ylabel('log-density')
plt.ylim(-30,5)
plt.legend()
plt.show()









