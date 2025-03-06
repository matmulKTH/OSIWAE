import numpy as np
import models
from scipy.stats import norm
from scipy.integrate import simps
from particle_algorithms import OSIWAE,RML
from extra import Sampler
import matplotlib.pyplot as plt
import tensorflow as tf
import proposals

def conf_bounds(data):
    mean_values = np.mean(data, axis=0)
    lower_bound = np.quantile(data, 0.025, axis=0)
    upper_bound = np.quantile(data, 0.975, axis=0)
    return lower_bound,upper_bound,mean_values




# Number of iterations 
max_t = 100
# Number of particles
N = 100
# Number of propsals 
L = 5
M = 100

num_runs = 1
all_err_A_rml = np.zeros((num_runs,max_t))
all_err_B_rml = np.zeros((num_runs,max_t))
all_err_A_ovsmc = np.zeros((num_runs,max_t))
all_err_B_ovsmc = np.zeros((num_runs,max_t))
all_err_A_osiwae = np.zeros((num_runs,max_t))
all_err_B_osiwae = np.zeros((num_runs,max_t))


model = models.MultivariateLinearGaussian(ovf_model=True)
A_true = model.A
B_true = model.B
np.savetxt('A_true.txt', A_true)
np.savetxt('B_true.txt', B_true)
model.generate_data(max_n = max_t,path_to_save='model_data.txt')



dim = 10
dim_obs = 10
for i in range(num_runs):
    A_true = np.loadtxt('A_true.txt')
    B_true = np.loadtxt('B_true.txt')
    data = np.loadtxt('model_data.txt', delimiter=',')

    x = data[:dim, :]  # First `dim` rows correspond to x
    y = data[dim:dim + dim_obs, :]  # Next `dim_obs` rows correspond to y

    y = tf.constant(y)

    model = models.MultivariateLinearGaussian(ovf_model=True)
    model.set_data(x,y)
    model.set_true_A_B(A_true, B_true)

    # Run OSIWAE
    # Initialize with random diagonals for A and B
    model.set_parameters()

    # Define neural network proposal and optimizer
    lr_model = 0.001
    lr_proposal = 0.001
    optimizer_proposal = tf.keras.optimizers.Adam(learning_rate=lr_proposal)
    optimizer_model = tf.keras.optimizers.Adam(learning_rate=lr_model)
    sampler = Sampler(model, backward_sampler='full', mini_batch=100)

    proposal = proposals.DeepGaussian(model,124,124)

    
    pf_osiwae = OSIWAE(optimizer_proposal, optimizer_model, model, proposal,sampler, N, L, M, smoothing=True,
                alpha=0.5, beta=0.6, n_draws=1, ancestor=True, mini_batch=10)
    pf_osiwae.run(verbose_frequency=100)

    # Keep errors for A and B and filtered states estimtates
    err_A_osiwae = pf_osiwae.parameters_evolution[0]
    all_err_A_osiwae[i] = err_A_osiwae
    err_B_osiwae = pf_osiwae.parameters_evolution[1]
    all_err_B_osiwae[i] = err_B_osiwae
    states_osiwae = pf_osiwae.state

    # Run OVSMC
    print('OVSMC')
    model.set_parameters()

    # Define neural network proposal and optimizer
    lr_model = 0.001
    lr_proposal = 0.001
    optimizer_proposal = tf.keras.optimizers.Adam(learning_rate=lr_proposal)
    optimizer_model = tf.keras.optimizers.Adam(learning_rate=lr_model)
    sampler = Sampler(model, mini_batch=100)

    proposal = proposals.DeepGaussian(model,124,124)
    
    pf_ovsmc = OSIWAE(optimizer_proposal, optimizer_model, model, proposal,sampler, N, L, M, smoothing=False,
                alpha=1, beta=1, n_draws=2, ancestor=False, mini_batch=100)
    pf_ovsmc.run(verbose_frequency=100)

    # Keep errors for A and B and filtered states estimtates
    err_A_ovsmc = pf_ovsmc.parameters_evolution[0]
    all_err_A_ovsmc[i] = err_A_ovsmc
    err_B_ovsmc = pf_ovsmc.parameters_evolution[1]
    all_err_B_ovsmc[i] = err_B_ovsmc
    states_ovsmc = pf_ovsmc.state

    # Run RML
    print('RML')
    model.set_parameters()

    # Define neural network proposal and optimizer
    lr_model = 0.001
    optimizer_rml = tf.keras.optimizers.Adam(learning_rate=lr_model)
    
    sampler = Sampler(model,backward_sampler = 'full', mini_batch=100)

    
    pf_rml = RML(model, optimizer_rml, N,alpha=0.5, beta=0.6, n_draws=1, sampler=sampler, ancestor=True, mini_batch=100)
    pf_rml.run(verbose_frequency=1000)

    # Keep errors for A and B and filtered states estimtates
    err_A_rml = pf_rml.parameters_evolution[0]
    all_err_A_rml[i] = err_A_rml
    err_B_rml = pf_rml.parameters_evolution[1]
    all_err_B_rml[i] = err_B_rml 

    states_rml = pf_rml.state
    




#%% Plot results
lower_bound_rml_A,upper_bound_rml_A,mean_values_rml_A = conf_bounds(all_err_A_rml)
lower_bound_rml_B,upper_bound_rml_B,mean_values_rml_B = conf_bounds(all_err_B_rml)

lower_bound_ovsmc_A,upper_bound_ovsmc_A,mean_values_ovsmc_A = conf_bounds(all_err_A_ovsmc)
lower_bound_ovsmc_B,upper_bound_ovsmc_B,mean_values_ovsmc_B = conf_bounds(all_err_B_ovsmc)

lower_bound_osiwae_A,upper_bound_osiwae_A,mean_values_osiwae_A = conf_bounds(all_err_A_osiwae)
lower_bound_osiwae_B,upper_bound_osiwae_B,mean_values_osiwae_B = conf_bounds(all_err_B_osiwae)


plt.figure(figsize=(10, 6), dpi=300)
timesteps = np.arange(0,max_t)
# Plot for Matrix A
plt.title('Parameter Error of $A$', fontsize=18, pad=15)
plt.fill_between(timesteps, lower_bound_ovsmc_A, upper_bound_ovsmc_A, color='lightblue', alpha=0.5, label='OVSMC (5%-95% quantiles)')
plt.plot(timesteps, mean_values_ovsmc_A, color='blue', linewidth=2, label='Mean OVSMC')

plt.fill_between(timesteps, lower_bound_rml_A, upper_bound_rml_A, color='#ffcccc', alpha=0.5, label='RML (5%-95% quantiles)')
plt.plot(timesteps, mean_values_rml_A, color='red', linewidth=2, label='Mean RML')

plt.fill_between(timesteps, lower_bound_osiwae_A, upper_bound_osiwae_A, color='lightgreen', alpha=0.5, label='OSIWAE (5%-95% quantiles)')
plt.plot(timesteps, mean_values_osiwae_A, color='green', linewidth=2, label='Mean OSIWAE')

# Labels and legend
plt.ylabel('Mean Absolute Error', fontsize=16)
plt.xlabel('Time Steps $t$', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.legend(fontsize=14, loc='upper right', frameon=True, edgecolor='gray', framealpha=0.9)
plt.savefig('parameter_error_A.pdf', bbox_inches='tight')
plt.xlim(0,max_t)
plt.show()

# Set figure size and DPI for the next plot
plt.figure(figsize=(10, 6), dpi=300)

# Plot for Matrix B
plt.title('Parameter Error of $B$', fontsize=18, pad=15)
plt.fill_between(timesteps, lower_bound_ovsmc_B, upper_bound_ovsmc_B, color='lightblue', alpha=0.5, label='OVSMC (5%-95% quantiles)')
plt.plot(timesteps, mean_values_ovsmc_B, color='blue', linewidth=2, label='Mean OVSMC')

plt.fill_between(timesteps, lower_bound_rml_B, upper_bound_rml_B, color='#ffcccc', alpha=0.5, label='RML (5%-95% quantiles)')
plt.plot(timesteps, mean_values_rml_B, color='red', linewidth=2, label='Mean RML')


plt.fill_between(timesteps, lower_bound_osiwae_B, upper_bound_osiwae_B, color='lightgreen', alpha=0.5, label='OSIWAE (5%-95% quantiles)')
plt.plot(timesteps, mean_values_osiwae_B, color='green', linewidth=2, label='Mean OSIWAE')

# Labels and legend
plt.ylabel('Mean Absolute Error', fontsize=16)
plt.xlabel('Time Steps $t$', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.legend(fontsize=14, loc='upper right', frameon=True, edgecolor='gray', framealpha=0.9)
#plt.savefig('parameter_error_B.pdf', bbox_inches='tight')

plt.xlim(0,max_t)
plt.show()

# %%
