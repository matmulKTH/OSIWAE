import models
import matplotlib.pyplot as plt 
import random 
import proposals 
from extra import Sampler
from particle_algorithms import OSIWAE
from tqdm import tqdm
import tensorflow as tf


random.seed(42)

a_true = 0.8
b_true = 1.
sigma_true = 0.5
sigma_obs_true = 1.2

a_init = -0.9
sigma_init = 0.1

n = 2

N = 100
M = 100
L = 5

model = models.LinearGaussian1D(a_true, b_true, sigma_true, sigma_obs_true)
model.generate_data(max_n=n + 1)

model.set_parameters(a_init, b_true, sigma_init, sigma_obs_true)



lr_model = 0.0015
lr_proposal = 0.0015
optimizer_proposal = tf.keras.optimizers.Adam(learning_rate=lr_proposal)
optimizer_model = tf.keras.optimizers.Adam(learning_rate=lr_model)

proposal = proposals.DeepGaussian(model, 3,2)

sampler = Sampler(model,backward_sampler='full',mini_batch=None)
pf_siwae = OSIWAE(optimizer_proposal,optimizer_model,model,proposal, sampler, N, L, M, smoothing=True,
                 alpha=0.5, beta=0.5, n_draws=1, ancestor=True, mini_batch=10)
pf_siwae.run(verbose_frequency=1000)

a_siwae = pf_siwae.parameters_evolution[0]
sigma_siwae = pf_siwae.parameters_evolution[1]