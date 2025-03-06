import numpy as np 
import models
import tensorflow as tf
import proposals
from extra import Sampler 
from particle_algorithms import RML,OSIWAE
import matplotlib.pyplot as plt 

def calculate_landmark_error(true_landmarks, estimated_landmarks):
    """
    Calculate the error between true landmark positions and estimated ones.

    Parameters:
    true_landmarks (np.ndarray): Array of size 8x2 representing the true positions of the landmarks.
    estimated_landmarks (np.ndarray): Array of size 16xt_max representing the estimated positions for all timesteps.
                                      Format: [x1, y1, x2, y2, ..., x8, y8] for each timestep (t_max timesteps).

    Returns:
    np.ndarray: An array of size 8xt_max containing the error (Euclidean distance) for each landmark and timestep.
    """
    num_landmarks = true_landmarks.shape[0] 
    num_timesteps = estimated_landmarks.shape[1]  

    
    estimated_landmarks_reshaped = estimated_landmarks.reshape(num_landmarks, 2, num_timesteps)


    errors = np.zeros((num_landmarks, num_timesteps))

    # Calculate the error for each landmark at each timestep
    for t in range(num_timesteps):
        for i in range(num_landmarks):
            # Euclidean distance between the true position and the estimated position at each timestep
            errors[i, t] = np.linalg.norm(true_landmarks[i] - estimated_landmarks_reshaped[i, :, t])

    return errors   
def compute_stats(data):
    mean_values = np.mean(data, axis=1)
    min_values = np.min(data, axis=1)
    max_values = np.max(data, axis=1)
    return mean_values, min_values, max_values

# Algorithm parameters
N = 1000
L = 5
M = 1000
num_runs = 2

# Model parameters 
max_t = 500
num_landmarks = 8
lr_model = 1e-3
lr_proposal = 1e-3

all_mean_err_rml = np.zeros((num_runs,max_t))
all_mean_err_ovsmc = np.zeros((num_runs,max_t))
all_mean_err_osiwae = np.zeros((num_runs,max_t))

model = models.SLAMModel(num_landmarks = num_landmarks)
model.generate_data(max_n=max_t)
    
#Generate the same data for all 
x_all = model.x
y_all = model.y
    
landmarks = model.estimated_landmarks
landmarks_true = model.true_landmarks

for i in range(num_runs):
    # Create new model with the same data but different starting values 
    model = models.SLAMModel(num_landmarks = num_landmarks,initial_true_landmarks=landmarks_true)
    model.set_data(x_all, y_all)
    landmarks_init = model.estimated_landmarks

    #Generate the same data for all 
    x_all = model.x
    y_all = model.y

    #OSIWAE
    model_osiwae = models.SLAMModel(num_landmarks=num_landmarks,initial_true_landmarks=landmarks_true,initial_estimated_landmarks=landmarks)
    model_osiwae.set_data(x_all, y_all)
    
    optimizer_model = tf.keras.optimizers.Adam(learning_rate=lr_model)
    optimizer_proposal = tf.keras.optimizers.Adam(learning_rate=lr_model)
    proposal = proposals.DeepGaussian(model_osiwae, 12,12)
    
    sampler = Sampler(model, backward_sampler='full', mini_batch=100)
    
    pf_slam_osiwae = OSIWAE(optimizer_proposal, optimizer_model, model_osiwae, proposal, sampler, N, L, M, smoothing=True,
                alpha=0.6, beta=0.8, n_draws=1, ancestor=True, mini_batch = 100)
    pf_slam_osiwae.run(verbose_frequency=1000)
    
    
    #OVSMC
    model_ovsmc = models.SLAMModel(num_landmarks = num_landmarks,initial_true_landmarks=landmarks_true,initial_estimated_landmarks=landmarks)

    
    model_ovsmc.set_data(x_all, y_all)
    
    proposal = proposals.DeepGaussian(model_ovsmc, 12,12)
    optimizer_model = tf.keras.optimizers.Adam(learning_rate=lr_model)
    optimizer_proposal = tf.keras.optimizers.Adam(learning_rate=lr_model)
    
    sampler = Sampler(model_ovsmc, mini_batch=100)
    
    pf_slam_ovsmc = OSIWAE(optimizer_proposal, optimizer_model, model_ovsmc, proposal,sampler, N, L, M, smoothing=False,
                alpha=1.0, beta=1.0, n_draws=1, ancestor=True, mini_batch = 100)
    pf_slam_ovsmc.run(verbose_frequency=1000)

    #RML
    model_rml = models.SLAMModel(num_landmarks = num_landmarks,initial_true_landmarks=landmarks_true,initial_estimated_landmarks=landmarks)
    model_rml.set_data(x_all, y_all)


    sampler = Sampler(model_rml, backward_sampler='full', mini_batch=None)
    optimizer_model = tf.keras.optimizers.Adam(learning_rate=lr_model)
    
    
    proposal = proposals.Bootstrap(model_rml)
    pf_slam_rml = RML(model_rml, optimizer_model, N,alpha=0.6, beta=0.8, n_draws=1, sampler=sampler, ancestor=True, mini_batch=100)
    pf_slam_rml.run(verbose_frequency = 1000)


    # Calculate landmark error for each algorithm

    landmarks_osiwae = pf_slam_osiwae.parameters_evolution
    landmarks_rml = pf_slam_rml.parameters_evolution
    landmarks_ovsmc =pf_slam_ovsmc.parameters_evolution

    true_landmarks_osiwae = model_osiwae.true_landmarks.numpy()
    true_landmarks_rml = model_rml.true_landmarks.numpy()
    true_landmarks_ovsmc = model_ovsmc.true_landmarks.numpy()

    init_landmarks_osiwae = tf.reshape(pf_slam_osiwae.model.init_guesses, (num_landmarks, 2)).numpy()
    init_landmarks_rml = tf.reshape(pf_slam_rml.model.init_guesses, (num_landmarks, 2)).numpy()
    init_landmarks_ovsmc = tf.reshape(pf_slam_ovsmc.model.init_guesses, (num_landmarks, 2)).numpy()

    landmark_err_osiwae = calculate_landmark_error(true_landmarks_osiwae, landmarks_osiwae)
    landmark_err_rml = calculate_landmark_error(true_landmarks_rml, landmarks_rml)
    landmark_err_ovsmc = calculate_landmark_error(true_landmarks_ovsmc,landmarks_ovsmc)

    # Calculate the mean error over all landmarks
    mean_err_ovsmc = np.mean(landmark_err_ovsmc, axis=0)
    mean_err_osiwae = np.mean(landmark_err_osiwae, axis=0)
    mean_err_rml = np.mean(landmark_err_rml, axis=0)

    all_mean_err_rml[i] = mean_err_rml
    all_mean_err_ovsmc[i] = mean_err_ovsmc
    all_mean_err_osiwae[i] = mean_err_osiwae
    

    plt.plot(mean_err_ovsmc,label = 'ovsmc',color = 'tab:blue')
    plt.plot(mean_err_rml,label = 'rml',color = 'tab:orange')
    plt.plot(mean_err_osiwae,label = 'osiwae',color = 'tab:green')
    plt.legend()
    plt.show()


rml_mean, rml_min, rml_max = compute_stats(all_mean_err_rml)
ovsmc_mean, ovsmc_min, ovsmc_max = compute_stats(all_mean_err_ovsmc)
osiwae_mean, osiwae_min, osiwa_max = compute_stats(all_mean_err_osiwae)











