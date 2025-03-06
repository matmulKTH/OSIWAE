import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers
from tqdm import tqdm
import matplotlib.pyplot as plt 

class LinearGaussian1D:
    """
    Univariate linear Gaussian model

    X(0) ~ N(0, sigma^2/(1-a^2)

    X(n+1) ~ N(aX(n), sigma^2)
    Y(n) ~ N(bX(n), sigma_obs^2)
    """

    def __init__(self, a=0.98, b=1, sigma=0.2, sigma_obs=1, optimal_init_pf=False):

        """
        Initializes the parameters

        :param optimal_init_pf: (boolean) if True it indicates whether to use the target distribution
                inside the method 'initialize_particle_filter', otherwise according to N(0, sigma^2/(1-a^2))
        """

        # set model parameters
        self.set_parameters(a, b, sigma, sigma_obs)

        # variance for the locally optimal proposal
        self.sigma_optimal = self.sigma * self.sigma_obs / tf.math.sqrt(tf.math.square(self.b)
                                                                        * tf.math.square(self.sigma)
                                                                        + tf.math.square(self.sigma_obs))

        self.max_n = None
        self.x = None
        self.y = None
        self.dim = 1
        self.dim_obs = 1
        self.optimal_init_pf = optimal_init_pf

    def set_parameters(self, a, b, sigma, sigma_obs):

        """
        Sets the parameters and makes them trainable tf variables
        """

        self.a = tf.Variable(a, dtype=tf.float64, trainable=True)
        self.b = tf.Variable(b, dtype=tf.float64, trainable=True)
        self.sigma = tf.Variable(sigma, dtype=tf.float64, trainable=True)
        self.sigma_obs = tf.Variable(sigma_obs, dtype=tf.float64, trainable=True)

    def generate_data(self, max_n=10001, path_to_save=None):

        """
        Generates latent Markov chain and observations

        :param max_n: int for data sequence length
        :param path_to_save: if not None, string that indicates the path to save the numpy data
        """

        self.max_n = max_n
        self.x = np.zeros(max_n)
        self.x[0] = self.sigma / np.sqrt(1 - self.a ** 2) * np.random.randn()
        for m in range(1, max_n):
            self.x[m] = self.a * self.x[m - 1] + self.sigma * np.random.randn()

        self.y = np.random.normal(self.b * self.x, self.sigma_obs).reshape(1, max_n)
        self.y = tf.constant(self.y)
        
        
        
        x_np = self.x.numpy() if isinstance(self.x, tf.Tensor) else self.x
        y_np = self.y.numpy() if isinstance(self.y, tf.Tensor) else self.y
        
        if path_to_save is not None:
            to_save = np.concatenate([x_np.reshape(1, max_n), y_np.reshape(1, max_n)])
            np.savetxt(path_to_save, to_save, delimiter=',')

    def set_data(self, x, y):

        """
        Inputs data from external source

        :param x: 1-D array for the latent process
        :param y: 1-D array for observations
        """
        self.max_n = x.shape[0]

        self.x = x
        self.y = y
        self.y = tf.constant(self.y)
        
        
        
        
        
    def initialize_particle_filter(self, N):

        """
        Initializes N particles for the SMC algorithm
        :param N: int for the number of particles
        :return: particles ([N,1] tensor) and associated log-weights ([N,1] tensor)
        """

        if self.optimal_init_pf:
            # sample according to the true posterior p(x(0) | y(0))
            var0 = tf.math.square(self.sigma) / (1 - tf.math.square(self.a))
            mu_optimal = self.b * var0 * self.y[:, 0] / (tf.math.square(self.b) * var0 + tf.math.square(self.sigma_obs))
            sigma_optimal = tf.math.sqrt(var0) * self.sigma_obs / tf.math.sqrt(tf.math.square(self.b) * var0
                                                                               + tf.math.square(self.sigma_obs))
            particles = tf.random.normal([N, 1], mu_optimal, sigma_optimal, dtype=tf.float64)
            self.compute_kalman(1)
            logWeights = tf.ones([N], dtype=tf.float64) * self.log_likelihood
        else:
            # sample according to the prior p(x(0))
            particles = tf.random.normal([N, 1], 0, self.sigma / tf.math.sqrt(1 - tf.math.square(self.a)),
                                         dtype=tf.float64)

            
            # particles = tf.constant([[0.33160, -0.0700, -0.11357]],dtype=tf.float64)

            # # Transposing the tensor
            # particles = tf.transpose(particles)
            
            logWeights = self.log_emission_density(particles, 0)
        return particles, logWeights

    def log_latent_transition_density(self, x_old, x_new, n):

        """
        Computes the log-density of N(a*x_old, sigma^2) evaluated in x_new

        :param x_old: [N, 1] tensor
        :param x_new: [N, 1] tensor
        :param n: int indicating the iteration (not used here)
        :return: [N] tensor of log-densities
        """


        result = tf.squeeze(tfd.Normal(self.a * x_old, self.sigma).log_prob(x_new),axis = -1)

        return result

    def log_emission_density(self, x, n, x_old=None):

        """
        Computes the log-density of N(b*x, sigma_obs^2) evaluated in y[n]

        :param x: [N, 1] tensor
        :param n: int indicating the iteration
        :param x_old: [N, 1] of previous states (not used)
        :return: [N] tensor of log-densities
        """
        

        
        return tf.squeeze(tfd.Normal(self.b * x, self.sigma_obs).log_prob(self.y[:, n]), axis=-1)

    def log_target_transition_density(self, x_old, x_new, n):

        """
        Computes the log-numerator of the weights as the sum log_latent_transition_density and log_emission_density

        :param x_old: [N, 1] tensor
        :param x_new: [N, 1] tensor
        :param n: int
        :return: [N] tensor
        """
        
        
        log_latent_transition_density = self.log_latent_transition_density(x_old, x_new, n)
        log_emission_density = self.log_emission_density(x_new, n)

        result = log_latent_transition_density + log_emission_density 

    
        return result

    def max_log_target_transition_density(self, x_new, n):
        return tfd.Normal(0, self.sigma).log_prob(0) + self.log_emission_density(x_new, n)

    def propagate_latent_prior(self, x, n):

        """
        Propagation of particles according to bootstrap proposal

        :param x: [N, 1] tensor of current resampled particles
        :param n: int indicating the iteration (not used)
        :return: [N, 1] tensor
        """

        return self.a * x + self.sigma * tf.random.normal([x.shape[0], 1], dtype=tf.float64)

    def propagate_optimal(self, x, n):

        """
        Propagation of particles according to locally optimal proposal

        :param x: [N, 1] tensor of current resampled particles
        :param n: int indicating the iteration
        :return: [N, 1] tensor
        """

        self.mu_optimal = (tf.math.square(self.sigma_obs) * self.a * x + self.b * tf.math.square(self.sigma) * self.y[:,
                                                                                                               n]) \
                          / (tf.math.square(self.b) * tf.math.square(self.sigma) + tf.math.square(self.sigma_obs))
        return self.mu_optimal + self.sigma_optimal * tf.random.normal([x.shape[0], 1], dtype=tf.float64)

    def log_optimal_proposal_density(self, x):

        """
        Computes the log-densities of the optimal proposal evaluated in the propagated particles

        :param x: [N, 1] tensor of propagated particles
        :return: [N] tensor of log-densities
        """

        return tf.squeeze(tfd.Normal(self.mu_optimal, self.sigma_optimal).log_prob(x), axis=-1)

    def list_trainable(self):

        """
        :return: list of variables to be learned
        """

        return [self.a, self.sigma]

    def variables_to_watch(self):
        """
        :return: list of values to monitor
        """
        return [self.a.numpy(), self.sigma.numpy()]

    def compute_kalman(self, max_n=None):

        """
        Computes the Kalman filter and the marginal log-likelihood p(x(0:max_n))

        :param max_n: int for maximum number of iterations
        """

        if max_n is None:
            max_n = self.max_n

        a = self.a.numpy()
        b = self.b.numpy()
        var = self.sigma.numpy() ** 2
        var_obs = self.sigma_obs.numpy() ** 2

        self.x_pred = np.zeros(max_n)
        self.var_pred = np.zeros(max_n)
        self.x_filt = np.zeros(max_n)
        self.var_filt = np.zeros(max_n)
        self.innov = np.zeros(max_n)
        self.innov_var = np.zeros(max_n)

        for n in range(max_n):

            if n == 0:
                self.var_pred[0] = var / (1 - a ** 2)
            else:
                self.x_pred[n] = a * self.x_filt[n - 1]
                self.var_pred[n] = a ** 2 * self.var_filt[n - 1] + var

            self.innov[n] = self.y[0, n] - b * self.x_pred[n]
            self.innov_var[n] = b ** 2 * self.var_pred[n] + var_obs

            K = self.var_pred[n] * b / self.innov_var[n]

            self.x_filt[n] = self.x_pred[n] + K * self.innov[n]
            self.var_filt[n] = self.var_pred[n] * (1 - K * b)

        self.log_likelihood = -max_n * self.dim_obs * 0.5 * np.log(2 * np.pi) \
                              - 0.5 * np.sum(np.log(self.innov_var) + self.innov ** 2 / self.innov_var)

        return

    def __str__(self):
        return "One-dimensional Linear Gaussian State-Space Model"


class MultivariateLinearGaussian:
    """
    Multivariate linear Gaussian model

    X(0) ~ N(0, Sigma_0)

    X(n+1) ~ N(AX(n), Rdiag Rdiag^T)
    Y(n) ~ N(BX(n), Sdiag Sdiag^T)
    """

    def __init__(self, A=None, B='sparse', R=0.1, S=0.25, cov0=None, dim=2, dim_obs=2, optimal_init_pf=False, ovf_model=False, seed_ovf=None):

        """

        :param A: dim x dim matrix, if None it triggers a default model
        :param B: dim_obs x dim matrix, if 'sparse' and A is None it creates a truncated identity matrix
        :param Rdiag: dim x dim matrix
        :param Sdiag: dim_obs x dim_obs matrix
        :param cov0: dim x dim initial covariance matrix
        :param dim: int, dimension of the latent process
        :param dim_obs: int, dimension of the observation process
        :param optimal_init_pf: (boolean) if True it indicates whether to use the target distribution
                inside the method 'initialize_particle_filter', otherwise according to N(0, cov0)
        """

        self.ovf_model = ovf_model
        self.seed_ovf = seed_ovf
        self.set_parameters(A, B, R, S, cov0, dim, dim_obs)

        # compute the covariance for the locally optimal proposal
        self.K = tf.linalg.inv(
            tf.linalg.matmul(tf.linalg.matmul(self.B, self.cov), tf.transpose(self.B)) + self.cov_obs)
        self.K = tf.linalg.matmul(tf.linalg.matmul(self.cov, tf.transpose(self.B)), self.K)
        self.cov_optimal = self.cov - tf.linalg.matmul(tf.linalg.matmul(self.K, self.B), self.cov)
        self.chol = tf.linalg.cholesky(self.cov_optimal)


        self.max_n = None
        self.x = None
        self.y = None
        self.optimal_init_pf = optimal_init_pf

    def set_parameters(self, A=None, B=None, R=0.1, S=0.25, cov0=None, dim=None, dim_obs=None):

        """
        Sets the parameters, or initialize them according to a default model if input are None,
        and makes them tf variables
        """

        if A is None and self.ovf_model is False:
            alpha = 0.42
            A = np.fromfunction(lambda i, j: alpha ** (np.abs(i - j) + 1), (dim, dim))
            if B == 'sparse':
                B = np.fromfunction(lambda i, j: (i == j) * 1, (dim_obs, dim))
            else:
                np.random.seed(0)
                B = np.random.randn(dim_obs, dim)
            Rdiag = np.ones(dim)
            Sdiag = 0.5 * np.ones(dim_obs)
            cov0 = np.eye(dim)
            self.A = tf.Variable(A, dtype=tf.float64, trainable=False)
            self.B = tf.Variable(B, dtype=tf.float64, trainable=False)

        elif self.ovf_model:
            dim = 10
            dim_obs = 10
            # Use the model of Campbell et al. (2021)
            if A is not None and B is not None: 
                a_diag = np.diag(A)
                b_diag = np.diag(B)
                self.Adiag = tf.Variable(a_diag, dtype=tf.float64, trainable=True)
                self.Bdiag = tf.Variable(b_diag, dtype=tf.float64, trainable=True)
            elif self.seed_ovf is None:
                self.Adiag = tf.Variable(np.random.uniform(0.5, 1, size=dim), dtype=tf.float64, trainable=True)
                self.Bdiag = tf.Variable(np.random.uniform(0.5, 1, size=dim), dtype=tf.float64, trainable=True)
            else:
                self.Adiag = tf.Variable(np.diag(np.load('data_campbell/'+self.seed_ovf+'/F_init.npy')), dtype=tf.float64, trainable=True)
                self.Bdiag = tf.Variable(np.diag(np.load('data_campbell/'+self.seed_ovf+'/G_init.npy')), dtype=tf.float64, trainable=True)
            self.A = tf.eye(dim, dtype=tf.float64) * self.Adiag
            self.B = tf.eye(dim, dtype=tf.float64) * self.Bdiag
            
            # Latent and observation noise 
            Rdiag =  R* np.ones(dim)
            Sdiag = S * np.ones(dim_obs)
        
            
            
            cov0 = 0.1**2 * np.eye(dim)
            
        else:
            self.A = tf.Variable(A, dtype=tf.float64, trainable=False)
            self.B = tf.Variable(B, dtype=tf.float64, trainable=False)


        # set model parameters
        self.cov0 = tf.Variable(cov0, dtype=tf.float64, trainable=False)
        self.Rdiag = tf.Variable(Rdiag, dtype=tf.float64, trainable=False)
        self.cov = tf.Variable(np.diag(Rdiag) @ np.diag(Rdiag).T, dtype=tf.float64, trainable=False)
        self.Sdiag = tf.Variable(Sdiag, dtype=tf.float64, trainable=False)
        self.cov_obs = tf.Variable(np.diag(Sdiag) @ np.diag(Sdiag).T, dtype=tf.float64, trainable=False)
        self.dim = self.A.shape[0]
        self.dim_obs = self.B.shape[0]

    def generate_data(self, max_n=10001, path_to_save=None):

        """
        Generates latent Markov chain and observations

        :param max_n: int for data sequence length
        :param path_to_save: if not None, string that indicates the path to save the numpy data
        """

        self.max_n = max_n

        self.x = np.zeros((self.dim, max_n))
        self.x[:, 0] = np.random.multivariate_normal(np.zeros(self.dim), self.cov0)
        for m in range(1, max_n):
            self.x[:, m] = self.A.numpy() @ self.x[:, m - 1] + self.Rdiag.numpy() * np.random.randn(self.dim)

        self.y = np.zeros((self.dim_obs, max_n))
        self.y = self.B.numpy() @ self.x + self.Sdiag.numpy().reshape(self.dim_obs, 1) * np.random.randn(self.dim_obs,
                                                                                                         max_n)
        self.y = tf.constant(self.y)

        if path_to_save is not None:
            # write data
            to_save = np.concatenate([self.x, self.y])
            np.savetxt(path_to_save, to_save, delimiter=',')

    def set_data(self, x, y):

        """
        Inputs data from external source

        :param x: 2-D array for the latent process
        :param y: 2-D array for observations
        """

        self.x = x
        self.y = y
        self.y = tf.constant(self.y)
        self.max_n = y.shape[1]

    def initialize_particle_filter(self, N):

        """
        Initializes N particles for the SMC algorithm
        :param N: int for the number of particles
        :return: particles ([N,1] tensor) and associated log-weights ([N,1] tensor)
        """

        if self.optimal_init_pf:
            # sample according to the true posterior p(x(0) | y(0))
            K = tf.linalg.inv(tf.linalg.matmul(tf.linalg.matmul(self.B, self.cov0),
                                               tf.transpose(self.B)) + self.cov_obs)
            K = tf.linalg.matmul(tf.linalg.matmul(self.cov0, tf.transpose(self.B)), K)

            mu_optimal = tf.squeeze(tf.linalg.matmul(K, tf.reshape(self.y[:, 0], [self.dim_obs, 1])))

            cov_optimal = self.cov0 - tf.linalg.matmul(tf.linalg.matmul(K, self.B), self.cov0)
            chol0 = tf.linalg.cholesky(cov_optimal)
            particles = tfd.MultivariateNormalTriL(loc=mu_optimal, scale_tril=chol0).sample(N)
            self.compute_kalman(1)
            logWeights = tf.ones([N], dtype=tf.float64) * self.log_likelihood
        else:
            # sample according to the prior p(x(0))
            particles = tfd.MultivariateNormalTriL(scale_tril=tf.linalg.cholesky(self.cov0)).sample(N)
            logWeights = self.log_emission_density(particles, 0)
        return particles, logWeights

    def log_latent_transition_density(self, x_old, x_new, n):

        """
        Computes the log-density of N(A*x_old, Rdiag*Rdiag^T) evaluated in x_new

        :param x_old: [N, dim] tensor
        :param x_new: [N, dim] tensor
        :param n: int indicating the iteration (not used here)
        :return: [N, 1] tensor of log-densities
        """
        #Doesnt work with tf.function 
        if self.ovf_model:
            self.A = tf.eye(self.dim, dtype=tf.float64) * self.Adiag
        # self.A = tf.eye(self.dim, dtype=tf.float64) * self.Adiag

        mu = tf.linalg.matvec(self.A, x_old)
        return_val = tfd.MultivariateNormalDiag(mu, self.Rdiag).log_prob(x_new)

        return return_val
#    @tf.function
    def log_emission_density(self, x, n, x_old=None):

        """
        Computes the log-density of N(B*x, Sdiag*Sdiag^T) evaluated in y[n]

        :param x: [N, dim] tensor
        :param n: int indicating the iteration
        :param x_old: [N, dim] of previous states (not used)
        :return: [N, 1] tensor of log-densities
        """


        if self.ovf_model:
            self.B = tf.eye(self.dim, dtype=tf.float64) * self.Bdiag

        

        mu = tf.linalg.matvec(self.B, x)
        return_val = tfd.MultivariateNormalDiag(mu, self.Sdiag).log_prob(self.y[:, n])

        return return_val

    def log_target_transition_density(self, x_old, x_new, n):

        """
        Computes the log-numerator of the weights as the sum log_latent_transition_density and log_emission_density

        :param x_old: [N, dim] tensor
        :param x_new: [N, dim] tensor
        :param n: int
        :return: [N, 1] tensor
        """


        log_emission_dens = self.log_emission_density(x_new, n)

        log_latent_transition_density = self.log_latent_transition_density(x_old, x_new, n)


        result = log_latent_transition_density + log_emission_dens

        return result

    def max_log_target_transition_density(self, x_new, n):
        mu = tf.zeros((x_new.numpy().shape[0], self.dim), dtype=tf.float64)
        return tfd.MultivariateNormalDiag(mu, self.Rdiag).log_prob(mu) + self.log_emission_density(x_new, n)

    def propagate_latent_prior(self, x, n):

        """
        Propagation of particles according to bootstrap proposal

        :param x: [N, dim] tensor of current resampled particles
        :param n: int indicating the iteration (not used)
        :return: [N, dim] tensor
        """

        mu = tf.transpose(tf.linalg.matmul(tf.eye(self.dim, dtype=tf.float64) * self.Adiag, x, transpose_b=True))
        return mu + tf.math.multiply(tf.expand_dims(self.Rdiag, axis=0),
                                     tf.random.normal([x.shape[0], self.dim], dtype=tf.float64))

    def propagate_optimal(self, x, n):

        """
        Propagation of particles according to locally optimal proposal

        :param x: [N, dim] tensor of current resampled particles
        :param n: int indicating the iteration
        :return: [N, dim] tensor
        """

        Ax = tf.linalg.matmul(self.A, x, transpose_b=True)
        self.mu_optimal = tf.transpose(Ax) \
                          + tf.transpose(tf.linalg.matmul(self.K, tf.reshape(self.y[:, n], [self.dim_obs, 1])
                                                          - tf.linalg.matmul(self.B, Ax)))
        return self.mu_optimal + tfd.MultivariateNormalTriL(scale_tril=self.chol).sample(self.mu_optimal.shape[0])

    def log_optimal_proposal_density(self, x):

        """
        Computes the log-densities of the optimal proposal evaluated in the propagated particles

        :param x: [N, dim] tensor of propagated particles
        :return: [N, 1] tensor of log-densities
        """

        return tfd.MultivariateNormalTriL(self.mu_optimal, self.chol).log_prob(x)

    def list_trainable(self):

        """
        :return: list of variables to be learned
        """
        if self.ovf_model:
            return [self.Adiag, self.Bdiag]
        else:
            return []

    def variables_to_watch(self):

        """
        :return: list of values to monitor
        """
        if self.ovf_model:
            err_A = np.mean(np.abs(self.A_diag_true - self.Adiag.numpy()))
            err_B = np.mean(np.abs(self.B_diag_true - self.Bdiag.numpy()))

            return [err_A, err_B]
        else:
            return []

    def set_true_A_B(self,Adiag_true = None,Bdiag_true = None):

        """
        Sets the true values to compare the learned diagonals of A and B when training the model of Campbell et al. (2021).
        If self.seed_ovf is None the method should be called before re-initializing the parameters.
        """
        if Adiag_true is not None and Bdiag_true is not None:
            self.A_diag_true = Adiag_true
            self.B_diag_true = Bdiag_true
            print('h')
            
        else: 
            if self.seed_ovf is None:
                self.A_diag_true = self.Adiag.numpy()
                self.B_diag_true = self.Bdiag.numpy()
            else:
                self.A_diag_true = np.diag(np.load('data_campbell/'+self.seed_ovf+'/F.npy'))
                self.B_diag_true = np.diag(np.load('data_campbell/'+self.seed_ovf+'/G.npy'))

    def compute_kalman(self, max_n=None):

        """
        Computes the Kalman filter and the marginal log-likelihood p(x(0:max_n))

        :param max_n: int for maximum number of iterations
        """

        if max_n is None:
            max_n = self.max_n

        cov = self.cov.numpy()
        cov_obs = self.cov_obs.numpy()
        A_diag = self.A_diag_true
        B_diag = self.B_diag_true
        
        A = np.eye(self.dim) * A_diag
        B = np.eye(self.dim) * B_diag
        


        self.x_pred = np.zeros((self.dim, max_n))
        self.cov_pred = np.zeros((self.dim, self.dim, max_n))
        self.x_filt = np.zeros((self.dim, max_n))
        self.cov_filt = np.zeros((self.dim, self.dim, max_n))
        self.innov = np.zeros((self.dim_obs, max_n))
        self.det_innov_cov = np.zeros(max_n)
        self.inv_innov_cov = np.zeros((self.dim_obs, self.dim_obs, max_n))

        for n in range(max_n):

            if n == 0:
                self.cov_pred[:, :, 0] = self.cov0.numpy()
            else:
                self.x_pred[:, n] = A @ self.x_filt[:, n - 1]
                self.cov_pred[:, :, n] = A @ self.cov_filt[:, :, n - 1] @ A.T + cov

            self.innov[:, n] = self.y[:, n] - B @ self.x_pred[:, n]
            innov_cov = B @ self.cov_pred[:, :, n] @ B.T + cov_obs
            self.det_innov_cov[n] = np.linalg.det(innov_cov)
            self.inv_innov_cov[:, :, n] = np.linalg.inv(innov_cov)

            K = self.cov_pred[:, :, n] @ B.T @ self.inv_innov_cov[:, :, n]

            self.x_filt[:, n] = self.x_pred[:, n] + K @ self.innov[:, n]
            self.cov_filt[:, :, n] = self.cov_pred[:, :, n] - K @ B @ self.cov_pred[:, :, n]

        s = np.sum(np.log(self.det_innov_cov))
        for n in range(max_n):
            s += self.innov[:, n].T @ self.inv_innov_cov[:, :, n] @ self.innov[:, n]

        self.log_likelihood = -max_n * self.dim_obs * 0.5 * np.log(2 * np.pi) - 0.5 * s

        return

    def __str__(self):
        return "Multivariate linear Gaussian State-Space Model"


class GrowthModel:
    '''Class of growth model'''

    def __init__(self, alpha_vec=np.array([0.5, 25., 8.]), b=0.05, sigma=np.sqrt(10), sigma_obs=1.):
        # set model parameters
        self.set_parameters(alpha_vec, b, sigma, sigma_obs)

        self.max_n = None
        self.x = None
        self.y = None
        self.dim = 1
        self.dim_obs = 1

    def set_parameters(self, alphas, b, sigma, sigma_obs):
        self.alpha0 = tf.Variable(alphas[0], dtype=tf.float64, trainable=True)
        self.alpha1 = tf.Variable(alphas[1], dtype=tf.float64, trainable=True)
        self.alpha2 = tf.Variable(alphas[2], dtype=tf.float64, trainable=True)
        self.b = tf.Variable(b, dtype=tf.float64, trainable=True)
        self.sigma = tf.Variable(sigma, dtype=tf.float64, trainable=True)
        self.sigma_obs = tf.Variable(sigma_obs, dtype=tf.float64, trainable=True)

    def prior_mean(self, x, n):
        return self.alpha0 * x + self.alpha1 * x / (1 + x**2) \
                + self.alpha2 * tf.math.cos(1.2 * (self.dummy_range[n] - 1))

    def generate_data(self, max_n=10001, path_to_save=None):
        # generate hidden Markov chain and observations
        self.max_n = max_n
        self.dummy_range = tf.range(1, max_n+1, dtype=tf.float64)

        self.x = np.zeros(max_n)
        self.x[0] = 0.1
        for m in range(1, max_n):
            self.x[m] = self.prior_mean(self.x[m - 1], m).numpy() + self.sigma.numpy() * np.random.randn()

        self.y = np.random.normal(self.b.numpy() * self.x ** 2, self.sigma_obs.numpy()).reshape(1, max_n)
        self.y = tf.constant(self.y, dtype=tf.float64)

        if path_to_save is not None:
            # write data
            to_save = np.concatenate([self.x.reshape(1, max_n), self.y.numpy().reshape(1, max_n)])
            np.savetxt(path_to_save, to_save, delimiter=',')

    def set_data(self, x, y):
        # input external data
        self.x = x
        self.y = y
        self.y = tf.constant(self.y, dtype=tf.float64)
        self.max_n = len(y)

    def initialize_particle_filter(self, N):
        # initialize N particles
        particles = 0.1 * tf.ones([N, 1], dtype=tf.float64)
        logWeights = self.log_emission_density(particles, 0)
        return particles, logWeights

    def log_latent_transition_density(self, x_old, x_new, n):
        return tf.squeeze(tfd.Normal(self.prior_mean(x_old, n), self.sigma).log_prob(x_new), axis=-1)

    def log_emission_density(self, x, n, x_old=None):
        return tf.squeeze(tfd.Normal(self.b * tf.math.square(x), self.sigma_obs).log_prob(self.y[:, n]), axis=-1)

    def log_target_transition_density(self, x_old, x_new, n):
        return self.log_latent_transition_density(x_old, x_new, n) + self.log_emission_density(x_new, n)

    def propagate_latent_prior(self, x, n):
        return self.prior_mean(x, n) + self.sigma * tf.random.normal([x.shape[0], 1], dtype=tf.float64)

    def compute_ekf_parameters(self, x, n):
        k = 2 * self.sigma**2 * self.b * self.prior_mean(x, n)/(4 * self.sigma**2 * self.b**2 * self.prior_mean(x,n)**2
                                                                + self.sigma_obs**2)
        mu = self.prior_mean(x, n) + k * (self.y[:,n] - self.b * self.prior_mean(x, n)**2)
        sigma = self.sigma * self.sigma_obs / tf.sqrt(4 * self.sigma**2 * self.b**2 * self.prior_mean(x,n)**2
                                                                + self.sigma_obs**2)
        return mu, sigma
    def propagate_optimal(self, x, n):

        """
        Propagation of particles according to EKF proposal

        :param x: [N, 1] tensor of current resampled particles
        :param n: int indicating the iteration
        :return: [N, 1] tensor
        """

        self.mu_optimal, self.sigma_optimal = self.compute_ekf_parameters(x, n)
        return self.mu_optimal + self.sigma_optimal * tf.random.normal([x.shape[0], 1], dtype=tf.float64)

    def log_optimal_proposal_density(self, x_old, x_new, n):

        """
        Computes the log-densities of the optimal proposal evaluated in the propagated particles

        :param x: [N, 1] tensor of propagated particles
        :return: [N] tensor of log-densities
        """
        self.mu_optimal, self.sigma_optimal = self.compute_ekf_parameters(x_old, n)
        return tf.squeeze(tfd.Normal(self.mu_optimal, self.sigma_optimal).log_prob(x_new), axis=-1)

    def list_trainable(self):
        return [self.alpha0,self.b,self.sigma]

    def variables_to_watch(self):
        return [self.alpha0.numpy(),self.b.numpy(),self.sigma.numpy()]

    def __str__(self):
        return "Growth Model"
    


class SLAMModel:
    """
        A simple 2D SLAM model for estimating the trajectory of a robot and the positions of landmarks.
    """

    def __init__(self, num_landmarks=5, sigma_motion=0.2, sigma_dist=0.1, sigma_angle=0.1, 
             initial_true_landmarks=None, initial_estimated_landmarks=None):
        """
        Initializes the SLAM model. Landmarks can be randomly generated or provided as arguments.
        """
        self.set_parameters(sigma_motion, sigma_dist, sigma_angle)
        self.num_landmarks = num_landmarks
        
        
        
        radius = 5  # Radius of the circle
        center = np.array([15, 15], dtype=np.float64)  # Center of the circle (optional)
        
        # If true landmarks are provided, use them; otherwise, generate random landmarks
        if initial_true_landmarks is not None:
            self.true_landmarks = tf.Variable(initial_true_landmarks, dtype=tf.float64)
        else:
            angles = np.linspace(0, 2 * np.pi, num_landmarks, endpoint=False)

            # Initialize the landmarks on the circle
            landmarks = np.array([[center[0] + radius * np.cos(theta), 
                                    center[1] + radius * np.sin(theta)] for theta in angles], dtype=np.float64)
            self.true_landmarks = tf.Variable(landmarks, dtype=tf.float64)
        
        # If estimated landmarks are provided, use them; otherwise, generate based on true landmarks with noise
        if initial_estimated_landmarks is not None:
            print("init_guesses")
            self.init_guesses = tf.convert_to_tensor(initial_estimated_landmarks, dtype=tf.float64)
        else:
            self.init_guesses = self.true_landmarks + np.random.randn(num_landmarks, 2) * 1
        
        # Flatten the estimated landmarks and make them trainable
        flattened_tensor = tf.reshape(self.init_guesses, [-1])
        self.estimated_landmarks = [tf.Variable(coord, dtype=tf.float64, trainable=True) for coord in flattened_tensor.numpy()]
        self.max_n = None
        self.x = None
        self.y = None
        self.dim = 2
        self.dim_obs = self.num_landmarks * 2

        # Center of the landmarks
        self.true_midpoint = np.mean(self.true_landmarks, axis=0)
        
    def set_parameters(self, sigma_motion, sigma_dist,sigma_angle):
        """
        Sets the parameters and makes them fixed (non-trainable) tf variables.
        """
        self.sigma_motion = tf.Variable(sigma_motion, dtype=tf.float64, trainable=False)
        self.sigma_dist = tf.Variable(sigma_dist, dtype=tf.float64, trainable=False)
        self.sigma_angle = tf.Variable(sigma_angle, dtype=tf.float64, trainable=False)
        
    
    
    def generate_data(self, max_n=100, path_to_save=None):
        self.max_n = max_n
        self.x = tf.Variable(tf.zeros((2,max_n), dtype=tf.float64))

        self.y = tf.Variable(tf.zeros((self.dim_obs, max_n), dtype=tf.float64))

        # Initialize the robot at the center of the landmarks
        current_position = tf.Variable(self.true_midpoint, dtype=tf.float64)
        self.x[:,0].assign(current_position)
        min_bound = 0.0
        max_bound = 30.0
        # Simulate random walk trajectory
        for t in range(1, max_n):
            # Generate a random Gaussian step for the random walk
            random_step = tf.random.normal(shape=current_position.shape, mean=0.0, stddev=self.sigma_motion, dtype=tf.float64)

            # Update the robot's position with the random step
            new_position = current_position +  random_step
            
            # Truncuate the distribution s.t the robot stays in a specifique bound
            new_position = tf.clip_by_value(new_position, min_bound, max_bound)
            
            self.x[:,t].assign(new_position)
            current_position.assign(new_position)

            # Simulate observations of all landmarks
            for i in range(self.num_landmarks):
                landmark = self.true_landmarks[i]
                distance = tf.norm(current_position - landmark) + tf.random.normal([], stddev=self.sigma_dist, dtype=tf.float64)
                angle_obs = tf.atan2(landmark[1] - current_position[1], landmark[0] - current_position[0]) + tf.random.normal([], stddev=self.sigma_angle, dtype=tf.float64)
                # self.y[i, 0, t].assign(distance)
                # self.y[i, 1, t].assign(angle_obs)
                
                index = 2 * i  # Index in the flattened observation vector
                self.y[index, t].assign(distance)
                self.y[index + 1, t].assign(angle_obs)
                

        if path_to_save is not None:
            np.savez_compressed(path_to_save, trajectory=self.x.numpy(), landmarks=self.true_landmarks.numpy(), observations=self.y.numpy())


    def set_data(self, x, y):
        """
        Inputs data from an external source.
        
        :param x: 2D array for the true trajectory of the robot.
        :param y: 2D array for observations (distances and angles to landmarks).
        """
        self.max_n = x.shape[1]
        self.x = tf.constant(x, dtype=tf.float64)
        self.y = tf.constant(y, dtype=tf.float64)

    def initialize_particle_filter(self, N):
        """
        Initializes N particles for the SMC algorithm.
        :param N: int for the number of particles.
        :return: particles ([N,2] tensor) and associated log-weights ([N,1] tensor).
        """
        # Sample according to the prior p(x(0))
        

        particles = tf.random.normal(shape=[N, 2], mean=self.true_midpoint, stddev=0.1, dtype=tf.float64)
        # Concatenate along the last axis (axis=1) to create a tensor of shape [N, 2]
        log_weights = self.log_emission_density(particles, 0)
        return particles, log_weights



    def log_latent_transition_density(self, x_old, x_new, n):
        """
        Computes the log-density of N(predicted_x_new, sigma_motion^2) evaluated in x_new

        :param x_old: [N, 2] tensor representing the old states (positions)
        :param x_new: [N, 2] tensor representing the new states (positions)
        :param n: int indicating the iteration (not used here)
        :return: [N] tensor of log-densities
        """        
        log_trans = tfd.MultivariateNormalDiag(loc=x_old, scale_diag=tf.ones_like(x_old) * self.sigma_motion).log_prob(x_new)
        
        return log_trans
    

    def angle_difference(self,a, b):
        diff = a - b
        return (diff + np.pi) % (2 * np.pi) - np.pi 
    
    def log_emission_density(self, x, n, x_old=None):
        # Ensure x is of shape [N, K, 2]
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)  # Shape: [N, 1, 2]
    
        N, K, _ = x.shape  # N is the number of particles, K is the number of positions per particle
    
        # Get estimated landmark positions
        landmark_coords = tf.reshape(
            tf.stack(self.estimated_landmarks), [self.num_landmarks, 2]
        )
    
        # Expand dimensions for broadcasting
        landmark_coords_expanded = tf.reshape(
            landmark_coords, [1, 1, self.num_landmarks, 2]
        )  # Shape: [1, 1, num_landmarks, 2]
        x_expanded = tf.expand_dims(x, axis=2)  # Shape: [N, K, 1, 2]
    
        # Corrected difference: landmarks - particles
        diff = landmark_coords_expanded - x_expanded  # Shape: [N, K, num_landmarks, 2]
    
        # Compute distances and angles
        distances = tf.norm(diff, axis=-1)  # Shape: [N, K, num_landmarks]
        angles = tf.atan2(diff[..., 1], diff[..., 0])  # Shape: [N, K, num_landmarks]
    
        # Get observed distances and angles
        observed_y = self.y[:, n]  # Shape: [d_obs]
        observed_distances = observed_y[::2]  # Shape: [num_landmarks]
        observed_angles = observed_y[1::2]    # Shape: [num_landmarks]
    
        # Expand dimensions for broadcasting
        observed_distances = observed_distances[tf.newaxis, tf.newaxis, :]  # Shape: [1, 1, num_landmarks]
        observed_angles = observed_angles[tf.newaxis, tf.newaxis, :]        # Shape: [1, 1, num_landmarks]
    
        # Compute angle errors with wrapping
        angle_errors = self.angle_difference(angles, observed_angles)  # Shape: [N, K, num_landmarks]
    
        # Define Gaussian distributions for distances and angles
        distance_dist = tfd.Normal(loc=distances, scale=self.sigma_dist)
        angle_dist = tfd.Normal(loc=0.0, scale=self.sigma_angle)
    
        # Compute log probabilities
        log_prob_distances = distance_dist.log_prob(observed_distances)  # Shape: [N, K, num_landmarks]
        log_prob_angles = angle_dist.log_prob(angle_errors)              # Shape: [N, K, num_landmarks]
    
        # Sum log probabilities across landmarks
        total_log_prob = log_prob_distances + log_prob_angles
        log_prob = tf.reduce_sum(total_log_prob, axis=-1)  # Shape: [N, K]
        log_prob = tf.squeeze(log_prob)
        return log_prob

    def log_target_transition_density(self, x_old, x_new, n):
        """
        Computes the log-numerator of the weights as the sum of log_latent_transition_density and log_emission_density.
        
        :param x_old: [N, 2] tensor of previous states.
        :param x_new: [N, 2] tensor of current states.
        :param n: int indicating the iteration.
        :return: [N] tensor of log-densities.
        """
        log_emission_density = self.log_emission_density(x_new, n)
        if len(x_old.shape) == 3:
            log_emission_density = tf.expand_dims(log_emission_density, axis=-1)
        
        
        
        
        log_latent_transition_density = self.log_latent_transition_density(x_old, x_new, n)
        

        
        result = log_latent_transition_density + log_emission_density

        
        return result
    
    

    def propagate_latent_prior(self, x, n):
        """
        Propagation of particles according to the same random walk used during data generation.
    
        :param x: [N, 2] tensor of current resampled particles
        :param n: int indicating the iteration (not used)
        :return: [N, 2] tensor of propagated particles
        """
        # Propagate particles with motion noise (Gaussian random walk)
        noise = tf.random.normal(shape=x.shape, mean=0.0, stddev=self.sigma_motion, dtype=tf.float64)
        new_particles = x + noise
    
        return new_particles
    


    def list_trainable(self):
        """
        :return: List of variables to be learned (trainable).
        """
        return self.estimated_landmarks

    def variables_to_watch(self):
        """
        :return: List of values to monitor.
        """
        return [var.numpy() for var in self.estimated_landmarks]

    def plot_results(self, estimated_state=None):
        """
        Plots the true and estimated trajectories and landmarks using the class attributes.
        """
        plt.figure(figsize=(10, 8))

        # Plot true trajectory
        plt.plot(self.x[0,:-1], self.x[1,:-1], 'g-', label='True Trajectory')
        plt.scatter(self.x[0, 0], self.x[1,0], s=100, color='orange', edgecolor='black', label="Start Point")

        # Plot estimated trajectory if provided
        if estimated_state is not None:
            plt.plot(estimated_state[0, :-1], estimated_state[1, :-1], 'r-', label='Estimated Trajectory')

        # Plot initial guesses for landmarks
        plt.scatter(self.init_guesses[:, 0], self.init_guesses[:, 1], c='orange', marker='s', label='Initial Guesses')

        # Plot true landmarks
        plt.scatter(self.true_landmarks[:, 0], self.true_landmarks[:, 1], c='red', marker='x', label='True Landmarks')

        # Plot estimated landmarks
        estimated_landmark_coords = np.array([var.numpy() for var in self.estimated_landmarks]).reshape(-1, 2)
        plt.scatter(estimated_landmark_coords[:, 0], estimated_landmark_coords[:, 1], c='blue', marker='o', label='Estimated Landmarks')

        # Connect true and estimated landmarks for better comparison
        for i in range(self.num_landmarks):
            plt.plot([self.true_landmarks[i, 0], estimated_landmark_coords[i, 0]], [self.true_landmarks[i, 1], estimated_landmark_coords[i, 1]], 'k--', lw=1)

        plt.legend()
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title('SLAM: Random Walk Trajectory and Landmarks')
        plt.grid(True)
        plt.show()

    def __str__(self):
        return "2D SLAM Model for Estimating Robot Trajectory and Landmarks"




