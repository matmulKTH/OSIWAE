import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.nn import softmax
import copy
#%% OSIWAE
class OSIWAE:
    def __init__(self, optimizer_proposal, optimizer_model,model, proposal, sampler, N, L, M, smoothing=False,
                  alpha=1., beta=1., n_draws=1, ancestor=True, mini_batch=1000):

        """
        :param model: model object, see models.py
        :param proposal: proposal object, see proposals.py
        :param L: number of particles to compute the proposal ELBO (small)
        :param M: number of particles to compute the model ELBO (large)
        """

        self.optimizer_proposal = optimizer_proposal
        self.optimizer_model = optimizer_model
        self.model = model
        self.n_trainable_model = len(self.model.list_trainable())
        self.proposal = proposal
        self.n_trainable_proposal = len(self.proposal.list_trainable())
        self.N = N
        self.L = L
        self.M = M
        self.smoothing = smoothing
        self.logN = tf.math.log(tf.cast(N, tf.float64))
        self.logL = tf.math.log(tf.cast(L, tf.float64))
        self.logM = tf.math.log(tf.cast(M, tf.float64))
        self.alpha = alpha
        self.beta = beta
        self.n_draws = n_draws
        self.sampler = sampler
        self.ancestor = ancestor
        self.mini_batch = mini_batch
        assert N % mini_batch == 0
        self.n_batches = int(N/mini_batch)

    def run(self, n=None, show_progress_bar=True, verbose_frequency=50):

        self.optimizer_proposal.build(self.proposal.list_trainable())
        self.optimizer_model.build(self.model.list_trainable())

        if n is None or n > self.model.max_n:
            n = self.model.max_n

        self.initialize_monitoring_variables(n)

        self.xPart, self.logWeights = [tf.stop_gradient(item) for item in self.model.initialize_particle_filter(self.N)]  # initialize particles and weights
        self.normWeights = softmax(self.logWeights, axis=0)  # normalized weights
        self.state[:,0] = tf.reduce_sum(self.xPart * self.normWeights[:, tf.newaxis], axis=0)
        self.ess[0] = 1 / tf.reduce_sum(tf.math.square(self.normWeights))
        
        if self.smoothing:
            self.t_stats = self.init_grad_term(tf.stop_gradient(self.xPart))
            # print(len(self.t_stats))
            # print(self.t_stats[0].shape)
            # initialize Enoch indices
            self.enoch_indX = tf.range(self.N)

        if show_progress_bar:
            iter_range = tqdm(range(1, n), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        else:
            iter_range = range(1, n)

        for m in iter_range:

            if self.smoothing:
                self.update_parameters_smoothing(tf.constant(m))
            else:
                self.update_parameters_ovsmc(tf.constant(m))

            self.smc_step(tf.constant(m))
            self.state[:,m] = tf.reduce_sum(self.xPart * self.normWeights[:, tf.newaxis], axis=0)
            if verbose_frequency > 0 and m % verbose_frequency == 1:
                print('\nIteration ' + str(m) + ', total and moving average ELBO proposal = '
                      + str((np.round(self.avg_elbo_prop, 4), np.round(np.mean(self.list_elbos_prop[-50:]), 4))))
                print('Iteration ' + str(m) + ', total and moving average ELBO model = '
                      + str((np.round(self.avg_elbo_model, 4), np.round(np.mean(self.list_elbos_model[-50:]), 4))))

    def initialize_monitoring_variables(self, n):
        """
        Initializes some variables to keep track of variables and ELBO through iterations

        :param n: int, number of iterations to monitor (equal to the total number of iterations)
        :param logWeights: [N, 1] tensor, needed for storing the first ELBO
        """

        self.ess = np.zeros(n)  # monitors the normalized effective sample size at each iteration

        # monitor the evolution of model and proposal parameters
        self.parameters_evolution = np.zeros((len(self.model.variables_to_watch())
                                              + len(self.proposal.variables_to_watch()), n))
        self.parameters_evolution[:len(self.model.variables_to_watch()), 0] = self.model.variables_to_watch()
        self.parameters_evolution[len(self.model.variables_to_watch()):, 0] = self.proposal.variables_to_watch()
        self.nn_weights = []
        self.nn_weights.append(copy.deepcopy(self.proposal.weights_nn()))
        self.state = np.zeros((self.model.dim,n))
        self.avg_elbo_prop = 0  # online ELBO mean
        self.list_elbos_prop = []  # list of ELBOs
        self.avg_elbo_model = 0  # online ELBO mean
        self.list_elbos_model = []  # list of ELBOs

        return

    def update_parameters_ovsmc(self, m):

        indX = self.sampler.resampling(self.normWeights, self.M)
        xPartRes = tf.gather(self.xPart, indices=indX, axis=0)

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
            g.watch(self.proposal.list_trainable()+self.model.list_trainable())
            xPartIWAE = self.proposal.propagate(xPartRes, m)
            logWeightsIWAE = self.compute_log_weights(xPartRes, xPartIWAE, m)

            # if self.optimizer_proposal is not None:
            logWeightsIWAE_prop = logWeightsIWAE[:self.L]
            max_logW_prop = tf.stop_gradient(tf.reduce_max(logWeightsIWAE_prop))
            W_prop = tf.math.exp(logWeightsIWAE_prop - max_logW_prop)
            IWAE_elbo_prop = max_logW_prop + tf.math.log(tf.reduce_sum(W_prop)) - self.logL
            negative_mean_IWAE_elbo_prop = tf.scalar_mul(-1., IWAE_elbo_prop)

            # if self.optimizer_model is not None:
            max_logW_model = tf.stop_gradient(tf.reduce_max(logWeightsIWAE))
            W_model = tf.math.exp(logWeightsIWAE - max_logW_model)
            IWAE_elbo_model = max_logW_model + tf.math.log(tf.reduce_sum(W_model)) - self.logM
            negative_mean_IWAE_elbo_model = tf.scalar_mul(-1., IWAE_elbo_model)


        grad_IWAE_elbo_prop = g.gradient(negative_mean_IWAE_elbo_prop, self.proposal.list_trainable(),
                                    unconnected_gradients='zero')
        grad_IWAE_elbo_model = g.gradient(negative_mean_IWAE_elbo_model, self.model.list_trainable(),
                                          unconnected_gradients='zero')

        self.optimizer_proposal.apply_gradients(zip(grad_IWAE_elbo_prop, self.proposal.list_trainable()))
        self.optimizer_model.apply_gradients(zip(grad_IWAE_elbo_model, self.model.list_trainable()))

        self.parameters_evolution[:self.n_trainable_model, m] = self.model.variables_to_watch()
        self.parameters_evolution[self.n_trainable_model:, m] = self.proposal.variables_to_watch()
        self.avg_elbo_prop += (IWAE_elbo_prop.numpy() - self.avg_elbo_prop) / m.numpy()
        self.list_elbos_prop.append(IWAE_elbo_prop)
        self.avg_elbo_model += (IWAE_elbo_model.numpy() - self.avg_elbo_model) / m.numpy()
        self.list_elbos_model.append(IWAE_elbo_model)
    
    @tf.function
    def gradient_proposal(self, xPartRes, m):
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as g:
            g.watch(self.proposal.list_trainable())
            xPartN = self.proposal.propagate(xPartRes,m)
            logWeights = self.compute_log_weights(xPartRes,xPartN, m)
            max_logW = tf.stop_gradient(tf.reduce_max(logWeights))
            W_prop = tf.math.exp(logWeights - max_logW)
            elbo = max_logW + tf.math.log(tf.reduce_sum(W_prop)) - self.logL
            negative_elbo = -elbo
        return g.gradient(negative_elbo, self.proposal.list_trainable(), unconnected_gradients='zero'), tf.stop_gradient(elbo)


    @tf.function
    def gradient_model(self, xPartRes, xPartResN, m):
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as g:
            g.watch(self.model.list_trainable())
            xPartN = self.proposal.propagate(xPartRes,m)
            logWeights = self.compute_log_weights(xPartRes, xPartN, m)
            max_logW = tf.stop_gradient(tf.reduce_max(logWeights))
            partial_sum_W = tf.reduce_sum(tf.math.exp(logWeights - max_logW))
            partial_log_sum_W = max_logW + tf.math.log(partial_sum_W)
            smoothing_elbos = self.compute_elbos(xPartResN,partial_log_sum_W, m)
            mean_elbo = tf.reduce_mean(smoothing_elbos)
            negative_mean_elbo = -mean_elbo
        gradients = g.gradient(negative_mean_elbo, self.model.list_trainable(),
                               unconnected_gradients='zero')
        return gradients, tf.stop_gradient(smoothing_elbos)

    def update_parameters_smoothing(self, m):

        indX = self.sampler.resampling(self.normWeights, self.M - 1)
        xPartRes = tf.gather(self.xPart, indices=indX, axis=0)
        indXN = self.sampler.resampling(self.normWeights, self.N)
        xPartResN = tf.gather(self.xPart, indices=indXN, axis=0)
        
        
        grad_IWAE_elbo_prop, IWAE_elbo_prop = self.gradient_proposal(xPartRes[:self.L], m)

        self.optimizer_proposal.apply_gradients(zip(grad_IWAE_elbo_prop, self.proposal.list_trainable()))


        self.avg_elbo_prop += (IWAE_elbo_prop.numpy() - self.avg_elbo_prop) / m.numpy()
        self.list_elbos_prop.append(IWAE_elbo_prop)

        grad_IWAE_elbo_model, smoothing_elbos = self.gradient_model(xPartRes, xPartResN, m)
        gradients_model = []
        for i, grad in enumerate(grad_IWAE_elbo_model):
            t_stat = tf.gather(self.t_stats[i], indices=indXN, axis=0)
            t_stat_mean = tf.reduce_mean(t_stat, axis=0)

            add_grad = self.M * tf.reduce_mean(
                tf.reshape(smoothing_elbos, [self.N] + [1 for _ in t_stat.shape[:-1]])
                * (t_stat - t_stat_mean), axis=0)
            gradients_model.append(grad - add_grad)
        
        self.optimizer_model.apply_gradients(zip(gradients_model, self.model.list_trainable()))

        self.parameters_evolution[:self.n_trainable_model, m] = self.model.variables_to_watch()
        self.parameters_evolution[self.n_trainable_model:, m] = self.proposal.variables_to_watch()
        weights_nn = self.proposal.weights_nn()
        self.nn_weights.append(copy.deepcopy(weights_nn))
        self.avg_elbo_model += (tf.reduce_sum(self.normWeights * smoothing_elbos).numpy() - self.avg_elbo_model) / m.numpy()
        self.list_elbos_model.append(tf.reduce_sum(self.normWeights * smoothing_elbos))

    
    def compute_elbos(self, xPartRes, partial_log_sum_W, m):
        
        xPartN = self.proposal.propagate(xPartRes, m)
            
        logWeight_variable = self.compute_log_weights(xPartRes, xPartN, m)
        max_logW = tf.stop_gradient(tf.math.maximum(partial_log_sum_W, logWeight_variable))
        sum_weights = tf.math.exp(logWeight_variable - max_logW) + tf.math.exp(partial_log_sum_W - max_logW)
        all_elbos = max_logW + tf.math.log(sum_weights) - self.logM

        return all_elbos

    @tf.function
    def compute_elbo_mini_batch(self, partial_log_sum_W, logWeight_variable):
        max_logW = tf.stop_gradient(tf.math.maximum(partial_log_sum_W, logWeight_variable))
        sum_weights = tf.math.exp(logWeight_variable - max_logW) + tf.math.exp(partial_log_sum_W - max_logW)
        all_elbos = max_logW + tf.math.log(sum_weights) - self.logM
        return tf.reduce_mean(all_elbos, axis=1)

    def additive_term(self, x_new, x_old, m):
        grads = [self.grad_log_weights_model(x_new[self.mini_batch * k:self.mini_batch * (k + 1)],
                  x_old[self.mini_batch * k:self.mini_batch * (k + 1)], m) for k in range(self.n_batches)]
        return [tf.concat([g[k] for g in grads], axis=0) for k in range(self.n_trainable_model)]
    
    @tf.function
    def grad_log_weights_model(self, x_new, x_old, m):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
            g.watch(self.model.list_trainable())
            out = self.model.log_target_transition_density(x_old, x_new, m)
        grads = g.jacobian(out, self.model.list_trainable(), unconnected_gradients='zero')
        return grads

    def init_grad_term(self, x):
        grads = [self.grad_g0(x[self.mini_batch * k:self.mini_batch * (k + 1)]) for k in range(self.n_batches)]
        return [tf.concat([g[k] for g in grads], axis=0) for k in range(self.n_trainable_model)]


    def grad_g0(self, x):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
            g.watch(self.model.list_trainable())
            out = self.model.log_emission_density(x, 0)
        return g.jacobian(out, self.model.list_trainable(), unconnected_gradients='zero')

    def smc_step(self, m):

        resampled = False
        self.ess[m] = 1 / tf.reduce_sum(tf.math.square(self.normWeights))
        if self.ess[m] <= self.alpha * self.N:
            resampled = True
            indX = self.sampler.resampling(self.normWeights, self.N)
        else:
            indX = tf.range(self.N)

        # Propagate and reweight
        xPartRes = tf.gather(self.xPart, indices=indX, axis=0)
        
        
        xPartN = tf.stop_gradient(self.proposal.propagate(xPartRes, m))
        logWeightsN = self.compute_log_weights(xPartRes, xPartN, m)
            
        
        if not resampled:
            # If we don't resample we need to add the previous log weights to the current weights
            logWeightsN += self.logWeights

        if self.smoothing:
            self.enoch_indX = tf.gather(self.enoch_indX, indices=indX)
            # We need smoothing expectations
            if resampled and np.unique(self.enoch_indX).shape[0] <= self.beta * self.N:
                if self.n_draws > 0:
                    ## Pure PaRIS (if self.ancestor is False)
                    backward_indX = self.sampler.backward_sampling(self.logWeights, self.xPart, xPartN, indX,
                                                                    m, tf.constant(self.n_draws))
                    if self.ancestor:
                        backward_indX = tf.concat([backward_indX, tf.expand_dims(indX, axis=1)], axis=1)
                    t_stats_new = []

                    grads = self.additive_term(tf.expand_dims(xPartN, axis=1),
                                                tf.gather(self.xPart, backward_indX, axis=0), m)
                    for it in range(self.n_trainable_model):
                        t_stats_new.append(tf.gather(self.t_stats[it], backward_indX, axis=0) + grads[it])

                    self.t_stats = [tf.math.reduce_mean(item, axis=1) for item in t_stats_new]

                else:
                    raise ValueError('Unable to compute additive term: select n_draws > 0 or ancestor = True')

                self.enoch_indX = tf.range(self.N)
            else:
                t_stats_new = []
                grads = self.additive_term(xPartN, xPartRes, m)
                for it in range(self.n_trainable_model):
                    t_stats_new.append(tf.gather(self.t_stats[it], indX, axis=0) + grads[it])
                self.t_stats = t_stats_new

        self.normWeights = tf.stop_gradient(softmax(logWeightsN))

        # update particles and weights
        self.xPart = tf.stop_gradient(xPartN)
        self.logWeights = tf.stop_gradient(logWeightsN)


    
    @tf.function
    def compute_log_weights(self, x_old, x_new, m):
        """
        Computes the log-weights in the new algorithm with a warm up period where we have a bootstrap proposal the first 5000 steps

        :param x_old: [N, model.dim] tensor for current particles
        :param x_new: [N, model.dim] tensor for new propagated particles
        :param n: int for the current iteration
        :return: [N, 1] tensor of log-weights
        """

        return self.model.log_target_transition_density(x_old, x_new, m) - self.proposal.log_proposal_density(x_old,x_new,m)
    
    
    
    
    def __str__(self):
        return '\n\nSequential IWAE'

#%% RML 
class RML:
    """
    Standard bootstrap particle filter-based RML algorithm
    """

    def __init__(self, model, optimizer, N, alpha=1., beta=1., n_draws=2, sampler=None, ancestor=False, mini_batch=100):

        """

        :param model: model object, see models.py
        :param proposal: proposal object, see proposals.py
        :param add_func: target function object for filter expectation, see target_functions.py
        :param L: number of particles to compute the proposal ELBO (small)
        :param N: number of particles to compute the model ELBO (large)
        """

        self.model = model
        self.n_trainable_model = len(self.model.list_trainable())
        self.optimizer = optimizer
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.n_draws = n_draws
        self.sampler = sampler
        self.ancestor = ancestor
        self.mini_batch = mini_batch
        assert N % mini_batch == 0
        self.n_batches = int(N / mini_batch)

    def run(self, n=None, show_progress_bar=True, verbose_frequency=100):

        self.optimizer.build(self.model.list_trainable())

        if n is None or n > self.model.max_n:
            n = self.model.max_n

        self.parameters_evolution = np.zeros((len(self.model.variables_to_watch()), n))
        self.parameters_evolution[:, 0] = self.model.variables_to_watch()

        self.xPart, self.logWeights = [tf.stop_gradient(item) for item in self.model.initialize_particle_filter(
            self.N)]  # initialize particles and weights
        self.normWeights = softmax(self.logWeights, axis=0)  # normalized weights
        
        self.state = np.zeros((self.model.dim,n))
        self.state[:,0] = tf.reduce_sum(self.xPart * self.normWeights[:, tf.newaxis], axis=0)
        self.t_stats = self.init_t_stats()
        self.enoch_indX = tf.range(self.N)

        if show_progress_bar:
            iter_range = tqdm(range(1, n), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        else:
            iter_range = range(1, n)

        for m in iter_range:

            m = tf.constant(m)

            ess = 1 / tf.reduce_sum(tf.math.square(self.normWeights))
            resampled = False
            if ess <= self.alpha * self.N:
                resampled = True
                indX = self.sampler.resampling(self.normWeights, self.N)
            else:
                indX = tf.range(self.N)

            xPartRes = tf.gather(self.xPart, indices=indX, axis=0)
            xPartN = tf.stop_gradient(self.model.propagate_latent_prior(xPartRes, m))
            gradients_log_g, log_g = self.gradient_log_g(xPartN, m)
            if resampled:
                logWeightsN = log_g
                predWeights = tf.ones([self.N], dtype=tf.float64)/tf.constant(self.N, dtype=tf.float64)
            else:
                # If we don't resample we need to add the previous log weights to the current weights
                logWeightsN = log_g + self.logWeights
                predWeights = self.normWeights


            normWeightsN = softmax(logWeightsN)

            expected_gradients_log_g = []
            for i in range(self.n_trainable_model):
                expected_gradients_log_g.append(tf.reduce_sum(tf.reshape(normWeightsN, [self.N] + [1 for _ in gradients_log_g[i].shape[:-1]])
                                                  * gradients_log_g[i], axis=0))

            self.enoch_indX = tf.gather(self.enoch_indX, indices=indX)
            # We need smoothing expectations
            if resampled and np.unique(self.enoch_indX).shape[0] <= self.beta * self.N:
                if self.sampler.backward_sampler is None:
                    ## FFBSm
                    back_probs = softmax(tf.expand_dims(self.logWeights, axis=0)
                                         + self.model.log_target_transition_density(tf.expand_dims(self.xPart, axis=0),
                                                                                    tf.expand_dims(xPartN, axis=1), m))
                    t_stats_new = []
                    grads = self.additive_term(tf.expand_dims(xPartN, axis=1),
                                               tf.expand_dims(self.xPart, axis=0), m)
                    for it in range(self.n_trainable_model):
                        t_stats_new.append(tf.reduce_sum(tf.math.multiply(tf.expand_dims(back_probs, axis=-1),
                                                                          tf.expand_dims(self.t_stats[it], axis=0) +
                                                                          grads[it]), axis=1))
                    self.t_stats = t_stats_new

                else:
                    # Take one backward draw for each particle
                    if self.n_draws > 0:
                        ## Pure PaRIS (if self.ancestor is False)
                        backward_indX = self.sampler.backward_sampling(self.logWeights, self.xPart, xPartN, indX,
                                                                       m, tf.constant(self.n_draws))
                        if self.ancestor:
                            backward_indX = tf.concat([backward_indX, tf.expand_dims(indX, axis=1)], axis=1)
                        t_stats_new = []

                        grads = self.additive_term(tf.expand_dims(xPartN, axis=1),
                                                   tf.gather(self.xPart, backward_indX, axis=0), m)
                        for it in range(self.n_trainable_model):
                            t_stats_new.append(tf.gather(self.t_stats[it], backward_indX, axis=0) + grads[it])

                        self.t_stats = [tf.math.reduce_mean(item, axis=1) for item in t_stats_new]

                    elif self.ancestor:
                        ## Poor man's smoother
                        t_stats_new = []
                        grads = self.additive_term(xPartN, xPartRes, m)
                        for it in range(self.n_trainable_model):
                            t_stats_new.append(tf.gather(self.t_stats[it], indX, axis=0) + grads[it])
                        self.t_stats = t_stats_new
                    else:
                        raise ValueError('Unable to compute additive term: select n_draws > 0 or ancestor = True')

                self.enoch_indX = tf.range(self.N)
            else:
                t_stats_new = []
                grads = self.additive_term(xPartN, xPartRes, m)
                for it in range(self.n_trainable_model):
                    t_stats_new.append(tf.gather(self.t_stats[it], indX, axis=0) + grads[it])
                self.t_stats = t_stats_new

            diffWeights = normWeightsN - predWeights
            final_gradients = []
            for i, grad in enumerate(expected_gradients_log_g):
                final_gradients.append(- grad - tf.reduce_sum(tf.reshape(diffWeights,
                                        [self.N] + [1 for _ in self.t_stats[i].shape[:-1]]) * self.t_stats[i], axis=0))

            self.optimizer.apply_gradients(zip(final_gradients, self.model.list_trainable()))

            self.xPart = tf.stop_gradient(xPartN)
            self.logWeights = tf.stop_gradient(logWeightsN)
            self.normWeights = tf.stop_gradient(normWeightsN)
            self.state[:,m] = tf.reduce_sum(self.xPart * self.normWeights[:, tf.newaxis], axis=0)
            self.parameters_evolution[:, m] = self.model.variables_to_watch()

            if verbose_frequency > 0 and m % verbose_frequency == 1:
                print(self.model.variables_to_watch())

    @tf.function
    def gradient_log_g(self, xPartN, m):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
            g.watch(self.model.list_trainable())
            log_g = self.model.log_emission_density(xPartN, m)
        return g.jacobian(log_g, self.model.list_trainable(), unconnected_gradients='zero'), tf.stop_gradient(log_g)

    def init_t_stats(self):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
            g.watch(self.model.list_trainable())
            out = tf.zeros([self.N], dtype=tf.float64)

        return g.jacobian(out, self.model.list_trainable(), unconnected_gradients='zero')

    @tf.function
    def additive_term(self, x_new, x_old, m):
        grads = [self.grad_log_g_and_m(x_new[self.mini_batch * k:self.mini_batch * (k + 1)],
                 x_old[self.mini_batch * k:self.mini_batch * (k + 1)], m) for k in range(self.n_batches)]
        return [tf.concat([g[k] for g in grads], axis=0) for k in range(self.n_trainable_model)]

    @tf.function
    def grad_log_g_and_m(self, x_new, x_old, m):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
            g.watch(self.model.list_trainable())
            out = self.model.log_emission_density(x_old, m-1) + self.model.log_latent_transition_density(x_old, x_new, m)

        return g.jacobian(out, self.model.list_trainable(), unconnected_gradients='zero')

    def __str__(self):
        return '\n\nRecursive maximum likelihood algorithm'

