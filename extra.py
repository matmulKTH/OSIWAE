import numpy as np
import tensorflow as tf
from tensorflow.nn import softmax

class Sampler:

    def __init__(self, model, resampling_type='multinomial', backward_sampler=None, mini_batch=None):

        self.model = model
        self.resampling_type = resampling_type
        self.backward_sampler = backward_sampler
        self.mini_batch = mini_batch
    @staticmethod
    def samples_from_categorical(w, size):
        bins = tf.math.cumsum(w)
        u_samples = tf.random.uniform([size], dtype=tf.float64)
        return tf.searchsorted(bins, u_samples)

    @staticmethod
    def samples_from_many_categoricals(w_matrix, n_draws):

        bins = tf.math.cumsum(w_matrix, axis=1)

        u_samples = tf.random.uniform([w_matrix.shape[0], n_draws], dtype=tf.float64)

        return tf.searchsorted(bins, u_samples)

    def resampling(self, w, size):
        """
        Computes the resampling indices.

        :param w: 1-d array/tensor of normalized weights
        :param size: int or tuple with shape of the samples needed
        :return: [size] tensor of resampled indices
        """

        if self.resampling_type == 'multinomial':
            return self.samples_from_categorical(w, size)
        elif self.resampling_type == 'residual':
            floors = tf.math.floor(size*w)
            residuals = (size*w-floors)/(size-tf.reduce_sum(floors))
            floors = tf.cast(floors, tf.int32)
            out1 = tf.repeat(tf.range(size), floors)
            out2 = self.samples_from_categorical(residuals, size-out1.shape[0])
            return tf.concat([out1, out2], axis=0)
        elif self.resampling_type == 'stratified':
            bins = tf.math.cumsum(w)
            u_samples = (tf.range(size, dtype=tf.float64) + tf.random.uniform([size], dtype=tf.float64)) / size
            return tf.searchsorted(bins, u_samples)
        elif self.resampling_type == 'systematic':
            bins = tf.math.cumsum(w)
            u_samples = (tf.range(size, dtype=tf.float64) + tf.random.uniform([1], dtype=tf.float64)) / size
            return tf.searchsorted(bins, u_samples)


    def backward_sampling(self, logWeights, xPart, xPartN, indX, n, n_draws):

        """
        Compute backward sampling in the particle filter.

        :param logWeights: [N, 1] tensor of old log-weights
        :param xPart: [N, dim] tensor of old particles
        :param xPartN: [N, dim] tensor of new particles
        :param indX: [N] tensor of resampling indices of the old particles
        :param n: int for iteration number
        :param n_draws: int for number of backward draws required
        :return: [N, n_draws] tensor of backward indices
        """

        if self.backward_sampler is None:
            raise ValueError('Choose the type of backward sampler')

        if self.backward_sampler == 'full' and self.mini_batch is None:

            '''Full backward draws'''
            
            log_target = self.model.log_target_transition_density(tf.expand_dims(xPart, axis=0),
                                                                 tf.expand_dims(xPartN, axis=1), n)
            

            
            exp_logWeights = tf.expand_dims(logWeights, axis=0)
            

            
            back_probs = softmax(exp_logWeights + log_target)

            
            
            backward_indX = self.samples_from_many_categoricals(back_probs, n_draws)

            
        elif self.backward_sampler == 'full' and self.mini_batch is not None:

            assert xPartN.shape[0] % self.mini_batch == 0
            n_batch = int(xPartN.shape[0] / self.mini_batch)

            backward_indX = tf.concat([self.mini_backward_sampling(logWeights, xPart,
                        xPartN[self.mini_batch*k:self.mini_batch*(k+1)], n, n_draws) for k in range(n_batch)], axis=0)

        else:
            raise ValueError('Backward sampler not in list')
        return backward_indX

#    @tf.function
    def mini_backward_sampling(self, logWeights, xPart, xPartN, n, n_draws):
        back_probs = softmax(tf.expand_dims(logWeights, axis=0)
                             + self.model.log_target_transition_density(tf.expand_dims(xPart, axis=0),
                                                                        tf.expand_dims(xPartN, axis=1), n))
        
        
        return self.samples_from_many_categoricals(back_probs, n_draws)
