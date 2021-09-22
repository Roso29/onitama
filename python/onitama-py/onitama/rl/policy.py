import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from stable_baselines.common.tf_layers import conv, conv_to_fc, linear
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.distributions import ProbabilityDistributionType, CategoricalProbabilityDistribution


def cnn_extractor_onitama(scaled_images, n_obs, n_filters_out=50, filter_size=5, **kwargs):
    """
    CNN with 5 x 5 x 50 (50 = 25 x 2) outputs, that is masked by 2nd half of inputs
    :param scaled_images: (TensorFlow Tensor) Image input placeholder (Batch size x Obs shape)
    :param n_obs: number of final dimension that is for observations, rest are not passed as inputs and are
    used as mask, pass -1 for using all inputs
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    # split into mask and obs
    if n_obs != -1:
        scaled_images = scaled_images[:, :, :, :n_obs]
    layer_1 = activ(
        conv(scaled_images, 'c1', n_filters=32, filter_size=filter_size, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(
        conv(layer_1, 'c2', n_filters=64, filter_size=filter_size, stride=1, pad='SAME', init_scale=np.sqrt(2),
             **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=n_filters_out, filter_size=filter_size, stride=1, pad='SAME',
                         init_scale=np.sqrt(2), **kwargs))
    layer_3_flat = conv_to_fc(layer_3)
    return layer_3_flat


def apply_mask(values, mask_flat, mask_to=tf.float32.min):
    # if it's masked, tf.float32.min, if it's valid then 1, to sample only valid
    masked = tf.where(mask_flat > 0, values, tf.fill(tf.shape(values), mask_to))
    return masked


class DQNMaskedCNNPolicy(DQNPolicy):
    """
    Policy object that implements a DQN policy, using a feed forward neural network.
    Uses part of inputs to mask the output action probability distribution.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param layer_norm: (bool) enable layer normalisation
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 obs_phs=None, layer_norm=False, dueling=False, act_fun=tf.nn.relu, **kwargs):
        super(DQNMaskedCNNPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                 n_batch, dueling=dueling, reuse=reuse,
                                                 scale=True, obs_phs=obs_phs)

        self.n_obs = 9

        if layers is None:
            layers = [64, 64]

        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("action_value"):
                extracted_features = cnn_extractor_onitama(self.processed_obs, self.n_obs)
                action_scores = tf_layers.fully_connected(extracted_features, num_outputs=self.n_actions)

            if self.dueling:
                with tf.variable_scope("state_value"):
                    state_out = extracted_features
                    for layer_size in layers:
                        state_out = tf_layers.fully_connected(state_out, num_outputs=layer_size, activation_fn=None)
                        if layer_norm:
                            state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
                        state_out = act_fun(state_out)
                    state_score = tf_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores, axis=1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores

        # TODO: should be applied before q or after (ie. right before softmax)?
        mask_inp = self.processed_obs[:, :, :, self.n_obs:]
        mask_flat = conv_to_fc(mask_inp)
        # get a mask as tf.float32.min for invalid and 1s for valid
        self.mask = apply_mask(tf.ones_like(mask_flat), mask_flat)
        # get masked q values
        masked_q = apply_mask(q_out, mask_flat)
        self.q_values = masked_q
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        """
        Returns the actions, q_values, states for a single step

        :param obs: (np.ndarray float or int) The current observation of the environment
        :param state: (np.ndarray float) The last states (used in recurrent policies)
        :param mask: (np.ndarray float) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray int, np.ndarray float, np.ndarray float) actions, q_values, states
        """
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Unefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])
        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: (np.ndarray float or int) The current observation of the environment
        :param state: (np.ndarray float) The last states (used in recurrent policies)
        :param mask: (np.ndarray float) The last masks (used in recurrent policies)
        :return: (np.ndarray float) the action probability
        """
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})


class ACMaskedCNNPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, **kwargs):
        super(ACMaskedCNNPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        self.n_obs = 9

        with tf.variable_scope("model", reuse=reuse):
            mask = self.processed_obs[:, :, :, self.n_obs:]
            pi_latent = vf_latent = cnn_extractor_onitama(self.processed_obs, self.n_obs)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._pdtype = MaskedCategoricalProbabilityDistributionType(ac_space.n)
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, mask, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class MaskedCategoricalProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, n_cat):
        """
        The probability distribution type for categorical input

        :param n_cat: (int) the number of categories
        """
        self.n_cat = n_cat

    def probability_distribution_class(self):
        return CategoricalProbabilityDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, mask, init_scale=1.0, init_bias=0.0):
        pdparam = linear(pi_latent_vector, 'pi', self.n_cat, init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', self.n_cat, init_scale=init_scale, init_bias=init_bias)
        # apply mask
        mask_flat = conv_to_fc(mask)
        pdparam = apply_mask(pdparam, mask_flat)
        q_values = apply_mask(q_values, mask_flat)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [self.n_cat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return tf.int64