import random
import collections # useful data structures
import numpy as np
import jax
import jax.numpy as jnp # jax numpy
import haiku as hk # jax neural network library
import warnings
warnings.filterwarnings('ignore')

QLearnParams = collections.namedtuple("Params", ["online", "target"])

# Q-learn-state
QLearnState = collections.namedtuple("LearnerState", ["count", "optim_state"])

# NamedTuple to store transitions
Transition = collections.namedtuple("Transition", ["obs", "action", "reward", "next_obs", "done"])

# Actor state stores the current number of timesteps
QActorState = collections.namedtuple("ActorState", ["count"])
# Implement a function takes q-values as input and returns the greedy_action
def select_greedy_action(q_values):

  # YOUR CODE
  action = jnp.argmax(q_values)
  # END YOUR CODE

  return action

def build_network(num_actions: int, layers=[20, 20]) -> hk.Transformed:
  """Factory for a simple MLP network for approximating Q-values."""

  def q_network(obs):
    network = hk.Sequential(
        [hk.Flatten(),
         hk.nets.MLP(layers + [num_actions])])
    return network(obs)

  return hk.without_apply_rng(hk.transform(q_network))

def compute_squared_error(pred, target):

  return  jnp.square(pred-target)


# Bellman target
def compute_bellman_target(reward, done, next_q_values, gamma=0.99):

  bellman_target = jax.lax.cond(
    done == 0.0,  # condition
    lambda _: reward + gamma * jnp.max(next_q_values),  # if done == 0.0
    lambda _: jnp.array(reward, dtype=jnp.float32),  # else (convert reward to float32)
    operand=None  # no shared input required
   )
  return bellman_target

def q_learning_loss(q_values, action, reward, done, next_q_values):

    chosen_action_q_value = q_values[action] # q_value of action, use array indexing
    bellman_target        = compute_bellman_target(reward, done, next_q_values, gamma=0.99)
    squared_error         = compute_squared_error(chosen_action_q_value, bellman_target)

    return squared_error
def select_random_action(key, num_actions):

    action = jax.random.randint(key = key,shape= (), minval = 0, maxval=num_actions)

    return action


EPSILON_DECAY_TIMESTEPS = 10000 # decay epsilon over 3000 timesteps
EPSILON_MIN = 0.1 # 10% exploration
def get_epsilon(num_timesteps):
  # YOUR CODE
  epsilon = 1 - num_timesteps/EPSILON_DECAY_TIMESTEPS # decay epsilon

  epsilon = jax.lax.select(
      epsilon < EPSILON_MIN,
      EPSILON_MIN, # if less than min then set to min
      epsilon # else don't change epsilon
  )
  # END YOUR CODE

  return epsilon
def select_epsilon_greedy_action(key, q_values, num_timesteps):
    num_actions = len(q_values) # number of available actions

    # YOUR CODE HERE
    epsilon = get_epsilon(num_timesteps) # get epsilon value

    should_explore = jax.random.uniform(key, minval=0.0, maxval=1.0) <= epsilon  # hint: a boolean expression to check if some random number is less than epsilon

    action = jax.lax.select(
        should_explore,
        select_random_action(key, num_actions) , # if should explore
        select_greedy_action(q_values) # if should be greedy
    )
    # END YOUR CODE

    return action

class TransitionMemory(object):
  """A simple Python replay buffer."""

  def __init__(self, max_size=10_000, batch_size=256):
    self.batch_size = batch_size
    self.buffer = collections.deque(maxlen = max_size)

  def push(self, transition):
    # add transition to the replay buffer
    self.buffer.append(
        (transition.obs, transition.action, transition.reward,
          transition.next_obs, transition.done)
          )

  def is_ready(self):
    return self.batch_size <= len(self.buffer)

  def sample(self):
    # Randomly sample a batch of transitions from the buffer
    random_replay_sample = random.sample(self.buffer, self.batch_size)

    # Batch the transitions together
    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*random_replay_sample)

    return Transition(
        np.stack(obs_batch).astype("float32"),
        np.asarray(action_batch).astype("int32"),
        np.asarray(reward_batch).astype("float32"),
        np.stack(next_obs_batch).astype("float32"),
        np.asarray(done_batch).astype("float32")
                      )

def update_target_params(learn_state, online_weights, target_weights, update_frequency=100):

  """A function to update target params every 100 training steps"""

  target = jax.lax.cond(
      jnp.mod(learn_state.count, update_frequency) == 0,
      lambda x, y: x,
      lambda x, y: y,
      online_weights,
      target_weights
  )

  params = QLearnParams(online_weights, target)

  return params
