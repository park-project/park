from park import logger


# Env-related abstractions

class Env(object):
    """
    The main park class. The interface follows OpenAI gym
    https://gym.openai.com, which encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.

    The main API methods that users of this class need to know are:
    
        observe
        step
        reset
        seed

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    """

    # Set this in some subclasses
    metadata = {'env.name': 'abstract_env'}
    reward_range = (-float('inf'), float('inf'))

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the space.
        """
        raise NotImplementedError

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        """
        logger.warn('Could not seed environment ' + self.metadata['env.name'])
        return

# Space-related abstractions

class Space(object):
    """
    Defines the observation and action spaces, so you can write generic
    code that applies to any Env. For example, you can choose a random
    action.
    """
    def __init__(self, struct=None, shape=None, dtype=None):
        import numpy as np # takes about 300-400ms to import, load lazily
        self.struct = None if struct is None else struct  # tensor, graph, etc.
        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)

    def sample(self):
        """
        Uniformly randomly sample a random element of this space
        """
        raise NotImplementedError

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError
