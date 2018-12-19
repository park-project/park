import numpy as np
import random


"""
Separate the random number generator from the environment.
This is used for all random sample in the space native methods.
We expect new algorithms to have their own rngs.
"""

np_random = np.random.RandomState()
np_random.seed(42)

# native random, complement functions like O(1) choice from np_random
nrng = random.Random()
nrng.seed(42)
