import random
import params
from config import Action, Query

class ActionSpace(Space):
    def sample(self):
        n = random.randint(1, params.NDIMS)
        dims = random.sample(range(params.NDIMS), n)
        cols = []
        for i in range(n-1):
            cols.append(random.randint(1, 100))
        return Action(dims, cols)

    def contains(self, a):
        valid = True
        valid &= (len(a.dimensions) <= params.NDIMS)
        for d in a.dimensions:
            valid &= isinstance(d, int)
            valid &= (d < params.NDIMS && d >= 0)
        valid &= (len(a.columns) == len(a.dimensions)-1)
        for c in a.columns:
            valid &= isinstance(c, int)
            valid &= (c > 0)
        return valid

class DataObsSpace(Space):
    def sample(self):
        pass

    def contains(self, s):
        return s.data_iterator is not None

class QueryObsSpace(Space):
    def sample(self):
        pass

    def contains(self, s):
        return isinstance(s, Query) && s.valid()


