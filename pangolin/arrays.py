from pangolin import *
#import pangolin.automap

from pangolin.automap_simple import automap

class Array(RV):
    """
    Fill up lists and call automap when full
    """

    def __init__(self, shape, check=True):
        # do not call super!
        self.check = check
        if isinstance(shape,int):
            shape = (shape,)
        assert isinstance(shape,tuple)
        self.init_shape = shape
        self.rvs = np.empty(shape, dtype=RV)
        self.num_empty = np.prod(shape)
        self._shape = None

    def __setitem__(self, idx, value):
        value = makerv(value)
        assert isinstance(value, RV)
        assert self.rvs[idx] is None, "can only assign into slot of Array once"

        if hasattr(value, "cond_dist"):
            # guess shape from first assignment
            expected_shape = self.init_shape + value.shape
            if self._shape is None:
                self._shape = expected_shape
            else:
                assert self._shape == expected_shape

            # make copy of RV
            self.rvs[idx] = RV(value.cond_dist, *value.parents)
            # make sure no one else uses rv
            value.clear()
            self.num_empty -= 1
            if self.num_empty == 0:
                self.activate()

    def activate(self):
        #print("ACTIVATING")
        #print(f"{self.rvs=}")
        rv = automap(self.rvs.tolist(),check=self.check)
        super().__init__(rv.cond_dist, *rv.parents)
