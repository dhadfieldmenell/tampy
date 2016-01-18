from IPython import embed as shell
import numpy as np

class Matrix(object):
    """
    The matrix class is useful for tracking object poses.
    """
    def __init__(self, *args):
        raise NotImplementedError("Override this.")

class Vector2d(Matrix):
    """
    The NAMO domain uses the Vector2d class to track poses of objects in the grid.
    """
    def __init__(self, vec):
        if type(vec) is str:
            if not vec.endswith(")"):
                vec += ")"
            vec = eval(vec)
        self._vec = np.array(vec)
        assert len(self._vec) == 2
        self._vec = self._vec.reshape((2, 1))

    def shape(self):
        return self._vec.shape

    def __getitem__(self, i):
        e = self._vec[i]
        if not isinstance(e, np.ndarray):
            return e
        if len(e) != 2:
            return e
        return Vector2d(e)

    def __eq__(self, other):
        return np.array_equal(self._vec, other._vec)

    def __repr__(self):
        return repr(self._vec)
