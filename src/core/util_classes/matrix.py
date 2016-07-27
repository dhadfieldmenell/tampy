from IPython import embed as shell
import numpy as np

class Matrix(np.ndarray):
    """
    The matrix class is useful for tracking object poses.
    """
    def __new__(cls, *args, **kwargs):
        raise NotImplementedError("Override this.")

class Vector(Matrix):
    """
    The vector class.
    """
    def __new__(cls, vec, dim):
        if type(vec) is str:
            if not vec.endswith(")"):
                vec += ")"
            vec = eval(vec)
        obj = np.array(vec)
        # deals with case where obj is zero-dimensional
        assert len(np.atleast_1d(obj)) == dim
        obj = obj.reshape((dim, 1))
        return obj

class Vector1d(Vector):
    def __new__(cls, vec):
        return super(Vector1d, cls).__new__(cls, vec, 1)

class Vector2d(Vector):
    """
    The NAMO domain uses the Vector2d class to track poses of objects in the grid.
    """
    def __new__(cls, vec):
        return super(Vector2d, cls).__new__(cls, vec, 2)

class Vector3d(Vector):
    def __new__(cls, vec):
        return super(Vector3d, cls).__new__(cls, vec, 3)

class Vector7d(Vector):
    def __new__(cls, vec):
        return super(Vector7d, cls).__new__(cls, vec, 7)

class PR2PoseVector(Vector3d):
    """
    The PR2 domain uses the PR2PoseVector class to base poses of pr2.
    """
    pass

class PR2ArmPose(Vector7d):
    """
    The PR2's arm pose is a 7d vector.
    """
    def __new__(cls, vec):
        return super(Vector2d, cls).__new__(cls, vec, 7)

class Value(Vector1d):
    pass
