import yaml
from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation as R

class Parameters:
    def __init__(self, filename):
        with open(filename, 'r', encoding='utf-8') as stream:
            entries = yaml.load(stream, Loader=yaml.SafeLoader)
        self.__dict__.update(entries)

class Analysis(Enum):
    LM          = "LM"
    PRESCRDISPL = "PrescrDispl"
    PENALTY     = "Penalty"

def matrix2xyzquat(
    matrix: np.ndarray, # 4x4 transformation matrix
    offset: np.ndarray = np.zeros(3) # offset to add to the initial position, in the not-transformed system
    ):

    # Extract translation
    xyz = np.asarray([matrix[0,3], matrix[1,3], matrix[2,3]])

    # Extract rotation
    rot_matrix = matrix[:3,:3]
    r = R.from_matrix(rot_matrix)
    quat = r.as_quat()

    # If offset is not zero, apply it
    if not np.array_equal(offset, np.zeros(3)):
        offset_transformed = offset.dot(rot_matrix.T)
        xyz += offset_transformed

    # Create translation + quaternion
    xyzquat = np.append(xyz, quat)
    return xyzquat