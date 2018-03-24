import numpy as np
from scipy.io import loadmat
import glob

def gather_data(path, eye='right'):
    
    mat_files = glob.glob(f'{path}/**/*.mat', recursive=True)
    mat_files.sort()
    
    indices = []
    IMG = []
    POSE = []
    GAZE = []
    for file in mat_files:
        matfile = loadmat(file)
        
        img = matfile['data'][eye][0, 0]['image'][0, 0]
        pose = matfile['data'][eye][0, 0]['pose'][0, 0]
        gaze = matfile['data'][eye][0, 0]['gaze'][0, 0]
        file_idx = file.split('/')[-2], file.split('/')[-1].split('.')[0]
        
        indices.extend([[*file_idx, jpg[0][0]] for jpg in matfile['filenames']])
        IMG.extend(img)
        POSE.extend(pose)
        GAZE.extend(gaze)
    
    indices = np.array(indices)
    IMG = np.array(IMG).reshape((-1, 36, 60, 1))
    POSE = np.array(POSE)
    GAZE = np.array(GAZE)
    
    return indices, IMG, POSE, GAZE


def gaze3Dto2D(array, stack=True):
    """
    theta = asin(-y)
    phi = atan2(-x, -z)
    """
    if array.ndim == 2:
        assert array.shape[1] == 3
        x, y, z = (array[:, i]for i in range(3))
    elif array.ndim == 1:
        assert array.shape[0] == 3
        x, y, z = (array[i] for i in range(3))
        
    theta = np.arcsin(-y)
    phi = np.arctan2(-x, -z)
    
    if not stack:
        return theta, phi
    elif stack:
        return np.stack((theta, phi)).T
    
    
def gaze2Dto3D(array, z=-1.0):
    # TODO implement gaze 2D to 3D
    theta = array[:, 0]
    phi = array[:, 1]
    
    pass


def pose3Dto2D(array):
    """
    M = Rodrigues((x,y,z))
    Zv = (the third column of M) ???
    theta = asin(Zv[1])
    phi = atan2(Zv[0], Zv[2])
    """
    from scipy.special import legendre
    Rodrigues = legendre(3)
    M = Rodrigues(array)
    theta = np.arcsin(M[:, 1])
    phi = np.arctan2(M[:, 0], M[:, 2])
    
    return np.stack((theta, phi)).T


def pose2Dto3D(array):
    # TODO implement pose 2D to 3D
    pass


def euclidean_dist(x, y):
    return np.sum(((x - y)**2), axis=-1)
