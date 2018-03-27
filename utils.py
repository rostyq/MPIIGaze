import numpy as np
from scipy.io import loadmat
import glob
from cv2 import Rodrigues
from tqdm import tqdm

def gather_eye_data(path, eye='right'):
    
    mat_files = glob.glob(f'{path}/**/*.mat', recursive=True)
    mat_files.sort()
    
    indices = []
    images = []
    poses = []
    gazes = []
    for file in tqdm(mat_files):
        matfile = loadmat(file)
        
        file_idx = file.split('/')[-2], file.split('/')[-1].split('.')[0]
        
        indices.extend([[*file_idx, jpg[0][0], eye] for jpg in matfile['filenames']])
        images.extend(matfile['data'][eye][0, 0]['image'][0, 0])
        poses.extend(matfile['data'][eye][0, 0]['pose'][0, 0])
        gazes.extend(matfile['data'][eye][0, 0]['gaze'][0, 0])
    
    indices = np.array(indices)
    images = np.array(images).reshape((-1, 36, 60, 1))
    poses = np.array(poses)
    gazes = np.array(gazes)
    
    return indices, images, poses, gazes


def gather_all_data(path):
    
    mat_files = glob.glob(f'{path}/**/*.mat', recursive=True)
    mat_files.sort()
    
    index = dict(left=list(), right=list())
    image = dict(left=list(), right=list())
    pose = dict(left=list(), right=list())
    gaze = dict(left=list(), right=list())
    
    for file in tqdm(mat_files):
        # read file
        matfile = loadmat(file)
        
        # file name
        file_idx = file.split('/')[-2], file.split('/')[-1].split('.')[0]
        for eye in ['left', 'right']:
            index[eye].extend([[*file_idx, jpg[0][0], eye] for jpg in matfile['filenames']])
            image[eye].extend(matfile['data'][eye][0, 0]['image'][0, 0])
            pose[eye].extend(matfile['data'][eye][0, 0]['pose'][0, 0])
            gaze[eye].extend(matfile['data'][eye][0, 0]['gaze'][0, 0])
    
    index = np.stack(tuple(index.values())).reshape((-1, 4))
    image = np.stack(tuple(image.values())).reshape((-1, 36, 60, 1))
    pose = np.stack(tuple(pose.values())).reshape((-1, 3))
    gaze = np.stack(tuple(gaze.values())).reshape((-1, 3))
    return index, image, pose, gaze


def stack_eyes_data(left_array, right_array):
    return np.stack((left_array, right_array)).T.flatten()


def gather_data(path):
    data = (gather_eye_data(path, eye=eye) for eye in ['right', 'left'])
    for left, right in zip(data[0], data[1]):
        stack_eyes_data(left, right)

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
    def convert_pose(vect):
        M, _ = Rodrigues(np.array(vect).astype(np.float32))
        Zv = M[:, 2]
        theta = np.arctan2(Zv[0], Zv[2])
        phi = np.arcsin(Zv[1])
        return np.array([theta, phi])
    
    return np.apply_along_axis(convert_pose, 1, array)



def pose2Dto3D(array):
    # TODO implement pose 2D to 3D
    pass


def euclidean_dist(x, y):
    return np.sum(((x - y)**2), axis=-1)
