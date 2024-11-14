
import os
import scipy.spatial
import torch
from random import randint

import sys
sys.path.append("./ext/gaussian-splatting/")
import numpy as np

from scene_4ddress.dataset_readers import orbit_camera, focal2fov, fov2focal

from scene_4ddress.cameras import Camera

import pickle

from utils.sh_utils import RGB2SH

import scipy

from utils.general_utils import build_scaling_rotation, strip_symmetric

import smplx



def apply_labels_to_colors(labels, gaussians):

    with torch.no_grad():
        labels_to_colors = {
            1: [1.0, 0.0, 0.0],
            2: [0.0, 1.0, 0.0],
            3: [0.0, 0.0, 1.0],
            4: [1.0, 1.0, 0.0],
            5: [1.0, 0.0, 1.0],
            6: [0.0, 1.0, 1.0],
            7: [0.5, 0.5, 0.5],
        }

        def get_features(label):
            _color = torch.tensor(labels_to_colors[label], dtype=torch.float32).reshape(1, 3, 1)
            fused_color = RGB2SH(torch.tensor(np.asarray(_color)).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3, (3 + 1) ** 2)).float().cuda()
            features[:, :3, :] = torch.repeat_interleave(fused_color, (3 + 1) ** 2, dim=2)

            _features_dc = features[:,:,0:1].transpose(1, 2)
            _features_rest = features[:,:,1:].transpose(1, 2)
            return _features_dc, _features_rest

        for idx, scan_label in enumerate(labels):
            if scan_label == 0:
                continue

            features_dc, features_rest = get_features(scan_label)

            gaussians._features_dc[idx] = features_dc
            gaussians._features_rest[idx] = features_rest           
        
    return gaussians


def create_camera(width=940, height=1280, elevation=0, azimuth=0, radius=7.0):
    
    frontal_camera_c2w = orbit_camera(elevation=elevation, azimuth=azimuth, radius=radius)

    fovx = 0.25
    
    fovy = focal2fov(fov2focal(fovx, width), height)

    frontal_camera_c2w[:3, 1:3] *= -1
    
    w2c = np.linalg.inv(frontal_camera_c2w)

    R = np.transpose(w2c[:3,:3])
    T = w2c[:3, 3]
    

    FovY = fovy 
    FovX = fovx

    image = torch.zeros((3, height, width), dtype=torch.float32, device="cuda")
    
    camera_name = "{}-camera"
    if azimuth == 0:
        camera_name = camera_name.format("frontal")
    elif azimuth == 90:
        camera_name = camera_name.format("right")
    elif azimuth == 180:
        camera_name = camera_name.format("back")
    elif azimuth == 270:
        camera_name = camera_name.format("left")
    else:
        camera_name = camera_name.format(azimuth)

    return Camera(colmap_id=camera_name, R=R, T=T, 
                  FoVx=FovX, FoVy=FovY, 
                  image=image, gt_alpha_mask=None,
                  image_name=camera_name, uid=camera_name, data_device="cuda")


# load data from pkl_dir
def load_pickle(pkl_dir):
    return pickle.load(open(pkl_dir, "rb"))


def rotate_gaussians_along_y(gaussians, y_angle, isDegree=True):
    '''
    Rotate gaussians along y axis
    '''
    from gaussian_utils import GaussianTransformUtils
    
    if isDegree:
        y_angle = np.deg2rad(y_angle)

    matrix = GaussianTransformUtils.ry(y_angle)

    rotate_gaussians(gaussians, matrix)


def rotate_gaussians(gaussians, matrix):
    '''
    Rotate gaussians along y axis
    '''
    from gaussian_utils import GaussianTransformUtils
    
    with torch.no_grad():
        features = torch.cat([gaussians._features_dc, gaussians._features_rest], dim=1)

        xyz, rotations, features = GaussianTransformUtils.rotate_by_wxyz_quaternions(
            gaussians._xyz,
            gaussians._rotation,
            features,
            rotation_matrix=matrix.cuda()
        )
        gaussians._xyz = xyz
        gaussians._rotation = rotations
        next_dc = features[:, 0, :].unsqueeze(1)
        gaussians._features_dc = next_dc
        next_rest = features[:, 1:, :]
        gaussians._features_rest = next_rest
        
def vectorized_rotate_gaussians(gaussians, matrix, mask):
    '''
    Rotate gaussians along y axis
    '''
    from gaussian_utils import GaussianTransformUtils
    
    with torch.no_grad():
        features = torch.cat([gaussians._features_dc[mask], gaussians._features_rest[mask]], dim=1)

        xyz, rotations, features = GaussianTransformUtils.vectorized_rotate_by_wxyz_quaternions(
            gaussians._xyz[mask].float(),
            gaussians._rotation[mask].float(),
            features.float(),
            rotation_matrix=matrix.cuda().float()
        )
        gaussians._xyz[mask] = xyz
        gaussians._rotation[mask] = rotations
        next_dc = features[:, 0, :].unsqueeze(1)
        gaussians._features_dc[mask] = next_dc
        next_rest = features[:, 1:, :]
        gaussians._features_rest[mask] = next_rest
        
        

def rescale_gaussians(gaussians, scale):
    '''
    Rescale gaussians
    '''
    from gaussian_utils import GaussianTransformUtils

    with torch.no_grad():
        xyz, scalings = GaussianTransformUtils.rescale(
            gaussians._xyz,
            gaussians._scaling,
            scale
        )
        gaussians._xyz = xyz
        gaussians._scaling = scalings

def translate_gaussians(gaussians, translation):
    '''
    Translate gaussians
    '''
    from gaussian_utils import GaussianTransformUtils

    x, y, z = translation

    with torch.no_grad():
        gaussians._xyz = GaussianTransformUtils.translation(gaussians._xyz, x, y, z)

def remove_samples_from_gaussians(gaussians, mask):
    with torch.no_grad():
        gaussians._xyz = gaussians._xyz[mask]
        gaussians._features_dc = gaussians._features_dc[mask]
        gaussians._features_rest = gaussians._features_rest[mask]
        gaussians._scaling = gaussians._scaling[mask]
        gaussians._rotation = gaussians._rotation[mask]
        gaussians._opacity = gaussians._opacity[mask]
    
def copy_gaussians_to_target(src_gaussians, trgt_gaussians, mask):
    with torch.no_grad():
        trgt_gaussians._xyz = torch.cat([trgt_gaussians._xyz, src_gaussians._xyz[mask]], dim=0)
        trgt_gaussians._features_dc = torch.cat([trgt_gaussians._features_dc, src_gaussians._features_dc[mask]], dim=0)
        trgt_gaussians._features_rest = torch.cat([trgt_gaussians._features_rest, src_gaussians._features_rest[mask]], dim=0)
        trgt_gaussians._scaling = torch.cat([trgt_gaussians._scaling, src_gaussians._scaling[mask]], dim=0)
        trgt_gaussians._rotation = torch.cat([trgt_gaussians._rotation, src_gaussians._rotation[mask]], dim=0)
        trgt_gaussians._opacity = torch.cat([trgt_gaussians._opacity, src_gaussians._opacity[mask]], dim=0)

    return trgt_gaussians



    