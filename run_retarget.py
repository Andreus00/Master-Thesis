#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import hydra.conf
import smplx.lbs
import torch
import gc
import pickle

import sys

import smplx

sys.path.append("./ext/gaussian-splatting/")
sys.path.append("./NICP/")
sys.path.append("./NICP/data/")
sys.path.append("./NICP/data/preprocess_voxels/")

from gaussian_renderer import render
from scene_4ddress import IsotropicGaussianModel, GaussianModel, GaussianAvatar
from gaussian_utils import GaussianTransformUtils
from argparse import ArgumentParser
from arguments import PipelineParams

from NICP.utils_cop.prior import MaxMixturePrior
from NICP.utils_cop.SMPL import SMPL

import matplotlib.pyplot as plt
import numpy as np

import trimesh

from utils.sh_utils import RGB2SH

from PIL import Image

from swap_utils import *

from lvd_templ.data.datamodule_AMASS import MetaData

import hydra
import glob

import pytorch_lightning as pl
from nn_core.serialization import NNCheckpointIO

from utils.general_utils import build_scaling_rotation, strip_symmetric

from lvd_templ.evaluation.utils import vox_scan, fit_LVD, selfsup_ref, SMPL_fitting, fit_cham, fit_plus_D


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def remove_samples_from_gaussians(gaussians, mask):
    with torch.no_grad():
        gaussians._xyz = gaussians._xyz[mask]
        gaussians._features_dc = gaussians._features_dc[mask]
        gaussians._features_rest = gaussians._features_rest[mask]
        gaussians._scaling = gaussians._scaling[mask]
        gaussians._rotation = gaussians._rotation[mask]
        gaussians._opacity = gaussians._opacity[mask]

## Function to load checkpoint
def get_model(chk):
    # Recovering the Path to the checkpoint
    chk_zip = glob.glob(os.path.join(chk, 'checkpoints/*.zip'))[0]
    
    # Restoring the network configurations using the Hydra Settings
    tmp = hydra.core.global_hydra.GlobalHydra.instance().clear()
    tmp = hydra.initialize(config_path=str(chk))
    cfg_model = hydra.compose(config_name="config")
    
    # Recovering the metadata
    train_data = hydra.utils.instantiate(cfg_model.nn.data.datasets.train, mode="test")
    MD = MetaData(class_vocab=train_data.class_vocab)
    
    # Instantiating the correct nentwork
    model: pl.LightningModule = hydra.utils.instantiate(cfg_model.nn.module, _recursive_=False, metadata=MD)
    
    # Restoring the old checkpoint
    old_checkpoint = NNCheckpointIO.load(path=chk_zip)
    module = model._load_model_state(checkpoint=old_checkpoint, metadata=MD).to(device)
    module.model.eval()
    
    return module, MD, train_data, cfg_model

def load_model():
    origin, xaxis = [0, 0, 0], [1, 0, 0]
    alpha = np.pi/2
        
    Rx = trimesh.transformations.rotation_matrix(alpha, xaxis, origin)
    
    ### Get SMPL model
    os.chdir('NICP')
    SMPL_model = SMPL('neutral_smpl_with_cocoplus_reg.txt', obj_saveable = True).cuda()
    prior = MaxMixturePrior(prior_folder='utils_cop/prior/', num_gaussians=8) 
    prior.to(device)
    
    #### Restore Model
    model_name = '1ljjfnbx'
    chk = os.path.join('storage', model_name)
    module, MD, train_data, cfg_model = get_model(chk)
    module.cuda()
    os.chdir('..')

    ret_dict = {
        "SMPL_model": SMPL_model,
        "prior": prior,
        "module": module,
        "MD": MD,
        "train_data": train_data,
        "cfg_model": cfg_model,
        "Rx": Rx,
        "chk": chk
    }

    return ret_dict


def register_shapes(scan_src, scan_name, out_dir, cfg, SMPL_model, prior, module, MD, train_data, cfg_model, Rx=None, **kwargs):
    ### REGISTRATIONS FOR ALL THE INPUT SHAPES
    print('--------------------------------------------')
    out_cham_s, smpld_vertices = None, None

    inv_Rx = np.linalg.inv(Rx)

    # Basic Name --> You can add "tag" if you want to differentiate the runs
    out_name = 'out' + cfg['core'].tag
    
    ### Get Resolution and GT_IDXS of the experiment
    res = MD.class_vocab()['occ_res']
    gt_points = MD.class_vocab()['gt_points']
    gt_idxs = train_data.idxs
    type = cfg_model['nn']['data']['datasets']['type']
    grad = cfg_model['nn']['module']['grad']

    # Scans name format
    if(cfg['core'].challenge != 'demo'):
        name = os.path.basename(os.path.dirname(scan_name))
    else:
        name = os.path.basename(scan_name)

    # Canonicalize the input point cloud and prepare input of IF-NET
    if Rx is not None:
        scan_src.apply_transform(Rx)
    
    voxel_src, mesh_src, (centers, total_size) = vox_scan(scan_src, res, style=type, grad=grad)
    
    # Save algined mesh
    if not(os.path.exists(out_dir)):
        os.mkdir(out_dir)
    k = mesh_src.export(out_dir + '/aligned.ply')
    
    
    # IF N-ICP is requested, run it
    if cfg['core'].ss_ref:
        # We add a name to specify the NF-ICP is performed
        out_name = out_name + '_ss'
        module.train()
        selfsup_ref(module, torch.tensor(np.asarray(scan_src.vertices)), voxel_src, gt_points, steps=cfg['core'].steps_ss, lr_opt=cfg['core'].lr_ss)
        module.eval()

    # You can initialize LVD in different points in space. Default is at the origin
    if cfg['core'].init:
        picker = np.int32(np.random.uniform(0,len(mesh_src.vertices),gt_points))
        init = torch.unsqueeze(torch.tensor(np.asarray(mesh_src.vertices[picker]),dtype=torch.float32),0)
    else:
        init = torch.zeros(1, gt_points, 3).cuda()
    
    # Fit LVD
    reg_src =  fit_LVD(module, gt_points, voxel_src, iters=cfg['lvd'].iters, init=init)
        
    # FIT SMPL Model to the LVD Prediction
    out_s, params = SMPL_fitting(SMPL_model, reg_src, gt_idxs, prior, iterations=2000)
    params_np = {}
    for p in params.keys():
        params_np[p] = params[p].detach().cpu().numpy()
        
    # Save intermidiate output 
    # NOTE: You may want to remove this if you are interested only
    # in the final registration
    T = trimesh.Trimesh(vertices = out_s, faces = SMPL_model.faces) 
    T.export(out_dir + '/' + out_name + '.ply')   #SHREC: 85
    np.save(out_dir + '/loss_' + out_name + '.npy',params_np)
    
    # SMPL Refinement with Chamfer            
    if cfg['core'].cham_ref:
        # Mark the registration as Chamfer Refined
        out_name = out_name + '_cham_' + str(cfg['core'].cham_bidir)
        
        # CHAMFER REGISTRATION
        # cham_bidir = 0  -> Full and clean input
        # cham_bidir = 1  -> Partial input
        # cham_bidir = -1 -> Noise input
        out_cham_s, params = fit_cham(SMPL_model, out_s, mesh_src.vertices, prior,params,cfg['core'].cham_bidir)
        
        # Save Output
        T = trimesh.Trimesh(vertices = out_cham_s, faces = SMPL_model.faces)
        T.apply_scale(total_size)
        T.apply_translation(centers)
        # T.apply_transform(inv_Rx)
        T.export(out_dir + '/' + out_name + '.ply')
        
        # save smpl parameters
        np.save(out_dir + '/beta_' + out_name + '.npy', params['beta'].detach().cpu().numpy())
        np.save(out_dir + '/pose_' + out_name + '.npy', params['pose'].detach().cpu().numpy())
        np.save(out_dir + '/trans_' + out_name + '.npy', params['trans'].detach().cpu().numpy())
        np.save(out_dir + '/scale_' + out_name + '.npy', params['scale'].detach().cpu().numpy())
        
        
        # Update the name
        out_s = out_cham_s
    
    # SMPL Refinement with +D
    if cfg['core'].plusD:
        smpld_vertices, faces, params = fit_plus_D(out_s, SMPL_model, mesh_src.vertices, subdiv= 1, iterations=300)
        T = trimesh.Trimesh(vertices = smpld_vertices, faces = faces)
        T.apply_scale(total_size)
        T.apply_translation(centers)
        out_name_grid = out_name + '_+D'
        T.export(out_dir + '/' + out_name_grid + '.ply')
    gc.collect()


    return centers, total_size, inv_Rx


def gs_to_trimesh(gs):
    xyz = gs._xyz.detach().cpu().numpy()

    faces = np.arange(0, (len(xyz))).reshape(-1, 1)
    faces = np.hstack([faces, faces, faces])
    
    mesh = trimesh.Trimesh(vertices=xyz, faces=faces, face_normals=np.ones_like(xyz), process=False)
    
    return mesh


def load_smpl_data(base_path):
    smpl_mesh = trimesh.load(os.path.join(base_path, "out_ss_cham_0.ply"))
    smpl_betas = np.load(os.path.join(base_path, "beta_out_ss_cham_0.npy"))
    smpl_pose = np.load(os.path.join(base_path, "pose_out_ss_cham_0.npy"))
    smpl_scale = np.load(os.path.join(base_path, "scale_out_ss_cham_0.npy"))
    smpl_trans = np.load(os.path.join(base_path, "trans_out_ss_cham_0.npy"))
    
    with torch.no_grad():
        from NICP.utils_cop.SMPL import SMPL
        smpl_model = SMPL(
            "utils_cop/shapedirs_300.npy", 
            num_betas=10,
            base_path="NICP",
            ).forward(betas=torch.tensor(smpl_betas, dtype=torch.float32),
              body_pose=torch.tensor(smpl_pose, dtype=torch.float32),
              global_orient=torch.tensor([0, 0, 0], dtype=torch.float32),
              transl=torch.tensor(smpl_trans, dtype=torch.float32),
            )
    
    return smpl_mesh, smpl_model

def load_smpl_data_from_nicp_registration(path):
    smpl_mesh = trimesh.load(os.path.join(path, "out_ss_cham_0.ply"))
    smpl_betas = np.load(os.path.join(path, "beta_out_ss_cham_0.npy"))
    smpl_pose = np.load(os.path.join(path, "pose_out_ss_cham_0.npy"))
    smpl_scale = np.load(os.path.join(path, "scale_out_ss_cham_0.npy"))
    smpl_trans = np.load(os.path.join(path, "trans_out_ss_cham_0.npy"))
    
    return {
        "mesh": smpl_mesh,
        "betas": smpl_betas,
        "body_pose": smpl_pose,
        "scale": smpl_scale,
        "transl": smpl_trans
    }
    
def load_smpl_data_from_registration(path, use_base_smpl=False):
    
    mesh_pkl = load_pickle(path)
    mesh_pkl["betas"] = torch.asarray(mesh_pkl["betas"]).detach().unsqueeze(0) if not use_base_smpl else torch.zeros(1, 10)
    mesh_pkl["body_pose"] = torch.asarray(mesh_pkl["body_pose"]).detach().unsqueeze(0)
    mesh_pkl["transl"] = torch.asarray(mesh_pkl["transl"]).detach().unsqueeze(0)
    mesh_pkl["global_orient"] = torch.asarray(mesh_pkl["global_orient"]).detach().unsqueeze(0)
    
    smplx_model = smplx.SMPL("smpl/smpl300/SMPL_MALE.pkl", num_betas=10)
    with torch.no_grad():
        smpl_vertices = smplx_model.forward(betas=mesh_pkl["betas"], body_pose=mesh_pkl["body_pose"], global_orient=mesh_pkl["global_orient"]).vertices[0]
    
    # create a trimesh object from the SMPL output
    triangles = smpl_vertices[smplx_model.faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1)[:, None]
    vertex_normals = np.zeros_like(smpl_vertices)
    for i, face in enumerate(smplx_model.faces):
        vertex_normals[face] += face_normals[i]
    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=1)[:, None]
    smpl_mesh = trimesh.Trimesh(vertices=smpl_vertices, faces=smplx_model.faces, vertex_normals=vertex_normals, process=False)
    return smpl_mesh, mesh_pkl
    

def adjust_gaussians_covariance(gaussians):
    '''
    Given an isotropic gaussian model, this function transforms it into an anisotropic gaussian model
    and adjusts the covariance matrix of each gaussian in order to fix holes.
    '''
    anisotropic_gaussians = GaussianModel(3)
    anisotropic_gaussians.from_gaussians(gaussians)
    
    xyz = gaussians._xyz.detach().cpu().numpy()
    kdtree = scipy.spatial.cKDTree(xyz)
    
    # Find neighbors
    neigh_dist, neigh_idxs = kdtree.query(xyz, k=5)  # k=5 to include the point itself
    
    # Get neighbor positions
    neigh_pos = xyz[neigh_idxs[:, 1:]]  # exclude the point itself
    
    print(neigh_pos[:10])
    
    # Calculate deviations from the mean position
    deviations = neigh_pos - xyz[:, None]
    print(deviations[:10])
    # weights = neigh_dist[:, 1:] / np.sum(neigh_dist[:, 1:], axis=1)[:, None]
    # Calculate the covariance matrices
    cov_matrices = np.einsum('...ij,...ik->...ijk', deviations, deviations)
    
    # perform mean to get the covariance matrix
    cov_matrices = np.mean(cov_matrices, axis=1)
    # cov_matrices = np.mean(cov_matrices * weights[:, :, None, None], axis=1)
    
    cov_squared = cov_matrices @ cov_matrices
    
    # Perform eigen decomposition to get the scaling (eigenvalues) and rotation (eigenvectors)
    U, Sigma, VT = np.linalg.svd(cov_matrices, hermitian=True)
    
    print(U.shape, Sigma.shape, VT.shape)
    rotation_matrices = U @ VT
    
    
    scale = np.sqrt(Sigma)
    
    # permute xyz of eigenvalues to zxy
    # eigenvalues = eigenvalues[:, [2, 0, 1]]
    
    rotation_matrices = scipy.spatial.transform.Rotation.from_matrix(rotation_matrices).as_matrix()
    
    # check if all the matrices are eyes
    # Ensure eigenvalues are not too small
    # min_val = 1e-3
    # eigenvalues = torch.asarray([np.diag(np.sqrt(x)) for x in eigenvalues])
    # eigenvalues = np.maximum(eigenvalues, min_val)
    
    # Convert rotations to quaternions
    quaternions = torch.asarray(scipy.spatial.transform.Rotation.from_matrix(rotation_matrices).as_quat()).cuda().float()

    with torch.no_grad():
        anisotropic_gaussians._scaling = anisotropic_gaussians.scaling_inverse_activation(torch.tensor(scale, dtype=torch.float, device="cuda"))
        
        # anisotropic_gaussians._rotation = torch.tensor(quaternions, dtype=torch.float32, device="cuda")
        anisotropic_gaussians._rotation = torch.nn.functional.normalize(quaternions)
        
        # vectorized_rotate_gaussians(anisotropic_gaussians, torch.tensor(rotation_matrices, dtype=torch.float32, device="cuda"), mask=torch.ones(len(anisotropic_gaussians._xyz), dtype=torch.bool, device="cuda"))
    
    return anisotropic_gaussians

def simple_adjust_gaussians_covariance(gaussians: IsotropicGaussianModel):
    '''
    Given an isotropic gaussian model, this function transforms it into an anisotropic gaussian model
    and adjusts the covariance matrix of each gaussian in order to fix holes.
    
    The adjacency matrix is used to determine which gaussians are neighbors.
    It is an NxN matrix where N is the number of gaussians in the model.
    The element (i, j) is 1 if the i-th gaussian is a neighbor of the j-th gaussian.
    '''
    anisotropic_gaussians = GaussianModel(3)
    anisotropic_gaussians.from_gaussians(gaussians)
    
    # Algorithm:
    # 1. find the neighbors of each gaussian
    # 2. calculate the maximum x, y and z distances between the neighbors
    # 3. set the scaling of the gaussian to the mean distances
    
    xyz = gaussians._xyz.detach().cpu().numpy()
    kdtree = scipy.spatial.cKDTree(xyz)
    
    # find neighbors
    NUM_NEIGHBORS = 10
    neigh_dist, neigh_idxs = kdtree.query(xyz, k=NUM_NEIGHBORS)
    
    neigh_pos = xyz[neigh_idxs[:, 1:]]
    
    mean_dist = np.mean(neigh_dist[:, 1:], axis=1)
    
    min_dist = 1e-4
    mean_dist = np.maximum(mean_dist, min_dist)
    print(mean_dist)
    
       
    with torch.no_grad():
        anisotropic_gaussians._scaling = anisotropic_gaussians.scaling_inverse_activation(torch.tensor(mean_dist, dtype=torch.float32, device="cuda"))
        # anisotropic_gaussians._rotation = torch.tensor(quaternions, dtype=torch.float32, device="cuda")
    
    return anisotropic_gaussians
    

def run_retarget(cfg, pipe):
    RUN_NICP = True
    LOAD_NICP = True
    LOAD_DATA_REG = False
    LOAD_DATA_REG_TARGET = False

    source_labels = os.path.join("4D-DRESS", cfg.src_user, cfg.src_outfit, cfg.src_take, "Semantic", "labels_with_faces", f"labels_with_faces_{cfg.src_sample}.pkl") 
    target_labels = os.path.join("4D-DRESS", cfg.trg_user, cfg.trg_outfit, cfg.trg_take, "Semantic", "labels_with_faces", f"labels_with_faces_{cfg.trg_sample}.pkl")
    source_scan_labels = load_pickle(source_labels)["scan_labels"]
    target_scan_labels = load_pickle(target_labels)["scan_labels"]


    source_clothes = source_scan_labels == 3
    target_clothes = target_scan_labels != 3

    source_gaussians = IsotropicGaussianModel(3)
    ply_path = os.path.join("output", cfg.source_model, "point_cloud", "iteration_3000", "point_cloud.ply")
    source_gaussians.load_ply(ply_path)
    
    target_gaussians = IsotropicGaussianModel(3)
    ply_path = os.path.join("output", cfg.target_model, "point_cloud", "iteration_3000", "point_cloud.ply")
    target_gaussians.load_ply(ply_path)
    
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    camera_back = create_camera(azimuth=0)
    camera_left = create_camera(azimuth=90)
    camera_front = create_camera(azimuth=180)
    camera_right = create_camera(azimuth=270)

    bg = background

    if not LOAD_DATA_REG and not LOAD_DATA_REG_TARGET:
        rotate_gaussians_along_y(source_gaussians, 90)
        rotate_gaussians_along_y(target_gaussians, 90)
    elif LOAD_DATA_REG_TARGET:
        rotate_gaussians_along_y(source_gaussians, 90)
        
    # rotate_gaussians_along_y(target_gaussians, -160)
    

    source_trimesh = gs_to_trimesh(source_gaussians)
    target_trimesh = gs_to_trimesh(target_gaussians)

    if RUN_NICP:
        # Read the configuration file
        os.chdir('NICP')
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(config_path="conf_test")
        pipe_cfg = hydra.compose(config_name="default")
        os.chdir('..')
        
        model_pkg = load_model()
        
        src_nicp_path = os.path.join("output", cfg.source_model, "NICP")
        if not os.path.exists(src_nicp_path):
            os.makedirs(src_nicp_path)
        
        trg_nicp_path = os.path.join("output", cfg.target_model, "NICP")
        if not os.path.exists(trg_nicp_path):
            os.makedirs(trg_nicp_path)

        if not cfg.only_target_nicp:
            centers, total_size, inv_Rx = register_shapes(source_trimesh, cfg.source_model, src_nicp_path, pipe_cfg, **model_pkg)
            source_trimesh.apply_scale(total_size)
            source_trimesh.apply_translation(centers)
            source_trimesh.apply_transform(inv_Rx)

        centers, total_size, inv_Rx = register_shapes(target_trimesh, cfg.target_model, trg_nicp_path, pipe_cfg, **model_pkg)
        target_trimesh.apply_scale(total_size)
        target_trimesh.apply_translation(centers)
        target_trimesh.apply_transform(inv_Rx)

        source_smpl_pkl = load_smpl_data_from_nicp_registration(os.path.join("output", cfg.source_model, "NICP"))
        source_smpl = source_smpl_pkl.pop("mesh")
        target_smpl_pkl = load_smpl_data_from_nicp_registration(os.path.join("output", cfg.target_model, "NICP"))
        target_smpl = target_smpl_pkl.pop("mesh")
    elif LOAD_NICP:
        source_smpl_pkl = load_smpl_data_from_nicp_registration(os.path.join("output", cfg.source_model, "NICP"))
        source_smpl = source_smpl_pkl.pop("mesh")
        target_smpl_pkl = load_smpl_data_from_nicp_registration(os.path.join("output", cfg.target_model, "NICP"))
        target_smpl = target_smpl_pkl.pop("mesh")
        # source_smpl = trimesh.load(os.path.join("output", cfg.source_model, "NICP", "out_ss_cham_0.ply"))    # smpl fitting
        # target_smpl = trimesh.load(os.path.join("output", cfg.target_model, "NICP", "out_ss_cham_0.ply"))    # smpl fitting
    elif LOAD_DATA_REG:
        # THIS ONLY WORKS WITH SOURCE 11 AND TARGET 100 OF 00122/Inner/Take2
        # load NICP for source, load registration pose for target
        src_mesh_pkl_path = os.path.join(cfg.dataset_path, cfg.src_user, cfg.src_outfit, cfg.src_take, "SMPL", f"mesh-{cfg.src_sample}_smpl.pkl")
        source_smpl, source_smpl_pkl = load_smpl_data_from_registration(src_mesh_pkl_path)
        # source_smpl.vertices -= np.mean(source_smpl.vertices, axis=0)
        # source_smpl.vertices[:, 0] += 0.02
        # source_smpl.vertices[:, 1] += 0.05
        # with torch.no_grad():
        #     source_gaussians._xyz -= torch.mean(source_gaussians._xyz, dim=0)
        trg_mesh_pkl_path = os.path.join(cfg.dataset_path, cfg.trg_user, cfg.trg_outfit, cfg.trg_take, "SMPL", f"mesh-{cfg.trg_sample}_smpl.pkl")
        target_smpl, target_smpl_pkl = load_smpl_data_from_registration(trg_mesh_pkl_path)
        # target_smpl.vertices -= np.mean(target_smpl.vertices, axis=0)
        # target_smpl.vertices[:, 1] += 0.06
        # with torch.no_grad():
        #     target_gaussians._xyz -= torch.mean(target_gaussians._xyz, dim=0)
        # exit()
        # # move every gaussian to the closest point on the smpl target model
        # with torch.no_grad():
        #     dist, idx = target_smpl.kdtree.query(target_gaussians._xyz.detach().cpu())
        #     target_gaussians._xyz[i] = torch.asarray(target_smpl.vertices[idx])


        # import matplotlib.pyplot as pltù
        # fig = plt.figure()
        
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(source_smpl.vertices[:, 0], source_smpl.vertices[:, 1], source_smpl.vertices[:, 2])
        # ax.scatter(source_gaussians._xyz[:, 0].detach().cpu(), source_gaussians._xyz[:, 1].detach().cpu(), source_gaussians._xyz[:, 2].detach().cpu())
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(-1, 1)
        # plt.show()
        # plt.cla()
        # exit()
        
    elif LOAD_DATA_REG_TARGET:
        # Here I use the NICP for the source model and the dataset registration pose for the target model
        source_smpl_pkl = load_smpl_data_from_nicp_registration(os.path.join("output", cfg.source_model, "NICP"))
        source_smpl = source_smpl_pkl.pop("mesh")
        
        trg_mesh_pkl_path = os.path.join(cfg.dataset_path, cfg.trg_user, cfg.trg_outfit, cfg.trg_take, "SMPL", f"mesh-{cfg.trg_sample}_smpl.pkl")
        target_smpl, target_smpl_pkl = load_smpl_data_from_registration(trg_mesh_pkl_path, use_base_smpl=True)
    else:
        raise ValueError("No registration method selected")
    

    # use pyrender render the source smpl model and the source trimesh model
    import pyrender

    source_smpl_mesh = pyrender.Mesh.from_trimesh(source_smpl)
    source_gs_mesh = pyrender.Mesh.from_points(source_trimesh.vertices)

    target_smpl_mesh = pyrender.Mesh.from_trimesh(target_smpl)
    target_gs_mesh = pyrender.Mesh.from_points(target_trimesh.vertices)

    SHOW_SMPL_FIT_PREVIEW = True
    if SHOW_SMPL_FIT_PREVIEW:
        scene = pyrender.Scene()

        el1 = scene.add(target_smpl_mesh)
        el2 = scene.add(target_gs_mesh)

        pyrender.Viewer(scene, use_raymond_lighting=True)

        scene.remove_node(el1)
        scene.remove_node(el2)

        el1 = scene.add(source_smpl_mesh)
        el2 = scene.add(source_gs_mesh)

        pyrender.Viewer(scene, use_raymond_lighting=True)

        scene.remove_node(el1)
        scene.remove_node(el2)

    RUN_RETARGET = 2    # choose between different retargeting methods
    if RUN_RETARGET == 1:
        
        knn_num = 11

        # create a KDTree for source
        source_kdtree = scipy.spatial.cKDTree(source_smpl.vertices)

        # 1. Mapping between the clothes (t') and the chamfer-ss (t)
        source_clothes_xyz = source_trimesh.vertices[source_clothes]
        smpl_dist_for_clothes, smpl_idx_for_clothes = source_kdtree.query(source_clothes_xyz, k=knn_num)

        # # 2. Calculate vector between the source clothes and the chamfer-ss
        # offsets_vector = source_clothes_xyz - source_smpl.vertices[smpl_idx_for_clothes]
        
        if knn_num == 1:
            smpl_idx_for_clothes = smpl_idx_for_clothes.reshape(-1, 1)
            smpl_dist_for_clothes = smpl_dist_for_clothes.reshape(-1, 1)

        smpl_dist_for_clothes = np.repeat(smpl_dist_for_clothes[:, :, np.newaxis], 3, axis=2)
        avg_src_verts = np.average(source_smpl.vertices[smpl_idx_for_clothes], weights=smpl_dist_for_clothes, axis=1)
        avg_trg_verts = np.average(target_smpl.vertices[smpl_idx_for_clothes], weights=smpl_dist_for_clothes, axis=1)
        avg_src_normals = np.average(source_smpl.vertex_normals[smpl_idx_for_clothes], weights=smpl_dist_for_clothes, axis=1)
        avg_trg_normals = np.average(target_smpl.vertex_normals[smpl_idx_for_clothes], weights=smpl_dist_for_clothes, axis=1)

        offsets_vector = source_clothes_xyz - avg_src_verts
        

        base_vector = avg_trg_verts

        # 2.1 Calculate the rotation matrix for each gaussian between the two smpl models
        angles_between_vectors = np.arccos(np.einsum("ij, ij -> i", avg_src_normals, avg_trg_normals))
        axis = np.cross(avg_src_normals, avg_trg_normals, axisa=1, axisb=1)
        rot_mat = torch.asarray(scipy.spatial.transform.Rotation.from_rotvec(np.einsum("i, ij -> ij", angles_between_vectors, axis)).as_matrix())


        # 3. Create new gaussians for the target clothes, in all those points where the source clothes are
        transfered_clothes_xyz = torch.asarray(base_vector)
        transfered_clothes_features_dc = source_gaussians._features_dc[source_clothes].clone()
        transfered_clothes_features_rest = source_gaussians._features_rest[source_clothes].clone()
        transfered_clothes_scaling = source_gaussians._scaling[source_clothes].clone()
        transfered_clothes_opacity = source_gaussians._opacity[source_clothes].clone()
        transfered_clothes_rotation = source_gaussians._rotation[source_clothes].clone()


        # 4. Add offset to the clothes
        transfered_clothes_xyz += offsets_vector

        # 5. Remove the points close to the body from the target model
        target_clothes = target_scan_labels != 3

        if LOAD_DATA_REG_TARGET:
            # remove all the gaussians from the target model and add SMPL gaussinas
            remove_samples_from_gaussians(target_gaussians, torch.ones(target_clothes.shape, dtype=torch.bool))
            def simple_mesh_to_pcd(mesh: trimesh.Trimesh):

                res = {}
                vertices = np.asarray(mesh.vertices).reshape(-1, 3)
                normals = np.asarray(mesh.vertex_normals).reshape(-1, 3)
                scales = np.ones((vertices.shape[0], 1)) * -7.0
                opacity = np.ones((vertices.shape[0], 1))
                colors = torch.ones((vertices.shape[0], 3), dtype=torch.float32)
                
                res["scales"] = scales
                res["opacity"] = opacity
                from utils.graphics_utils import BasicPointCloud
                res["pointcloud"] = BasicPointCloud(points=vertices, colors=colors, normals=normals)
                return res
            smpl_pointcloud_data = simple_mesh_to_pcd(target_smpl)
            target_gaussians.create_from_pcd(smpl_pointcloud_data["pointcloud"], 1, smpl_pointcloud_data["scales"], smpl_pointcloud_data["opacity"])
        else:
            remove_samples_from_gaussians(target_gaussians, target_clothes)

        # add the new clothes to the target model
        target_gaussians._xyz = torch.cat([target_gaussians._xyz, transfered_clothes_xyz.detach().float().to(device)], dim=0)
        target_gaussians._features_dc = torch.cat([target_gaussians._features_dc, transfered_clothes_features_dc.detach().float().to(device)], dim=0)
        target_gaussians._features_rest = torch.cat([target_gaussians._features_rest, transfered_clothes_features_rest.detach().float().to(device)], dim=0)
        target_gaussians._scaling = torch.cat([target_gaussians._scaling, transfered_clothes_scaling.detach().float().to(device)], dim=0)
        target_gaussians._opacity = torch.cat([target_gaussians._opacity, transfered_clothes_opacity.detach().float().to(device)], dim=0)
        target_gaussians._rotation = torch.cat([target_gaussians._rotation, transfered_clothes_rotation.detach().float().to(device)], dim=0)
        
        # create a mask for the last added clothes
        mask = torch.zeros(len(target_gaussians._xyz), dtype=torch.bool, device="cuda")
        mask[-len(transfered_clothes_xyz):] = True
        # rotate clothes
        vectorized_rotate_gaussians(target_gaussians, rot_mat, mask)
        
        # target_gaussians = source_gaussians
        # TODO: pass neigbors based on mesh faces?
        source_gaussians = adjust_gaussians_covariance(source_gaussians)
        target_gaussians = adjust_gaussians_covariance(target_gaussians)
    elif RUN_RETARGET == 2:
        # This method uses the GaussianAvatar to store and manipulate gaussians
        # based on smpl and LBS
        
        # 1. Create a GaussianAvatar for the source model and the target model
        source_avatar = GaussianAvatar(3, source_gaussians, source_smpl, source_smpl_pkl, source_scan_labels, RUN_NICP or LOAD_NICP)
        source_avatar.inverse_lbs()
        target_avatar = GaussianAvatar(3, target_gaussians, target_smpl, target_smpl_pkl, target_scan_labels, RUN_NICP or LOAD_NICP)
        target_avatar.inverse_lbs()
        
        # 2. Call the transfer_clothes function to transfer the clothes from the source to the target
        # source_avatar.transfer_clothes(target_avatar)
        
    
    # render the target model
    for camera in [camera_front, camera_left, camera_right, camera_back]:
        with torch.no_grad():
            target_pkg = render(camera, target_avatar, pipe, bg)
            target_image = target_pkg["render"]
            
            source_pkg = render(camera, source_avatar, pipe, bg)
            source_image = source_pkg["render"]
            
        
        target_image = target_image.cpu().numpy().transpose(1, 2, 0)
        source_image = source_image.cpu().numpy().transpose(1, 2, 0)
        cat_images = np.concatenate([source_image, target_image], axis=1)
        plt.imshow(cat_images)
        plt.show()
        if cfg.save_output:
            '''
            Save the output under the folder of the target
            '''
            if not os.path.exists(os.path.join("output", cfg.target_model, "render")):
                os.makedirs(os.path.join("output", cfg.target_model, "render"))
            plt.imsave(os.path.join("output", cfg.target_model, "render", f"render-{camera.image_name}.png"), cat_images)

from arguments import ParamGroup
class ExperimentParams(ParamGroup):
    def __init__(self, parser):
        self.source_model = ""  # path to the Gaussian Splats of the source model
        self.target_model = ""  # path to the Gaussian Splats of the target model
        self.dataset_path = "4D-DRESS"  # path to the dataset
        self.src_user = "00122" # user of the source model
        self.src_outfit = "Inner"   # outfit of the source model
        self.src_take = "Take2" # take of the source model
        self.src_sample = ""    # sample of the source model
        self.trg_user = "00122" # user of the target model
        self.trg_outfit = "Inner"   # outfit of the target model
        self.trg_take = "Take2" # take of the target model
        self.trg_sample = ""    # sample of the target model
        
        self.save_output = True
        self.only_target_nicp = False
        super().__init__(parser, "Pipeline Parameters")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    pp = PipelineParams(parser)
    ep = ExperimentParams(parser)
    args = parser.parse_args(sys.argv[1:])
    
    print("Source model: " + args.source_model)
    print("Target model: " + args.target_model)

    run_retarget(ep.extract(args), pp.extract(args))

    # All done
    print("\nTraining complete.")
