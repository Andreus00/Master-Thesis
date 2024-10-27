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
import tqdm

import sys

import smplx

sys.path.append("./ext/gaussian-splatting/")
sys.path.append("./NICP/")
sys.path.append("./NICP/data/")
sys.path.append("./NICP/data/preprocess_voxels/")

from gaussian_renderer import render
from scene_4ddress import IsotropicGaussianModel, GaussianModel, GaussianAvatar, GaussianAvatarCopy, GaussianAvatarCopy2
from gaussian_utils import GaussianTransformUtils
from argparse import ArgumentParser
from arguments import PipelineParams
from arguments_4d_dress import ExperimentParams

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

SAVE_PIPELINE_STEPS_RENDERS = True
save_pipeline_out = "pipeline_step_renderings/"

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
    SMPL_model = SMPL('neutral_smpl_with_cocoplus_reg_augmented.txt', obj_saveable = True).cuda()
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

nicp_fitting_animation = True

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
        smpld_vertices, faces, params = fit_plus_D(out_s, SMPL_model, mesh_src.vertices, subdiv=0, iterations=300)
        T = trimesh.Trimesh(vertices = smpld_vertices, faces = faces)
        T.apply_scale(total_size)
        T.apply_translation(centers)
        out_name_grid = out_name + '_+D'
        T.export(out_dir + '/' + out_name_grid + '.ply')
    gc.collect()


    return centers, total_size, inv_Rx


def gs_to_trimesh(gs, downsample_percent=1.):
    xyz = gs._xyz.detach().cpu().numpy()
    
    if downsample_percent < 1.0:
        num_downsample = int(len(xyz) * downsample_percent)
        idxs = np.random.choice(len(xyz), num_downsample, replace=False)
        xyz = xyz[idxs]
    

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
    
def load_smpl_data_from_registration(path, smplx_model, use_base_smpl=False, device="cuda"):
    
    if smplx_model is None:
        smplx_model = smplx.SMPL("smpl/smpl300/SMPL_MALE.pkl", num_betas=10)
    smplx_model.to(device)

    mesh_pkl = load_pickle(path)

    mesh_pkl["betas"] = torch.asarray(mesh_pkl["betas"]).detach().unsqueeze(0) if not use_base_smpl else torch.zeros(1, 10)
    mesh_pkl["body_pose"] = torch.asarray(mesh_pkl["body_pose"]).detach().unsqueeze(0)
    mesh_pkl["transl"] = torch.asarray(mesh_pkl["transl"]).detach().unsqueeze(0)
    mesh_pkl["global_orient"] = torch.asarray(mesh_pkl["global_orient"]).detach().unsqueeze(0)    
    
    with torch.no_grad():
        betas = mesh_pkl["betas"].float().to(device)
        body_pose = mesh_pkl["body_pose"].float().to(device)
        global_orient = mesh_pkl["global_orient"].float().to(device)
        
        smpl_vertices = smplx_model.forward(betas=betas,
                                            body_pose=body_pose,
                                            global_orient=global_orient).vertices[0].cpu()
    
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
    
@torch.no_grad()
def adjust_gaussians_covariance(gaussians, neigh_idxs=None):
    '''
    Given an isotropic gaussian model, this function transforms it into an anisotropic gaussian model
    and adjusts the covariance matrix of each gaussian in order to fix holes.
    '''
    anisotropic_gaussians = GaussianModel(3)
    anisotropic_gaussians.from_gaussians(gaussians)
    
    xyz = gaussians._xyz.detach().cpu().numpy()
    kdtree = scipy.spatial.cKDTree(xyz)
    
    # Find neighbors
    if neigh_idxs is None:
        neigh_dist, neigh_idxs = kdtree.query(xyz, k=5)  # k=5 to include the point itself
        neigh_idxs = neigh_idxs[:, 1:]
    
    # Get neighbor positions
    neigh_pos = xyz[neigh_idxs]  # exclude the point itself

    # Calculate deviations from the mean position
    deviations = neigh_pos - xyz[:, None]
    # weights = neigh_dist[:, 1:] / np.sum(neigh_dist[:, 1:], axis=1)[:, None]
    # Calculate the covariance matrices
    cov_matrices = np.einsum('...ij,...ik->...ikj', deviations, deviations)
    
    # perform mean to get the covariance matrix
    cov_matrices = np.mean(cov_matrices, axis=1)
    # cov_matrices = np.mean(cov_matrices * weights[:, :, None, None], axis=1)
    
    cov_squared = cov_matrices @ cov_matrices
    
    # Perform eigen decomposition to get the scaling (eigenvalues) and rotation (eigenvectors)
    U, Sigma, VT = np.linalg.svd(cov_squared, hermitian=True)
    
    rotation_matrices = U @ VT
    
    
    scale = Sigma
    
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

def load_clothes_labels(src_user, src_outfit, src_take, src_sample, use_mesh_reconstruction=False):
    sample_name = f"labels_with_faces_{src_sample}.pkl" if use_mesh_reconstruction else f"labels_with_faces_gs_{src_sample}.pkl"
    scan_labels_path = os.path.join("4D-DRESS", src_user, src_outfit, src_take, "Semantic", "labels_with_faces", sample_name) 
    scan_labels = torch.asarray(load_pickle(scan_labels_path)["scan_labels"])

    return scan_labels

def create_clothes_mask(scan_labels, clothes_label_ids):
    # create source and target clothes masks
    source_clothes = torch.zeros(len(scan_labels), dtype=torch.bool)
    for label_id in clothes_label_ids:
        source_clothes = torch.logical_or(source_clothes, (scan_labels == label_id))
    
    return source_clothes
    
def load_gaussian(model, trg_y_rotation=0, use_mesh_reconstruction=False):
    gaussians = IsotropicGaussianModel(3)
    
    def get_last_iteration(model_path):
        pcd_folder_name = "point_cloud" if use_mesh_reconstruction else "point_cloud"
        p = os.path.join("output", model_path, pcd_folder_name)
        models = [x for x in os.listdir(p) if os.path.isdir(os.path.join(p, x))]
        models.sort()
        return os.path.join(p, models[-1], "point_cloud.ply")
    
    ply_path = get_last_iteration(model)
    gaussians.load_ply(ply_path)
    
    rotate_gaussians_along_y(gaussians, trg_y_rotation)
    
    return gaussians


def create_gaussian_avatar(params, run_nicp=True, use_mesh_reconstruction=False):
    '''
    Create a Gaussian Avatar from the given parameters.
    First load scan labels, then load the isotropic gaussian model and create the mesh.
    Run NICP if necessary.
    Finally create the Gaussian Avatar.
    '''
    scan_labels = load_clothes_labels(params["user"], params["outfit"], params["take"], params["sample"], use_mesh_reconstruction=use_mesh_reconstruction)
    
    gaussians = load_gaussian(params["model"], params["y_rotation"], use_mesh_reconstruction=use_mesh_reconstruction)
    
    trimesh_mesh = gs_to_trimesh(gaussians, downsample_percent=1.0)
    
    if run_nicp:
        os.chdir('NICP')
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(config_path="conf_test")
        pipe_cfg = hydra.compose(config_name="default")
        os.chdir('..')
        
        model_pkg = load_model()
        
        Rx = model_pkg["Rx"]
        
        nicp_path = os.path.join("output", params["model"], "NICP")
        if not os.path.exists(nicp_path):
            os.makedirs(nicp_path)

        _, _, Rx_inv = register_shapes(trimesh_mesh, params["model"], nicp_path, pipe_cfg, **model_pkg)

        smpl_pkl = load_smpl_data_from_nicp_registration(os.path.join("output", params["model"], "NICP"))
        smpl_mesh = smpl_pkl.pop("mesh")
    else:
        origin, xaxis = [0, 0, 0], [1, 0, 0]
        alpha = np.pi/2
        Rx = trimesh.transformations.rotation_matrix(alpha, xaxis, origin)
        
        smpl_pkl = load_smpl_data_from_nicp_registration(os.path.join("output", params["model"], "NICP"))
        smpl_mesh = smpl_pkl.pop("mesh")
    
    # smpl_pkl["global_orient"] = torch.tensor([1, 0, 0], dtype=torch.float32).unsqueeze(0)
    
    # smpl_mesh.apply_transform(Rx_inv)
    rotate_gaussians(gaussians, torch.asarray(Rx[:3, :3]).float().cuda())
    
    return GaussianAvatarCopy2(3, gaussians, smpl_mesh, smpl_pkl, scan_labels)

@torch.no_grad()
def polish_gaussian_avatar(avatar: GaussianAvatarCopy2):
    # use the adjust_covaariance function to adjust the gaussians
    # to calculate neighbor indexes, use the gauss2smpl mapping inside avatar
    # to get the smpl vertices, and then use faces of the smpl model to get the neighbors of 
    # each vertex
    
    gauss2smpl = avatar.gauss2smpl[:, 1].flatten().detach().cpu().numpy()
    faces = torch.asarray(avatar.smpl.faces)
    smpl_neigh_idxs = []
    
    min_neigh = np.inf
    for i in range(len(avatar.smpl.v_template)):
        neighbors = set((faces[torch.sum(faces==i, dim=1).bool(), :]).flatten().tolist())
        neighbors.remove(i)
        smpl_neigh_idxs.append(list(neighbors))
        if len(neighbors) < min_neigh:
            min_neigh = len(neighbors)
    
    for i in range(len(smpl_neigh_idxs)):
        smpl_neigh_idxs[i] = smpl_neigh_idxs[i][:min_neigh]
        
    neigh_idxs = torch.tensor(smpl_neigh_idxs, dtype=torch.long)[gauss2smpl]
    
    # g2 = adjust_gaussians_covariance(avatar, neigh_idxs=neigh_idxs)
    # avatar._scaling = g2._scaling
    # avatar._rotation = g2._rotation
    # del g2

def run_retarget_and_animate(cfg, pipe, src_clothes_label_ids, trg_clothes_label_ids, run_nicp):    
    
    params = {
        "user": cfg.src_user,
        "outfit": cfg.src_outfit,
        "take": cfg.src_take,
        "sample": cfg.src_sample,
        "y_rotation": cfg.src_y_rotation,
        "model": cfg.source_model,
    }
    
    src_avatar = create_gaussian_avatar(params, run_nicp=run_nicp, use_mesh_reconstruction=cfg.use_mesh_reconstruction)
    
    # polish_gaussian_avatar(src_avatar)
        
    params = {
        "user": cfg.trg_user,
        "outfit": cfg.trg_outfit,
        "take": cfg.trg_take,
        "sample": cfg.trg_sample,
        "y_rotation": cfg.trg_y_rotation,
        "model": os.path.join(cfg.target_model, cfg.trg_sample),
    }
    
    trg_avatar = create_gaussian_avatar(params, run_nicp=run_nicp, use_mesh_reconstruction=cfg.use_mesh_reconstruction)
    
    if SAVE_PIPELINE_STEPS_RENDERS:
        src_avatar.show_phantom_gaussians()
        trg_avatar.show_phantom_gaussians()
        
        if not os.path.exists(save_pipeline_out):
            os.makedirs(save_pipeline_out)
        
        bg_color = [0.99, 0.99, 0.99]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        cameras = [create_camera(azimuth=x) for x in [0, 90, 180, 270]]
    
        # render the target model
        for idx, camera in enumerate(cameras):
            with torch.no_grad():
                target_pkg = render(camera, trg_avatar, pipe, bg)
                target_image = target_pkg["render"]
                
                source_pkg = render(camera, src_avatar, pipe, bg)
                source_image = source_pkg["render"]
                
            
            target_image = target_image.cpu().numpy().transpose(1, 2, 0)
            source_image = source_image.cpu().numpy().transpose(1, 2, 0)
            cat_images = np.concatenate([source_image, target_image], axis=0)
            cat_images = np.clip(cat_images, 0, 1)
            
            plt.imsave(os.path.join(save_pipeline_out, f"tposed-{idx}.png"), cat_images)
        
        exit()
            
    # retarget clothes
    src_clothes = create_clothes_mask(src_avatar.scan_labels, src_clothes_label_ids)
    trg_clothes = create_clothes_mask(trg_avatar.scan_labels, trg_clothes_label_ids).logical_not()
    
    src_avatar.transfer_clothes_to(trg_avatar, src_clothes, trg_clothes)
    
    # load animation data from target
    trg_folder = os.path.join("output", f"{cfg.trg_user}_{cfg.trg_outfit}_{cfg.trg_take}")
    
    models = [x for x in os.listdir(trg_folder) if os.path.isdir(os.path.join(trg_folder, x)) and x.startswith("f")]
    models.sort()
    
    smplx_model = smplx.SMPL("smpl/smpl300/SMPL_MALE.pkl", num_betas=10)
    
    for target_model in tqdm.tqdm(models):

        trg_mesh_pkl_path = os.path.join(cfg.dataset_path, cfg.trg_user, cfg.trg_outfit, cfg.trg_take, "SMPL", f"mesh-{target_model}_smpl.pkl")
        target_smpl, target_smpl_pkl = load_smpl_data_from_registration(trg_mesh_pkl_path, use_base_smpl=False, smplx_model=smplx_model)
        
        
        theta = torch.cat([target_smpl_pkl["global_orient"], target_smpl_pkl["body_pose"]], dim=1)
        trg_avatar.lbs(theta)
            
        bg_color = [0.99, 0.99, 0.99]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        cameras = [create_camera(azimuth=x) for x in [0, 90, 180, 270]]
        
        # render the target model
        for camera in cameras:
            with torch.no_grad():
                target_pkg = render(camera, trg_avatar, pipe, bg)
                target_image = target_pkg["render"]
                
                source_pkg = render(camera, src_avatar, pipe, bg)
                source_image = source_pkg["render"]
                
            
            target_image = target_image.cpu().numpy().transpose(1, 2, 0)
            source_image = source_image.cpu().numpy().transpose(1, 2, 0)
            cat_images = np.concatenate([source_image, target_image], axis=1)
            cat_images = np.clip(cat_images, 0, 1)
            
            if cfg.save_output:
                '''
                Save the output under the folder of the target
                '''
                render_folder_name = "lbs_render_on_data_smpl_knn_augmented"
                render_folder_name = f"{render_folder_name}_src_{cfg.src_user}_{cfg.src_outfit}_{cfg.src_take}"
                if not os.path.exists(os.path.join("output", os.path.join(cfg.target_model, target_model), render_folder_name)):
                    os.makedirs(os.path.join("output", os.path.join(cfg.target_model, target_model), render_folder_name))
                plt.imsave(os.path.join("output", os.path.join(cfg.target_model, target_model), render_folder_name, f"render-{camera.image_name}.png"), cat_images)


def run_retarget(cfg, pipe, src_clothes_label_ids, trg_clothes_label_ids):
    RUN_NICP = True
    LOAD_NICP = False

    source_scan_labels = load_clothes_labels(cfg.src_user, cfg.src_outfit, cfg.src_take, cfg.src_sample)
    target_scan_labels = load_clothes_labels(cfg.trg_user, cfg.trg_outfit, cfg.trg_take, cfg.trg_sample)

    # create source and target clothes masks
    source_clothes = create_clothes_mask(source_scan_labels, src_clothes_label_ids)
    target_clothes = create_clothes_mask(target_scan_labels, trg_clothes_label_ids)
    
    target_gaussians = load_gaussian(cfg.target_model, cfg.src_y_rotation)
    source_gaussians = load_gaussian(cfg.source_model, cfg.trg_y_rotation)
          
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
        
        Rx = model_pkg["Rx"]
        
        src_nicp_path = os.path.join("output", cfg.source_model, "NICP")
        if not os.path.exists(src_nicp_path):
            os.makedirs(src_nicp_path)
        
        trg_nicp_path = os.path.join("output", cfg.target_model, "NICP")
        if not os.path.exists(trg_nicp_path):
            os.makedirs(trg_nicp_path)

        if not cfg.only_target_nicp:
            register_shapes(source_trimesh, cfg.source_model, src_nicp_path, pipe_cfg, **model_pkg)

        register_shapes(target_trimesh, cfg.target_model, trg_nicp_path, pipe_cfg, **model_pkg)


        source_smpl_pkl = load_smpl_data_from_nicp_registration(os.path.join("output", cfg.source_model, "NICP"))
        source_smpl = source_smpl_pkl.pop("mesh")
        target_smpl_pkl = load_smpl_data_from_nicp_registration(os.path.join("output", cfg.target_model, "NICP"))
        target_smpl = target_smpl_pkl.pop("mesh")
    elif LOAD_NICP:
        origin, xaxis = [0, 0, 0], [1, 0, 0]
        alpha = np.pi/2
        
        Rx = trimesh.transformations.rotation_matrix(alpha, xaxis, origin)
        
        source_smpl_pkl = load_smpl_data_from_nicp_registration(os.path.join("output", cfg.source_model, "NICP"))
        source_smpl = source_smpl_pkl.pop("mesh")
        target_smpl_pkl = load_smpl_data_from_nicp_registration(os.path.join("output", cfg.target_model, "NICP"))
        target_smpl = target_smpl_pkl.pop("mesh")
    else:
        raise ValueError("No registration method selected")
    

    # use pyrender render the source smpl model and the source trimesh model
    import pyrender


    # Rotate the gaussians to align them with NICP registration
    rotate_gaussians(source_gaussians, torch.asarray(Rx[:3, :3]).float().cuda())
    rotate_gaussians(target_gaussians, torch.asarray(Rx[:3, :3]).float().cuda())

    SHOW_SMPL_FIT_PREVIEW = False
    if SHOW_SMPL_FIT_PREVIEW:
        
        source_smpl_mesh = pyrender.Mesh.from_trimesh(source_smpl)
        # source_gs_mesh = pyrender.Mesh.from_points(source_trimesh.vertices)
        source_gs_mesh = pyrender.Mesh.from_points(gs_to_trimesh(source_gaussians).vertices)

        target_smpl_mesh = pyrender.Mesh.from_trimesh(target_smpl)
        # target_gs_mesh = pyrender.Mesh.from_points(target_trimesh.vertices)
        target_gs_mesh = pyrender.Mesh.from_points(gs_to_trimesh(target_gaussians).vertices)
        
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

    # This method uses the GaussianAvatar to store and manipulate gaussians
    # based on smpl and LBS
    
    # 1. Create a GaussianAvatar for the source model and the target model
    source_avatar = GaussianAvatarCopy2(3, source_gaussians, source_smpl, source_smpl_pkl, source_scan_labels)
    
    target_avatar = GaussianAvatarCopy2(3, target_gaussians, target_smpl, target_smpl_pkl, target_scan_labels)
    
    # 2. Call the transfer_clothes function to transfer the clothes from the source to the target
    source_avatar.transfer_clothes_to(target_avatar, source_clothes, target_clothes)
    
    target_beta = torch.tensor(target_smpl_pkl["betas"], dtype=torch.float, device="cpu")
    target_theta = torch.tensor(target_smpl_pkl["body_pose"], dtype=torch.float, device="cpu")
    target_avatar.lbs(target_beta, target_theta)
    
    # Rotate back gaussians to display them correctly
    
    # if RUN_NICP:
    #     rotate_gaussians(source_avatar, torch.asarray(np.linalg.inv(Rx[:3, :3])).float())
    rotate_gaussians(target_avatar, torch.asarray(np.linalg.inv(Rx[:3, :3])).float())
    
    
    bg_color = [0.99, 0.99, 0.99]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    cameras = [create_camera(azimuth=x) for x in [0, 90, 180, 270]]
    
    # render the target model
    for camera in cameras:
        with torch.no_grad():
            target_pkg = render(camera, target_avatar, pipe, bg)
            target_image = target_pkg["render"]
            
            source_pkg = render(camera, source_avatar, pipe, bg)
            source_image = source_pkg["render"]
            
        
        target_image = target_image.cpu().numpy().transpose(1, 2, 0)
        source_image = source_image.cpu().numpy().transpose(1, 2, 0)
        cat_images = np.concatenate([source_image, target_image], axis=1)
        # plt.imshow(cat_images)
        # plt.show()
        if cfg.save_output:
            '''
            Save the output under the folder of the target
            '''
            render_folder_name = "lbs_render_on_smpl"
            render_folder_name = f"{render_folder_name}_src_{cfg.src_user}_{cfg.src_outfit}_{cfg.src_take}"
            if not os.path.exists(os.path.join("output", cfg.target_model, render_folder_name)):
                os.makedirs(os.path.join("output", cfg.target_model, render_folder_name))
            plt.imsave(os.path.join("output", cfg.target_model, render_folder_name, f"render-{camera.image_name}.png"), cat_images)

from typing import List

        

# if __name__ == "__main__":
#     # Set up command line argument parser
#     parser = ArgumentParser(description="Training script parameters")
#     pp = PipelineParams(parser)
#     ep = ExperimentParams(parser)
#     # parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing the trained gaussian models")
#     parser.add_argument('--src_clothes_label_ids', nargs='+', type=int, required=True)
#     parser.add_argument('--trg_clothes_label_ids', nargs='+', type=int, required=True)
#     # parser.add_argument('--retarget_and_animate', default=False, action='store_true', help="Run NICP for the first target, then use LBS to retarget other frames.")
#     args = parser.parse_args(sys.argv[1:])
    
#     ep = ep.extract(args)
    
#     if not (ep.src_user and ep.src_outfit and ep.src_take and ep.trg_user and ep.trg_outfit and ep.trg_take):
#         raise ValueError("Please specify source and target user, outfit and take")
    
#     trg_folder = os.path.join("output", f"{ep.trg_user}_{ep.trg_outfit}_{ep.trg_take}")
    
#     models = [x for x in os.listdir(trg_folder) if os.path.isdir(os.path.join(trg_folder, x))]
#     models.sort()
#     src_folder = os.path.join("output", f"{ep.src_user}_{ep.src_outfit}_{ep.src_take}")
#     source_model = sorted(os.listdir(src_folder))[0]
#     ep.src_sample = source_model
        
#     ep.source_model = f"{ep.src_user}_{ep.src_outfit}_{ep.src_take}/{source_model}"
    
    
#     for idx, target_model in enumerate(models):
#         print(f"Processing {target_model}")
#         ep.trg_sample = target_model
#         ep.target_model = f"{ep.trg_user}_{ep.trg_outfit}_{ep.trg_take}/{target_model}"

#         run_retarget(ep, pp.extract(args), args.src_clothes_label_ids, args.trg_clothes_label_ids)
#         if idx <= 0:
#             ep.only_target_nicp = True  # set to true after the first iteration to avoid re-running the source NICP

#     # All done
#     print("\nTraining complete.")



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    pp = PipelineParams(parser)
    ep = ExperimentParams(parser)
    # parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing the trained gaussian models")
    parser.add_argument('--src_clothes_label_ids', nargs='+', type=int, required=True)
    parser.add_argument('--trg_clothes_label_ids', nargs='+', type=int, required=True)
    parser.add_argument('--run_nicp', default=False, action='store_true', help="Run NICP registration")
    # parser.add_argument('--retarget_and_animate', default=False, action='store_true', help="Run NICP for the first target, then use LBS to retarget other frames.")
    args = parser.parse_args(sys.argv[1:])
    
    ep = ep.extract(args)
    
    if not (ep.src_user and ep.src_outfit and ep.src_take and ep.trg_user and ep.trg_outfit and ep.trg_take):
        raise ValueError("Please specify source and target user, outfit and take")
    
    trg_folder = os.path.join("output", f"{ep.trg_user}_{ep.trg_outfit}_{ep.trg_take}")
    
    models = [x for x in os.listdir(trg_folder) if os.path.isdir(os.path.join(trg_folder, x)) and x.startswith("f")]
    models.sort()
    
    src_folder = os.path.join("output", f"{ep.src_user}_{ep.src_outfit}_{ep.src_take}")
    source_model = sorted([el for el in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, el)) and el.startswith("f")])[0]
    ep.src_sample = source_model
        
    ep.source_model = f"{ep.src_user}_{ep.src_outfit}_{ep.src_take}/{source_model}"
    
    
    print(f"Processing...")
    ep.trg_sample = models[0]
    ep.target_model = f"{ep.trg_user}_{ep.trg_outfit}_{ep.trg_take}"

    run_retarget_and_animate(ep, pp.extract(args), args.src_clothes_label_ids, args.trg_clothes_label_ids, args.run_nicp)

    # All done
    print("\nTraining complete.")
    
    
def main(pp, ep, args):
    
    if not (ep.src_user and ep.src_outfit and ep.src_take and ep.trg_user and ep.trg_outfit and ep.trg_take):
        raise ValueError("Please specify source and target user, outfit and take")
    
    trg_folder = os.path.join("output", f"{ep.trg_user}_{ep.trg_outfit}_{ep.trg_take}")
    
    models = [x for x in os.listdir(trg_folder) if os.path.isdir(os.path.join(trg_folder, x)) and x.startswith("f")]
    models.sort()
    
    src_folder = os.path.join("output", f"{ep.src_user}_{ep.src_outfit}_{ep.src_take}")
    source_model = sorted([el for el in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, el)) and el.startswith("f")])[0]
    ep.src_sample = source_model
        
    ep.source_model = f"{ep.src_user}_{ep.src_outfit}_{ep.src_take}/{source_model}"
    
    
    print(f"Processing...")
    ep.trg_sample = models[0]
    ep.target_model = f"{ep.trg_user}_{ep.trg_outfit}_{ep.trg_take}"

    run_retarget_and_animate(ep, pp, args.src_clothes_label_ids, args.trg_clothes_label_ids, args.run_nicp)

    # All done
    print("\nTraining complete.")