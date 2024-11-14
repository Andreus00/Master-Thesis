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

from __future__ import annotations
from NICP.utils_cop.SMPL import SMPL
from torch import Tensor
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from gaussian_utils import remove_samples_from_gaussians
import scipy
import torch.nn.functional as F
from typing import List
# import smplx

from torch.autograd import Variable

from typing import Tuple

# import smplx.lbs

class GaussianModel:

    def setup_functions(self):
                        
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, scales: np.array=None, opacities:np.array=None):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0


        if scales is None:
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        else:
            scales = torch.tensor(scales).float().cuda()
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        if opacities is None:
            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        else:
            opacities = torch.tensor(opacities).float().cuda()

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
    
    def from_gaussians(self, model: Union[IsotropicGaussianModel, GaussianModel]):
        self._xyz = model._xyz
        self._features_dc = model._features_dc
        self._features_rest = model._features_rest
        if model._scaling.shape[1] == 1:
            self._scaling = model._scaling.repeat(1, 3)
        else:
            self._scaling = model._scaling
        self._rotation = model._rotation
        self._opacity = model._opacity
        self.max_radii2D = model.max_radii2D
        self.xyz_gradient_accum = model.xyz_gradient_accum
        self.denom = model.denom
        self.optimizer = model.optimizer
        self.percent_dense = model.percent_dense
        self.spatial_lr_scale = model.spatial_lr_scale

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        

class GaussianAvatar(GaussianModel):
    '''
    An extension of the Gaussian model that keeps a smpl model inside and uses it to
    perform LBS on gaussians, based on the underlying smpl.
    '''
    def setup_functions(self):
        super().setup_functions()
        self.T_lbs = None
        self.kdtree = None
        
        def compute_covariance_from_scaling_rotation_lbs(scaling, scaling_modifier, rotation):
            '''
            Build the covariance matrix from the scaling and rotation.
            This version uses the LBS matrix to transform the covariance matrix
            based on the kinematic chain.
            '''
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            left_cov = torch.matmul(self.T_lbs[0], actual_covariance)
            actual_covariance = torch.matmul(left_cov, self.T_lbs[0].transpose(1, 2))
            symm = strip_symmetric(actual_covariance)
            return symm
        
        # self.covariance_activation = compute_covariance_from_scaling_rotation_lbs
        
    def __init__(self, sh_degree : int, isot_gauss: IsotropicGaussianModel, smpl_mesh, smpl_model_pkl, scan_labels, using_nicp):
        super().__init__(sh_degree)
        self.from_gaussians(isot_gauss)
        self.smpl_mesh = smpl_mesh
        self.betas = smpl_model_pkl["betas"]    # contains pose and shape info
        self.body_pose = smpl_model_pkl["body_pose"]
        self.transl = smpl_model_pkl["transl"]
        self.scale = smpl_model_pkl["scale"] if "scale" in smpl_model_pkl else 1.0
        if using_nicp:
            self.smpl = SMPL(model_path='NICP/neutral_smpl_with_cocoplus_reg.txt', base_path="NICP")
            self.inverse_lbs = self.inverse_lbs_nicp
        else:
            import smplx
            self.smpl = smplx.SMPL("smpl/smpl300/SMPL_NEUTRAL.pkl", num_betas=10)
            self.inverse_lbs = self.inverse_lbs_smpl
        self.scan_labels = scan_labels
        self.kdtree = None
        self.gauss2smpl = None
        self.T_inv = None
        self.run_knn()
        # self.get_T_lbs()
        
    '''
    i do something like this:
    knn between smpl and gaussian splat
        with torch.no_grad():
            results = knn_points(gs_points, template_points, K=1)
            dists, idxs = results.dists, results.idx
    Bs = shape_offsets[idxs]
    Ps = pose_offsets[idxs]

    add blendshape and poseshape offsets to mean
    gs_points = gs_points + Bs + Ps

    i also get per joint gaussian:

            lbs_weights = smpl.lbs_weights[idxs]
            num_joints = 55
            lbs_weights = torch.squeeze(lbs_weights, 1)
            T = torch.matmul(lbs_weights, A.view(1, num_joints, 16)).view(1, -1, 4, 4)
    later i apply to rot of gaussians: (T is used as T_lbs)

            def compute_covariance_from_scaling_rotation_lbs(scaling, scaling_modifier, rotation, T_lbs):
                L = build_scaling_rotation(scaling_modifier * scaling, rotation)
                actual_covariance = L @ L.transpose(1, 2)
                left_cov = torch.matmul(T_lbs[0], actual_covariance)
                actual_covariance = torch.matmul(left_cov, T_lbs[0].transpose(1, 2))
                symm = strip_symmetric(actual_covariance)
                return symm 
    T is T_lbs
    I dont change the spherical harmonic coefs for now
    '''
    
    def run_knn(self):
        self.kdtree = scipy.spatial.cKDTree(self.smpl_mesh.vertices)
        dists, idxs = self.kdtree.query(self.get_xyz.detach().cpu().numpy(), k=1)
        self.gauss2smpl = idxs
    
    
    def batch_global_rigid_transformation(self, Rs, Js, parent, rotate_base = False):
        N = Rs.shape[0]
        if rotate_base:
            np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype = np.float)
            np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
            rot_x = Variable(torch.from_numpy(np_rot_x).float()).to(Rs.device)
            root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
        else:
            root_rotation = Rs[:, 0, :, :]
        Js = torch.unsqueeze(Js, -1)

        def make_A(R, t):
            R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
            t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1)).to(R.device)], dim = 1)
            return torch.cat([R_homo, t_homo], 2)
        
        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]

        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(Rs[:, i], j_here)
            res_here = torch.matmul(results[parent[i]], A_here)
            results.append(res_here)

        results = torch.stack(results, dim = 1)

        new_J = results[:, :, :3, 3]
        Js_w0 = torch.cat([Js, Variable(torch.zeros(N, 24, 1, 1)).to(Rs.device)], dim = 2)
        init_bone = torch.matmul(results, Js_w0)
        init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
        A = results - init_bone

        return new_J, A
    
    @torch.no_grad()
    def inverse_lbs_nicp(self, theta_in_rodrigues=True) -> Tuple[Tensor, Tensor]:
        from smplx.lbs import batch_rodrigues
        beta = torch.asarray(self.betas)[:, :10]
        theta = torch.asarray(self.body_pose)
        v_template = torch.asarray(self.smpl.v_template).float()
        shapedirs = torch.asarray(self.smpl.shapedirs)
        posedirs = torch.asarray(self.smpl.posedirs)
        J_regressor = torch.asarray(self.smpl.J_regressor)
        parents = torch.asarray(self.smpl.parents)
        lbs_weights = torch.asarray(self.smpl.weight)
        
        
        device = beta.device
        self.smpl.cur_device = torch.device(device.type, device.index)

        num_batch = beta.shape[0]
        size_beta = beta.shape[1]

        blendshapes = None
        if size_beta == 10:
            blendshapes = torch.matmul(beta, self.smpl.shapedirs).view(-1, self.smpl.size[0], self.smpl.size[1])
            v_shaped = blendshapes + self.smpl.v_template
        elif size_beta == 300:
            blendshapes = torch.matmul(beta, self.smpl.shapedirs_300).view(-1, self.smpl.size[0], self.smpl.size[1])
            v_shaped = blendshapes + self.smpl.v_template

        Jx = torch.matmul(v_shaped[:, :, 0], J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        if theta.shape[1] < 72:
            new_theta = torch.zeros(theta.shape[0], 72, device = theta.device)
            new_theta[:, :theta.shape[1]] = theta
            theta = new_theta
        elif theta.shape[1] > 72:
            theta = theta[:, :72]
            
        if theta_in_rodrigues:
            Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        else: #theta is already rotations
            Rs = theta.view(-1,24,3,3)

        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.smpl.e3).view(-1, 207)
        poseshapes = torch.matmul(pose_feature, posedirs).view(-1, self.smpl.size[0], self.smpl.size[1])
        v_posed = poseshapes + v_shaped
        J_transformed, A = self.batch_global_rigid_transformation(Rs, J, self.smpl.parents, rotate_base = False)

        W = lbs_weights.expand(num_batch,*lbs_weights.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)
        
        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.smpl.cur_device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]

        
        smpl_mesh = torch.asarray(self.smpl_mesh.vertices)
        
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(-1, 1)
        # ax.scatter(verts[0, :, 0], verts[0, :, 1], verts[0, :, 2], color='r')
        # ax.scatter(smpl_mesh[..., 0], smpl_mesh[..., 1], smpl_mesh[..., 2], color='b')
        # plt.show()
        # plt.close()
        
        # align smpl_mesh and verts
        with torch.no_grad():
            self._xyz += torch.asarray(self.transl).squeeze()[[2,0,1]]
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(-1, 1)
        # ax.scatter(verts[0, :, 0], verts[0, :, 1], verts[0, :, 2], color='r')
        # ax.scatter(smpl_mesh[..., 0], smpl_mesh[..., 1], smpl_mesh[..., 2], color='b')
        # ax.scatter(self._xyz[:, 0].detach().cpu(), self._xyz[:, 1].detach().cpu(), self._xyz[:, 2].detach().cpu(), color='black')
        # plt.show()
        # plt.close()
        
        T_inv = torch.linalg.inv(T).float()
        
        
        with torch.no_grad():
            xyz = self._xyz.detach().cpu().clone()
            T_inv_gauss = T_inv[:, self.gauss2smpl, :, :]
            v_homo_gauss =v_homo[:, self.gauss2smpl, 3:, :].reshape(-1, 1)

            xyz = torch.cat((xyz,  v_homo_gauss), dim=-1).float().reshape(1, -1, 4, 1)
            xyz_mult = torch.matmul(T_inv_gauss, xyz)
            xyz = xyz_mult.reshape(-1, 4)[:, :3]
            Ps = poseshapes[:, self.gauss2smpl]
            Bs = blendshapes[:, self.gauss2smpl]
            xyz_psbs = -(Ps + Bs).squeeze()#[self.gauss2smpl, :]
            xyz += xyz_psbs
            self.T_inv = T_inv
            self._xyz = xyz.cuda()
            
        
        
        # import matplotlib.pyplot as plt
        # # plot verts
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(-1, 1)
        # ax.scatter(verts[0, :, 0], verts[0, :, 1], verts[0, :, 2], color='r')
        # ax.scatter(self.smpl_mesh.vertices[..., 0], self.smpl_mesh.vertices[..., 1], self.smpl_mesh.vertices[..., 2], color='b')
        # ax.scatter(smpl_mesh_can[..., 0], smpl_mesh_can[..., 1], smpl_mesh_can[..., 2], color='green')
        # ax.scatter(self._xyz[:, 0].detach().cpu(), self._xyz[:, 1].detach().cpu(), self._xyz[:, 2].detach().cpu(), color='black')
        # plt.show()
        # plt.close()
        

    
    def lbs_gs(self, beta, theta, theta_in_rodrigues=True) -> Tuple[Tensor, Tensor]:
        '''
        Run LBS on the gaussians.
        '''
        with torch.no_grad():
            from smplx.lbs import batch_rodrigues
            v_template = torch.asarray(self.smpl.v_template).float()
            shapedirs = torch.asarray(self.smpl.shapedirs)
            posedirs = torch.asarray(self.smpl.posedirs)
            J_regressor = torch.asarray(self.smpl.J_regressor)
            parents = torch.asarray(self.smpl.parents)
            lbs_weights = torch.asarray(self.smpl.weight)
            
            
            device = beta.device
            self.smpl.cur_device = torch.device(device.type, device.index)

            num_batch = beta.shape[0]
            size_beta = beta.shape[1]
            
            _xyz = self._xyz.clone().cpu()

            blendshapes = None
            if size_beta == 10:
                blendshapes = torch.matmul(beta, self.smpl.shapedirs).view(-1, self.smpl.size[0], self.smpl.size[1])
            elif size_beta == 300:
                blendshapes = torch.matmul(beta, self.smpl.shapedirs_300).view(-1, self.smpl.size[0], self.smpl.size[1])
                
            v_shaped = blendshapes[:, self.gauss2smpl, :] + _xyz
            
            v_shaped_smpl = blendshapes + self.smpl.v_template

            Jx = torch.matmul(v_shaped_smpl[:, :, 0], J_regressor)
            Jy = torch.matmul(v_shaped_smpl[:, :, 1], J_regressor)
            Jz = torch.matmul(v_shaped_smpl[:, :, 2], J_regressor)
            J = torch.stack([Jx, Jy, Jz], dim = 2)
            if theta.shape[1] < 72:
                new_theta = torch.zeros(theta.shape[0], 72, device = theta.device)
                new_theta[:, :theta.shape[1]] = theta
                theta = new_theta
            elif theta.shape[1] > 72:
                theta = theta[:, :72]
                
            if theta_in_rodrigues:
                Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
            else: #theta is already rotations
                Rs = theta.view(-1,24,3,3)

            pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.smpl.e3).view(-1, 207)
            poseshapes = torch.matmul(pose_feature, posedirs).view(-1, self.smpl.size[0], self.smpl.size[1])

            v_posed = poseshapes[:, self.gauss2smpl, :] + v_shaped
            J_transformed, A = self.batch_global_rigid_transformation(Rs, J, self.smpl.parents, rotate_base = False)

            W = lbs_weights.expand(num_batch,*lbs_weights.shape[1:])
            T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)
            
            v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.smpl.cur_device)], dim = 2)
            v_homo = torch.matmul(T[:, self.gauss2smpl, :], torch.unsqueeze(v_posed_homo, -1))

            verts = v_homo[:, :, :3, 0]

            self._xyz = verts.cuda().squeeze()
            
            self.T_inv = None


    def translate_xyz_by_smpl_offsets(self, offsets):
        '''
        Apply an offset [6890, 1] to the gaussians. Exploit knn previously calculated to do so.
        '''
        with torch.no_grad():
            self._xyz[:] += offsets[self.gauss2smpl].cuda()
           
    
    def is_in_t_pose(self):
        return self.T_inv is not None
    
    def get_clothes_mask(self, clothes: int):
        return self.scan_labels != clothes
    
    @torch.no_grad()
    def transfer_clothes_to(self, target_gaussians: GaussianAvatar, source_clothes_mask, target_clothes_mask):
                
        
        remove_samples_from_gaussians(target_gaussians, target_clothes_mask)
        
        target_gaussians._xyz = torch.cat((target_gaussians._xyz, self._xyz[source_clothes_mask, :]))
        target_gaussians._features_dc = torch.cat((target_gaussians._features_dc, self._features_dc[source_clothes_mask, :]))
        target_gaussians._features_rest = torch.cat((target_gaussians._features_rest, self._features_rest[source_clothes_mask, :]))
        target_gaussians._scaling = torch.cat((target_gaussians._scaling, self._scaling[source_clothes_mask, :]))
        target_gaussians._rotation = torch.cat((target_gaussians._rotation, self._rotation[source_clothes_mask, :]))
        target_gaussians._opacity = torch.cat((target_gaussians._opacity, self._opacity[source_clothes_mask, :]))
        
        target_gaussians.run_knn()

from typing import Union

class GaussianAvatar2(GaussianModel):
    '''
    An extension of the Gaussian model that keeps a smpl model inside and uses it to
    perform LBS on gaussians, based on the underlying smpl.
    '''
    def setup_functions(self):
        super().setup_functions()
        
        def compute_covariance_from_scaling_rotation_lbs(scaling, scaling_modifier, rotation):
            '''
            Build the covariance matrix from the scaling and rotation.
            This version uses the LBS matrix to transform the covariance matrix
            based on the kinematic chain.
            '''
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            left_cov = torch.matmul(self.T_lbs[0], actual_covariance)
            actual_covariance = torch.matmul(left_cov, self.T_lbs[0].transpose(1, 2))
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.inverse_lbs = self.inverse_lbs_nicp
        self.covariance_activation = compute_covariance_from_scaling_rotation_lbs
    
    @torch.no_grad()
    def clean_gaussians(self):
        '''
        Some gaussians are none (idk why). Let's clean them.
        
        self.denom = model.denom
        self.optimizer = model.optimizer
        self.percent_dense = model.percent_dense
        self.spatial_lr_scale = model.spatial_lr_scale
        '''
        mask = ~self._xyz.isnan().any(dim=1)
        self._xyz = self._xyz[mask]
        self._features_dc = self._features_dc[mask]
        self._features_rest = self._features_rest[mask]
        self._scaling = self._scaling[mask]
        self._rotation = self._rotation[mask]
        self._opacity = self._opacity[mask]
        self.scan_labels = self.scan_labels[mask.cpu()]
        
        self.gauss2smpl = self.gauss2smpl[mask]
        
        self.T_lbs_canon = self.T_lbs_canon[:, mask]
        self.canonical_xyz = self.canonical_xyz[mask]
        self.T_lbs = self.T_lbs[:, mask]
        
        return mask
        
        
    def __init__(self, sh_degree : int, gauss: Union[IsotropicGaussianModel, GaussianModel], smpl_mesh, smpl_model_pkl, scan_labels):
        super().__init__(sh_degree)
        self.device = torch.device("cuda")
        self.from_gaussians(gauss)

        self.smpl: SMPL = SMPL(model_path='NICP/neutral_smpl_with_cocoplus_reg_augmented.txt', base_path="NICP", obj_saveable=True)
        self.v_template = torch.asarray(self.smpl.v_template, device=self.device).float()
        self.shapedirs = torch.asarray(self.smpl.shapedirs, device=self.device)
        self.shapedirs_300 = torch.asarray(self.smpl.shapedirs_300, device=self.device)
        self.posedirs = torch.asarray(self.smpl.posedirs, device=self.device)
        self.J_regressor = torch.asarray(self.smpl.J_regressor, device=self.device)
        self.parents = torch.asarray(self.smpl.parents, device=self.device)
        self.lbs_weights = torch.asarray(self.smpl.weight, device=self.device)
        self.e3 = torch.asarray(self.smpl.e3, device=self.device)
        
        self.betas = torch.asarray(smpl_model_pkl["betas"], device=self.device)    # contains pose and shape info
        self.scan_labels = scan_labels
        
        # Data for lbs
        self.kdtree = None
        self.gauss2smpl = None
        
        # LBS data for compute_covariance_from_scaling_rotation_lbs
        self.T_lbs = None
        
        # Canonical data
        self.T_lbs_canon = None
        self.canonical_xyz = None
        
        # KNN
        self.k = 3
        
        _, posed_gauss2smpl = self.run_knn(smpl_mesh.vertices)
        
        self.inverse_lbs(posed_gauss2smpl, smpl_model_pkl, smpl_mesh)
    
    
    def batch_index_select(self, data, inds):
        bs, nv = data.shape[:2]
        device = data.device
        inds = inds + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
        data = data.reshape(bs*nv, *data.shape[2:])
        return data[inds.long()]

    def smplx_lbsmap_top_k(self, verts_transform, points, template_points, source_points=None, K=1, addition_info=None):
        '''ref: https://github.com/JanaldoChen/Anim-NeRF
        Args:  
        '''
        from pytorch3d.ops.knn import knn_points
        bz, np, _ = points.shape
        with torch.no_grad():
            results = knn_points(points, template_points, K=K)
            dists, idxs = results.dists, results.idx
        neighbs_dist = dists * 10e3
        neighbs = idxs.squeeze(0)
        
        weight_std = 0.1
        weight_std2 = 2. * weight_std ** 2
        xyz_neighbs_lbs_weight = self.lbs_weights[:, neighbs, :].squeeze(1) # (bs, n_rays*K, k_neigh, 24)

        xyz_neighbs_weight_conf = torch.exp(-torch.sum(torch.abs(xyz_neighbs_lbs_weight - xyz_neighbs_lbs_weight[..., 0:1, :]), dim=-1)/weight_std2) # (bs, n_rays*K, k_neigh)
        xyz_neighbs_weight_conf = torch.gt(xyz_neighbs_weight_conf, 0.).float()
        xyz_neighbs_weight = torch.exp(-neighbs_dist) # (bs, n_rays*K, k_neigh)
        xyz_neighbs_weight *= xyz_neighbs_weight_conf
        xyz_neighbs_weight = xyz_neighbs_weight / xyz_neighbs_weight.sum(-1, keepdim=True) # (bs, n_rays*K, k_neigh)

        
        
        xyz_neighbs_transform = verts_transform[:, neighbs] #self.batch_index_select(verts_transform, neighbs) # (bs, n_rays*K, k_neigh, 4, 4)
        xyz_transform = torch.sum(xyz_neighbs_weight.unsqueeze(-1).unsqueeze(-1) * xyz_neighbs_transform, dim=2) # (bs, n_rays*K, 4, 4)
        xyz_dist = torch.sum(xyz_neighbs_weight * neighbs_dist, dim=2, keepdim=True) # (bs, n_rays*K, 1)

        if addition_info is not None: #[bz, nv, 3]
            xyz_neighbs_info = addition_info[:, neighbs] # self.batch_index_select(addition_info, neighbs)
            # print(xyz_neighbs_info.shape)
            xyz_info = torch.sum(xyz_neighbs_weight.unsqueeze(-1)* xyz_neighbs_info, dim=2) 
            return xyz_dist, xyz_transform, xyz_info
        else:
            return xyz_dist, xyz_transform
    
    def run_knn(self, smpl_verts):
        kdtree = scipy.spatial.cKDTree(smpl_verts)
        dists, idxs = kdtree.query(self.get_xyz.detach().cpu().numpy(), k=self.k, distance_upper_bound=1000.)
        idxs = torch.asarray(idxs, device=self.device)
        if len(idxs.shape) == 1:
            idxs = idxs.unsqueeze(1)
        
        return kdtree, idxs
    
    
    def batch_global_rigid_transformation(self, Rs, Js, parent, rotate_base = False):
        N = Rs.shape[0]
        if rotate_base:
            np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype = np.float)
            np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
            rot_x = Variable(torch.from_numpy(np_rot_x).float()).to(Rs.device)
            root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
        else:
            root_rotation = Rs[:, 0, :, :]
        Js = torch.unsqueeze(Js, -1)

        def make_A(R, t):
            R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
            t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1)).to(R.device)], dim = 1)
            return torch.cat([R_homo, t_homo], 2)
        
        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]

        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(Rs[:, i], j_here)
            res_here = torch.matmul(results[parent[i]], A_here)
            results.append(res_here)

        results = torch.stack(results, dim = 1)

        new_J = results[:, :, :3, 3]
        Js_w0 = torch.cat([Js, Variable(torch.zeros(N, 24, 1, 1)).to(Rs.device)], dim = 2)
        init_bone = torch.matmul(results, Js_w0)
        init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
        A = results - init_bone

        return new_J, A
    
    
    @torch.no_grad()
    def inverse_lbs_nicp(self, gauss2smpl, smpl_model_pkl, smpl_mesh, theta_in_rodrigues=True) -> Tuple[Tensor, Tensor]:
        from smplx.lbs import batch_rodrigues
        
        # Carica tutto su GPU
        device = self.device
        
        smpl_mesh = torch.asarray(smpl_mesh.vertices, device=self.device)
        
        beta = torch.tensor(smpl_model_pkl["betas"], device=device)[:, :10]
        theta = torch.tensor(smpl_model_pkl["body_pose"], device=device)
        
        # TODO: integrate this
        if "global_orient" in smpl_model_pkl:
            global_orient = torch.tensor(smpl_model_pkl["global_orient"], device=device)
            theta = torch.cat((global_orient, theta), dim=1)
        else:
            # global_orient = torch.zeros(beta.shape[0], 3, device=device)
            theta = theta # torch.cat((global_orient, theta), dim=1)
        transl = torch.tensor(smpl_model_pkl["transl"], device=device)
        
        num_batch = beta.shape[0]
        size_beta = beta.shape[1]
        
        shapedirs = self.shapedirs if size_beta == 10 else self.shapedirs_300
        
        # Calcoli su GPU
        if size_beta in {10, 300}:
            blendshapes = torch.matmul(beta, shapedirs).view(-1, self.smpl.size[0], self.smpl.size[1])
        else:
            raise ValueError(f"Invalid size_beta: {size_beta}")
        v_shaped = blendshapes + self.v_template

        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)
        
        if theta.shape[1] < 72:
            new_theta = torch.zeros(theta.shape[0], 72, device=device)
            new_theta[:, :theta.shape[1]] = theta
            theta = new_theta
        elif theta.shape[1] > 72:
            theta = theta[:, :72]
            
        if theta_in_rodrigues:
            Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        else:  # theta is already rotations
            Rs = theta.view(-1, 24, 3, 3)

        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        poseshapes = torch.matmul(pose_feature, self.posedirs).view(-1, self.smpl.size[0], self.smpl.size[1])
        v_posed = poseshapes + v_shaped
        J_transformed, A = self.batch_global_rigid_transformation(Rs, J, self.parents, rotate_base=False)

        W = self.lbs_weights.expand(num_batch, *self.lbs_weights.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)
        
        
        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=device)], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))
        
        v_final = v_homo[:, :, :3].squeeze(-1).detach()
        
        # Align smpl_mesh and verts. This misalignment is given from a wrong angle of the SMPL model used in NICP and my SMPL model
        # TODO: Fix this as this part is necessary because the forward lbs is not giving the exact same shape as the NICP result
        
        if 'global_orient' not in smpl_model_pkl:
            offsets = (v_final - smpl_mesh).squeeze()
            self.translate_xyz_by_smpl_offsets(offsets, gauss2smpl[:, 0])
        

        T_inv: torch.Tensor = torch.linalg.inv(T).float()
        
        _, T_inv_gauss, poseshapes_blendshapes = self.smplx_lbsmap_top_k(verts_transform=T_inv, 
                                                                         points=self._xyz.reshape(num_batch, -1, 3), 
                                                                         template_points=v_final, 
                                                                         K=self.k, 
                                                                         addition_info=poseshapes+blendshapes)


        self._xyz = self._xyz.to(device)  # Assicuriamoci che self._xyz sia su GPU
        # T_inv_gauss = T_inv[:, gauss2smpl, :, :].mean(dim=2)
        v_homo_gauss = v_homo[:, gauss2smpl, 3:, :].mean(dim=2).reshape(-1, 1)

        self._xyz = torch.cat((self._xyz, v_homo_gauss), dim=-1).float().reshape(1, -1, 4, 1)
        xyz_mult = torch.matmul(T_inv_gauss, self._xyz)
        self._xyz = xyz_mult.reshape(-1, 4)[:, :3]
        
        xyz_psbs = (poseshapes_blendshapes).squeeze()
        self._xyz -= xyz_psbs
        
        self.canonical_xyz = self._xyz.detach().clone()
        
        # setup knn_tree, gauss2smpl, T_lbs, Bs and Ps
        self.kdtree, self.gauss2smpl = self.run_knn(self.v_template.cpu())
        self.T_lbs_canon = T_inv_gauss[..., :3, :3] # torch.eye(3).reshape(1, 1, 3, 3).repeat_interleave(num_batch, dim=0).repeat_interleave(self._xyz.shape[0], dim=1).to(device)
        self.T_lbs = self.T_lbs_canon.clone()
        
        self.clean_gaussians()
        
    def remove_skin(self):
        '''
        Find all those gaussians that are part of the skin and whose knn[0] is
        a part of the body covered by clothes
        '''
        clothes_mask = self.get_clothes_mask(0)    # mask that selects all the clothes points
        skin_mask = ~clothes_mask          # mask that selects all the skin points
        covered_skin_smpl_idxs = set(self.gauss2smpl[clothes_mask, :].flatten().unique().cpu().tolist())
        import tqdm
        for idx, el in tqdm.tqdm(enumerate(self.gauss2smpl[:, 0])):
            if skin_mask[idx]:
                if el.item() in covered_skin_smpl_idxs:
                    skin_mask[idx] = False
        
        valid_mask = torch.logical_or(skin_mask, clothes_mask)
        # now remove all those gaussians that are covered by clothes
        
        self._xyz = self._xyz[valid_mask]
        self.canonical_xyz = self.canonical_xyz[valid_mask]
        self._features_dc = self._features_dc[valid_mask]
        self._features_rest = self._features_rest[valid_mask]
        self._scaling = self._scaling[valid_mask]
        self._rotation = self._rotation[valid_mask]
        self._opacity = self._opacity[valid_mask]
        self.scan_labels = self.scan_labels[valid_mask.cpu()]
        
        self.gauss2smpl = self.gauss2smpl[valid_mask]
        self.T_lbs_canon = self.T_lbs_canon[:, valid_mask]
        self.T_lbs = self.T_lbs[:, valid_mask]
        
        
    @torch.no_grad()
    def fix_skin_and_clothes(self, verts, gauss2smpl=None):
        '''
        fix skin where it collides with clothes and where it is missing.
        To do so, I first remove all those skin gaussians that are in contact with clothes.
        Then I take all those smpl points that are not covered by the skin, and I add fake 
        skin gaussians on them.
        '''
        if gauss2smpl is None:
            gauss2smpl = self.gauss2smpl
        
        verts = self.v_template
        
        skin_mask = self.get_clothes_mask(0)
        other_mask = ~skin_mask
        covered_body_idxs = gauss2smpl[..., 0][other_mask]
        smpl_other_points = verts[covered_body_idxs].detach().cpu().numpy()
        
        import trimesh
        trimes_smpl_other = trimesh.Trimesh(vertices=verts.cpu(), faces=self.smpl.faces)
        norms = trimes_smpl_other.vertex_normals
        
        # now use vertex normals to create the new gaussians on smpl_other_points
        # the normals are the direction of the gaussians
        
        other_gaussians_xyz = torch.asarray(smpl_other_points, device=self.device, dtype=torch.float32)
        other_gaussians_features_dc = torch.ones((other_gaussians_xyz.shape[0], 1, 3), device=self.device, dtype=torch.float32)
        other_gaussians_features_rest = torch.ones((other_gaussians_xyz.shape[0], 15, 3), device=self.device, dtype=torch.float32)
        other_gaussians_scaling = torch.asarray([-6.0, -6.0, -7.5], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(other_gaussians_xyz.shape[0], 1)
        import scipy.spatial.transform
        rot = scipy.spatial.transform.Rotation.from_euler('xyz', norms[covered_body_idxs.cpu()]).as_quat()
        other_gaussians_rotation = torch.tensor(rot, device=self.device, dtype=torch.float32)# .unsqueeze(0).repeat(other_gaussians_xyz.shape[0], 1)
        other_gaussians_opacity = torch.ones((other_gaussians_xyz.shape[0], 1), device=self.device, dtype=torch.float32)
        
        self._xyz = torch.cat((self._xyz, other_gaussians_xyz), dim=0)
        self._features_dc = torch.cat((self._features_dc, other_gaussians_features_dc), dim=0)
        self._features_rest = torch.cat((self._features_rest, other_gaussians_features_rest), dim=0)
        self._scaling = torch.cat((self._scaling, other_gaussians_scaling), dim=0)
        self._rotation = torch.cat((self._rotation, other_gaussians_rotation), dim=0)
        self._opacity = torch.cat((self._opacity, other_gaussians_opacity), dim=0)
        
        self.scan_labels = torch.cat((self.scan_labels, torch.zeros((other_gaussians_xyz.shape[0]), device=self.scan_labels.device)))
        
        # add a mask of -1 to the body reconstruction
        gauss2smpl = torch.cat((gauss2smpl, -torch.ones_like(gauss2smpl[other_mask])), dim=0)
        return gauss2smpl
        
        

    
    @torch.no_grad()
    def lbs(self, theta, beta=None, theta_in_rodrigues=True) -> Tuple[Tensor, Tensor]:
        '''
        Run LBS on the gaussians.
        This function sets self.T_lbs
        '''
        from smplx.lbs import batch_rodrigues
        
        if beta is None:
            beta = self.betas
        
        beta = beta.to(self.device)
        theta = theta.to(self.device)
        
        # self.smpl.cur_device = torch.device(device.type, device.index)

        num_batch = beta.shape[0]
        size_beta = beta.shape[1]
                
        template = self.v_template

        blendshapes = None
        if size_beta == 10:
            blendshapes = torch.matmul(beta, self.shapedirs).view(-1, self.smpl.size[0], self.smpl.size[1])
        elif size_beta == 300:
            blendshapes = torch.matmul(beta, self.shapedirs_300).view(-1, self.smpl.size[0], self.smpl.size[1])
            
        v_shaped = blendshapes + template

        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        
        if theta.shape[1] < 72:
            new_theta = torch.zeros(theta.shape[0], 72, device=self.device)
            new_theta[:, :theta.shape[1]] = theta
            theta = new_theta
        elif theta.shape[1] > 72:
            theta = theta[:, :72]
            
        if theta_in_rodrigues:
            Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        else: #theta is already rotations
            Rs = theta.view(-1,24,3,3)

        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        poseshapes = torch.matmul(pose_feature, self.posedirs).view(-1, self.smpl.size[0], self.smpl.size[1])

        v_posed = poseshapes + v_shaped
        J_transformed, A = self.batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W = self.lbs_weights.expand(num_batch,*self.lbs_weights.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)
        
        # v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=self.device)], dim = 2)
        # v_homo = torch.matmul(T[:, self.gauss2smpl, :].mean(dim=2), torch.unsqueeze(v_posed_homo, -1))

        # verts = v_homo[:, :, :3, 0]
        
        xyzs = self.canonical_xyz.clone().unsqueeze(0)

        _, T, psbs = self.smplx_lbsmap_top_k(
            verts_transform=T,
            points=xyzs,
            template_points=self.v_template.unsqueeze(0),
            K=self.k,
            addition_info=poseshapes+blendshapes
        )
        xyzs += psbs
        xyzs = torch.cat([xyzs, torch.ones((num_batch, xyzs.shape[1], 1), device=self.device)], dim = 2)
        xyzs = torch.matmul(T, xyzs.unsqueeze(-1))[:, :, :3, 0]
        self._xyz = xyzs.squeeze(0)
        self.T_lbs = torch.matmul(T[..., :3, :3], self.T_lbs_canon)
            

    def translate_xyz_by_smpl_offsets(self, offsets, gauss2smpl=None):
        '''
        Apply an offset [6890, 1] to the gaussians. Exploit knn previously calculated to do so.
        '''
        if gauss2smpl is None:
            gauss2smpl = self.gauss2smpl
        with torch.no_grad():
            self._xyz[:] += offsets[gauss2smpl]
        
    
    def get_clothes_mask(self, clothes: int):
        return self.scan_labels != clothes
    
    
    @torch.no_grad()
    def remove_samples_from_gaussians(self, mask):
        self._xyz = self._xyz[mask]
        self.canonical_xyz = self.canonical_xyz[mask]
        self._features_dc = self._features_dc[mask]
        self._features_rest = self._features_rest[mask]
        self._scaling = self._scaling[mask]
        self._rotation = self._rotation[mask]
        self._opacity = self._opacity[mask]
        
        self.gauss2smpl = self.gauss2smpl[mask]
        self.T_lbs = self.T_lbs[:, mask]
        self.T_lbs_canon = self.T_lbs_canon[:, mask]
        self.scan_labels = self.scan_labels[mask]
        
    
    @torch.no_grad()
    def transfer_clothes_to(self, target_gaussians: GaussianAvatar2, source_clothes_mask, target_clothes_mask):
                
        target_gaussians.remove_samples_from_gaussians(target_clothes_mask)
        # self.remove_samples_from_gaussians(self.scan_labels == 0)
        
        
        target_gaussians._xyz = torch.cat((target_gaussians._xyz, self._xyz[source_clothes_mask, :].clone()))
        target_gaussians.canonical_xyz = torch.cat((target_gaussians.canonical_xyz, self._xyz[source_clothes_mask, :].clone()))
        target_gaussians._features_dc = torch.cat((target_gaussians._features_dc, self._features_dc[source_clothes_mask, :].clone()))
        target_gaussians._features_rest = torch.cat((target_gaussians._features_rest, self._features_rest[source_clothes_mask, :].clone()))
        target_gaussians._scaling = torch.cat((target_gaussians._scaling, self._scaling[source_clothes_mask, :].clone()))
        target_gaussians._rotation = torch.cat((target_gaussians._rotation, self._rotation[source_clothes_mask, :].clone()))
        target_gaussians._opacity = torch.cat((target_gaussians._opacity, self._opacity[source_clothes_mask, :].clone()))

        # transfer new T_lbs, T_lbs_canon and scan_labels
        target_gaussians.T_lbs = torch.cat((target_gaussians.T_lbs, self.T_lbs[:, source_clothes_mask, ...].clone()), dim=1)
        target_gaussians.T_lbs_canon = torch.cat((target_gaussians.T_lbs_canon, self.T_lbs_canon[:, source_clothes_mask, ...].clone()), dim=1)
        target_gaussians.scan_labels = torch.cat((target_gaussians.scan_labels, self.scan_labels[source_clothes_mask].clone()))
                
        # calculate new knn
        target_gaussians.kdtree, target_gaussians.gauss2smpl = target_gaussians.run_knn(torch.asarray(self.smpl.v_template))

        target_gaussians.remove_skin()
    
    @torch.no_grad()
    def color_to_segmentations(self):
        mapping = {
            0: torch.asarray([177, 177, 177], device=self.device, dtype=torch.float) / 255, # #B1B1B1 grey
            1: torch.asarray([255, 128, 0], device=self.device, dtype=torch.float) / 255, # #FD4747 orange
            2: torch.asarray([188, 0, 255], device=self.device, dtype=torch.float) / 255, # #BC00FF purple
            3: torch.asarray([253, 71, 71], device=self.device, dtype=torch.float) / 255, # #FD4747 red
            4: torch.asarray([70, 249, 69], device=self.device, dtype=torch.float) / 255, # #47FD47 green
            5: torch.asarray([0, 178, 255], device=self.device, dtype=torch.float) / 255, # #00B2FF blue
        }
        
        # create a new tensor with the same shape as the scan_labels
        new_colors = torch.zeros((self.scan_labels.shape[0], 3), device=self.device)
        
        # for each label, assign the corresponding color
        for i in range(len(mapping)):
            new_colors[self.scan_labels == i] = mapping[i]
            
        fused_color = RGB2SH(new_colors.cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        
    @torch.no_grad()
    def show_phantom_gaussians(self):
        opacity_mask = self._opacity < 0.00001
        new_opacities = torch.ones_like(self._opacity) * 0.0
        new_opacities[opacity_mask] = 1.0
        self._opacity = nn.Parameter(new_opacities)
        
        

class IsotropicGaussianModel:
    '''
    Gaussian model with isotropic scaling.
    
    Instead of keeping scales as a 3D vector, we keep a single scalar value.
    We then repeat this scalar value to create a 3D vector for the scaling.
    '''

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        repeat_scaling = self._scaling.repeat(1, 3)
        return self.scaling_activation(repeat_scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, scales: np.array=None, opacities:np.array=None):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0


        if scales is None:
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[...,None].reshape(-1, 1)
        else:
            scales = torch.tensor(scales).float().cuda()
            assert scales.shape[1] == 1, "Isotropic model should have a single scaling value"
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        if opacities is None:
            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        else:
            opacities = torch.tensor(opacities).float().cuda()

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1