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
import scipy
import torch.nn.functional as F
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
            
        if scales.shape[1] == 1:
            scales = scales.repeat(1, 3)
        
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
        
    def from_gaussians(self, gaussian_model):
        self._xyz = gaussian_model._xyz
        self._features_dc = gaussian_model._features_dc
        self._features_rest = gaussian_model._features_rest
        self._scaling = gaussian_model._scaling.repeat(1, 3)
        self._rotation = gaussian_model._rotation
        self._opacity = gaussian_model._opacity
        self.max_radii2D = gaussian_model.max_radii2D
        self.xyz_gradient_accum = gaussian_model.xyz_gradient_accum
        self.denom = gaussian_model.denom
        self.optimizer = gaussian_model.optimizer
        self.percent_dense = gaussian_model.percent_dense
        self.spatial_lr_scale = gaussian_model.spatial_lr_scale

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
        
        self.covariance_activation = compute_covariance_from_scaling_rotation_lbs
        
    def __init__(self, sh_degree : int, gaussians: IsotropicGaussianModel, smpl_mesh, smpl_model_pkl, scan_labels, using_nicp, gender='neutral'):
        super().__init__(sh_degree)
        self.from_gaussians(gaussians)
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
            self.smpl = smplx.SMPL(f"smpl/smpl300/SMPL_{gender.upper()}.pkl", num_betas=10)
            self.inverse_lbs = self.inverse_lbs_smpl
        self.scan_labels = scan_labels
        self.kdtree = None
        self.gauss2smpl = None
        self.T_inv = None
        self.Ps = None
        self.Bs = None
        self.run_knn()
        # self.get_T_lbs()
        
    
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
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.scatter(verts[0, :, 0], verts[0, :, 1], verts[0, :, 2], color='r')
        ax.scatter(smpl_mesh[..., 0], smpl_mesh[..., 1], smpl_mesh[..., 2], color='b')
        plt.show()
        plt.close()
        
        # align smpl_mesh and verts
        verts_mean = torch.mean(verts.squeeze(0), dim=0)
        smpl_mesh_mean = torch.mean(smpl_mesh, dim=0)
        offsets = (verts - smpl_mesh).squeeze()
        print(verts_mean.shape)
        print(smpl_mesh_mean.shape)
        smpl_mesh -= (smpl_mesh_mean - verts_mean)
        self.translate_xyz_by_smpl_offsets(offsets)
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.scatter(verts[0, :, 0], verts[0, :, 1], verts[0, :, 2], color='r')
        ax.scatter(smpl_mesh[..., 0], smpl_mesh[..., 1], smpl_mesh[..., 2], color='b')
        ax.scatter(self._xyz[:, 0].detach().cpu(), self._xyz[:, 1].detach().cpu(), self._xyz[:, 2].detach().cpu(), color='black')
        plt.show()
        plt.close()
        
        T_inv = torch.linalg.inv(T).float()
        
        smpl_mesh = smpl_mesh.unsqueeze(0).unsqueeze(-1)
        
        smpl_mesh_2 = torch.cat((smpl_mesh, v_homo[:, :, 3:, :]), dim=2).float()
        
        smpl_mesh_can = torch.matmul(T_inv, smpl_mesh_2)
        smpl_mesh_can = smpl_mesh_can[:, :, :3, :].reshape(-1, 3)
        smpl_mesh_can -= (poseshapes + blendshapes).squeeze()
        # apply T_inv and pose/shape blendshapes to the gaussians
        
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
            self.Bs = blendshapes
            self.Ps = poseshapes
            self._xyz = xyz.cuda()
            
        
        
        import matplotlib.pyplot as plt
        # plot verts
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.scatter(verts[0, :, 0], verts[0, :, 1], verts[0, :, 2], color='r')
        ax.scatter(self.smpl_mesh.vertices[..., 0], self.smpl_mesh.vertices[..., 1], self.smpl_mesh.vertices[..., 2], color='b')
        ax.scatter(smpl_mesh_can[..., 0], smpl_mesh_can[..., 1], smpl_mesh_can[..., 2], color='green')
        ax.scatter(self._xyz[:, 0].detach().cpu(), self._xyz[:, 1].detach().cpu(), self._xyz[:, 2].detach().cpu(), color='black')
        plt.show()
        plt.close()
        

        # return verts, joints, Rs
    
    
    def inverse_lbs_smpl_(self, theta_in_rodrigues=True) -> Tuple[Tensor, Tensor]:
        ''' Performs Linear Blend Skinning with the given shape and pose parameters

            Parameters
            ----------
            betas : torch.tensor BxNB
                The tensor of shape parameters
            pose : torch.tensor Bx(J + 1) * 3
                The pose parameters in axis-angle format
            v_template torch.tensor BxVx3
                The template mesh that will be deformed
            shapedirs : torch.tensor 1xNB
                The tensor of PCA shape displacements
            posedirs : torch.tensor Px(V * 3)
                The pose PCA coefficients
            J_regressor : torch.tensor JxV
                The regressor array that is used to calculate the joints from
                the position of the vertices
            parents: torch.tensor J
                The array that describes the kinematic tree for the model
            lbs_weights: torch.tensor N x V x (J + 1)
                The linear blend skinning weights that represent how much the
                rotation matrix of each part affects each vertex
            pose2rot: bool, optional
                Flag on whether to convert the input pose tensor to rotation
                matrices. The default value is True. If False, then the pose tensor
                should already contain rotation matrices and have a size of
                Bx(J + 1)x9
            dtype: torch.dtype, optional

            Returns
            -------
            verts: torch.tensor BxVx3
                The vertices of the mesh after applying the shape and pose
                displacements.
            joints: torch.tensor BxJx3
                The joints of the model
        '''
        from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform
        betas = torch.asarray(self.betas)
        pose = torch.asarray(self.body_pose)
        v_template = torch.asarray(self.smpl.v_template).float()
        shapedirs = torch.asarray(self.smpl.shapedirs)
        posedirs = torch.asarray(self.smpl.posedirs)
        J_regressor = torch.asarray(self.smpl.J_regressor)
        parents = torch.asarray(self.smpl.parents)
        lbs_weights = torch.asarray(self.smpl.lbs_weights)
        #print all shapes
        print("Betas:", betas.shape)
        print("Pose:", pose.shape)
        print("V_template:", v_template.shape)
        print("Shapedirs:", shapedirs.shape)
        print("Posedirs:", posedirs.shape)
        print("J_regressor:", J_regressor.shape)
        print("Parents:", parents.shape)
        print("LBS Weights:", lbs_weights.shape)
        batch_size = max(betas.shape[0], pose.shape[0])
        device, dtype = betas.device, betas.dtype

        # Add shape contribution
        blendshapes = blend_shapes(betas, shapedirs)
        v_shaped = v_template + blendshapes

        # Get the joints
        # NxJx3 array
        J = vertices2joints(J_regressor, v_shaped)

        # 3. Add pose blend shapes
        # N x J x 3 x 3
        ident = torch.eye(3, dtype=dtype, device=device)
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        poseshapes = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)

        v_posed = poseshapes + v_shaped
        # 4. Get the global joint location
        J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                dtype=dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        verts = v_homo[:, :, :3, 0]

        
        smpl_mesh = torch.asarray(self.smpl_mesh.vertices)
        
        
        # align smpl_mesh and verts
        offsets = (verts - smpl_mesh).squeeze()
        smpl_mesh += offsets
        # self.translate_xyz_by_smpl_offsets(offsets)
        
        
        T_inv = torch.linalg.inv(T).float()
        
        smpl_mesh = smpl_mesh.unsqueeze(0).unsqueeze(-1)
        
        smpl_mesh_2 = torch.cat((smpl_mesh, v_homo[:, :, 3:, :]), dim=2).float()
        print("AAA", smpl_mesh_2.shape)
        
        smpl_mesh_can = torch.matmul(T_inv, smpl_mesh_2)
        smpl_mesh_can = smpl_mesh_can[:, :, :3, :].reshape(-1, 3)
        smpl_mesh_can -= (poseshapes + blendshapes).squeeze()
        print("BBB", smpl_mesh_can.shape)
        # apply T_inv and pose/shape blendshapes to the gaussians
        
        with torch.no_grad():
            xyz = self._xyz.detach().cpu().clone()
            T_inv_gauss = T_inv[:, self.gauss2smpl, :, :]
            v_homo_gauss =v_homo[:, self.gauss2smpl, 3:, :].reshape(-1, 1)
            print(v_homo_gauss.shape)
            print(xyz.shape)
            xyz = torch.cat((xyz,  v_homo_gauss), dim=-1).float().reshape(1, -1, 4, 1)
            xyz_mult = torch.matmul(T_inv_gauss, xyz)
            xyz = xyz_mult.reshape(-1, 4)[:, :3]
            xyz_psbs = (poseshapes + blendshapes).squeeze()[self.gauss2smpl, :]
            print(xyz_psbs.shape)
            xyz += xyz_psbs
            self.T_inv = T_inv
            self.Bs = blendshapes
            self.Ps = poseshapes
            self._xyz = xyz.cuda()
            
        
        
        import matplotlib.pyplot as plt
        # plot verts
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.scatter(verts[0, :, 0], verts[0, :, 1], verts[0, :, 2], color='r')
        # ax.scatter(self.smpl_mesh.vertices[..., 0], self.smpl_mesh.vertices[..., 1], self.smpl_mesh.vertices[..., 2], color='b')
        # ax.scatter(self._xyz[:, 0].detach().cpu(), self._xyz[:, 1].detach().cpu(), self._xyz[:, 2].detach().cpu(), color='black')
        plt.show()
        plt.close()
        

        # return verts, joints, Rs

    def translate_xyz_by_smpl_offsets(self, offsets):
        '''
        Apply an offset [6890, 1] to the gaussians. Exploit knn previously calculated to do so.
        '''
        with torch.no_grad():
            self._xyz[:] += offsets[self.gauss2smpl].cuda()
            
    
    def inverse_lbs_smpl(self
    ) -> Tuple[Tensor, Tensor]:
        from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform
        betas = torch.asarray(self.betas)[:, :10]
        pose = torch.asarray(self.body_pose)
        v_template = torch.asarray(self.smpl.v_template)
        shapedirs = torch.asarray(self.smpl.shapedirs)
        posedirs = torch.asarray(self.smpl.posedirs)
        J_regressor = torch.asarray(self.smpl.J_regressor)
        parents = torch.asarray(self.smpl.parents)
        lbs_weights = torch.asarray(self.smpl.lbs_weights)
        #print all shapes
        global_orient = torch.zeros((betas.shape[0], 3), device=betas.device, dtype=betas.dtype)
        pose = torch.cat([global_orient, pose], dim=1)
        print("Betas:", betas.shape)
        print("Pose:", pose.shape)
        print("V_template:", v_template.shape)
        print("Shapedirs:", shapedirs.shape)
        print("Posedirs:", posedirs.shape)
        print("J_regressor:", J_regressor.shape)
        print("Parents:", parents.shape)
        print("LBS Weights:", lbs_weights.shape)
        

        # FIRST CALCULATE POSE_BLEND_SHAPES, SHAPE_BLEND_SHAPES, AND THE ROTATION MATRICES
        batch_size = max(betas.shape[0], pose.shape[0])
        device, dtype = betas.device, betas.dtype

        # Calculate shape contribution
        Bs = blend_shapes(betas, shapedirs)
        v_shaped = v_template + Bs

        # Get the joints
        # NxJx3 array
        J = vertices2joints(J_regressor, v_shaped)

        # 3. Add pose blend shapes
        # N x J x 3 x 3
        ident = torch.eye(3, dtype=dtype, device=device)
        if True:
            rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
                [batch_size, -1, 3, 3])
            
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            # (N x P) x (P, V * 3) -> N x V x 3
            pose_offsets = torch.matmul(
                pose_feature, posedirs).view(batch_size, -1, 3)
        else:
            pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
            rot_mats = pose.view(batch_size, -1, 3, 3)

            pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                        posedirs).view(batch_size, -1, 3)

        v_posed = pose_offsets + v_shaped
        # 4. Get the global joint location
        J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                dtype=dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        verts = v_homo[:, :, :3, 0]
        rest = v_homo[:, :, -1, 0].reshape(-1, 1)
        
        # NOW INVERT THE ROTATION MATRICES, MULTIPLY THEM WITH THE CURRENT SMPL,
        # AND REMOVE POSE/SHAPE OFFEST
        print("mean shape",  torch.mean(torch.asarray(self.smpl_mesh.vertices), dim=0).shape)
        current_verts = (torch.asarray(self.smpl_mesh.vertices).float() - torch.mean(torch.asarray(self.smpl_mesh.vertices), dim=0)).float()
        current_verts = torch.cat((current_verts, rest),dim=1)
        T_inv = torch.inverse(T)
        verts_homo = torch.matmul(T_inv, torch.unsqueeze(current_verts, dim=-1))
        verts_inv = verts_homo[:, :, :3, 0]

        verts_canonical = verts_inv - pose_offsets - Bs
        
        import matplotlib.pyplot as plt
        # scatter verts
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.scatter(verts_canonical[0, :, 0], verts_canonical[0, :, 1], verts_canonical[0, :, 2], color='b')
        ax.scatter(verts[0, :, 0], verts[0, :, 1], verts[0, :, 2], color='r')
        ax.scatter(v_template[:, 0], v_template[:, 1], v_template[:, 2], color='g')
        ax.scatter(current_verts[:, 0], current_verts[:, 1], current_verts[:, 2], color='black')
        plt.show()
        plt.close()
        exit()

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
        if training_args.block_position:
            training_args.position_lr_init = 0.0
            training_args.position_lr_final = 0.0
            training_args.position_lr_delay_mult = 0.0
            training_args.position_lr_max_steps = 0
            
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
        new_scaling = self._scaling[selected_pts_mask].repeat(N,1) / (0.8*N)
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