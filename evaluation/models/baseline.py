import sys
sys.path.append('./ext/gaussian-splatting')
import torch
from arguments_4d_dress import ModelParams4dDress, OptimizationParams4dDress
from arguments import PipelineParams
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene_4ddress import IsotropicGaussianModel, GaussianModel
import tqdm
from random import randint
from scene_4ddress.gaussian_model_lbs import GaussianAvatar2
import smplx
import trimesh
import argparse
import numpy as np
from PIL import Image
import os

import matplotlib.pyplot as plt

from .model import Model
from typing import Union, Dict,NamedTuple
import random

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: Dict[str, np.array]
    image_path: str
    image_name: str
    width: int
    height: int


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

from torch import nn
import math

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, images, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 width=None, height=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")


        if images is not None:
            self.original_image = {k: image.clamp(0.0, 1.0).to(self.data_device) for k, image in images.items()}
            self.image_width = self.original_image["white"].shape[2]
            self.image_height = self.original_image["white"].shape[1]

            if gt_alpha_mask is not None:
                gt_alpha_mask = gt_alpha_mask.to(self.data_device)
                self.original_image = {k: image * gt_alpha_mask for k, image in images.items()}
            else:
                self.original_image = {k: image * torch.ones((1, self.image_height, self.image_width), device=self.data_device) for k, image in images.items()}

        else:
            self.original_image = torch.zeros((3, height, width), device=self.data_device)
            assert width is not None and height is not None, "Width and Height must be provided if images are not provided"
            self.image_width = width
            self.image_height = height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def loadCam(args, id, cam_info, resolution_scale):

    white_image, black_image = cam_info.image["white"].permute(2, 0, 1), cam_info.image["black"].permute(2, 0, 1)

    resized_image_rgb_white =white_image

    gt_image_white = resized_image_rgb_white[:3, ...]
    loaded_mask = None

    if resized_image_rgb_white.shape[1] == 4:
        loaded_mask = resized_image_rgb_white[3:4, ...]
        
    resized_image_rgb_black = black_image
    
    gt_image_black = resized_image_rgb_black[:3, ...]

    gt_image = {"white": gt_image_white.to(args.data_device), "black": gt_image_black.to(args.data_device)}

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  images=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


class Scene:

    gaussians : Union[IsotropicGaussianModel, GaussianModel]

    def __init__(self, args, gaussians : IsotropicGaussianModel, input_data):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        self.train_cameras = {}
        self.test_cameras = {}
        
        scales  = None
        opacities = None
        scene_info = None
        
        scales, opacities, scene_info = read4dDressInfo(args, input_data)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        resolution_scale = 1.
        print("Loading Training Cameras")
        self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
        print("Loading Test Cameras")
        self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if scales is not None and opacities is not None:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, scales, opacities)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]



def read4dDressInfo(cfg, input_data) -> SceneInfo:
    cameras = input_data['cameras']
    smpl_params = input_data['source']['smpl']
    gender = input_data['source']['basic_info']['gender'].upper()
    width, height = cameras["intrinsics"][0, 2]*2, cameras["intrinsics"][1, 2]*2

    
    body_pose = torch.asarray(smpl_params["body_pose"]).reshape(1, -1)
    betas = torch.asarray(smpl_params["betas"]).reshape(1, -1)
    global_orient = torch.asarray(smpl_params["global_orient"]).reshape(1, -1)
    transl = torch.asarray(smpl_params["transl"]).reshape(1, -1)
    model_path = f"smpl/smpl300/SMPL_{gender.upper()}.pkl"
        
    smpl_model = smplx.SMPL(model_path=model_path)
    from smplx.utils import SMPLOutput
    res: SMPLOutput = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl, return_verts=True)
    
    vertices = res.vertices.detach().cpu().numpy()[0]
    # rotation = input_data['source']['basic_info']['rotation']
    # vertices = np.matmul(rotation, vertices.T).T
    faces = smpl_model.faces
    
    # subdivide smpl to have more points
    from trimesh.remesh import subdivide
    vertices, faces = subdivide(vertices, faces)
    vertices, faces = subdivide(vertices, faces)
    colors = np.ones((vertices.shape[0], 3))
    normals = np.zeros((vertices.shape[0], 3))
    pcd = BasicPointCloud(points=vertices, colors=colors, normals=normals)
    scales = np.ones((vertices.shape[0], 1)) * -6.5
    opacity = np.ones((vertices.shape[0], 1))


    cam_infos = []

    for idx in range(len(cameras["extrinsics"])):

        c2w = cameras["extrinsics"][idx]
        intrinsics = cameras["intrinsics"]

        FovX = intrinsics[0, 0]
        
        FovY = intrinsics[1, 1]

        c2w[:3, 1:3] *= -1
        
        w2c = np.linalg.inv(c2w)

        R = np.transpose(w2c[:3,:3])
        T = w2c[:3, 3]

        alpha_image_white_bg = input_data['source']['imgs_white'][idx]
        alpha_image_black_bg = input_data['source']['imgs_black'][idx]

        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image={"white": alpha_image_white_bg, "black": alpha_image_black_bg},
            image_path="image_path",
            image_name=f"capture-f00011.png",
            width=width,
            height=height
        )
        cam_infos.append(cam_info)

    
    nerf_normalization = getNerfppNorm(cam_infos)

    ply_path = "points3d.ply"   # we don't use this later

    return scales, opacity, SceneInfo(
        point_cloud=pcd,
        train_cameras=cam_infos,
        test_cameras=[],
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )


class Baseline(Model):

    def reconstruct(self, input_data):
        parser = argparse.ArgumentParser()
        dataset = ModelParams4dDress(parser)
        opt = OptimizationParams4dDress(parser)
        self.pipe = PipelineParams(parser)
        dataset.subj = input_data['source']['subject']
        dataset.outfit = input_data['source']['outfit']
        dataset.seq = input_data['source']['take']
        dataset.init_from_mesh = False
        dataset.init_from_smpl = True
        setattr(dataset, 'images', 'images')
        setattr(dataset, 'resolution', -1)
        setattr(dataset, 'white_background', False)
        setattr(dataset, 'source_path', "test")
        setattr(dataset, 'model_path', "reconstruction_putput")
        first_iter = 0
        if opt.use_isotropic_gaussians:
            gaussians = IsotropicGaussianModel(dataset.sh_degree)
        else:
            gaussians = GaussianModel(dataset.sh_degree)
        smpl_params = input_data['source']['smpl']
        smpl_params['gender'] = input_data['source']['basic_info']['gender']
        self.scene = Scene(dataset, gaussians, input_data=input_data)

        gaussians.training_setup(opt)

        bg_labels = ["white", "black"]
        background = {
            bg_labels[0]: torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
            bg_labels[1]: torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
            }

        viewpoint_stack = None
        ema_loss_for_log = 0.0
        progress_bar = tqdm.tqdm(range(first_iter, opt.iterations), desc="Training progress")
        first_iter += 1
        for iteration in range(first_iter, opt.iterations + 1):

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
            if opt.random_background:
                bg_label = bg_labels[randint(0, 1)]
            elif dataset.white_background:
                bg_label = bg_labels[0]
            else:
                bg_label = bg_labels[1]
                
            bg = background[bg_label]

            render_pkg = render(viewpoint_cam, gaussians, self.pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image[bg_label].cuda()
            
            # if iteration % 10 == 0:
            #     print("viewpoint_cam: ", viewpoint_cam.__dict__)
            #     import matplotlib.pyplot as plt
            #     cat = torch.cat([image.detach().cpu().permute(1, 2, 0), gt_image.cpu().detach().permute(1, 2, 0)], axis=1)
            #     plt.imshow(cat)
            #     plt.show()
            
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()

            # iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Densification
                if (not opt.block_densification) and (iteration < opt.densify_until_iter):
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        # size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        size_threshold = opt.size_threshold # filter on scales
                        extent = self.scene.cameras_extent * opt.camera_extent_multiplier # filter on scales
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

        self.scene.save(iteration)
        smpl_model = smplx.SMPL(f"smpl/smpl300/SMPL_{input_data['source']['basic_info']['gender'].upper()}.pkl")
        betas = torch.tensor(input_data['source']['smpl']['betas'])
        body_pose = torch.tensor(input_data['source']['smpl']['body_pose'])
        global_orient = torch.tensor(input_data['source']['smpl']['global_orient'])
        transl = torch.tensor(input_data['source']['smpl']['transl'])

        run_nicp = False
        
        if run_nicp:
            # os.chdir('NICP')
            # hydra.core.global_hydra.GlobalHydra.instance().clear()
            # hydra.initialize(config_path="conf_test")
            # pipe_cfg = hydra.compose(config_name="default")
            # os.chdir('..')
            
            # model_pkg = load_model()
            
            # Rx = model_pkg["Rx"]
            
            # nicp_path = os.path.join("output", params["model"], "NICP")
            # if not os.path.exists(nicp_path):
            #     os.makedirs(nicp_path)

            # register_shapes(trimesh_mesh, params["model"], nicp_path, pipe_cfg, **model_pkg)

            # smpl_pkl = load_smpl_data_from_nicp_registration(os.path.join("output", params["model"], "NICP"))
            # smpl_mesh = smpl_pkl.pop("mesh")
            # origin, xaxis = [0, 0, 0], [1, 0, 0]
            # alpha = np.pi/2
            # Rx = trimesh.transformations.rotation_matrix(alpha, xaxis, origin)
            
            # smpl_pkl = load_smpl_data_from_nicp_registration(os.path.join("output", params["model"], "NICP"))
            # smpl_mesh = smpl_pkl.pop("mesh")
            pass
        else:
            smpl_verts = smpl_model(betas=betas, 
                                body_pose=body_pose, 
                                global_orient=global_orient, 
                                transl=transl, 
                                return_verts=True).vertices[0].detach().cpu().numpy()
            
            faces = smpl_model.faces
            
            from trimesh.remesh import subdivide
            verts, faces = subdivide(smpl_verts, faces)
            
            smpl_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            smpl_mesh.fix_normals()
        
        scan_labels = torch.ones((self.scene.gaussians._xyz.shape[0], ), device="cuda")
        self.avatar = GaussianAvatar2(3, self.scene.gaussians, smpl_mesh=smpl_mesh, smpl_model_pkl=input_data['source']['smpl'], scan_labels=scan_labels)
        
        print("RECONSTRUCTION DONE")
        print("Number of Gaussians: ", self.scene.gaussians._xyz.shape[0])


    def pose(self, pose):
        self.avatar.lbs(theta=pose)

    def render(self, camera_info, device="cuda"):
        cams = []
        extrinsics = camera_info["extrinsics"]
        intrinsics = camera_info["intrinsics"]
        
        width = int(intrinsics[0, 2]*2)
        height = int(intrinsics[1, 2]*2)
        
        loaded_mask = None
        for _id, c2w in extrinsics.items():
            
            FovX = intrinsics[0, 0]
            
            FovY = intrinsics[1, 1]
            c2w = c2w.copy()
            c2w[:3, 1:3] *= -1
            
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3,:3])
            T = w2c[:3, 3]
            
            cams.append(Camera(colmap_id=_id, R=R, T=T, 
                  FoVx=FovX, FoVy=FovY, 
                  images=None, gt_alpha_mask=loaded_mask,
                  image_name=f'image-{_id}', uid=_id, 
                  data_device=device, 
                  width=width, height=height))
            
        renders = []
        
        for cam in cams:
            render_pkg = render(cam, self.avatar, self.pipe, torch.tensor([1, 1, 1], dtype=torch.float32, device=device))
            renders.append(render_pkg["render"])
        renders = torch.stack(renders)
        return renders
    
