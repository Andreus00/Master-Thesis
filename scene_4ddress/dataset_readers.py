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
import sys
from PIL import Image
from typing import NamedTuple
from scene_4ddress.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import smplx

from typing import Dict

import torch

import trimesh
import pyrender

import matplotlib.pyplot as plt

import pickle

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

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

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

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



def open_pickle(*arg):
    pickle_file = os.path.join(*arg)
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    return data

def open_image(path):
    image = Image.open(path)
    return image

def triangle_area(p1, p2, p3):
    return np.linalg.norm(np.cross(p2 - p1, p3 - p1)) / 2

# load data from pkl
def load_pickle(pkl_dir):
    return pickle.load(open(pkl_dir, "rb"))

# preprocess scan_mesh: scale, centralize, rotation, offset
def preprocess_scan_mesh(mesh, mcentral=False, bbox=True, rotation=None, offset=None, scale=1.0):
    # get scan vertices mass center
    mcenter = np.mean(mesh['vertices'], axis=0)
    # get scan vertices bbox center
    bmax = np.max(mesh['vertices'], axis=0)
    bmin = np.min(mesh['vertices'], axis=0)
    bcenter = (bmax + bmin) / 2
    # centralize scan data around mass center
    if mcentral:
        mesh['vertices'] -= mcenter
    # centralize scan data around bbox center
    elif bbox:
        mesh['vertices'] -= bcenter
    # scale scan vertices
    mesh['vertices'] /= scale
    # rotate scan vertices
    if rotation is not None:
        mesh['vertices'] = np.matmul(rotation, mesh['vertices'].T).T
    # offset scan vertices
    if offset is not None:
        mesh['vertices'] += offset
    # return scan data, centers, scale
    return mesh, {'mcenter': mcenter, 'bcenter': bcenter}, scale

# load scan mesh with texture from pkl
def load_scan_mesh(mesh_fn, rotation=None, offset=None, scale=1.0, normalize_points=False):
    # locate atlas_fn
    atlas_fn = mesh_fn.replace('mesh-', 'atlas-')
    # load scan mesh and atlas data
    mesh_data, atlas_data = load_pickle(mesh_fn), load_pickle(atlas_fn)
    
    if normalize_points:
        mesh_vertices_mean = np.mean(mesh_data['vertices'], axis=0)
        mesh_data['vertices'] -= mesh_vertices_mean
    # load scan uv_coordinate and uv_image as TextureVisuals
    uv_image = Image.fromarray(atlas_data).transpose(method=Image.Transpose.FLIP_TOP_BOTTOM).convert("RGB")
    texture_visual = trimesh.visual.texture.TextureVisuals(uv=mesh_data['uvs'], image=uv_image)
    # pack scan data as trimesh
    scan_trimesh = trimesh.Trimesh(
        vertices=mesh_data['vertices'],
        faces=mesh_data['faces'],
        vertex_normals=mesh_data['normals'],
        visual=texture_visual,
        process=False,
    )
    # pack scan data as mesh
    scan_mesh = {
        'vertices_origin': scan_trimesh.vertices.copy(),
        'vertices': scan_trimesh.vertices.copy(),
        'faces': scan_trimesh.faces,
        'edges': scan_trimesh.edges,
        'colors': scan_trimesh.visual.to_color().vertex_colors,
        'normals': scan_trimesh.vertex_normals,
        'uvs': mesh_data['uvs'],
        'uv_image': np.array(uv_image),
        'uv_path': atlas_fn,
    }
    # preprocess scan mesh: scale, centralize, normalize, rotation, offset
    scan_mesh, center, scale = preprocess_scan_mesh(scan_mesh, mcentral=False, bbox=False, rotation=rotation, offset=offset, scale=scale)
    return scan_trimesh, scan_mesh, center, scale

def mesh_to_pointcloud(mesh, segmentation_labels=None):

    res = {}
    vertices = np.asarray(mesh['vertices']).reshape(-1, 3)
    normals = np.asarray(mesh['normals']).reshape(-1, 3)
    faces = mesh['faces']
    scales = np.ones((vertices.shape[0], 1)) * -6.5
    opacity = np.ones((vertices.shape[0], 1))
    colors = []
    uv_image = np.flip(mesh['uv_image'], axis=0)
    for uv in mesh['uvs']:
        col = uv_image[int(uv[1] * uv_image.shape[0]), int(uv[0] * uv_image.shape[1])]
        colors.append(col)
    
    colors = np.array(colors) / 255.0
    
    points_on_faces = []
    normals_on_faces = []
    scales_on_faces = []
    opacity_on_faces = []
    colors_on_faces = []
    segmentation_labels_on_faces = []
    for face in faces:
        p1, p2, p3 = vertices[face]
        mean_point = (p1 + p2 + p3) / 3
        points_on_faces.append(mean_point)
        mean_normal = (normals[face[0]] + normals[face[1]] + normals[face[2]]) / 3
        normals_on_faces.append(mean_normal)
        scales_on_faces.append(scales[face[0]])
        opacity_on_faces.append(opacity[face[0]])
        mean_color = (colors[face[0]] + colors[face[1]] + colors[face[2]]) / 3
        colors_on_faces.append(mean_color)

    if segmentation_labels is not None:
        # get the most common label for each face
        for face in faces:
            l1, l2, l3 = segmentation_labels[face[0]], segmentation_labels[face[1]], segmentation_labels[face[2]]
            label = int(np.argmax(np.bincount([l1, l2, l3])))
            segmentation_labels_on_faces.append(label)
        
        segmentation_labels = np.concatenate([segmentation_labels, segmentation_labels_on_faces], axis=0)
        res["segmentation_labels"] = segmentation_labels


    
    points_on_faces = np.array(points_on_faces)
    normals_on_faces = np.array(normals_on_faces)
    scales_on_faces = np.array(scales_on_faces)
    opacity_on_faces = np.array(opacity_on_faces)
    colors_on_faces = np.array(colors_on_faces)

    vertices = np.concatenate([vertices, points_on_faces], axis=0)
    normals = np.concatenate([normals, normals_on_faces], axis=0)
    scales = np.concatenate([scales, scales_on_faces], axis=0)
    opacity = np.concatenate([opacity, opacity_on_faces], axis=0)
    colors = np.concatenate([colors, colors_on_faces], axis=0)
    
    res["scales"] = scales
    res["opacity"] = opacity
    res["pointcloud"] = BasicPointCloud(points=vertices, colors=colors, normals=normals)
    return res, segmentation_labels_on_faces


def open_mesh(path, scan_rot=None, offset=None):
    # pkl_obj = open_pickle(path)
    scan_trimesh, scan_mesh, center, scale = load_scan_mesh(path)
    pcd_pack = mesh_to_pointcloud(scan_mesh)
    scales, opacity, pcd = pcd_pack["scales"], pcd_pack["opacity"], pcd_pack["pointcloud"]
    return scales, opacity, pcd


class CameraInfo4dDress(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    mask: np.array
    image_path: str
    image_name: str
    width: int
    height: int


def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))
    
def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)

def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R

# elevation & azimuth to pose (cam2world) matrix
def orbit_camera(elevation, azimuth, radius=10, is_degree=True, target=None, opengl=True):
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T

import cv2

def render_scene(scene, fovy, c2w, renderer):
    camera = pyrender.PerspectiveCamera(yfov=fovy)
    camera_node = scene.add(camera, pose=c2w)

    color, depth = renderer.render(scene)

    scene.remove_node(camera_node)
    
    return Image.fromarray(color.astype(np.uint8))
    
    image = np.concatenate([image, np.ones((image.shape[0], image.shape[1], 1)) * 255], axis=-1)

    mask = np.where(depth > 0, 255, 0).astype(np.uint8)
    
    _image = Image.fromarray(image.astype(np.uint8))
    _mask = Image.fromarray(mask.astype(np.uint8))
    

    alpha_image = Image.new("RGBA", _image.size, (0, 0, 0, 255))
    alpha_image.paste(_image, (0, 0), _mask)
    
    return Image.fromarray(image.astype(np.uint8))

def create_cameras(width, height, cfg):
    azimuths = np.linspace(0, 360, cfg.num_cams, endpoint=False).tolist() * len(cfg.elevations)
    elevations = [[elevation] * cfg.num_cams for elevation in cfg.elevations]
    elevations = [item for sublist in elevations for item in sublist]
    
    extrinsixcs = {}
    
    FovX = cfg.FovX
    FovY = focal2fov(fov2focal(FovX, width), height)
    
    intrinsics = np.asarray([
        [FovX, 0, width // 2],
        [0, FovY, height // 2],
        [0, 0, 1]
    ])

    for idx in range(len(azimuths)):

        c2w = orbit_camera(elevations[idx], azimuths[idx], radius=7.0, is_degree=True, target=None, opengl=True)
        extrinsixcs[idx] = c2w
        
    return {'extrinsixcs': extrinsixcs, 'intrinsics': intrinsics}

def read4dDressInfo(cfg, smpl_params=None, cameras=None) -> SceneInfo:
    if cameras is None:
        width, height = 940, 1280
    else:
        width, height = cameras["intrinsics"][0, 2]*2, cameras["intrinsics"][1, 2]*2
    scan_trimesh, scan_mesh, center, scale = load_scan_mesh(cfg.obj_path, normalize_points=True)
    mesh = pyrender.Mesh.from_trimesh(scan_trimesh, smooth=False)
    scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[1., 1., 1.])

    scene.add(mesh, pose=np.eye(4))

    segmentation_labels_file_name = [x for x in os.listdir(os.path.join(cfg.source_path, "Semantic/labels")) if cfg.id in x][0]
    segmentation_labels = open_pickle(cfg.source_path, "Semantic/labels", segmentation_labels_file_name)["scan_labels"]

    if cfg.init_from_mesh:
        # We init a pointcloud from the mesh
        pcd_pack, segmentation_labels = mesh_to_pointcloud(scan_mesh, segmentation_labels=segmentation_labels)
        scales, opacity, pcd, segmentation_labels = pcd_pack["scales"], pcd_pack["opacity"], pcd_pack["pointcloud"], pcd_pack["segmentation_labels"]
        
        if not os.path.exists(os.path.join(cfg.source_path, "Semantic/labels_with_faces/")):
            os.makedirs(os.path.join(cfg.source_path, "Semantic/labels_with_faces/"))
        
        lwf_filename = f"labels_with_faces_{cfg.id}.pkl"
            
        with open(os.path.join(cfg.source_path, "Semantic/labels_with_faces/", lwf_filename), "wb") as f:
            pickle.dump({"scan_labels": segmentation_labels}, f)
            print(f"Saved labels with faces to {lwf_filename}. Shape: {segmentation_labels.shape}")
    elif cfg.init_from_smpl:
        
        if smpl_params is not None:
            body_pose = smpl_params["body_pose"].reshape(1, -1)
            betas = smpl_params["betas"].reshape(1, -1)
            global_orient = smpl_params["global_orient"].reshape(1, -1)
            transl = smpl_params["transl"].reshape(1, -1)
            model_path = f"smpl/smpl300/SMPL_{smpl_params['gender'].upper()}.pkl"
        else:
            # load smpl from the dataset and use that to create the pointcloud
            smpl = open_pickle(cfg.source_path, "SMPL", f"mesh-{cfg.id}_smpl.pkl")
            body_pose = torch.asarray(smpl["body_pose"]).reshape(1, -1)
            betas = torch.asarray(smpl["betas"]).reshape(1, -1)
            global_orient = torch.asarray(smpl["global_orient"]).reshape(1, -1)
            transl = torch.asarray(smpl["transl"]).reshape(1, -1)
            model_path=f"smpl/smpl300/SMPL_{cfg.gender.upper()}.pkl"
        
        smpl_model = smplx.SMPL(model_path=model_path)
        from smplx.utils import SMPLOutput
        res: SMPLOutput = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient, return_verts=True)
        
        vertices = res.vertices.detach().cpu().numpy()[0]
        mesh_mean = np.mean(vertices, axis=0)
        mean = np.mean(scan_trimesh.vertices, axis=0)
        vertices -= mesh_mean - mean
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
        # pcd_pack, segmentation_labels = mesh_to_pointcloud(mesh=mesh, segmentation_labels=segmentation_labels)
        # scales, opacity, pcd, segmentation_labels = pcd_pack["scales"], pcd_pack["opacity"], pcd_pack["pointcloud"], pcd_pack["segmentation_labels"]
        
        
    else:
        # We init a pointcloud randomly
        num_pts = cfg.num_pts
        r = 0.5
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * r - (r/2)
        shs = np.random.random((num_pts, 3))
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )
        scales, opacity = np.ones((num_pts, 1)) * -6.5, np.ones((num_pts, 1))


    if cameras is None:
        cameras = create_cameras(width, height, cfg)

    cam_infos = []

    r = pyrender.OffscreenRenderer(width, height)

    for idx in range(len(cameras["extrinsixcs"])):

        c2w = cameras["extrinsixcs"][idx]
        intrinsics = cameras["intrinsics"]

        FovX = intrinsics[0, 0]
        
        FovY = intrinsics[1, 1]

        scene.bg_color = [255, 255, 255]
        alpha_image_white_bg = render_scene(scene, FovY, c2w, r)
        scene.bg_color = [0, 0, 0]
        alpha_image_black_bg = render_scene(scene, FovY, c2w, r)

        c2w[:3, 1:3] *= -1
        
        w2c = np.linalg.inv(c2w)

        R = np.transpose(w2c[:3,:3])
        T = w2c[:3, 3]

        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image={"white": alpha_image_white_bg, "black": alpha_image_black_bg},
            image_path="image_path",
            image_name=f"side--capture-f00011.png",
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


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "4D-DRESS" : read4dDressInfo
}