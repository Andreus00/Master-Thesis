import pyrender
import numpy as np
import trimesh
import pickle
from PIL import Image
import torch
from typing import NamedTuple
import math

import sys
sys.path.append('ext/gaussian-splatting')
from scene.gaussian_model import BasicPointCloud


def open_pickle(*arg):
    pickle_file = os.path.join(*arg)
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    return data

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


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

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

    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    
    # add alpha channel
    color = color.astype(np.float32)
    mask = (depth > 0).astype(np.uint8) * 255
    color = np.concatenate([color, mask[..., None]], axis=-1)

    scene.remove_node(camera_node)
    
    return Image.fromarray(color.astype(np.uint8))
    

'''
Prepare an evaluation dataset for avatar reconstruction.

Each sample in the dataset is a dictionary with the following keys:
- 'source_images': a list of 20 source images that will be used to reconstruct the avatar
- 'source_cameras': a list of 20 source cameras corresponding to the source images
- 'source_smpl':
    - 'betas': betas for the source SMPL model
    - 'pose': pose for the source SMPL model
    - 'global_orient': global orientation for the source SMPL model
    - 'translation': translation for the source SMPL model
- 'target_images': a list of 10 randomly sampled target images synthesized from the mesh
- 'target_cameras': a list of 10 target cameras corresponding to the target images
- 'target_smpl':
    - 'betas': betas for the target SMPL model
    - 'pose': pose for the target SMPL model
    - 'global_orient': global orientation for the target SMPL model
    - 'translation': translation for the target SMPL model

The dataset used for evaluation is 4D-Dress.

4D-Dress:
- 00122:
    - Inner:
        - Take1:
            - Capture:
                - 0004:
                    - images:
                        - capture-f00011.png
                        - capture-f00012.png
                        - ...
                    - masks:
                        - mask-f00011.png
                        - mask-f00012.png
                        - ...
                - 0028:
                - 0052:
                - 0076:
            - Meshes_pkl:
                - mesh-f00011.pkl
                - atlas-f00011.pkl
                - mesh-f00012.pkl
                - atlas-f00012.pkl
            - Semantic:
                - labels:
                    - label-f00011.png
                    - label-f00012.png
                    - ...
            - SMPL:
                - mesh-f00011_smpl.pkl
                - mesh-f00011_smpl.ply
                - mesh-f00012_smpl.pkl
                - mesh-f00012_smpl.ply
                - ...
            - SMPLX:
                - mesh-f00011_smplx.pkl
                - mesh-f00011_smplx.ply
                - mesh-f00012_smplx.pkl
                - mesh-f00012_smplx.ply
                - ...
            - basic_info.pkl
        - Take2:
        - Take3:
        - Take4:
        - ...
    - Outer:
- 00123:
- ...
    
'''
import os
import random
import pyrender
import smplx
import tqdm
dataset_path = '4D-DRESS'
output_dataset_path = '4D-DRESS-subset'

smpl_model_male = smplx.SMPL('smpl/smpl300/SMPL_MALE.pkl')
smpl_model_female = smplx.SMPL('smpl/smpl300/SMPL_FEMALE.pkl')

SURFACE_LABEL_TO_INT = {
    'upper': 3,
    'lower': 4,
    'outer': 5,
}

def extract_clothes(mesh, labels):
    # init label_meshes and face_labels
    vertices = mesh['vertices']
    faces = mesh['faces']
    colors = mesh['colors'] if 'colors' in mesh else None
    uvs = mesh['uvs'] if 'uvs' in mesh else None
    label_meshes = dict()
    face_labels = labels[faces]
    # loop over all labels
    for name, nl in SURFACE_LABEL_TO_INT.items():
        # skip empty label
        if np.sum(labels == nl) == 0: continue
        # find label faces: with label vertices == 3
        vertex_label_nl = np.where(labels == nl)[0]
        face_label_nl = np.where(np.sum(face_labels == nl, axis=-1) == 3)[0]
        # find correct indices
        correct_indices = (np.zeros(labels.shape[0]) - 1).astype(int)
        correct_indices[vertex_label_nl] = np.arange(vertex_label_nl.shape[0])
        # extract label_mesh[vertices, faces]
        label_meshes[name] = {'vertices': vertices[vertex_label_nl], 'faces': correct_indices[faces[face_label_nl]]}
        # extract label_mesh colors
        label_meshes[name]['colors'] = colors[vertex_label_nl] if colors is not None else None
        # extract label_mesh uvs
        label_meshes[name]['uvs'] = uvs[vertex_label_nl] if uvs is not None else None
    return label_meshes

def random_pick(subject_path, get_first=False):
    
    trg_takes = [take for take in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, take))]

    if get_first:
        trg_take = sorted(trg_takes)[0]
    else:
        trg_take = random.choice(trg_takes)

    # for each take, choose a random frame
    subject_path = os.path.join(subject_path, trg_take)
    
    # choose a random frame from the take
    frames_take_path = os.path.join(subject_path, 'Capture/0004/images/')
    frames_take = [frame for frame in os.listdir(frames_take_path)]
    if get_first:
        frame = sorted(frames_take)[0][8:14]
    else:
        frame = random.choice(frames_take)[8:14]
    
    # load basic info
    basic_info = open_pickle(subject_path, 'basic_info.pkl')
    scan_frames, scan_rotation = basic_info['scan_frames'], basic_info['rotation']
    
    scan_dir = os.path.join(subject_path, 'Meshes_pkl')
    smpl_dir = os.path.join(subject_path, 'SMPL')
    smplx_dir = os.path.join(subject_path, 'SMPLX')
    label_dir = os.path.join(subject_path, 'Semantic', 'labels')
    cloth_dir = os.path.join(subject_path, 'Semantic', 'clothes')
    
    scan_mesh_fn = os.path.join(scan_dir, 'mesh-{}.pkl'.format(frame))
    smpl_mesh_fn = os.path.join(smpl_dir, 'mesh-{}_smpl.ply'.format(frame))
    smplx_mesh_fn = os.path.join(smplx_dir, 'mesh-{}_smplx.ply'.format(frame))
    scan_label_fn = os.path.join(label_dir, 'label-{}.pkl'.format(frame))
    scan_cloth_fn = os.path.join(cloth_dir, 'cloth-{}.pkl'.format(frame))
    
    scan_mesh = open_pickle(scan_mesh_fn)
    
    smpl_pkl = open_pickle(smpl_mesh_fn.replace('.ply', '.pkl'))

    smpl_pkl['betas'] = torch.tensor(smpl_pkl['betas']).unsqueeze(0)
    smpl_pkl['body_pose'] = torch.tensor(smpl_pkl['body_pose']).unsqueeze(0)
    smpl_pkl['global_orient'] = torch.tensor(smpl_pkl['global_orient']).unsqueeze(0)
    smpl_pkl['transl'] = torch.tensor(smpl_pkl['transl']).unsqueeze(0) 
    
    transl = smpl_pkl['transl'].numpy()
    scan_mesh['vertices'] -= transl
    smpl_pkl['transl'] = torch.tensor([0, 0, 0]).unsqueeze(0)
    
    scan_mesh['uv_path'] = scan_mesh_fn.replace('mesh-f', 'atlas-f')
    atlas_data = open_pickle(scan_mesh['uv_path'])
    uv_image = Image.fromarray(atlas_data).transpose(method=Image.Transpose.FLIP_TOP_BOTTOM).convert("RGB")
    texture_visual = trimesh.visual.texture.TextureVisuals(uv=scan_mesh['uvs'], image=uv_image)
    # pack scan data as trimesh
    scan_trimesh = trimesh.Trimesh(
        vertices=scan_mesh['vertices'],
        faces=scan_mesh['faces'],
        vertex_normals=scan_mesh['normals'],
        visual=texture_visual,
        process=False,
    )
    scan_mesh['colors'] = scan_trimesh.visual.to_color().vertex_colors
    
    labels = open_pickle(scan_label_fn)
    
    clothes = extract_clothes(scan_mesh, labels['scan_labels'])
    
    # show each cloth
    clothes_trimesh = dict()
    for cloth_name, cloth_mesh in clothes.items():
        uvs = scan_mesh['uvs'][labels['scan_labels'] == SURFACE_LABEL_TO_INT[cloth_name]]
        texture_visual = trimesh.visual.texture.TextureVisuals(uv=uvs, image=uv_image)
        trim = trimesh.Trimesh(
            vertices=cloth_mesh['vertices'],
            faces=cloth_mesh['faces'],
            vertex_colors=cloth_mesh['colors'],
            visual=texture_visual,
            process=False,
        )
        clothes_trimesh[cloth_name] = trim
        
    # load smplx mesh
    smplx_trimesh = trimesh.load_mesh(smplx_mesh_fn)
    smplx_trimesh.vertices -= transl
    smplx_trimesh.colors = np.ones_like(smplx_trimesh.vertices) * 0.7
    
    return trg_take, frame, {'scan_mesh': scan_mesh, 'scan_mesh_fn': scan_mesh_fn, 'scan_trimesh': scan_trimesh, 'clothes_trimesh': clothes_trimesh, 'smpl_pkl': smpl_pkl, 'smplx_trimesh': smplx_trimesh, 'basic_info': basic_info, 'labels': labels}

def camera_parameters(num_cams=20, elevations=[0, -40], width=940, height=1280, fovx=0.25):
    cam_infos = {
        'extrinsics': {},
    }
    
    fovy = focal2fov(fov2focal(fovx, width), height)
    
    azimuths = np.linspace(0, 360, num_cams//len(elevations), endpoint=False).tolist() * len(elevations)
    elevations = [[elevation] * (num_cams//len(elevations)) for elevation in elevations]
    elevations = [item for sublist in elevations for item in sublist]
    
    cam_infos['intrinsics'] = np.asarray([[fovx, 0, width//2],
                                         [0, fovy, height//2],
                                         [0, 0, 1],
                                         ], dtype=np.float32)
    
    for i, (elevation, azimuth) in enumerate(zip(elevations, azimuths)):
        c2w = orbit_camera(elevation, azimuth, radius=7)
        cam_infos['extrinsics'][i] = c2w
    
    return cam_infos
    

def orbital_render(scan_mesh, num_cams=20, elevations=[0, -40], width=940, height=1280, cam_infos=None, bw_bg=True):
    
    scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[1., 1., 1.])

    scene.add(pyrender.Mesh.from_trimesh(scan_mesh), pose=np.eye(4))
    
    r = pyrender.OffscreenRenderer(width, height)
    
    if cam_infos is None:
        cam_infos = camera_parameters(num_cams, elevations, width, height)
    
    renders = {}
    
    for i in range(num_cams):
        c2w = cam_infos['extrinsics'][i]
        fovy = cam_infos['intrinsics'][1, 1]
        
        scene.bg_color = [0, 0, 0]
        image_black = render_scene(scene, fovy, c2w, r)
        if bw_bg:
            scene.bg_color = [255, 255, 255]
            image_white = render_scene(scene, fovy, c2w, r)
            
            renders[i] = {'image_white': image_white, 'image_black': image_black}
        else:
            renders[i] = image_black
    
    return renders


def sample_subject(subject, subject_path, outfit, get_first=False):
    # pick a random take and frame
    take, frame, src_pick = random_pick(subject_path, get_first=get_first)
    
    # render the scan mesh
    renders = orbital_render(src_pick['scan_trimesh'], bw_bg=False)
    
    # render the clothes and save the meshes
    clothes_renders = {}
    cloth_mesh_paths = {}
    for cloth_name, cloth_mesh in src_pick['clothes_trimesh'].items():
        path = os.path.join(output_dataset_path, 'meshes', cloth_name)
        if not os.path.exists(path):
            os.makedirs(path)
        cloth_mesh_path = os.path.join(path, '{}_{}_{}_{}_{}.ply'.format(subject, outfit, take, frame, cloth_name))
        cloth_mesh.export(cloth_mesh_path)
        cloth_mesh_paths[cloth_name] = cloth_mesh_path
        clothes_renders[cloth_name] = orbital_render(cloth_mesh, num_cams=4, elevations=[0], bw_bg=False)
        


    # save renders
    renders_paths = []
    base_path = os.path.join(output_dataset_path, 'images', 'avatar')
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for cid, render in renders.items():
        renders_paths.append(os.path.join(output_dataset_path, 'images/avatar', '{}_{}_{}_{}_{}.png'.format(subject, outfit, take, frame, cid)))
        render.save(renders_paths[-1])
    
    renders_path_clothes = {}
    for cloth_name, cloth_renders in clothes_renders.items():
        renders_path_clothes[cloth_name] = []
        for cid, render in cloth_renders.items():
            clothes_path = os.path.join(output_dataset_path, 'images', cloth_name)
            if not os.path.exists(clothes_path):
                os.makedirs(clothes_path)
            path = os.path.join(clothes_path, '{}_{}_{}_{}_{}_{}.png'.format(subject, outfit, take, frame, cid, cloth_name))
            renders_path_clothes[cloth_name].append(path)
            render.save(path)
                                       
    render_smplx = orbital_render(src_pick['smplx_trimesh'], bw_bg=False)
    
    # save smplx render
    renders_path_smplx = []
    smplx_path = os.path.join(output_dataset_path, 'images', 'smplx')
    if not os.path.exists(smplx_path):
        os.makedirs(smplx_path)
    for cid, render in render_smplx.items():
        path = os.path.join(smplx_path, '{}_{}_{}_{}_{}.png'.format(subject, outfit, take, frame, cid))
        renders_path_smplx.append(path)
        render.save(path)
        
    
    sample = {
        'subject': subject,
        'outfit': outfit,
        'take': take,
        'frame': frame,
        'imgs': renders_paths,
        'imgs_clothes': renders_path_clothes,
        'cloth_mesh_paths': cloth_mesh_paths,
        'basic_info': src_pick['basic_info'],
        'labels': src_pick['labels'],
        'smpl_renders': renders_path_smplx,
        "smpl": {
                "betas": src_pick['smpl_pkl']['betas'].numpy(),
                "body_pose": src_pick['smpl_pkl']['body_pose'].numpy(),
                "global_orient": src_pick['smpl_pkl']['global_orient'].numpy(),
                "transl": src_pick['smpl_pkl']['transl'].numpy(),
                },
    }
    
    return sample
    
    
if __name__ == '__main__':
    
    if not os.path.exists(output_dataset_path):
        os.makedirs(output_dataset_path)
        
    if not os.path.exists(os.path.join(output_dataset_path, 'images')):
        os.makedirs(os.path.join(output_dataset_path, 'images'))
    
    cam_info = camera_parameters()
    
    # create a file in the output dataset and save the camera parameters
    with open(os.path.join(output_dataset_path, 'cameras.pkl'), 'wb') as f:
        pickle.dump(cam_info, f)
    
    dataset = {}
    
    # subjects = os.listdir(dataset_path)
    samples = [x for x in sorted(os.listdir(dataset_path)) if os.path.isdir(os.path.join(dataset_path, x))]
    
    print(f'Samples: {samples}')
    print(f'Found {len(samples)} subjects')
    
    num_samples_per_subject = 10
    
    pbar = tqdm.tqdm(samples)
    
    for sample in pbar:
        
        sampled = 0
        
        while sampled < num_samples_per_subject:
            try:
                pbar.set_description(f'Sampled {sampled}/{num_samples_per_subject} from {sample}')
                get_first = sampled == 0
                
                outfit = sample[-5:]
                    
                subject = sample[:-5]
                
                subject_path = os.path.join(dataset_path, sample, outfit)
                
                src_sample = sample_subject(subject, subject_path, outfit, get_first=get_first)
                
                subject_id = f'{subject}_{outfit}'
                
                sample_id = f'{src_sample["take"]}_{src_sample["frame"]}'
                
                if not subject_id in dataset:
                    dataset[subject_id] = {sample_id: src_sample}
                else:
                    dataset[subject_id][sample_id] = src_sample
                
                sampled += 1
            except FileNotFoundError as e:
                print(e)
                continue
            
    with open(os.path.join(output_dataset_path, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)