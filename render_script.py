import os
import numpy as np
import pickle
import pyrender
import trimesh
from PIL import Image
import torch


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

def render_scene(scene, fovy, c2w, renderer):
    camera = pyrender.PerspectiveCamera(yfov=fovy)
    camera_node = scene.add(camera, pose=c2w)

    color, depth = renderer.render(scene)

    scene.remove_node(camera_node)
    
    return Image.fromarray(color.astype(np.uint8))

import math

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
        target[1] = 0.1
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T

def create_camera(width, height, azimuth=0, elevation=0):
    
    extrinsixcs = {}
    
    FovX = 0.25
    FovY = focal2fov(fov2focal(FovX, width), height)
    
    intrinsics = np.asarray([
        [FovX, 0, width // 2],
        [0, FovY, height // 2],
        [0, 0, 1]
    ])

    radius = 7
    c2w = orbit_camera(elevation, azimuth, radius=radius, is_degree=True, target=None, opengl=True)
    
        
    return FovY, c2w, radius

def render(path, azimuth, elevation=0):
    scan_trimesh, scan_mesh, center, scale = load_scan_mesh(path, normalize_points=True)
    
    mesh = pyrender.Mesh.from_trimesh(scan_trimesh, smooth=False)
    scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[1., 1., 1.])

    scene.add(mesh, pose=np.eye(4))
    
    FovY, c2w, r = create_camera(940, 1280, azimuth, elevation)
    
    return render_scene(scene, FovY, c2w, pyrender.OffscreenRenderer(940, 1280))


if False:   # render a single view for each subject
    subjs = ['00122', '00123', '00127', '00147']
    azimuth_fix = [90, 70, -20, -20]
    outfits = ['Inner', 'Outer']


    data_path_template = "4D-DRESS/{}/{}"

    for idx, sub in enumerate(subjs):
        azimuth = azimuth_fix[idx]
        for outfit in outfits:
            data_path = data_path_template.format(sub, outfit)
            
            # list takes and take the first one
            take = sorted([x for x in os.listdir(data_path) if x.startswith('Take')])[0]
            
            meshes_path = os.path.join(data_path, take, "Meshes_pkl")
            
            atlas_name = sorted(os.listdir(meshes_path))[0]
            mesh_name = f"mesh{atlas_name[5:]}"
            
            mesh_path = os.path.join(meshes_path, mesh_name)
            
            img = render(mesh_path, azimuth)
            
            # save image
            if not os.path.exists("./temp"):
                os.makedirs("./temp")
            img.save(f"./temp/{sub}_{outfit}.png")

elif False:  # render 360 degree view
    subj = '00122'
    azimuths = np.arange(-180, 180, 10)
    elevations = [0, -40]
    outfit = 'Inner'


    data_path_template = "4D-DRESS/{}/{}"
    for idx, azimuth in enumerate(azimuths):
        for elevation in elevations:
            data_path = data_path_template.format(subj, outfit)
            
            # list takes and take the first one
            take = sorted([x for x in os.listdir(data_path) if x.startswith('Take')])[0]
            
            meshes_path = os.path.join(data_path, take, "Meshes_pkl")
            
            atlas_name = sorted(os.listdir(meshes_path))[0]
            mesh_name = f"mesh{atlas_name[5:]}"
            
            mesh_path = os.path.join(meshes_path, mesh_name)
            
            img = render(mesh_path, azimuth, elevation)
            
            # save image
            if not os.path.exists(f"./temp/{subj}"):
                os.makedirs(f"./temp/{subj}")
            img.save(f"./temp/{subj}/{idx}_{elevation}.png")
            
else:   # render point cloud and a smpl mesh registered by NICP
    points_path = "output/00122_Inner_Take2/f00011/NICP/aligned.ply"
    smpl_path = "output/00122_Inner_Take2/f00011/NICP/out_ss_cham_0.ply"
    
    pp = trimesh.load(points_path, process=False)
    
    points = pp.vertices
    # downsample
    random_indices = np.random.choice(points.shape[0], 10000, replace=False)
    points = points[random_indices]
    # #6D84FC
    COLOR = np.array([252, 109, 109], dtype=np.uint8) / 255
    points = trimesh.points.PointCloud(points, colors=COLOR)
    
    # load smpl mesh
    smplmesh = trimesh.load(smpl_path, process=False)
    
    # apply scale to the smpl mesh
    smplmesh.vertices *= 0.53
    
    # move the smpl mesh toward the camera and down
    smplmesh.vertices[:, 2] += 0.11
    smplmesh.vertices[:, 1] += -0.01
    
    # create a scene with the point cloud and the smpl mesh
    scene = trimesh.Scene([points, smplmesh])
    scene.show()
            