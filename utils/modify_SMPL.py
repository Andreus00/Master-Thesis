from NICP.utils_cop.SMPL import SMPL
import numpy as np
import trimesh
from trimesh import remesh
import torch
import matplotlib.pyplot as plt


smpl: SMPL = SMPL(model_path='NICP/neutral_smpl_with_cocoplus_reg.txt', base_path="NICP", obj_saveable=True)

beta=torch.zeros((1, 10))
theta=torch.rand(1, 72) * 0.5 - 0.05

v, j, r = smpl(theta=theta, beta=beta, get_skin=True)
# trimesh.Trimesh(vertices=v[0].numpy(), faces=smpl.faces, process=False).show()

vertices, faces, attr = remesh.subdivide(vertices=np.asarray(smpl.v_template).reshape(-1, 3), faces=np.asarray(smpl.faces).reshape(-1, 3),
                face_index=None,
                vertex_attributes={})

vetices = torch.tensor(vertices).reshape(-1, 3)


smpl.v_template = torch.asarray(vertices)
smpl.faces = torch.asarray(faces)

# now, I have to modify all the matrices of smpl to match the new vertices
# To do so, for each new vertex, I have to find all the neighboring vertices, and
# add an entry to the matrices by linearly interpolating the values of the lines 
# corresponding to the neighboring vertices


# First create a dictionary with the neighbors of each new vertex
new_verts_neighbors = {}

for new_face in faces:
    for vert in new_face:
        # if it is a vertex of the original model, skip
        if vert < 6890:
            continue
        
        # it is a new vertex: I add its neighbors
        if vert not in new_verts_neighbors:
            new_verts_neighbors[vert] = set()
        for _vert in new_face:
            if vert != _vert and  _vert < 6890:
                new_verts_neighbors[vert].add(_vert)
                
# prepare the new matrices

new_J_regressor = torch.zeros((len(vertices), 24))
new_J_regressor[:6890] = smpl.J_regressor

new_joint_regressor = torch.zeros((len(vertices), 19))
new_joint_regressor[:6890] = smpl.joint_regressor

new_weight = torch.zeros((len(vertices), 24))
new_weight[:6890] = smpl.weight[0]

new_shapedirs = torch.zeros(10, len(vertices) * 3)
new_shapedirs[:, :6890 * 3] = smpl.shapedirs
new_shapedirs = new_shapedirs.reshape(10, len(vertices), 3)

new_shapedirs_300 = torch.zeros(300, len(vertices) * 3)
new_shapedirs_300[:, :6890 * 3] = smpl.shapedirs_300
new_shapedirs_300 = new_shapedirs_300.reshape(300, len(vertices), 3)

new_posedirs = torch.zeros(207, len(vertices) * 3)
new_posedirs[:, :6890 * 3] = smpl.posedirs
new_posedirs = new_posedirs.reshape(207, len(vertices), 3)


for vert, neighbors in new_verts_neighbors.items():
    neighbors = list(neighbors)
    point = vertices[vert]
    neighbors_verts = vertices[neighbors]
    
    # compute the weights of each neighbor based on the distance
    distances = np.linalg.norm(neighbors_verts - point, axis=1)
    weights = (1 - np.array(distances) / np.sum(distances)).reshape(-1, 1)
    
    # Interpolate the values of the neighbors for each matrix
    # J_regressor_i = torch.sum(new_J_regressor[neighbors] * weights, dim=0)
    # new_J_regressor[vert] = J_regressor_i
    joint_regressor_i = torch.sum(new_joint_regressor[neighbors] * weights, dim=0)
    new_joint_regressor[vert] = joint_regressor_i
    weight_i = torch.sum(new_weight[neighbors] * weights, dim=0)
    new_weight[vert] = weight_i
    
    
    # shapedirs_i = torch.sum([new_shapedirs[x] * weights[i] for i, x in enumerate(neighbors)], dim=1)
    shapedirs_i = torch.sum(new_shapedirs[:, neighbors] * weights[None, ...], dim=1)
    new_shapedirs[:, vert, :] = shapedirs_i
    shapedirs_300_i = torch.sum(new_shapedirs_300[:, neighbors] * weights[None, ...], dim=1)
    new_shapedirs_300[:, vert, :] = shapedirs_300_i
    posedirs_i = torch.sum(new_posedirs[:, neighbors] * weights[None, ...], dim=1)
    new_posedirs[:, vert, :] = posedirs_i
    

# update the smpl object
smpl.J_regressor = new_J_regressor
smpl.joint_regressor = new_joint_regressor
smpl.weight = new_weight.unsqueeze(0)
smpl.shapedirs = new_shapedirs.reshape(10, -1)
smpl.shapedirs_300 = new_shapedirs_300.reshape(300, -1)
smpl.posedirs = new_posedirs.reshape(207, -1)
smpl.size = (len(vertices), 3)

# now test SMPL

verts, joints, Rs = smpl(theta=theta, beta=beta, get_skin=True)

# trimesh.Trimesh(vertices=verts[0].numpy(), faces=faces, process=False).show()

# now save the new model
import json

with open("NICP/neutral_smpl_with_cocoplus_reg.txt", 'r') as reader:
            model = json.load(reader)
# print("MODEL 1")
# print([(k, np.asarray(v).shape) for k,v in model.items()])
model['v_template'] = smpl.v_template.tolist()
model['J_regressor'] = smpl.J_regressor.tolist()
model['cocoplus_regressor'] = smpl.joint_regressor.tolist()
model['weights'] = smpl.weight.squeeze().tolist()
model['shapedirs'] = smpl.shapedirs.T.reshape(-1, 3, 10).tolist()
model['shapedirs_300'] = smpl.shapedirs_300.T.reshape(-1, 3, 300).tolist() #.reshape(-1, 300).T.tolist()
model['posedirs'] = smpl.posedirs.T.reshape(-1, 3, 207).tolist()
model['f'] = smpl.faces.tolist()

# print("\nmodel 2")
# print([(k, np.asarray(v).shape) for k,v in model.items()])
# exit()

with open("NICP/neutral_smpl_with_cocoplus_reg_augmented.txt", 'w') as writer:
    json.dump(model, writer)
