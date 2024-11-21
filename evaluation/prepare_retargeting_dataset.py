'''
This script samples the Inner and Outer subjects to find coulpes of subjects that are similar 
in terms of their pose.
'''

import argparse
import os
import pickle
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import smplx
import random
import trimesh
import pyrender

import time

def read_dataset_folders(dataset_folder):
    dataset = sorted([folder for folder in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, folder))])
    subjects = {}
    for subject in dataset:
        *subject_id, garment = subject.split('_')
        subject_id = '_'.join(subject_id)
        if subject_id not in subjects:
            subjects[subject_id] = dict()
            subjects[subject_id][garment] = sorted(os.listdir(os.path.join(dataset_folder, subject)))
        else:
            subjects[subject_id][garment] = sorted(os.listdir(os.path.join(dataset_folder, subject)))
    return subjects

def read_smpl_params(path):
    smpl_params = dict()    # dict that maps frame name to smpl parameters
    takes = sorted([folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))])
    for take in takes:
        smpl_params[take] = dict()
        smpl_folder = os.path.join(path, take, 'SMPL')
        smpl_files = sorted([fp for fp in os.listdir(smpl_folder) if fp.endswith('.pkl')])
        for smpl_file in smpl_files:
            with open(os.path.join(smpl_folder, smpl_file), 'rb') as f:
                smpl_data = pickle.load(f)
                pose = smpl_data['body_pose']
                smpl_params[take][smpl_file.strip('.pkl')] = pose

    print(f'smpl_params_keys: {smpl_params.keys()}')
    return smpl_params


def compute_similarity(smpl_params1, smpl_params2):
    '''
    Given two matrices, where each row is a smpl pose parameter, we compute the similarity
    between the two matrices and find the closest sample for each row in the first matrix.
    '''
    smpl_params1 = torch.tensor(smpl_params1).to(device)
    smpl_params2 = torch.tensor(smpl_params2).to(device)
    
    # compute the similarity between the two matrices
    # smpl_params1: num_samples x 72
    # smpl_params2: num_samples x 72
    # similarity: num_samples x num_samples
    mult = torch.matmul(smpl_params1, smpl_params2.T)
    norm1 = torch.norm(smpl_params1, dim=1).unsqueeze(1)
    norm2 = torch.norm(smpl_params2, dim=1).unsqueeze(0)
    similarity = mult / (norm1 * norm2)
    
    # find the closest sample for each row in the first matrix
    # closest_samples: num_samples
    _, closest_samples = torch.max(similarity, dim=1)
    
    return closest_samples.cpu().numpy()


def visualize(vertices, faces, viewer, old_node=None):
    nodes = []
    
    viewer.render_lock.acquire()
    for i in range(len(vertices)):
        mesh = trimesh.Trimesh(vertices[i], faces[i])
        
        # update the viewer
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
        if old_node is not None:
            viewer.scene.remove_node(old_node[i])
        nodes.append(viewer.scene.add(pyrender_mesh))
        
    viewer.render_lock.release()
    time.sleep(0.3)
    
    return nodes
    


def find_similar_samples(subjects, dataset_folder):
    # for each subject, we have to find the similar samples for the inner and outer takes
    smpl = smplx.SMPL('./smpl/smpl/SMPL_m.pkl').to(device)
    
    scene = pyrender.Scene()
    viewer = pyrender.Viewer(scene, run_in_thread=True)
    
    current_nodes = current_node_outer = None
    
    for subject_id, garments in subjects.items():
        inner_path = os.path.join(dataset_folder, f'{subject_id}_Inner', 'Inner')
        inner_samples = read_smpl_params(inner_path)
        outer_path = os.path.join(dataset_folder, f'{subject_id}_Outer', 'Outer')
        outer_samples = read_smpl_params(outer_path)
        
        # for each sample in inner_samples, find the closest sample in outer_samples
        inner_sample_poses = []
        inner_keys = list()
        for take, poses in inner_samples.items():
            for key, pose in poses.items():
                inner_sample_poses.append(pose)
                inner_keys.append((take, key))
        outer_sample_poses = []
        outer_keys = list()
        for take, poses in outer_samples.items():
            for key, pose in poses.items():
                outer_sample_poses.append(pose)
                outer_keys.append((take, key))
        
        closest_samples = compute_similarity(inner_sample_poses, outer_sample_poses)
        
        # test by initializing the smpl model and visualizing the poses
        random_samples = np.random.randint(0, len(inner_samples), 5)
        
        for i in range(len(inner_sample_poses)):
            inner_sample = inner_sample_poses[i]
            outer_sample = outer_sample_poses[closest_samples[i]]
            
            inner_sample = torch.tensor(inner_sample).to(device)
            outer_sample = torch.tensor(outer_sample).to(device)
            
            inner_pose = inner_sample
            outer_pose = outer_sample
            
            inner_pose = inner_pose.view(1, -1)
            outer_pose = outer_pose.view(1, -1)
            
            inner_pose = inner_pose.view(1, -1)
            outer_pose = outer_pose.view(1, -1)
            
            inner_output = smpl(body_pose=inner_pose)
            outer_output = smpl(body_pose=outer_pose)
            
            inner_output = inner_output.vertices.detach().cpu().squeeze(0).numpy()
            outer_output = outer_output.vertices.detach().cpu().squeeze(0).numpy()
            
            # visualize the two poses
            print(f'inneroutput: {inner_output.shape} - faces: {smpl.faces.shape}')
            
            current_nodes = visualize((inner_output, outer_output), (smpl.faces, smpl.faces), viewer, current_nodes)
            
        # save the results
        subjects[subject_id]['Inner'] = inner_samples
        subjects[subject_id]['Outer'] = outer_samples
        
    return subjects
    
    
def main(cfg):
    # read the dataset folder
    dataset_folder = cfg.dataset
    
    subjects = read_dataset_folders(dataset_folder)
    
    # now we have the subjects.
    # for each one of them, we have to find similar samples for the inner and the outer takes.
    
    subjects = find_similar_samples(subjects, dataset_folder)
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    main(args)