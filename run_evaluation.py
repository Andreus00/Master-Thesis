'''
Run a reconstruction evaluation on a model
'''

import argparse
import os
import pickle
import json
import numpy as np
from PIL import Image
import torch
import sys
from evaluation.metric_utils.metrics import Metrics
import tqdm

from evaluation.models import models


def load_dataset(path):
    cameras_path = os.path.join(path, 'cameras.pkl')
    with open(cameras_path, 'rb') as f:
        cameras = pickle.load(f)
    
    dataset_path = os.path.join(path, 'dataset.pkl')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    return cameras, dataset

def prepare_batch(batch):
    '''
    Prepare a batch for the model
    '''
    def load_imgs(imgs_paths, format='RGBA'):
        imgs = []
        for img_path in imgs_paths:
            img = torch.asarray(np.asarray(Image.open(img_path).convert(format)).copy(), dtype=torch.float32)/255
            imgs.append(img)
        return torch.stack(imgs)
    
    source = batch['source']
    source['imgs_white'] = load_imgs(source['imgs_white'])
    source['imgs_black'] = load_imgs(source['imgs_black'])
    target = batch['target']
    target['imgs_white'] = load_imgs(target['imgs_white'], format="RGB")
    target['imgs_black'] = load_imgs(target['imgs_black'], format="RGB")
    return source, target

def clone_cameras(cameras):
    return {
        'extrinsics': {k: v.copy() for k, v in cameras['extrinsics'].items()},
        'intrinsics': cameras['intrinsics'].copy(),
    }

def main(args):
    dataset_path = args.dataset
    
    metrics = Metrics()
    
    cameras, dataset = load_dataset(dataset_path)
    
    ssim = 0
    psnr = 0
    lpips = 0
    
    pbar = tqdm.tqdm(dataset.items())
    
    for i, data in pbar:
        source_data, target_data = prepare_batch(data)
        
        # clone cameras
        train_cameras = clone_cameras(cameras)
        
        input_data = {
            'source': source_data,
            'cameras': train_cameras,
        }
        model = models[args.model]()
        
        model.reconstruct(input_data)
        
        target_pose = torch.asarray(target_data['smpl']['body_pose'])
        target_global_orient = torch.asarray(target_data['smpl']['global_orient'])
        target_full_pose = torch.cat([target_global_orient, target_pose], dim=1)
        model.pose(target_full_pose)
        
        test_cameras = clone_cameras(cameras)
        renders: torch.Tensor = model.render(test_cameras).cpu().detach()
        
        target_data = target_data['imgs_white'].cpu().detach().permute(0, 3, 1, 2)
        
        val = metrics.evaluate(renders, target_data)
        out_folder = f'eval_output/{args.dataset}/{args.model}/{i}'
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        for idx, (img_1, img_2) in enumerate(zip(renders, target_data)):
            img_1 = img_1.permute(1, 2, 0).numpy()
            img_2 = img_2.permute(1, 2, 0).numpy()
            
            img_1_name = f'{out_folder}/img_1_{idx}.png'
            img_2_name = f'{out_folder}/img_2_{idx}.png'
        
            Image.fromarray((img_1*255).astype(np.uint8)).save(img_1_name)
            Image.fromarray((img_2*255).astype(np.uint8)).save(img_2_name)
        
        ssim += val[0]
        psnr += val[1]
        lpips += val[2]
    
    ssim /= len(dataset)
    psnr /= len(dataset)
    lpips /= len(dataset)
    
    print(f'SSIM: {ssim}')
    print(f'PSNR: {psnr}')
    print(f'LPIPS: {lpips}')
    
    with open(os.path.join(dataset_path, f'{str(type(model).__name__)}_metrics.json'), 'w') as f:
        json.dump({str(type(model).__name__): {
            'ssim': ssim.item(),
            'psnr': psnr.item(),
            # 'lpips': lpips.item(),
        }}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline', help='Model name [baseline]')
    parser.add_argument('--dataset', type=str, default='4D-DRESS-reconstruction-evaluation', help='Path to dataset')
    args = parser.parse_args(sys.argv[1:])
    main(args)
    