import pickle
import argparse
import os
import numpy as np
import PIL.Image as Image

DATASET_DIR = "./4D-DRESS"

class DataSample:

    def __init__(self, image, mask, intrinsics, extrinsics) -> None:
        self.image: Image.Image = image
        self.mask: Image.Image = mask

        self.camera_intrinsics = intrinsics
        self.camera_extrinsics = extrinsics


def open_pickle(*arg):
    pickle_file = os.path.join(*arg)
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    return data

def open_image(*arg):
    image_file = os.path.join(*arg)
    image = Image.open(image_file)
    return image

def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    '''
    Apply a mask to an image
    '''
    black_base = Image.new('RGB', image.size, (0, 0, 0))
    return Image.composite(image, black_base, mask)


def load_data(dataset_dir, subj, outfit, seq):
    images_dir = os.path.join(dataset_dir, subj, outfit, seq)

    basic_info = open_pickle(images_dir, 'basic_info.pkl')
    scan_frames = basic_info['scan_frames']
    scan_rot = basic_info['rotation']
    cameras = open_pickle(images_dir, "Capture", "cameras.pkl")
    
    # For each camera, read the image, the mask and the camera
    data_samples = []
    for key, camera in cameras.items():
        first_el_frame = open_image(images_dir, "Capture", key, "images", f"capture-f{scan_frames[0]}.png")
        first_el_mask = open_image(images_dir, "Capture", key, "masks", f"mask-f{scan_frames[0]}.png")
        
        camera_intrinsics = camera['intrinsics']
        camera_extrinsics = camera['extrinsics']
        print(camera_intrinsics.shape)
        print(camera_intrinsics)
        print()
        print(camera_extrinsics.shape)
        print(camera_extrinsics)

        data_samples.append(DataSample(first_el_frame, first_el_mask, camera_intrinsics, camera_extrinsics))
        
    return data_samples




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', default='00122', help='subj name')
    parser.add_argument('--outfit', default='Inner', help='outfit name')
    parser.add_argument('--seq', default='Take2', help='seq name')
    args = parser.parse_args()

    training_data = load_data(DATASET_DIR, args.subj, args.outfit, args.seq)

