'''
Here we define the model for the evaluation.

The model must have a function that takes the input data to reconstruct the avatar,
and a function that given a pose and some cameras returns the images of the posed avatar.
'''
from .model import Model
import os
import PIL.Image as Image
import numpy as np
import torch

class HaveFun(Model):
    def __init__(self):
        self.images = None
        self.current_image = None

    def reconstruct(self, input_data):
        src = input_data['source']
        name = "_".join([src['subject'], src['outfit'], src['take'], src['frame']])
        base_path = "evaluation/models/HaveFun/out/4D-Dress"
        
        switch = {
            '00123_Inner_Take7_f00034': "001",
            '00147_Inner_Take3_f00100': "002",
            '00127_Inner_Take10_f00148': "003",
            '00122_Inner_Take2_f00115': "004",
            '00127_Outer_Take14_f00094': "005",
            '00123_Outer_Take8_f00138': "006",
            '00123_Outer_Take11_f00114': "007",
            '00123_Outer_Take13_f00053': "008",
            '00123_Outer_Take9_f00054': "009",
            '00147_Outer_Take16_f00118': "010",
        }
        
        results_path = f"evaluation/models/HaveFun/out/4D-Dress/{switch[name]}/camera_rotate_tpose_20view/results_deform_smplx"
        
        results = sorted([os.path.join(results_path, x) for x in os.listdir(results_path) if x.startswith("aeval") and x.endswith("rgb.png")])
        
        def load_image(path):
            return (torch.asarray(np.asarray(Image.open(path).convert('RGB'))).cpu().detach().permute(2, 0, 1)[:, :, 2:-2] / 255.0).float()
        
        self.images = torch.stack([load_image(x) for x in results], dim=0)


    def pose(self, pose):
        pass

    def render(self, cameras):
        return self.images