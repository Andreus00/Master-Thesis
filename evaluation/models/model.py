'''
Here we define the model for the evaluation.

The model must have a function that takes the input data to reconstruct the avatar,
and a function that given a pose and some cameras returns the images of the posed avatar.
It also has a function to retarget the clothes from another model.
'''

from typing import List
from __future__ import annotations

class Model:
    def __init__(self):
        pass

    def reconstruct(self, input_data):
        raise NotImplementedError()

    def pose(self, pose):
        raise NotImplementedError()

    def render(self, cameras):
        raise NotImplementedError()
    
    def retarget(self, source: Model, clothes: List[int]):
        raise NotImplementedError()