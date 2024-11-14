import trimesh
import numpy as np
import cv2
import copy
from PIL import Image

class SMPLMesh:

    def __init__(self, points, triangles, width, height) -> None:
        self.img_width = width
        self.img_height = height
        self.pcd = None
        self.background = None
        
        self.mesh = trimesh.Trimesh(vertices=points, faces=triangles, process=False)