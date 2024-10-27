from arguments import ParamGroup
import os
from argparse import ArgumentParser, Namespace
import sys

class ModelParams4dDress(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.subj = "00122"
        self.outfit = "Outer"
        self.seq = "Take9"
        self.gender = "neutral"
        self.eval = False
        self.num_pts = 10_000
        self.obj_path = ""
        self.id: str = ""
        self.init_from_mesh = False
        self.init_from_smpl = True
        self.num_cams = 10   # 10
        self.FovX = 0.20
        self.elevations = [0, -40]
        self.smpl_params = None
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g
    
its = 2000
# decomment this for small anisotropic gaussian splats
class OptimizationParams4dDress(ParamGroup):
    def __init__(self, parser):
        self.iterations = its # 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016 # * 4
        self.position_lr_delay_mult = 0.5
        self.position_lr_max_steps = its
        self.block_position = False  # this overwrites the position_lr parameters when set to True
        self.feature_lr = 0.0025
        self.opacity_lr = 0.0 # 0.05    # we don't optimize opacity as in GaussianAvatar paper
        self.scaling_lr = 0.0025
        self.rotation_lr = 0.001
        self.percent_dense = 1.0   # 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 100_000_000   # we don't want to reset opacity
        self.densify_from_iter = 200
        self.densify_until_iter = its - 1
        self.densify_grad_threshold = 0.00002 # 0.0002
        self.camera_extent_multiplier = 0.5
        self.size_threshold = 0.25
        self.block_densification = False # this overwrites the densification parameters when set to True
        self.random_background = True
        self.use_isotropic_gaussians = False
        super().__init__(parser, "Optimization Parameters")

# decomment this for isotropic gaussian splats
# class OptimizationParams4dDress(ParamGroup):
#     def __init__(self, parser):
#         self.iterations = its # 30_000
#         self.position_lr_init = 0.00016
#         self.position_lr_final = 0.0000016 # * 4
#         self.position_lr_delay_mult = 0.5
#         self.position_lr_max_steps = its
#         self.block_position = False  # this overwrites the position_lr parameters when set to True
#         self.feature_lr = 0.0025
#         self.opacity_lr = 0.0 # 0.05    # we don't optimize opacity as in GaussianAvatar paper
#         self.scaling_lr = 0.0025
#         self.rotation_lr = 0.001
#         self.percent_dense = 1.0   # 0.01
#         self.lambda_dssim = 0.2
#         self.densification_interval = 100
#         self.opacity_reset_interval = 100_000_000   # we don't want to reset opacity
#         self.densify_from_iter = 200
#         self.densify_until_iter = its - 1
#         self.densify_grad_threshold = 0.0002
#         self.camera_extent_multiplier = 1
#         self.size_threshold = None
#         self.block_densification = False # this overwrites the densification parameters when set to True
#         self.random_background = True
#         self.use_isotropic_gaussians = True
#         super().__init__(parser, "Optimization Parameters")

    def get_combined_args(parser : ArgumentParser):
        cmdlne_string = sys.argv[1:]
        cfgfile_string = "Namespace()"
        args_cmdline = parser.parse_args(cmdlne_string)

        try:
            cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
            print("Looking for config file in", cfgfilepath)
            with open(cfgfilepath) as cfg_file:
                print("Config file found: {}".format(cfgfilepath))
                cfgfile_string = cfg_file.read()
        except TypeError:
            print("Config file not found at")
            pass
        args_cfgfile = eval(cfgfile_string)

        merged_dict = vars(args_cfgfile).copy()
        for k,v in vars(args_cmdline).items():
            if v != None:
                merged_dict[k] = v
        return Namespace(**merged_dict)

class ExperimentParams(ParamGroup):
    def __init__(self, parser):
        self.source_model = ""  # path to the Gaussian Splats of the source model
        self.target_model = ""  # path to the Gaussian Splats of the target model
        self.dataset_path = "4D-DRESS"  # path to the dataset
        self.src_user = "" # "00122" # user of the source model
        self.src_outfit = "" # "Inner"   # outfit of the source model
        self.src_take = "" # "Take2" # take of the source model
        self.src_sample = ""    # sample of the source model
        self.src_y_rotation = 0
        self.trg_user = "" # "00122" # user of the target model
        self.trg_outfit = "" # "Inner"   # outfit of the target model
        self.trg_take = "" # "Take2" # take of the target model
        self.trg_sample = ""    # sample of the target model
        self.trg_y_rotation = 0
        
        self.save_output = True
        self.only_target_nicp = False
        self.use_mesh_reconstruction = False
        super().__init__(parser, "Pipeline Parameters")