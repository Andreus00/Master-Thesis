# This file has the sole purpose to find the best parameters for the reconstruction of avatars.


# It runs many reconstructions with different parameters and saves the results in a file.

# parameter grid definition for reconstruction

iterations = [2_000] # , 1_500, 2_000]
scaling_lr = [0.0025, 0.00125]
densify_grad_threshold = [0.00002, 0.00001]
size_threshold = [1, 0.25]
camera_extent_multiplier = [1.0, 0.5, 0.1]
use_isotropic_gaussians = [True, False]
source_path = ["4D-DRESS/00122/Inner/Take2/", "4D-DRESS/00123/Outer/Take8/"]


import sys
sys.path.append("./ext/gaussian-splatting/")
from argparse import Namespace, ArgumentParser
from arguments_4d_dress import ModelParams4dDress, OptimizationParams4dDress, ExperimentParams
from arguments import PipelineParams
from train_baseline_on_user import main as train_baseline
from run_retarget_lbs_on_user import main as run_retarget
from video_from_renders import main as run_video_reconstruction

skip = True # skip the first iteration

for it in iterations:
    for scal_lr in scaling_lr:
        for dens_grad_thr in densify_grad_threshold:
            for siz_thr in size_threshold:
                for cam_ext_mult in camera_extent_multiplier:
                    for use_iso_gauss in use_isotropic_gaussians:
                        for src_path in source_path:
                            print(f"Running reconstruction with it={it}, scal_lr={scal_lr}, dens_grad_thr={dens_grad_thr}, siz_thr={siz_thr}, cam_ext_mult={cam_ext_mult}, use_iso_gauss={use_iso_gauss}, src_path={src_path}")
                            args = Namespace()
                            args.ip = "127.0.0.1"
                            args.port = 6009
                            args.debug_from = -1
                            args.detect_anomaly = False
                            args.test_iterations = [it]
                            args.save_iterations = [it]
                            args.quiet = False
                            args.checkpoint_iterations = []
                            args.start_checkpoint = None
                            args.save_iterations.append(it)
                            args.source_path = src_path
                            
                            parser = ArgumentParser(description="Training script parameters")
                            pp = PipelineParams(parser)
                            lp = ModelParams4dDress(parser)
                            setattr(lp, "source_path", src_path)
                            setattr(lp, "model_path", "output/")
                            setattr(lp, "images", "images")
                            setattr(lp, "resolution", -1)
                            setattr(lp, "white_background", False)
                            op = OptimizationParams4dDress(parser)
                            
                            op.iterations = it
                            op.position_lr_max_steps = it
                            op.densify_until_iter = it - 1
                            op.scaling_lr = scal_lr
                            op.densify_grad_threshold = dens_grad_thr
                            op.size_threshold = siz_thr
                            op.camera_extent_multiplier = cam_ext_mult
                            op.use_isotropic_gaussians = use_iso_gauss
                            
                            train_baseline(lp, op, pp, args)
                        
                        print("Running retargeting and video reconstruction")
                        # run the retargeting
                        args = Namespace()
                        args.src_clothes_label_ids = [2, 3, 4, 5]
                        args.trg_clothes_label_ids = [2, 3, 4, 5]
                        args.run_nicp = False
                        
                        parser = ArgumentParser(description="Training script parameters")
                        ep = ExperimentParams(parser)
                        ep.src_user = "00123"
                        ep.src_outfit = "Outer"
                        ep.src_take = "Take8"
                        
                        ep.trg_user = "00122"
                        ep.trg_outfit = "Inner"
                        ep.trg_take = "Take2"
                        
                        run_retarget(pp, ep, args)
                        
                        # run video reconstruction
                        
                        args = Namespace()
                        args.src_user = "00123"
                        args.src_outfit = "Outer"
                        args.src_take = "Take8"
                        args.trg_user = "00122"
                        args.trg_outfit = "Inner"
                        args.trg_take = "Take2"
                        args.test_name = "lbs_render_on_data_smpl_knn_augmented_src"
                        args.out_name = f"it_{it}_slr_{scal_lr}_dgt_{dens_grad_thr}_st_{siz_thr}_cem_{cam_ext_mult}_uig_{use_iso_gauss}"
                        args.video_folder = "./reconstruction_tests/"
                        
                        run_video_reconstruction(args)
                        
                    
