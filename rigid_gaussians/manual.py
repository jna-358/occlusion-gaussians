#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from gaussian_renderer import render_occlusion
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import time


MIN_POINTS = 1000

def upload_image(image):
    image = Image.fromarray(image)
    image_io = BytesIO()
    image.save(image_io, format='JPEG')
    image_io.seek(0)

    url = "http://127.0.0.1:5000/upload"
    files = {'file': ('image.jpeg', image_io, 'image/jpeg')}
    data = {'description': 'Rendered image'}
    requests.post(url, files=files, data=data)

def get_angles():
    url = "http://127.0.0.1:5000/get_params"
    response = requests.get(url)
    response = response.json()
    keys = list(response.keys())
    keys.sort()
    angles = [int(response[key]) for key in keys]
    angles, scaling_mod = angles[:-1], angles[-1]
    angles = [(angles[i] / 100.0) * 2.0 * np.pi - (0 if i % 2 == 1 else np.pi) for i in range(len(angles))]
    scaling_mod /= 100.0
    return angles, scaling_mod

def make_rot_z(angle):
    return torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0.0],
        [torch.sin(angle), torch.cos(angle), 0.0],
        [0.0, 0.0, 1.0]
    ]).float().cuda()


def embed_rotm(rotm):
    res = torch.eye(4).float().cuda()
    res[:3, :3] = rotm
    return res

def relative_to_absolute(tforms, angles):
    absolute = torch.empty_like(tforms).float().cuda()
    absolute[0] = tforms[0] @ embed_rotm(make_rot_z(angles[0])) # scipy.spatial.transform.Rotation.from_euler("z", angles[0]).as_matrix())
    for i in range(1, tforms.shape[0]):
        absolute[i] = absolute[i-1] @ tforms[i] @ embed_rotm(make_rot_z(angles[i])) # scipy.spatial.transform.Rotation.from_euler("z", angles[i]).as_matrix())
    return absolute

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, device):
    with torch.no_grad():
        prepare_output_and_logger(dataset)
        gaussians = GaussianModel(dataset.sh_degree, min_num_points=500)
        scene = Scene(dataset, gaussians)

        # Load kinematics
        kinematics_file = os.path.join(dataset.source_path, "kinematics.npy")
        kinematics = np.load(kinematics_file)
        kinematics = torch.from_numpy(kinematics).float().cuda()

        if checkpoint:
            (model_params, _) = torch.load(checkpoint, map_location=torch.device("cuda", device))
            gaussians.restore(model_params, opt)

        render_indices = range(0, gaussians.get_num_rigid_bodies)

        # Set gaussians in the ee radius to white
        ee_size = 5e-4
        ee = torch.from_numpy(scene.endeffector).cuda().float()
        is_close = torch.norm(gaussians._xyz - ee[None, :], dim=1) < ee_size
        gaussians._features_dc[is_close & (gaussians.rigid_body_indices == 7), :, :] = 2.0

        # Add white sphere to the end effector
        num_pts_ee = 512
        ee_xyz = torch.randn((num_pts_ee, 3)).cuda()
        ee_xyz = ee_xyz / torch.norm(ee_xyz, dim=1, keepdim=True)
        ee_xyz = ee_xyz * ee_size
        ee_xyz = ee_xyz + torch.from_numpy(scene.endeffector[None,:]).cuda().float()
        ee_opacity = torch.ones((num_pts_ee, 1)).cuda()
        ee_features_dc = torch.ones((num_pts_ee, gaussians._features_dc.shape[1], gaussians._features_dc.shape[2])).cuda() * 2.0
        ee_features_rest = torch.zeros((num_pts_ee, gaussians._features_rest.shape[1], gaussians._features_rest.shape[2])).cuda()
        ee_rotations = torch.zeros((num_pts_ee, 4)).cuda()
        ee_rotations[:, 0] = 1.0
        ee_scale = torch.ones((num_pts_ee, 3)).cuda() * (-7.0)
        ee_rigid_body_index = torch.ones((num_pts_ee,), dtype=torch.int32).cuda() * 7
        
        gaussians._xyz = torch.cat([gaussians._xyz, ee_xyz], dim=0)
        gaussians._opacity = torch.cat([gaussians._opacity, ee_opacity], dim=0)
        gaussians._features_dc = torch.cat([gaussians._features_dc, ee_features_dc], dim=0)
        gaussians._features_rest = torch.cat([gaussians._features_rest, ee_features_rest], dim=0)
        gaussians._rotation = torch.cat([gaussians._rotation, ee_rotations], dim=0)
        gaussians._scaling = torch.cat([gaussians._scaling, ee_scale], dim=0)
        gaussians.rigid_body_indices = torch.cat([gaussians.rigid_body_indices, ee_rigid_body_index], dim=0)


        #bg_color = [0.5, 0.2, 0.2]
        bg_color = [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color, dtype=torch.float32).cuda()
        viewpoint_render = scene.getTrainCameras()[0]

        # angles = torch.tensor([-0.8,   2.692,  0.0,   4.142,  0.,   2.612,  0.]).cuda()
        angles_last = torch.tensor([-2.5,   2.692,  0.0,   4.142,  0.,   2.612,  0.]).cuda()
        scaling_mod_last = 1.0
        

        print(f"Rendering training images for {len(scene.getTrainCameras())} viewpoints")
        render_last = time.time()
        while True:
            angles, scaling_mod = get_angles()
            print(f"Angles: {angles}; Scaling modifier: {scaling_mod}", end="\r")
            angles = torch.tensor(angles).cuda()
            if torch.all(angles == angles_last) and scaling_mod == scaling_mod_last:
                if time.time() - render_last < 5:
                    continue
                else:
                    render_last = time.time()
            angles_last = angles
            scaling_mod_last = scaling_mod

            bg = torch.rand((3)).cuda() if opt.random_background else background
            
            # Make the rigid transforms
            tforms = relative_to_absolute(kinematics, angles)

            render_pkg = render_occlusion(
                    viewpoint_render, gaussians, 
                    pipe, bg, 
                    render_indices=render_indices,
                    scaling_modifier=scaling_mod,
                    override_color=True, endeffector=scene.endeffector,
                    ee_thresh=0.005,
                    overwrite_tforms=tforms
                    )
            # image_ee = render_pkg["render"].detach().cpu().numpy().transpose(1,2,0)
            img = render_pkg["render"].detach().cpu().numpy().transpose(1,2,0)
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255).astype(np.uint8)
            upload_image(img)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--debugpy", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])

    if args.debugpy:
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print("Debugger attached!")

    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.device, device=args.device)

    # Configure and run training
    training(lp.extract(args), op.extract(args), pp.extract(args), 
             args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, 
             args.debug_from, args.device)

    # All done
    print("\nTraining complete.")
