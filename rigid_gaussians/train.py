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
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, to_colormap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import cv2
import numpy as np
import lpips
import requests
from io import BytesIO
from PIL import Image

lpips_loss_fn_alex = None
lpips_loss_fn_vgg = None

MIN_POINTS = 1000

# Upload an image to the image viewer
def upload_image(image, description):
    image = Image.fromarray(image)
    image_io = BytesIO()
    image.save(image_io, format='JPEG')
    image_io.seek(0)

    url = "http://127.0.0.1:5000/upload"
    files = {'file': ('image.jpeg', image_io, 'image/jpeg')}
    data = {'description': description}
    requests.post(url, files=files, data=data)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, min_num_points=1500)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32).cuda()

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    render_manual = False  
    for iteration in range(first_iter, opt.iterations + 1):     

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3)).cuda() if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii, point_distances = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["distances"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                num_points = [torch.sum(gaussians.rigid_body_indices == i).item() for i in range(gaussians.get_num_rigid_bodies)]
                
                # Extract values from nn.Parameter
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "Points": num_points,
                    }  
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Save render and gt
            if (iteration % 10 == 1 and iteration < 250) or (iteration % 250 == 1) or iteration == opt.iterations:
                for i_view in range(1):
                    viewpoint_render = scene.getTrainCameras()[i_view]
                    bg = torch.rand((3)).cuda() if opt.random_background else background
                    gt_image = viewpoint_render.original_image.cuda()
                    sep_line = np.ones((gt_image.shape[1], 4, 3), dtype=np.float32)

                    joint_images = []

                    # Just the background
                    joint_images.append(
                        render(
                            viewpoint_render, gaussians, 
                            pipe, bg, render_indices=[0]
                            )["render"].detach().cpu().numpy().transpose(1,2,0))

                    
                    # Just the rigid bodies
                    joint_images.append(
                        render(
                            viewpoint_render, gaussians, 
                            pipe, bg, 
                            render_indices=range(1, gaussians.get_num_rigid_bodies),
                            override_color=True,
                            )["render"].detach().cpu().numpy().transpose(1,2,0))
                    
                    # Everything
                    joint_images.append(
                        render(
                            viewpoint_render, gaussians, 
                            pipe, bg, render_indices=None
                            )["render"].detach().cpu().numpy().transpose(1,2,0))
                    
                    
                    gt_image_np = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
                    err_np = np.sum(np.abs(joint_images[-1] - gt_image_np), axis=-1)
                    err_np = to_colormap(err_np, cmap="inferno")

                    # 50-50 blending of rigid bodies and the ground truth
                    blended = 0.5 * joint_images[1] + 0.5 * gt_image_np
                    joint_images = [np.concatenate((j, sep_line), axis=1) for j in joint_images]

                    resize_div = 1
                    image_out = np.concatenate((*joint_images, gt_image_np, sep_line, err_np), axis=1)
                    # image_out = np.concatenate((blended, sep_line, gt_image_copy), axis=1)
                    image_out = np.clip(image_out, 0.0, 1.0)
                    image_out = cv2.resize(image_out, (image_out.shape[1] // resize_div, image_out.shape[0] // resize_div))
                    image_out = (image_out * 255).astype(np.uint8)

                    progress_bar_height = 30
                    progrss_bar_width = image_out.shape[1]
                    progress_pixels = int(progrss_bar_width * iteration / opt.iterations)
                    progress_bar_arr = np.ones((progress_bar_height, progrss_bar_width, 3), dtype=np.uint8) * 255
                    progress_bar_arr[:, :progress_pixels] = [16, 122, 176]

                    image_out = np.concatenate((image_out, progress_bar_arr), axis=0)
                    image_str = f"Iteration {iteration} / {opt.iterations} - Loss: {ema_loss_for_log:.{7}f} - From left to right: rendered background, rendered manipulator, rendered scene, ground truth, absolute error"

                    upload_image(image_out, image_str)
                    # image_out = cv2.resize(image_out, (image_out.shape[1] // 2, image_out.shape[0] // 2))
                    os.makedirs(os.path.join(dataset.model_path, "render"), exist_ok=True)
                    cv2.imwrite(os.path.join(dataset.model_path, "render", f"iter_{iteration}_{i_view}.png"), image_out[..., ::-1])
                    assert not render_manual

            # Log and save
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if True and iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None # 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()


            # Evaluation
            if dataset.eval and (iteration in testing_iterations or iteration == opt.iterations):
                print(f"\n[ITER {iteration}] Evaluating")
                l1_test = 0.0
                psnr_test = 0.0 
                rsme_test = 0.0
                ssim_test = 0.0
                lpips_alex_test = 0.0
                lpips_vgg_test = 0.0
                for viewpoint in scene.getTestCameras():
                    # Render the image
                    image_pkg = render(viewpoint_render, gaussians, pipe, background)
                    image_render = torch.clamp(image_pkg["render"], 0.0, 1.0)
                    image_gt = viewpoint.original_image
                    l1_test += l1_loss(image_render, image_gt)
                    psnr_test += psnr(image_render, image_gt).mean()
                    rsme_test += torch.sqrt(torch.mean((image_render - image_gt) ** 2))
                    ssim_test += ssim(image_render, image_gt)
                    lpips_alex_test += lpips_loss_fn_alex(image_render*2.0-1.0, image_gt*2.0-1.0).mean()
                    lpips_vgg_test += lpips_loss_fn_vgg(image_render*2.0-1.0, image_gt*2.0-1.0).mean()

                l1_test /= len(scene.getTestCameras())
                psnr_test /= len(scene.getTestCameras())
                rsme_test /= len(scene.getTestCameras())
                ssim_test /= len(scene.getTestCameras())
                lpips_alex_test /= len(scene.getTestCameras())
                lpips_vgg_test /= len(scene.getTestCameras())

                # Save the final eval score
                if iteration == opt.iterations:
                    score_path = scene.model_path + "/score.npz"
                    np.savez(score_path, 
                             num_iterations=opt.iterations,
                             l1=l1_test.item(), 
                             psnr=psnr_test.item(), 
                             rsme=rsme_test.item(), 
                             ssim=ssim_test.item(), 
                             lpips_alex=lpips_alex_test.item(), 
                             lpips_vgg=lpips_vgg_test.item())



                print(f"[ITER {iteration}] L1 {l1_test:.2e} PSNR {psnr_test:.1f} RSME {rsme_test:.2e} SSIM {ssim_test:.1f} LPIPS_A {lpips_alex_test:.3f} LPIPS_V {lpips_vgg_test:.3f}")

            # Optimizer step
            if iteration < opt.iterations and not render_manual:
                gaussians.optimizer.step()
                # gaussians.optimizer_pose.step()

                gaussians.optimizer.zero_grad(set_to_none = True)
                # gaussians.optimizer_pose.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations) or (iteration == opt.iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                iteration_str = str(iteration) if iteration != opt.iterations else "final"
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + iteration_str + ".pth")
                scene.save(iteration_str)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("/output/rigid_gaussians/trained_models", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def main():
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
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--debugpy", action="store_true")
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

    # Setup LPIPS
    lpips_loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    lpips_loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    # Set up the training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

if __name__ == "__main__":
    main()