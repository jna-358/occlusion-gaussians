#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import argparse
import os
import subprocess
import time

args = [
    "smooth_random",
    "rigid_gaussians_train",
    "rigid_gaussians_interactive",
    "mlp_blender_train",
    "mlp_blender_eval",
    "mlp_real_world",
    "mlp_gaussians_train",
    "mlp_gaussians_eval",
    "mlp_application_nullspace",
    "mlp_application_trajectory",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arg", type=str, choices=args, help="Choose the script to run")
    args = parser.parse_args()

    match args.arg:
        case "smooth_random":
            os.chdir("/content/smooth_random")
            subprocess.run(["python", "full_pipeline.py"])
        case "rigid_gaussians_train":
            os.chdir("/content/rigid_gaussians")
            print("To view the training progress, visit http://127.0.0.1:5000/")
            time.sleep(2)
            subprocess.run(["python", "train.py", "-s", "/content/data/rgb-5min", "--eval"])
        case "rigid_gaussians_interactive":
            os.chdir("/content/rigid_gaussians")
            print("To view the training progress, visit http://127.0.0.1:5000/interactive")
            time.sleep(2)
            subprocess.run(["python", "manual.py", "-s", "/content/data/rgb-5min", "--start_checkpoint", "/content/data/pretrained_models/rigid_gaussians.pth"])
        case "mlp_blender_train":
            os.chdir("/content/occlusion_mlp")
            subprocess.run(["python", "train.py", "training_configs/synthetic_baseline.json"])
        case "mlp_blender_eval":
            os.chdir("/content/occlusion_mlp")
            subprocess.run(["python", "eval.py", "training_configs/synthetic_baseline.json", "/content/data/pretrained_models/uniform_synthetic_baseline_pretrained.pth"])
        case "mlp_real_world":
            os.chdir("/content/occlusion_mlp")
            subprocess.run(["python", "eval_duration.py"])
        case "mlp_gaussians_train":
            os.chdir("/content/occlusion_mlp")
            subprocess.run(["python", "train.py", "training_configs/rigid_gaussians.json"])
        case "mlp_gaussians_eval":
            os.chdir("/content/occlusion_mlp")
            subprocess.run(["python", "eval.py", "training_configs/rigid_gaussians.json", "/content/data/pretrained_models/rigid_gaussians_pretrained.pth"])
        case "mlp_application_nullspace":
            os.chdir("/content/occlusion_mlp")
            subprocess.run(["python", "nullspace_optimization_gs.py"])
        case "mlp_application_trajectory":
            os.chdir("/content/occlusion_mlp")
            subprocess.run(["python", "trajectory_optimization_gs.py"])
