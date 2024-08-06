import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
import pickle as pkl
from architectures.linear_model import LinearModel
import datetime
from utils.rotations import get_rotation_matrix_torch
import debugpy
import pandas as pd
import argparse

checkpoint = "/content/data/pretrained_models/rigid_gaussians_application_pretrained.pth"
source_dir = "/content/data/rgb-10min"
calibration_file = "/content/data/real-world/pose_calibration.npz"
simulation_file = "/content/data/real-world/simulation_combined.h5"

num_steps_max = 10000

def subsample_movement(data, dist=0.01):
    to_keep = np.zeros(data.shape[0], dtype=bool)
    i_last = 0
    to_keep[0] = True
    for i in range(1, data.shape[0]):
        if np.any(np.abs(data[i, :] - data[i_last, :]) > dist):
            to_keep[i] = True
            i_last = i
    return to_keep

def get_rotation_matrix_z_torch(angle):
    T_res = torch.eye(4)
    T_res[0, 0] = torch.cos(angle)
    T_res[0, 1] = -torch.sin(angle)
    T_res[1, 0] = torch.sin(angle)
    T_res[1, 1] = torch.cos(angle)
    return T_res

def main():
    # Create the model
    model = LinearModel(7, 2, 78)

    # Load the model
    model_state_dict = torch.load(checkpoint)
    model.load_state_dict(model_state_dict)
    model.eval()

    # Load the pose calibration
    data_calibration = np.load(calibration_file)
    pos_endeffector = data_calibration["p3d_est"]
    pos_endeffector = torch.from_numpy(pos_endeffector).float()

    # Load the kinematics
    kinematics_path = os.path.join(source_dir, "kinematics.npy")
    kinematics = np.load(kinematics_path)
    kinematics = torch.from_numpy(kinematics).float()
    
    def kinematic_fun(angles):
        T_res = kinematics[0] @ get_rotation_matrix_torch(angles[0], 2)
        for i_joint in range(1, kinematics.shape[0]):
            T_res = T_res @ kinematics[i_joint] @ get_rotation_matrix_torch(angles[i_joint], 2)

        pos = T_res @ pos_endeffector[:, None]
        return pos

    # Generate initial trajectory [-2.5    2.852  0.     3.732  0.     2.612  0.   ]
    num_points = 1024
    angles_start = np.array([-0.8,  2.852,  0.,     3.732,  0.,     2.612,  0.   ])
    angles_end   = np.array([-3.0,  2.852,  0.,     3.732,  0.,     2.612,  0.   ])

    trajectory_linear = np.linspace(angles_start, angles_end, num_points)
    X = torch.tensor(trajectory_linear).float()

    angles_start = torch.from_numpy(angles_start).float()
    angles_end = torch.from_numpy(angles_end).float()

    # Get initial end effector position
    initial_position = kinematic_fun(X[0, :])
    print(f"Initial position: {initial_position}")

    # Convert to parameter
    X_param = torch.nn.Parameter(X)

    # Optimizer
    optimizer = torch.optim.Adam([X_param], lr=1e-3)

    # Bookkeeping
    history_angles = []
    history_visibility = []

    # Optimize
    pbar = tqdm.tqdm(range(num_steps_max))
    for i_epoch in pbar:
        optimizer.zero_grad()

        # Forward pass
        y = model(X_param)
        visibility = y[:, -1]

        # Save first epoch
        if i_epoch == 0:
            history_angles.append(X_param.detach().numpy().copy())
            history_visibility.append(visibility.detach().numpy().copy())

        # Compute loss
        loss_visibility = torch.mean((visibility - 1.0) ** 2)

        loss_endpoints = torch.mean((X_param[0, :] - angles_start) ** 2) + torch.mean((X_param[-1, :] - angles_end) ** 2)
        loss_cohesion = torch.mean(torch.diff(X_param, dim=0)**2)
        loss = loss_visibility + loss_endpoints + loss_cohesion * 4e3

        # Stop if visibility is good enough
        if loss_visibility < 2e-4 or i_epoch == num_steps_max - 1:
            history_angles.append(X_param.detach().numpy().copy())
            history_visibility.append(visibility.detach().numpy().copy())
            break
        else:

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Print
            pbar.set_description(
                f"Visibility: {loss_visibility:.2e}; Endpoints: {loss_endpoints:.2e}; Cohesion: {loss_cohesion:.2e}"
            )

    # Save to file
    output_dir = "/content/output/occlusion_mlp/trajectory_optimization"
    os.makedirs(output_dir, exist_ok=True)
    time_str = datetime.datetime.now().strftime("%b%d_%H-%M")
    with open(
        output_path := os.path.join(output_dir, f"trajectory_{time_str}.pkl"), "wb"
    ) as f:
        pkl.dump(
            {
                "angles_pre": history_angles[0],
                "visibility_pre": history_visibility[0],
                "angles_post": history_angles[-1],
                "visibility_post": history_visibility[-1],
            },
            f,
        )
    full_output_path = os.path.abspath(output_path).replace("\\", "/")
    print(f"Saved trajectory to {full_output_path}")

    # Plot history
    plt.figure(figsize=(8, 4))
    for i_joint in range(7):
        plt.plot(history_angles[0][:, i_joint], label=f"Joint {i_joint}")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Angle")
    plt.title("Joint angles (before optimization)")
    plt.savefig(png_path := os.path.join(output_dir, f"trajectory_angles_pre.png"))
    print(f"Saved plot to {png_path}")

    plt.figure(figsize=(8, 4))
    for i_joint in range(7):
        plt.plot(history_angles[-1][:, i_joint], label=f"Joint {i_joint}")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Angle")
    plt.title("Joint angles (after optimization)")
    plt.savefig(png_path := os.path.join(output_dir, f"trajectory_angles_post.png"))
    print(f"Saved plot to {png_path}")

    plt.figure(figsize=(8, 4))
    plt.plot(history_visibility[0], label="Before optimization")
    plt.plot(history_visibility[-1], label="After optimization")
    plt.xlabel("Iteration")
    plt.ylabel("Visibility")
    plt.legend()
    plt.title("Visibility")
    plt.savefig(png_path := os.path.join(output_dir, f"trajectory_visibility.png"))
    print(f"Saved plot to {png_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debugpy", action="store_true")
    args = parser.parse_args()

    # Attach the debugger
    if args.debugpy:
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    main()
