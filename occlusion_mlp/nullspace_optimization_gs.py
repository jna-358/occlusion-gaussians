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

checkpoint = "/data/pretrained_models/rigid_gaussians_application_pretrained.pth"
source_dir = "/data/rgb-10min"
calibration_file = "/data/real-world/pose_calibration.npz"
simulation_file = "/data/real-world/simulation_combined.h5"

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
    
    # Find start angles from simulation
    data_simulation = pd.read_hdf(simulation_file)
    visibility_sim = data_simulation["visibility"].values
    angles_sim = data_simulation[[f"joint_{i}" for i in range(7)]].values
    idx_hidden = np.arange(angles_sim.shape[0])[visibility_sim < 0.5]
    idx_chose = idx_hidden[0]
    start_angles = angles_sim[idx_chose, :].tolist()

    # Generate some input data

    X = torch.tensor(start_angles).unsqueeze(0)

    # Get initial end effector position
    initial_position = kinematic_fun(X[0, :])
    print(f"Initial position: {initial_position}")

    # Convert to parameter
    X_param = torch.nn.Parameter(X)

    # Optimizer
    optimizer = torch.optim.Adam([X_param], lr=1e-3)

    # Bookkeeping
    angle_history = np.empty((num_steps_max, 7))
    visibility_history = np.empty(num_steps_max)
    pos_error_history = np.empty(num_steps_max)

    # Optimize
    pbar = tqdm.tqdm(range(num_steps_max))
    for i_epoch in pbar:
        optimizer.zero_grad()

        # Forward pass
        y = model(X_param)
        visibility = y[0, -1]

        # Print first epoch
        if i_epoch == 0:
            print(f"Initial visibility: {visibility:.2f}")

        # Compute kinematics
        target_position = kinematic_fun(X_param[0, :])
        loss_position = torch.sum((target_position - initial_position) ** 2)

        # Save history
        angle_history[i_epoch, :] = X_param.detach().numpy()[0]
        visibility_history[i_epoch] = visibility.detach().numpy()
        pos_error_history[i_epoch] = np.sqrt(loss_position.detach().numpy())

        # Compute loss
        loss = (1. - visibility)**2 + 1e3 * loss_position

        # Stop if visibility is good enough
        if visibility > 0.999:
            angle_history = angle_history[: i_epoch + 1, :]
            visibility_history = visibility_history[: i_epoch + 1]
            pos_error_history = pos_error_history[: i_epoch + 1]
            break

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Print
        pbar.set_description(
            f"Visibility: {visibility:.2f}; Position error: {loss_position:.2e}; Loss: {loss:.2e}"
        )

    # Print final angles
    angles_final = X_param.detach().numpy()[0].tolist()
    angles_final = [round(angle, 4) for angle in angles_final]
    print(f"Final angles: {angles_final}")

    # Subsample angle history
    to_keep = subsample_movement(angle_history, dist=0.001)
    angle_history = angle_history[to_keep, :]
    visibility_history = visibility_history[to_keep]
    pos_error_history = pos_error_history[to_keep]

    # Save to file
    output_dir = "data/trajectory_optimization"
    os.makedirs(output_dir, exist_ok=True)
    time_str = datetime.datetime.now().strftime("%b%d_%H-%M")
    with open(
        output_path := os.path.join(output_dir, f"trajectory_{time_str}.pkl"), "wb"
    ) as f:
        pkl.dump(
            {
                "angles": angle_history,
                "visibility": visibility_history,
            },
            f,
        )
    full_output_path = os.path.abspath(output_path).replace("\\", "/")
    print(f"Saved trajectory to {full_output_path}")

    # Plot history
    plt.figure(figsize=(8, 4))
    for i_joint in range(7):
        plt.plot(angle_history[:, i_joint], label=f"Joint {i_joint}")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Angle")
    plt.title("Joint angles")
    plt.savefig(png_path := os.path.join(output_dir, f"trajectory_angles.png"))
    print(f"Saved plot to {png_path}")

    plt.figure(figsize=(8, 4))
    plt.plot(visibility_history, label="Optimization")
    plt.xlabel("Iteration")
    plt.ylabel("Visibility")
    plt.legend()
    plt.title("Visibility")
    plt.savefig(png_path := os.path.join(output_dir, f"trajectory_visibility.png"))
    print(f"Saved plot to {png_path}")

    plt.figure(figsize=(8, 4))
    plt.plot(pos_error_history)
    plt.xlabel("Iteration")
    plt.ylabel("Position error / m")
    plt.title("Position error")
    plt.savefig(png_path := os.path.join(output_dir, f"trajectory_pos_error.png"))
    print(f"Saved plot to {png_path}")
    plt.show()


if __name__ == "__main__":
    # # Attach the debugger
    # debugpy.listen(5678)
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()

    main()
