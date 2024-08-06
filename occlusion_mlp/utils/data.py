import torch
import numpy as np
import pickle as pkl
import pandas as pd


def split_test_middle(X, Y, test_split=0.2, buffer_split=0.01):
    # Compute number of samples
    num_samples = X.shape[0]

    # Split into train and test set (train buffer test buffer train)
    num_test = int(test_split * num_samples)
    num_buffer_half = int(buffer_split * num_samples)
    num_train = num_samples - num_test - 2 * num_buffer_half
    num_train_half = num_train // 2

    is_train = np.zeros((num_samples,), dtype=bool)
    is_test = np.zeros((num_samples,), dtype=bool)
    is_train[:num_train_half] = True
    is_test[
        num_train_half + num_buffer_half : num_train_half + num_buffer_half + num_test
    ] = True
    is_train[num_train_half + num_buffer_half + num_test + num_buffer_half :] = True

    assert ~np.any(np.logical_and(is_train, is_test))

    X_train, Y_train = X[is_train, :], Y[is_train, :]
    X_test, Y_test = X[is_test, :], Y[is_test, :]

    return X_train, Y_train, X_test, Y_test

def split_test_random(X, Y, test_split=0.2, buffer_split=0.01):
    # Compute number of samples
    num_samples = X.shape[0]
    num_test = int(test_split * num_samples)
    num_buffer = int(buffer_split * num_samples)

    # Pick random index for splitting
    idx_split = int(np.round(np.random.rand() * (num_samples - 1)))
    print(f"Splitting at index {idx_split}")

    # Split into train and test set
    is_train = np.zeros((num_samples,), dtype=bool)
    is_test = np.zeros((num_samples,), dtype=bool)

    if idx_split + num_test <= num_samples:
        # Split without wrapping
        is_test[idx_split : idx_split + num_test] = True
        is_train[idx_split + num_test + num_buffer :] = True
        is_train[:max(0, idx_split - num_buffer)] = True
    else:
        # Split with wrapping
        is_test[idx_split:] = True
        is_test[:num_samples - idx_split] = True
        is_train[num_samples - idx_split + num_buffer : idx_split - num_buffer] = True

    assert ~np.any(np.logical_and(is_train, is_test))

    X_train, Y_train = X[is_train, :], Y[is_train, :]
    X_test, Y_test = X[is_test, :], Y[is_test, :]

    return X_train, Y_train, X_test, Y_test

def load_data(filename, split=None, device=None, random_split=False):
    # Load the data
    print(f"Loading {filename} with split={split} and device={device}")

    if filename.endswith(".pkl"):
        with open(filename, "rb") as f:
            data = pkl.load(f)
        joint_angles = data["joint_angles"].astype(np.float32)
        visibility = data["visibility"][:, 0].astype(np.float32)
    elif filename.endswith(".h5"):
        df = pd.read_hdf(filename)
        joint_angles = df[[f"joint_{i}" for i in range(7)]].values.astype(np.float32)
        
        if "visibility" in df:
            visibility = df["visibility"].values.astype(np.float32)
        else:
            visibility = np.invert(np.isnan(df["marker_x"].values)).astype(np.float32)
            joint_angles = np.deg2rad(joint_angles)
    else:
        raise ValueError("Unknown file format")

    if filename.endswith("_train.h5"):
        # Subsample the data
        joint_angles = joint_angles[::2]
        visibility = visibility[::2]

    # distance = data["distances"][:, 0].astype(np.float32)

    # if "position" in data.keys():
    #     target_positions = data["position"].astype(np.float32)
    # else:
    #     target_positions = np.zeros((joint_angles.shape[0], 3), dtype=np.float32)

    num_samples = joint_angles.shape[0]
    print(f"Loaded {num_samples} samples")

    # # Concatenate all properties
    # target_output = np.concatenate(
    #     [
    #         target_positions,
    #         np.transpose(distance[None, :]),
    #         np.transpose(visibility[None, :]),
    #     ],
    #     axis=1,
    # )

    X = joint_angles
    Y = visibility[:, None] # target_output

    # Remove NaN values
    nan_lines = np.logical_or(np.any(np.isnan(X), axis=1), np.any(np.isnan(Y), axis=1))
    X = X[np.invert(nan_lines), :]
    Y = Y[np.invert(nan_lines), :]

    if split is not None:
        # Check that split is a float
        assert isinstance(split, float), "split must be a float"

        # Split the data
        if random_split:
            X_train, Y_train, X_test, Y_test = split_test_random(X, Y, test_split=split)
        else:
            X_train, Y_train, X_test, Y_test = split_test_middle(X, Y, test_split=split)

        # Convert to torch tensors
        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        Y_train = torch.from_numpy(Y_train)
        Y_test = torch.from_numpy(Y_test)

        # Shift to GPU
        if device is not None:
            X_train = X_train.to(device)
            Y_train = Y_train.to(device)
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)
        return X_train, Y_train, X_test, Y_test
    else:
        # Convert to torch tensors
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        # Shift to GPU
        if device is not None:
            X = X.to(device)
            Y = Y.to(device)
        return X, Y
