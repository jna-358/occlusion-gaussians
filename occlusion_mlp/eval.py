import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from architectures.linear_model import LinearModel
from utils.data import load_data
from utils.random_names import generate_random_str
import ray
from ray.tune.search.optuna import OptunaSearch
import debugpy
import argparse
import json
import pandas as pd

def train(config_default={}, model_path=None):
    print("###############################################")
    print("################ TRAINING #####################")
    print("###############################################")

    # Check if this is a hyperparameter search
    is_tune_run = False
    print(f"Is tune run: {is_tune_run}")

    # Merge the two configs
    config = config_default

    # Load the data
    X, Y, X_val, Y_val = load_data(
        config["FILENAME"],
        split=config["TEST_SPLIT"],
        device=config["DEVICE"],
        random_split=config["RANDOM_SPLIT"],
    )

    print(f"Avg visibility: {torch.mean(Y[:,-1]) * 100.0:.2f} %")

    if config["BASELINE_DATA"] is None:
        X_test, Y_test = X_val, Y_val
    else:
        X_test, Y_test = load_data(config["BASELINE_DATA"], device=config["DEVICE"])

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(config["SEED"])

    # Init model and optimizer
    model = LinearModel(
        config["NUM_JOINTS"],
        config["HIDDEN_LAYERS"],
        config["NUM_HIDDEN"],
    ).to(config["DEVICE"])

    # Load model weights if provided
    assert model_path is not None
    print(f"Loading model weights from {model_path}")
    model.load_state_dict(torch.load(model_path))

    # Print model class name
    print(f"Model class: {model.__class__.__name__}")

    total_steps = 0

    with torch.no_grad():
        num_visible_train = torch.sum(Y[:, -1])
        num_hidden_train = torch.sum(1.0 - Y[:, -1])

        # Visibility weights
        weight_visible = (num_visible_train + num_hidden_train) / (
            2 * num_visible_train
        )
        weight_hidden = (num_visible_train + num_hidden_train) / (2 * num_hidden_train)

        weight_mask = torch.zeros((Y.shape[0],), dtype=torch.float32)
        weight_mask[Y[:, -1] < 0.5] = weight_hidden
        weight_mask[Y[:, -1] >= 0.5] = weight_visible

        weight_mask_val = torch.zeros((Y_val.shape[0],), dtype=torch.float32)
        weight_mask_val[Y_val[:, -1] < 0.5] = weight_hidden
        weight_mask_val[Y_val[:, -1] >= 0.5] = weight_visible

        # Shift to GPU
        weight_mask = weight_mask.to(config["DEVICE"])
        weight_mask_val = weight_mask_val.to(config["DEVICE"])

    # Bookkeeping for early stopping
    balanced_accuracy_1m_val_min = np.inf

    balanced_acc_1m_test = 1.0

    # Init BCE loss
    bce_loss = torch.nn.BCELoss(weight=weight_mask)

    confusion_matrix = {
        "tp": None,
        "tn": None,
        "fp": None,
        "fn": None,
    }

    pbar = range(config["NUM_EPOCHS"])
 
    # Test set evaluation
    with torch.no_grad():
        # Forward pass
        Y_pred_val = model(X_val)
        visibility_pred_val = Y_pred_val[:, -1]

        pp = visibility_pred_val > 0.5
        pn = visibility_pred_val <= 0.5

        p = Y_val[:, -1] > 0.5
        n = Y_val[:, -1] <= 0.5

        tp = torch.logical_and(pp, p)
        tn = torch.logical_and(pn, n)

        balanced_accuracy_1m_val = 1.0 - (
            ((torch.sum(tp) / torch.sum(p)) + (torch.sum(tn) / torch.sum(n))) / 2.0
        )
        
        # If new best validation score, compute test score
        Y_test_pred = model(X_test)
        visibility_pred_test = Y_test_pred[:, -1]

        pp = visibility_pred_test > 0.5
        pn = visibility_pred_test <= 0.5

        p = Y_test[:, -1] > 0.5
        n = Y_test[:, -1] <= 0.5

        num_total = Y_test.shape[0]

        tp = torch.logical_and(pp, p)
        tn = torch.logical_and(pn, n)
        fp = torch.logical_and(pp, n)
        fn = torch.logical_and(pn, p)

        # Save confusion matrix
        tp_num = torch.sum(tp).item()
        tn_num = torch.sum(tn).item()
        fp_num = torch.sum(fp).item()
        fn_num = torch.sum(fn).item()

        confusion_matrix["tp"] = tp_num / num_total
        confusion_matrix["tn"] = tn_num / num_total
        confusion_matrix["fp"] = fp_num / num_total
        confusion_matrix["fn"] = fn_num / num_total

        tpr = torch.sum(tp) / torch.sum(p)
        tnr = torch.sum(tn) / torch.sum(n)

        balanced_acc_1m_test = 1.0 - (tpr + tnr) / 2.0
        balanced_acc_1m_test = balanced_acc_1m_test.item()

        balanced_accuracy_1m_val_min = balanced_accuracy_1m_val.item()

    # Print confusion matrix
    print("\nConfusion matrix:")
    print(f"tp/all: {confusion_matrix['tp']*100:.2f} %")
    print(f"tn/all: {confusion_matrix['tn']*100:.2f} %")
    print(f"fp/all: {confusion_matrix['fp']*100:.2f} %")
    print(f"fn/all: {confusion_matrix['fn']*100:.2f} %")

    return balanced_accuracy_1m_val_min, balanced_acc_1m_test


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("model_weights", type=str)
    parser.add_argument("--debugpy", action="store_true")
    args = parser.parse_args()

    # Attach debugger
    if args.debugpy:
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    # Load config json
    with open(args.config, "r") as f:
        config = json.load(f)

    # Train
    balanced_accuracy_1m_val_min, balanced_acc_1m_test = train(config_default=config, model_path=args.model_weights)

    print("\nBalanced Accuracy:")
    print(f"b_acc_val: {100. * (1. - balanced_accuracy_1m_val_min):.2f} %")
    print(f"b_acc_test: {100. * (1. - balanced_acc_1m_test):.2f} %")