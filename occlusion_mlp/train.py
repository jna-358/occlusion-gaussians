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

def train(config_tune={}, config_default={}):
    print("###############################################")
    print("################ TRAINING #####################")
    print("###############################################")
    print(f"Config tune: {config_tune}")

    # Check if this is a hyperparameter search
    is_tune_run = True if config_tune else False
    print(f"Is tune run: {is_tune_run}")

    # Merge the two configs
    config = {}
    for key in config_default:
        if key in config_tune:
            config[key] = config_tune[key]
        else:
            config[key] = config_default[key]

    # Create checkpoint directory
    if config["SAVE_BEST"]:
        time_str = datetime.now().strftime("%b%d_%H-%M")
        checkpoint_dir = os.path.join("checkpoints", time_str)
        os.makedirs(checkpoint_dir, exist_ok=True)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    # Load checkpoint if it is a tune run
    if is_tune_run:
        loaded_checkpoint = ray.train.get_checkpoint()
        if loaded_checkpoint:
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                model_state, optimizer_state = torch.load(
                    os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
                )
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)

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
    if config["PBAR"]:
        pbar = tqdm(pbar)
    stopping_counter = 0
    for epoch in pbar:
        Y_pred: torch.Tensor

        # Forward pass
        Y_pred = model(X)
        visibility_pred = Y_pred[:, -1]
        Y_pred = Y_pred[:, :-1]

        visibility_loss_train = bce_loss(visibility_pred, Y[:, -1])

        combined_loss_train = visibility_loss_train

        # Backpropagation
        optimizer.zero_grad()
        combined_loss_train.backward()

        # Gradient descent
        optimizer.step()

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

            stopping_counter += 1
            if balanced_accuracy_1m_val.item() < balanced_accuracy_1m_val_min:
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
                stopping_counter = 0

                # Save the model weights
                if config["SAVE_BEST"]:
                    torch.save(
                        model.state_dict(),
                        f"trained_models/{config['NAME']}.pth",
                    )

            if config["PBAR"]:
                pbar.set_description(
                    f"Loss: {combined_loss_train.item():.2e} # b_acc_val: {balanced_accuracy_1m_val.item():.2e} # b_acc_val_min: {balanced_accuracy_1m_val_min:.2e} # b_acc_test_min: {balanced_acc_1m_test:.2e} # {int(100*(stopping_counter / config['EARLY_STOPPING']))}%"
                )
            total_steps += 1

            if stopping_counter >= config["EARLY_STOPPING"]:
                if config["PBAR"]:
                    pbar.close()
                print(
                    f"EARLY STOPPING after {stopping_counter} epochs without improvement."
                )
                break

            # Report to raytune
            if is_tune_run and (epoch % config["TUNE_REPORT_INTERVAL"] == 0):
                tune_dir = "tune_runs"
                if not os.path.exists(tune_dir):
                    os.makedirs(tune_dir)

                # Generate random filename
                model_dirname = generate_random_str(len=16)
                while os.path.exists(os.path.join(tune_dir, model_dirname)):
                    model_dirname = generate_random_str(len=16)

                # Create directory
                os.makedirs(os.path.join(tune_dir, model_dirname))

                # Save config
                torch.save(
                    (model.state_dict(), optimizer.state_dict()),
                    os.path.join(tune_dir, model_dirname, "checkpoint.pt"),
                )

                # Report to raytune
                print("Reporting to raytune")
                checkpoint = ray.train.Checkpoint.from_directory(
                    os.path.join(tune_dir, model_dirname)
                )
                ray.train.report(
                    {
                        "min_bal_acc_val": balanced_accuracy_1m_val_min,
                        "min_bal_acc_test": balanced_acc_1m_test,
                    },
                    checkpoint=checkpoint,
                )

    # Print confusion matrix
    print("Confusion matrix:")
    print(f"tp/all: {confusion_matrix['tp']*100:.2f} %")
    print(f"tn/all: {confusion_matrix['tn']*100:.2f} %")
    print(f"fp/all: {confusion_matrix['fp']*100:.2f} %")
    print(f"fn/all: {confusion_matrix['fn']*100:.2f} %")

    # Save the model weights
    if config["SAVE"]:
        torch.save(
            model.state_dict(), output_path := f"trained_models/{config['NAME']}.pth"
        )
        print(f"Saved model to {output_path}")

    if is_tune_run:
        return
    else:
        return balanced_accuracy_1m_val_min, balanced_acc_1m_test


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
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
    train(config_default=config)