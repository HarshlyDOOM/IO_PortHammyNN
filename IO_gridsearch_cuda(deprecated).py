import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from Old_IO_Files.IO_SHND import LatentIOModel
import pandas as pd
from itertools import product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

def build_io_window_dataset(theta, u, window_size=2):
    X_list, Y_list = [], []
    for t in range(window_size, len(theta) - 1):
        theta_window = theta[t - window_size + 1 : t + 1].flatten()
        u_window = u[t - window_size + 1 : t + 1].flatten()
        x = np.concatenate([theta_window, u_window])
        y = theta[t + 1]
        X_list.append(x)
        Y_list.append(y)
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)
    return TensorDataset(X, Y)

def compute_rmse(y_true, y_pred):
    from math import pi
    rmse_rad = torch.sqrt(F.mse_loss(y_pred, y_true)).item()
    rmse_deg = rmse_rad * (180 / pi)
    nrmse = (rmse_rad / y_true.std()).item() * 100
    return rmse_rad, rmse_deg, nrmse

def simulate_io_model(model, u, theta, na=2, nb=2, skip=50):
    pred = []
    true = []
    theta = theta.copy()
    for t in range(skip, len(theta) - 1):
        th_hist = theta[t - na + 1 : t + 1].flatten()
        u_hist = u[t - nb + 1 : t + 1].flatten()
        x = np.concatenate([th_hist, u_hist])
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        x_tensor.requires_grad = True
        y_pred = model(x_tensor).squeeze().item()
        pred.append(y_pred)
        true.append(theta[t + 1].item())
        theta[t + 1] = y_pred
    return np.array(pred), np.array(true)

class Dict2Class:
    def __init__(self, d): self.__dict__.update(d)

def train_model(model_label, proj_fn, latent_dim, hidden_dim, bln_units, u, theta):
    dataset = build_io_window_dataset(theta, u)
    N = len(dataset)
    N_train, N_val = int(0.8 * N), int(0.1 * N)
    N_test = N - N_train - N_val
    train_ds, val_ds, test_ds = random_split(dataset, [N_train, N_val, N_test])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    input_dim = next(iter(train_loader))[0].shape[1]

    args = {
        "proj_fn": proj_fn,
        "nx": latent_dim,
        "nh": hidden_dim,
        "nph": 64,
        "alpha": 0.01,
        "eps": 0.01,
        "rehu_factor": 0.01,
        "scale_fx": False
    }
    args = Dict2Class(args)

    model = LatentIOModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        bln_units=bln_units,
        dt=1.0,
        model_type=model_label,
        args=args if model_label == "LyaProj" else None
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss = float("inf")
    patience, counter = 15, 0
    best_model = None

    for epoch in range(200):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb.requires_grad = True
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb.requires_grad = True  # ✅ This is critical
            pred = model(xb)
            val_loss += F.mse_loss(pred, yb).item()
        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict(best_model)

    # Evaluation
    y_true, y_pred = [], []
    model.eval()
    for xb, yb in test_loader:
        xb = xb.to(device)
        xb.requires_grad = True  # ✅ Needed for SHND/HNN models
        pred = model(xb).cpu()
        y_true.append(yb.numpy())
        y_pred.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    pred_rmse = compute_rmse(torch.tensor(y_true), torch.tensor(y_pred))

    sim_pred, sim_true = simulate_io_model(model, u.copy(), theta.copy())
    sim_rmse = compute_rmse(torch.tensor(sim_true), torch.tensor(sim_pred))

    return pred_rmse, sim_rmse

if __name__ == "__main__":
    data = np.load("training-val-test-data.npz")
    theta = data["th"].reshape(-1, 1)
    u = data["u"].reshape(-1, 1)

    latent_dims = [4, 6, 8, 10]
    hidden_dims = [32, 64, 128]
    bln_strategies = {
        32: [[32, 32]],
        64: [[32, 64], [64, 64]],
        128: [[64, 128], [128, 128]]
    }

    configs = []
    for model_label in ["SHND", "LyaProj"]:
        for latent_dim, hidden_dim in product(latent_dims, hidden_dims):
            for bln in bln_strategies[hidden_dim]:
                proj = "NN-REHU" if model_label == "LyaProj" else ""
                configs.append((model_label, proj, latent_dim, hidden_dim, bln))

    results = []
    for model_label, proj, latent_dim, hidden_dim, bln_units in configs:
        print(f"Running {model_label} | Latent={latent_dim} | Hidden={hidden_dim} | BLN={bln_units}")
        pred_rmse, sim_rmse = train_model(
            model_label, proj, latent_dim, hidden_dim, bln_units, u, theta
        )
        results.append({
            "Model": model_label,
            "Proj": proj,
            "LatentDim": latent_dim,
            "HiddenDim": hidden_dim,
            "BLN": str(bln_units),
            "PredRMSE(rad)": pred_rmse[0],
            "PredRMSE(deg)": pred_rmse[1],
            "PredNRMSE(%)": pred_rmse[2],
            "SimRMSE(rad)": sim_rmse[0],
            "SimRMSE(deg)": sim_rmse[1],
            "SimNRMSE(%)": sim_rmse[2]
        })

    df = pd.DataFrame(results)
    df.to_csv("shnd_lyaproj_gridsearch_results.csv", index=False)
    print(df)
