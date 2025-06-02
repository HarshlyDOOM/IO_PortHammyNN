import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

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

def compute_rmse_table(y_true, y_pred, name="Model"):
    from math import pi
    rmse_rad = torch.sqrt(F.mse_loss(y_pred, y_true)).item()
    rmse_deg = rmse_rad * (180 / pi)
    nrmse = (rmse_rad / y_true.std()).item() * 100
    return {
        "Model": name,
        "Pred RMSE (rad)": rmse_rad,
        "Pred RMSE (deg)": rmse_deg,
        "Pred NRMSE (%)": nrmse
    }

class LatentIOModel(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.latent_dynamics = nn.Sequential(
            nn.Linear(latent_dim + 1, 32),
            nn.Tanh(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, u=None):
        z = self.encoder(x)
        if u is not None:
            z = self.latent_dynamics(torch.cat([z, u], dim=-1))
        else:
            z = self.latent_dynamics(torch.cat([z, torch.zeros_like(z[:, :1])], dim=-1))
        return self.decoder(z)

def train_io_model():
    data = np.load("training-val-test-data.npz")
    theta = data["th"].reshape(-1, 1)
    u = data["u"].reshape(-1, 1)

    dataset = build_io_window_dataset(theta, u, window_size=2)
    N = len(dataset)
    N_train, N_val = int(0.8 * N), int(0.1 * N)
    N_test = N - N_train - N_val
    train_ds, val_ds, test_ds = random_split(dataset, [N_train, N_val, N_test])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    input_dim = next(iter(train_loader))[0].shape[1]
    model = LatentIOModel(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss / len(train_loader):.6f}")

    # Evaluate
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            y_true.append(yb)
            y_pred.append(pred)
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    # Print Table
    result = compute_rmse_table(y_true, y_pred, name="IO_SHND (Latent)")
    print("\n# Evaluation Results\n")
    print("{:<20} {:>15} {:>15} {:>15}".format("Model", "Pred RMSE (rad)", "Pred RMSE (deg)", "Pred NRMSE (%)"))
    print("{:<20} {:>15.6f} {:>15.6f} {:>15.6f}".format(
        result["Model"], result["Pred RMSE (rad)"], result["Pred RMSE (deg)"], result["Pred NRMSE (%)"]
    ))

    # Save model
    os.makedirs("trained_models", exist_ok=True)
    torch.save(model.state_dict(), "trained_models/io_shnd.pt")
    print("Model saved to trained_models/io_shnd.pt")

if __name__ == "__main__":
    train_io_model()
