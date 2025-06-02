####IO_Train.py version######
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from IO_SHND import LatentIOModel # Assuming IO_SHND.py is in the same directory or PYTHONPATH

def build_io_window_dataset(theta, u, window_size=2): # na=nb=window_size
    X_list, Y_list = [], []
    for t in range(window_size -1, len(theta) - 1): 
        theta_w = theta[t - window_size + 1 : t + 1].flatten()
        u_w = u[t - window_size + 1 : t + 1].flatten()
        x_instance = np.concatenate([theta_w, u_w])
        y_instance = theta[t + 1] 
        X_list.append(x_instance)
        Y_list.append(y_instance)
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)
    return TensorDataset(X, Y)

def simulate_io_model(model, u_signal, theta_signal, window_size=2, skip_initial=50, device='cpu', best_model_path=None):
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    model.to(device)
    y_preds_list = []
    y_true_for_sim_list = []
    initial_theta_hist_segment = theta_signal[skip_initial - window_size : skip_initial, 0]
    initial_u_hist_segment = u_signal[skip_initial - window_size : skip_initial, 0]

    if len(initial_theta_hist_segment) < window_size or len(initial_u_hist_segment) < window_size:
        print(f"Error: Not enough data for initial history. Skip_initial: {skip_initial}, window_size: {window_size}")
        print(f"Theta segment length: {len(initial_theta_hist_segment)}, U segment length: {len(initial_u_hist_segment)}")
        return np.array([]), np.array([])
    theta_history_buffer = list(initial_theta_hist_segment)
    u_history_buffer = list(initial_u_hist_segment)
    
    with torch.no_grad():
        for t_predict_idx in range(skip_initial, len(theta_signal)):
            if len(theta_history_buffer) != window_size or len(u_history_buffer) != window_size:
                print(f"Critical Error: History buffer size mismatch at t_predict_idx={t_predict_idx}")
                break
            x_window_np = np.concatenate([
                np.array(theta_history_buffer), 
                np.array(u_history_buffer)      
            ]).flatten()
            x_window_tensor = torch.tensor(x_window_np, dtype=torch.float32).unsqueeze(0).to(device)
            predicted_theta_at_t_predict_idx = model(x_window_tensor).squeeze().item()
            y_preds_list.append(predicted_theta_at_t_predict_idx)
            true_theta_at_t_predict_idx = theta_signal[t_predict_idx, 0]
            y_true_for_sim_list.append(true_theta_at_t_predict_idx)

            if t_predict_idx < skip_initial + 5 : 
                print(f"Sim step t_predict={t_predict_idx}:")
                print(f"  Input Theta Hist: {theta_history_buffer}")
                print(f"  Input U Hist: {u_history_buffer}")
                print(f"  Pred Theta[{t_predict_idx}]: {predicted_theta_at_t_predict_idx:.4f}")
                print(f"  True Theta[{t_predict_idx}]: {true_theta_at_t_predict_idx:.4f}")

            theta_history_buffer.pop(0)
            theta_history_buffer.append(predicted_theta_at_t_predict_idx)
            if t_predict_idx < len(u_signal): 
                u_history_buffer.pop(0)
                u_history_buffer.append(u_signal[t_predict_idx, 0])
            else:
                if t_predict_idx < len(theta_signal) -1 : 
                     print(f"Note: Reached end of u_signal at t_predict_idx={t_predict_idx} while forming history for next step.")
    
    preds_np = np.array(y_preds_list)
    trues_np = np.array(y_true_for_sim_list)
    print("\n--- Inside simulate_io_model (End) ---")
    print(f"Length of y_preds_list: {len(y_preds_list)}")
    print(f"Length of y_true_for_sim_list: {len(y_true_for_sim_list)}")
    if len(preds_np) > 0 and len(trues_np) > 0:
        print("Sim Preds (first 5):", preds_np[:5])
        print("Sim True  (first 5):", trues_np[:5])
        print("Sim Preds (last 5):", preds_np[-5:])
        print("Sim True  (last 5):", trues_np[-5:])
        if np.allclose(preds_np, trues_np):
            print("CRITICAL WARNING: Simulation predictions and true values are identical in simulate_io_model!")
        else:
            print("Note: Simulation predictions and true values appear different in simulate_io_model.")
        diff = preds_np - trues_np
        print(f"Sim Error Stats: Mean(abs(diff))={np.mean(np.abs(diff)):.4e}, Max(abs(diff))={np.max(np.abs(diff)):.4e}")
    print("---------------------------------------\n")

    if len(y_preds_list) != len(y_true_for_sim_list):
        print(f"Warning: Mismatch in prediction ({len(y_preds_list)}) and true ({len(y_true_for_sim_list)}) lengths in simulation. Trimming.")
        min_len = min(len(y_preds_list), len(y_true_for_sim_list))
        y_preds_list = y_preds_list[:min_len]
        y_true_for_sim_list = y_true_for_sim_list[:min_len]
    return np.array(y_preds_list), np.array(y_true_for_sim_list)


def compute_metrics(y_true, y_pred):
    # --- ADDED DEBUG PRINTS ---
    # print("\n--- Inside compute_metrics ---")
    # print(f"Type of y_true: {type(y_true)}, Shape: {y_true.shape if isinstance(y_true, np.ndarray) else 'Not an ndarray'}")
    # print(f"Type of y_pred: {type(y_pred)}, Shape: {y_pred.shape if isinstance(y_pred, np.ndarray) else 'Not an ndarray'}")
    # if isinstance(y_true, np.ndarray) and len(y_true) > 0:
    #     print("y_true (first 5):", y_true[:5])
    # if isinstance(y_pred, np.ndarray) and len(y_pred) > 0:
    #     print("y_pred (first 5):", y_pred[:5])

    # if len(y_true) == 0 or len(y_pred) == 0:
    #     print("Warning: compute_metrics received empty array(s).")
    #     return np.nan, np.nan, np.nan
    
    # if np.allclose(y_true, y_pred):
    #     print("WARNING in compute_metrics: y_true and y_pred are identical or very close!")
    #     # This would correctly lead to RMSE near 0.
    # else:
    #     print("Note in compute_metrics: y_true and y_pred appear different.")
    # --- END ADDED DEBUG PRINTS ---

    err = y_pred - y_true
    rm_r = np.sqrt(np.mean(err**2))
    rm_d = rm_r / (2 * np.pi) * 360
    
    std_y_true = np.std(y_true)
    if std_y_true == 0:
        print("Warning in compute_metrics: np.std(y_true) is zero.")
        nrms = np.nan if rm_r > 1e-9 else 0 
    else:
        nrms = rm_r / std_y_true * 100
    
    print(f"Computed RMSE (rad): {rm_r:.4e}")
    print("---------------------------\n")
    return rm_r, rm_d, nrms

class Dict2Class:
    def __init__(self, d): self.__dict__.update(d)

def train_and_evaluate(model_label="pH_SHND", proj_fn="ICNN", device_str='cpu'): 
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # --- Data Loading ---
    data = np.load("training-val-test-data.npz")
    theta_full = data["th"].reshape(-1, 1)
    u_full = data["u"].reshape(-1, 1)

    # --- Dataset Preparation ---
    window_size_config = 2 
    full_dataset = build_io_window_dataset(theta_full, u_full, window_size=window_size_config)
    
    if len(full_dataset) == 0:
        print("Error: Dataset is empty. Check build_io_window_dataset logic and data length.")
        return {"Model": f"IO_{model_label}_Error", "Pred RMSE (rad)": np.nan}

    N_total_samples = len(full_dataset)
    train_split_ratio = 0.8
    val_split_ratio = 0.1
    n_train = int(train_split_ratio * N_total_samples)
    n_val = int(val_split_ratio * N_total_samples)
    n_test = N_total_samples - n_train - n_val

    ## Input Normalization for Training
    
    # X_all, Y_all = full_dataset.tensors[0], full_dataset.tensors[1]

    # X_train = X_all[:n_train]
    # X_val = X_all[n_train:n_train+n_val]
    # X_test = X_all[n_train+n_val:]
    # Y_train = Y_all[:n_train]
    # Y_val = Y_all[n_train:n_train+n_val]
    # Y_test = Y_all[n_train+n_val:]

    # # --- Normalize inputs (X) based on training data stats ---
    # x_mean = X_train.mean(dim=0, keepdim=True)
    # x_std = X_train.std(dim=0, keepdim=True) + 1e-8  # avoid division by zero

    # X_train = (X_train - x_mean) / x_std
    # X_val = (X_val - x_mean) / x_std
    # X_test = (X_test - x_mean) / x_std

    if n_test <= 0:
        print("Warning: Test set size is zero or negative. Adjusting validation split.")
        n_val = max(0, N_total_samples - n_train - 1) 
        n_test = N_total_samples - n_train - n_val
        if n_test <=0:
            raise ValueError("Cannot create non-empty train, val, and test sets with current ratios and data size.")

    print(f"Total windowed samples: {N_total_samples}")
    print(f"Train samples: {n_train}, Validation samples: {n_val}, Test samples: {n_test}")

    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, N_total_samples))

    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)
    test_ds = Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # --- Model Configuration (Change Parameterss Here) ---
    num_signals_in_window = 2 
    windowed_input_dim_encoder = window_size_config * num_signals_in_window
    latent_dim_shared = 8
    encoder_hidden_dim_shared = 64
    shnd_bln_units_config = [32, 32] 
    dt_simulation_step = 1.0 
    actual_physical_control_input_dim = 1

    args_for_lyaproj_config = Dict2Class({
        "proj_fn": proj_fn, "nx": latent_dim_shared, "nh": 64, "nph": 64,
        "alpha": 0.01, "eps": 0.01, "rehu_factor": 0.01, "scale_fx": False
    })
    shnd_hyperparams_config = { "shnd_mu": 0.1, "shnd_nu": 2.0, "shnd_eps": 0.01 }

    model = LatentIOModel(
        windowed_input_dim=windowed_input_dim_encoder,
        latent_dim=latent_dim_shared,
        encoder_hidden_dim=encoder_hidden_dim_shared, # For Encoder and Decoders
        shnd_bln_units=shnd_bln_units_config,         # For BiLipNet in SHND variants
        dt=dt_simulation_step,
        model_type=model_label, # THIS IS THE KEY SWITCH
        args_for_lyaproj=args_for_lyaproj_config if model_label == "LyaProj" else None,
        actual_control_input_dim=actual_physical_control_input_dim, # For pH_SHND
        shnd_mu=shnd_hyperparams_config["shnd_mu"], # Pass individually
        shnd_nu=shnd_hyperparams_config["shnd_nu"],
        shnd_eps=shnd_hyperparams_config["shnd_eps"]
    ).to(device)


    # --- Training Loop ---
    num_epochs = 300 

    # # Separate params with and without weight decay (Weight Decay on R(x) of Port-Hamiltonian Model)

    # if hasattr(model.latent_dynamics_core, "R_net"):
    #     decay_params = list(model.latent_dynamics_core.R_net.parameters())
    #     no_decay_params = [p for n, p in model.named_parameters() if not any(p is d for d in decay_params)]
    #     optimizer = torch.optim.Adam([
    #         {'params': decay_params, 'weight_decay': 1e-4},
    #         {'params': no_decay_params, 'weight_decay': 0.0}
    #     ], lr=1e-3)
    # else:
    #     # fallback for models without R_net (like AE_SHND)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)          
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Learning Rate Scheduler, as from the original work
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    print(f"\nTraining model: IO_{model_label}" + (f"_{proj_fn}" if model_label == "LyaProj" and proj_fn else ""))
    for epoch in range(num_epochs):
        best_val_loss = float('inf')
        patience = 10 # Maximum Limit for Early Stopping
        wait = 0
        best_model_path = f"trained_models/best_io_{model_label.lower()}_{proj_fn.lower().replace('-', '_') if model_label == 'LyaProj' and proj_fn else ''}.pt"
        model.train()
        train_loss_epoch = 0
        for xb, yb in train_loader: 
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
            scheduler.step()
            train_loss_epoch += loss.item()
        avg_train_loss_epoch = train_loss_epoch / len(train_loader)
        model.eval()
        val_loss_epoch = 0
        with torch.no_grad(): 
            for xb_val, yb_val in val_loader:
                xb_val, yb_val = xb_val.to(device), yb_val.to(device)
                pred_val = model(xb_val)
                val_loss_epoch += F.mse_loss(pred_val, yb_val).item()
        avg_val_loss_epoch = val_loss_epoch / len(val_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {avg_train_loss_epoch:.6f} | Val Loss: {avg_val_loss_epoch:.6f}")
        if avg_val_loss_epoch < best_val_loss:
            best_val_loss = avg_val_loss_epoch
            wait = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}, best val loss: {best_val_loss:.6f}")
                break

    # --- Final Evaluation on Test Set (One-step prediction) ---
    model.eval()
    y_true_test_list, y_pred_test_list = [], []
    with torch.no_grad():
        for xb_test, yb_test in test_loader:
            xb_test, yb_test = xb_test.to(device), yb_test.to(device)
            pred_test_batch = model(xb_test)
            y_true_test_list.append(yb_test.cpu().numpy())
            y_pred_test_list.append(pred_test_batch.cpu().numpy())
    y_true_test_np = np.concatenate(y_true_test_list)
    y_pred_test_np = np.concatenate(y_pred_test_list)
    
    # # --- ADDED DEBUG PRINTS for one-step prediction metrics ---
    # print("\n--- Data passed to compute_metrics for ONE-STEP PREDICTION ---")
    # print(f"Type of y_true_test_np: {type(y_true_test_np)}, Shape: {y_true_test_np.shape}")
    # print(f"Type of y_pred_test_np: {type(y_pred_test_np)}, Shape: {y_pred_test_np.shape}")
    # if len(y_true_test_np) > 0: print("y_true_test_np (first 5):", y_true_test_np[:5])
    # if len(y_pred_test_np) > 0: print("y_pred_test_np (first 5):", y_pred_test_np[:5])
    # print("------------------------------------------------------------\n")
    one_step_pred_metrics = compute_metrics(y_true_test_np, y_pred_test_np)
    # # --- END ADDED DEBUG PRINTS for one-step prediction metrics ---

    # --- Simulation ---
    sim_pred_np, sim_true_np = simulate_io_model(
        model, u_full.copy(), theta_full.copy(), 
        window_size=window_size_config, 
        skip_initial=n_train + n_val, 
        device=device_str,
        best_model_path=best_model_path
    )
    
    # # --- ADDED DEBUG PRINTS for simulation metrics ---
    # print("\n--- Data passed to compute_metrics for SIMULATION ---")
    # print(f"Type of sim_true_np: {type(sim_true_np)}, Shape: {sim_true_np.shape if isinstance(sim_true_np, np.ndarray) else 'Not an ndarray'}")
    # print(f"Type of sim_pred_np: {type(sim_pred_np)}, Shape: {sim_pred_np.shape if isinstance(sim_pred_np, np.ndarray) else 'Not an ndarray'}")
    # if isinstance(sim_true_np, np.ndarray) and len(sim_true_np) > 0: print("sim_true_np (first 5):", sim_true_np[:5])
    # if isinstance(sim_pred_np, np.ndarray) and len(sim_pred_np) > 0: print("sim_pred_np (first 5):", sim_pred_np[:5])
    # print("-----------------------------------------------------\n")
    # # ---END ADDED DEBUG PRINTS for simulation metrics ---
    
    simulation_metrics = compute_metrics(sim_true_np, sim_pred_np)

    # --- Save Model and Results ---
    os.makedirs("trained_models", exist_ok=True)
    model_filename_suffix = f"_{proj_fn.lower().replace('-', '_')}" if model_label == "LyaProj" and proj_fn else ""
    torch.save(model.state_dict(), f"trained_models/io_{model_label.lower()}{model_filename_suffix}.pt")

    results_dict = {
        "Model": f"IO_{model_label}" + (f"_{proj_fn}" if model_label == "LyaProj" and proj_fn else ""),
        "Pred RMSE (rad)": one_step_pred_metrics[0],
        "Pred RMSE (deg)": one_step_pred_metrics[1],
        "Pred NRMSE (%)":  one_step_pred_metrics[2],
        "Sim RMSE (rad)":  simulation_metrics[0],
        "Sim RMSE (deg)":  simulation_metrics[1],
        "Sim NRMSE (%)":   simulation_metrics[2],
    }
    return results_dict

# --- Main Execution Block ---
if __name__ == "__main__":
    if torch.cuda.is_available(): device_to_use = 'cuda'
    else: device_to_use = 'cpu'
    all_run_results = []
    
    # Port-Hamiltonian SHND
    all_run_results.append(train_and_evaluate(model_label="pH_SHND", device_str=device_to_use))
    
    # Autoencoder SHND
    all_run_results.append(train_and_evaluate(model_label="AE_SHND", device_str=device_to_use))
    
    # LyaProj Models (ensure LyaProj.py is fixed for grad handling)
    all_run_results.append(train_and_evaluate(model_label="LyaProj", proj_fn="ICNN", device_str=device_to_use))
    all_run_results.append(train_and_evaluate(model_label="LyaProj", proj_fn="NN-REHU", device_str=device_to_use))

    print("\n# Final Evaluation Table")
    header_cols = ["Model", "Pred RMSE (rad)", "Pred RMSE (deg)", "Pred NRMSE (%)",
                   "Sim RMSE (rad)", "Sim RMSE (deg)", "Sim NRMSE (%)"]
    print("| " + " | ".join(f"{col:<20}" for col in header_cols) + " |")
    print("|" + "----------------------|" * len(header_cols))
    for result_row in all_run_results:
        row_values_str = []
        for col_key in header_cols:
            val = result_row[col_key]
            if isinstance(val, float): row_values_str.append(f"{val:<20.4f}")
            else: row_values_str.append(f"{str(val):<20}")
        print("| " + " | ".join(row_values_str) + " |")

