import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, periodogram
from IO_SHND import LatentIOModel

def load_model(model_label, proj_fn=None, device='cpu'):
    latent_dim = 8
    encoder_hidden_dim = 64
    shnd_bln_units = [32, 32]
    dt = 1.0
    control_input_dim = 1

    class Dict2Class:
        def __init__(self, d): self.__dict__.update(d)

    args_for_lyaproj = Dict2Class({
        "proj_fn": proj_fn, "nx": latent_dim, "nh": 64, "nph": 64,
        "alpha": 0.01, "eps": 0.01, "rehu_factor": 0.01, "scale_fx": False
    }) if proj_fn else None

    model = LatentIOModel(
        windowed_input_dim=4,
        latent_dim=latent_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        shnd_bln_units=shnd_bln_units,
        dt=dt,
        model_type=model_label,
        args_for_lyaproj=args_for_lyaproj,
        actual_control_input_dim=control_input_dim,
        shnd_mu=0.1, shnd_nu=2.0, shnd_eps=0.01
    ).to(device)

    suffix = f"_{proj_fn.lower().replace('-', '_')}" if proj_fn else ""
    model_path = f"trained_models/Best Models/best_io_{model_label.lower()}_{proj_fn.lower().replace('-', '_') if model_label == 'LyaProj' and proj_fn else ''}.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def build_io_window_dataset(theta, u, window_size=2):
    X_list, Y_list = [], []
    for t in range(window_size - 1, len(theta) - 1):
        theta_w = theta[t - window_size + 1 : t + 1].flatten()
        u_w = u[t - window_size + 1 : t + 1].flatten()
        x_instance = np.concatenate([theta_w, u_w])
        y_instance = theta[t + 1]
        X_list.append(x_instance)
        Y_list.append(y_instance)
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)
    return X, Y

def evaluate_model_and_plot(model_label, proj_fn=None, device='cpu'):
    model = load_model(model_label, proj_fn, device)
    data = np.load("training-val-test-data.npz")
    theta = data["th"].reshape(-1, 1)
    u = data["u"].reshape(-1, 1)
    X, Y = build_io_window_dataset(theta, u, window_size=2)
    X_test, Y_test = X[-3501:], Y[-3501:]

    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy().flatten()
        targets = Y_test.cpu().numpy().flatten()
        residuals = targets - preds

        # Save residuals for later frequency overlay analysis
        model_tag = f"io_{model_label.lower()}"
        if proj_fn:
            model_tag += f"_{proj_fn.lower().replace('-', '_')}"
        save_path = f"trained_models/Best Models/residuals_{model_tag}.npy"
        np.save(save_path, residuals)
        print(f"Saved residuals to: {save_path}")

    freqs, psd = welch(residuals, nperseg=256)
    plt.figure()
    plt.semilogy(freqs, psd)
    title = f"Residual Spectrum - IO_{model_label}" + (f"_{proj_fn}" if proj_fn else "")
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("PSD")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"residual_spectrum_{model_label.lower()}{'_' + proj_fn.lower().replace('-', '_') if proj_fn else ''}.png")
    print(f"Saved: {title}")


# === MAIN EVALUATION ===
models_to_run = [
    ("pH_SHND", None),
    ("AE_SHND", None),
    ("LyaProj", "ICNN"),
    ("LyaProj", "NN-REHU")
]

for label, proj in models_to_run:
    evaluate_model_and_plot(label, proj)

data = np.load("training-val-test-data.npz")
u = data["u"].reshape(-1)[-3501:]  # Same length as test set
th = data["th"].reshape(-1)[-3501:]  # Same length as test set
fs = 1.0  # sampling rate

# Input excitation spectrum
frequencies_th, psd_th = welch(th, fs=fs, nperseg=1024)

# Adjust this based on where the input has support (you saw this visually around 0.06 in earlier plots)
excited_band_cutoff = 0.06

# Load residuals from saved .npy files
residuals = {
    "IO_pH_SHND": np.load("trained_models/Best Models/residuals_io_ph_shnd.npy"),
    "IO_AE_SHND": np.load("trained_models/Best Models/residuals_io_ae_shnd.npy"),
    "IO_LyaProj_ICNN": np.load("trained_models/Best Models/residuals_io_lyaproj_icnn.npy"),
    "IO_LyaProj_NN_REHU": np.load("trained_models/Best Models/residuals_io_lyaproj_nn_rehu.npy"),
}

# Plot overlay: residual spectrum vs input excitation
for model_name, resid in residuals.items():
    f_resid, psd_resid = welch(resid, fs=fs, nperseg=1024)

    plt.figure(figsize=(10, 5))
    plt.semilogy(f_resid, psd_resid, label=f"Prediction Residual - {model_name}")
    plt.semilogy(frequencies_th, psd_th, label="Output Spectrum (theta)", alpha=0.5)
    plt.axvspan(0, excited_band_cutoff, color='orange', alpha=0.3, label="Excited Band")
    plt.title(f"Prediction Residual vs Output Spectrum - {model_name}")
    plt.xlabel("Frequency")
    plt.ylabel("PSD (log scale)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"trained_models/Best Models/residual_vs_input_{model_name.lower()}.png")
    plt.show()    
