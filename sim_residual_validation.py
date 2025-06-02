import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
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

    model_path = f"trained_models/Best Models/best_io_{model_label.lower()}_{proj_fn.lower().replace('-', '_') if model_label == 'LyaProj' and proj_fn else ''}.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

@torch.no_grad()
def simulate_model(model, theta, u, device='cpu', window_size=2):
    N = 3501
    sim_preds = []

    # Start from skip_initial = len(theta) - N - (window_size - 1)
    start_idx = len(theta) - N - (window_size - 1)
    
    # Initialize history buffers
    theta_hist = theta[start_idx : start_idx + window_size, 0].tolist()
    u_hist     = u[start_idx : start_idx + window_size, 0].tolist()

    for _ in range(N):
        x_input = np.array(theta_hist + u_hist, dtype=np.float32)
        x_tensor = torch.from_numpy(x_input).unsqueeze(0).to(device)
        pred = model(x_tensor).squeeze().cpu().item()
        sim_preds.append(pred)

        # Update buffers
        theta_hist.pop(0)
        theta_hist.append(pred)
        u_hist.pop(0)
        next_u_idx = start_idx + len(sim_preds) + window_size - 1
        if next_u_idx < len(u):
            u_hist.append(u[next_u_idx, 0])
        else:
            u_hist.append(0.0)  # pad with 0 if out of range

    return np.array(sim_preds)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    data = np.load("training-val-test-data.npz")
    theta = data["th"].reshape(-1, 1)
    u = data["u"].reshape(-1, 1)

    # === Output Spectrum ===
    theta_test = theta[-3501:].flatten()
    fs = 1.0
    freq_theta, psd_theta = welch(theta_test, fs=fs, nperseg=1024)
    excited_band_cutoff = 0.06
    plt.figure(figsize=(10, 5))
    plt.semilogy(freq_theta, psd_theta, label="Output Spectrum (θ)", alpha=0.5)    
    plt.title("Output Spectrum ")
    plt.xlabel("Frequency")
    plt.ylabel("PSD (log scale)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout() 
    plt.savefig("OutputSpectrum.png")
    plt.close()

    configs = [
        ("pH_SHND", None),
        ("AE_SHND", None),
        ("LyaProj", "ICNN"),
        ("LyaProj", "NN-REHU")
    ]

    for model_label, proj_fn in configs:
        model = load_model(model_label, proj_fn, device)
        preds = simulate_model(model, theta, u, device=device, window_size=2)
        targets = theta[-3501:, 0]
        residuals = targets - preds

        f_resid, psd_resid = welch(residuals, fs=fs, nperseg=1024)
        print("Residual stats:", np.min(residuals), np.max(residuals), np.mean(residuals))
        print("Preds stats:", np.min(preds), np.max(preds), np.mean(preds))
        print("Targets stats:", np.min(targets), np.max(targets), np.mean(targets))

        plt.figure(figsize=(10, 5))
        plt.semilogy(f_resid, psd_resid, label=f"Sim Residual - IO_{model_label}" + (f"_{proj_fn}" if proj_fn else ""))
        plt.semilogy(freq_theta, psd_theta, label="Output Spectrum (θ)", alpha=0.5)
        plt.axvspan(0, excited_band_cutoff, color='orange', alpha=0.3, label="Excited Band")
        plt.title(f"Sim Residual vs Output Spectrum - IO_{model_label}" + (f"_{proj_fn}" if proj_fn else ""))
        plt.xlabel("Frequency")
        plt.ylabel("PSD (log scale)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"trained_models/Best Models/sim_residual_vs_output_io_{model_label.lower()}" +
                    (f"_{proj_fn.lower().replace('-', '_')}" if proj_fn else "") + ".png")
        plt.close()

if __name__ == "__main__":
    main()
