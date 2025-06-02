##WORKING FILES FOR pH-SHND sims######

###IO_SHND.py####
###PORT HAMILTONIAN STRUCTURES BELOW######
import torch
import torch.nn as nn
import torch.nn.functional as F
from PLNet import PLNet, BiLipNet # Make sure PLNet.py is in PYTHONPATH or same directory
from LyaProj import stable_dyn     # Make sure LyaProj.py is in PYTHONPATH or same directory

# ----------------------------------------
# Encoder (shared)
# ----------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    def forward(self, x):
        return self.net(x)

# ----------------------------------------
# Decoder (for LyaProj AND AE_SHND)
# ----------------------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim=1): # output_dim usually 1 for theta, SISO System
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, z):
        return self.net(z)

# ----------------------------------------
# AE_SHND_Core: Hamiltonian Dynamics WITHOUT explicit B(z)u input
# ----------------------------------------
class AE_SHND_Core(nn.Module):
    def __init__(self, latent_dim, bln_units, mu=0.1, nu=2.0, eps=0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.eps = eps

        # mu and nu are for BiLipNet, which defines H
        net = BiLipNet(self.latent_dim, bln_units, mu, nu)
        self.H_module = PLNet(net, use_bias=False) 

        # Networks for J and R (or their factors)
        self.JR_net = nn.Sequential(
            nn.Linear(self.latent_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2 * (self.latent_dim ** 2)) # J_factor, R_factor
        )
    
    def _prepare_tensor_for_grad(self, t):
        if not t.requires_grad: return t.requires_grad_(True)
        elif t.grad_fn is not None: return t.clone().detach().requires_grad_(True)
        return t

    def forward(self, z): # Takes only latent state, z. u is implicitly handled by encoder
        batch_size, d = z.shape
        if d != self.latent_dim:
            raise ValueError(f"Input z dimension {d} does not match latent_dim {self.latent_dim}")

        gH = None
        original_grad_state = torch.is_grad_enabled()
        try:
            if not original_grad_state: torch.set_grad_enabled(True)
            z_for_H = self._prepare_tensor_for_grad(z)
            H_val = self.H_module(z_for_H)
            gH = torch.autograd.grad(H_val.sum(), z_for_H, create_graph=True, retain_graph=True)[0]
        finally:
            if not original_grad_state: torch.set_grad_enabled(False)
        
        if gH is None: raise RuntimeError("gH was not computed in AE_SHND_Core.")

        JR_out = self.JR_net(z)
        J_factor = JR_out[:, :self.latent_dim**2].view(batch_size, self.latent_dim, self.latent_dim)
        R_factor = JR_out[:, self.latent_dim**2:].view(batch_size, self.latent_dim, self.latent_dim)

        J = J_factor - J_factor.transpose(1, 2) # J = S - S^T
        R = R_factor @ R_factor.transpose(1, 2) # R = L L^T

        dz = torch.bmm(J - R, gH.unsqueeze(-1)).squeeze(-1) - self.eps * gH
        return dz # Only returns dz

# ----------------------------------------
# pH_SHND_Core: Port-Hamiltonian Dynamics Core
# ----------------------------------------
class pH_SHND_Core(nn.Module): 
    def __init__(self, latent_dim, bln_units, control_input_dim, mu=0.1, nu=2.0, eps=0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.control_input_dim = control_input_dim
        self.eps = eps
        net = BiLipNet(self.latent_dim, bln_units, mu, nu)
        self.H_module = PLNet(net, use_bias=False) 
        self.JR_net = nn.Sequential( 
            nn.Linear(self.latent_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2 * (self.latent_dim ** 2))
        )
        # self.J_net = nn.Sequential(
        #     nn.Linear(latent_dim, 64), nn.Tanh(),
        #     nn.Linear(64, 64), nn.Tanh(),
        #     nn.Linear(64, latent_dim ** 2)
        # )

        # self.R_net = nn.Sequential(
        #     nn.Linear(latent_dim, 64), nn.Tanh(),
        #     nn.Linear(64, 64), nn.Tanh(),
        #     nn.Linear(64, latent_dim ** 2)
        # )        
        self.Bnet = nn.Sequential(  #The Bu and B^T term in port-Hamiltonian form
            nn.Linear(self.latent_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, self.latent_dim * self.control_input_dim)
        )

    def _prepare_tensor_for_grad(self, t):
        if not t.requires_grad: return t.requires_grad_(True)
        elif t.grad_fn is not None: return t.clone().detach().requires_grad_(True)
        return t

    def compute_dz_and_output(self, z, u_current_physical):  #Computing z_dot and y for Port-Hamiltonian
        batch_size, _ = z.shape
        gH = None
        original_grad_state = torch.is_grad_enabled()
        try:
            if not original_grad_state: torch.set_grad_enabled(True)
            z_for_H = self._prepare_tensor_for_grad(z)
            H_val = self.H_module(z_for_H)
            gH = torch.autograd.grad(H_val.sum(), z_for_H, create_graph=True, retain_graph=True)[0]
        finally:
            if not original_grad_state: torch.set_grad_enabled(False)
        if gH is None: raise RuntimeError("gH was not computed.")

        JR_out = self.JR_net(z)
        J_flat = JR_out[:, :self.latent_dim ** 2]
        R_factor_flat = JR_out[:, self.latent_dim ** 2:]
        J = J_flat.view(batch_size, self.latent_dim, self.latent_dim)
        J = J - J.transpose(1, 2)
        R_factor = R_factor_flat.view(batch_size, self.latent_dim, self.latent_dim)
        R = R_factor @ R_factor.transpose(1, 2)

        # J_flat = self.J_net(z)
        # R_factor_flat = self.R_net(z)
        # J = J_flat.view(batch_size, self.latent_dim, self.latent_dim)
        # J = J - J.transpose(1, 2)
        # R_factor = R_factor_flat.view(batch_size, self.latent_dim, self.latent_dim)
        # R = R_factor @ R_factor.transpose(1, 2)

        B_elements = self.Bnet(z)
        B = B_elements.view(batch_size, self.latent_dim, self.control_input_dim)
        Bu = torch.bmm(B, u_current_physical.unsqueeze(-1)).squeeze(-1)
        term_J_R_gH = torch.bmm(J - R, gH.unsqueeze(-1)).squeeze(-1)
        dz = term_J_R_gH + Bu - self.eps * gH
        y_pred = torch.bmm(B.transpose(1, 2), gH.unsqueeze(-1)).squeeze(-1)
        return dz, y_pred

    def forward(self, z, u_current_physical, dt=1.0):
        dz1, _ = self.compute_dz_and_output(z, u_current_physical)
        z1_intermediate = z + dt * dz1
        dz2, y_pred = self.compute_dz_and_output(z1_intermediate, u_current_physical)
        z_next = z + 0.5 * dt * (dz1 + dz2)
        return z_next, y_pred

# ----------------------------------------
# LyaProj Dynamics Core (remains the same)
# ----------------------------------------
class LatentLyaProj(nn.Module):
    # Assuming LyaProj.py/Dynamics.forward is fixed for grad handling)
    def __init__(self, args_for_lyaproj): 
        super().__init__()
        self.dyn_module = stable_dyn(args_for_lyaproj) 
    def forward(self, z, u=None, dt=None): 
        dz = self.dyn_module(z) 
        return z + dt * dz, None


# ----------------------------------------
# Unified Latent IO Model
# ----------------------------------------
class LatentIOModel(nn.Module):
    def __init__(self, windowed_input_dim, latent_dim, encoder_hidden_dim, 
                 shnd_bln_units, 
                 dt=1.0, model_type="pH_SHND", # Default to pH_SHND
                 args_for_lyaproj=None, 
                 actual_control_input_dim=1, 
                 # Hyperparams for SHND variants (mu, nu, eps for H, J, R)
                 shnd_mu=0.1, shnd_nu=2.0, shnd_eps=0.01):
        super().__init__()
        self.dt = dt
        self.model_type = model_type
        self.actual_control_input_dim = actual_control_input_dim # Used by pH_SHND

        self.encoder = Encoder(windowed_input_dim, latent_dim, encoder_hidden_dim)

        if model_type == "pH_SHND": # Port-Hamiltonian version
            self.latent_dynamics_core = pH_SHND_Core(latent_dim, shnd_bln_units, 
                                                  self.actual_control_input_dim,
                                                  mu=shnd_mu, nu=shnd_nu, eps=shnd_eps)
            self.decoder = None # pH_SHND outputs y_pred directly
        elif model_type == "AE_SHND": # Autoencoder SHND version
            self.latent_dynamics_core = AE_SHND_Core(latent_dim, shnd_bln_units,
                                                   mu=shnd_mu, nu=shnd_nu, eps=shnd_eps)
            self.decoder = Decoder(latent_dim, encoder_hidden_dim) # AE_SHND uses a decoder
        elif model_type == "LyaProj":
            assert args_for_lyaproj is not None
            if not hasattr(args_for_lyaproj, 'nx') or args_for_lyaproj.nx != latent_dim:
                if hasattr(args_for_lyaproj, 'nx'): 
                    print(f"Warning: LyaProj args.nx ({args_for_lyaproj.nx}) differs from latent_dim ({latent_dim}). Overriding.")
                args_for_lyaproj.nx = latent_dim
            self.latent_dynamics_core = LatentLyaProj(args_for_lyaproj)
            self.decoder = Decoder(latent_dim, encoder_hidden_dim)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def forward(self, x_window):
        z_t = self.encoder(x_window)
        current_physical_u_t = x_window[:, -self.actual_control_input_dim:]

        y_model_output = None
        if self.model_type == "pH_SHND":
            _z_next_ignored, y_model_output = self.latent_dynamics_core(z_t, current_physical_u_t, self.dt)
        elif self.model_type == "AE_SHND":
            # AE_SHND_Core.forward(z) returns dz
            # It doesn't use current_physical_u_t directly in its core dynamics call
            dz = self.latent_dynamics_core(z_t) # Pass only z_t
            # Simple Euler integration for z_next for AE_SHND, can be changed to RK if desired
            z_next = z_t + self.dt * dz 
            y_model_output = self.decoder(z_next)
        elif self.model_type == "LyaProj":
            # LatentLyaProj.forward(z, u, dt) returns (z_next, None)
            z_next, _ = self.latent_dynamics_core(z_t, current_physical_u_t, self.dt)
            y_model_output = self.decoder(z_next)
        
        if y_model_output is None:
            raise RuntimeError(f"y_model_output not set for model_type {self.model_type}")
        return y_model_output


