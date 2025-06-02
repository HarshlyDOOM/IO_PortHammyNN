# import torch
# import torch.nn.functional as F
# from torch import nn
# from ICNN import * 
# from utils import * 

# class Dynamics(nn.Module):
#     def __init__(self, fhat, V, alpha=0.01, scale_fx: bool = False):
#         super().__init__()
#         self.fhat = fhat
#         self.V = V
#         self.alpha = alpha
#         self.scale_fx = scale_fx

#     def forward(self, x):
#         fx = self.fhat(x)
#         if self.scale_fx:
#             fx = fx / fx.norm(p=2, dim=1, keepdim=True).clamp(min=1.0)

#         Vx = self.V(x)
#         gV = torch.autograd.grad([a for a in Vx], [x], create_graph=True, only_inputs=True)[0]
#         rv = fx - gV * (F.relu((gV*fx).sum(dim=1) + self.alpha*Vx[:,0])/(gV**2).sum(dim=1))[:,None]

#         return rv
    
# def stable_dyn(args):

#     nx, nh, nph = args.nx, args.nh, args.nph
#     # #Modifications
#     # physical_state_dim = 2 * args.n # Or pass this as a separate arg if n_links not in args
#     # fhat = nn.Sequential(
#     #     nn.Linear(nx, nh), nn.ReLU(),
#     #     nn.Linear(nh, nh), nn.ReLU(),
#     #     nn.Linear(nh, physical_state_dim) # Output dim is physical_state_dim
#     # )    
#     fhat = nn.Sequential(
#         nn.Linear(nx, nh), nn.ReLU(),
#         nn.Linear(nh, nh), nn.ReLU(),
#         nn.Linear(nh, nx)
#     )

#     if args.proj_fn == "PSICNN":
#         V = PosDefICNN([nx, nph, nph, 1], eps=args.eps, negative_slope=0.3)
#     elif args.proj_fn == "ICNN":
#         V = ICNN([nx, nph, nph, 1])
#     elif args.proj_fn == "PSD":
#         V = MakePSD(ICNN([nx, nph, nph, 1]), nx, eps=args.eps, d=1.0)
#     elif args.proj_fn == "PSD-REHU":
#         V = MakePSD(ICNN([nx, nph, nph, 1], activation=ReHU(float(args.rehu_factor))), nx, eps=args.eps, d=1.0)
#     elif args.proj_fn == "NN-REHU":
#         seq = nn.Sequential(
#                 nn.Linear(nx, nph,), nn.ReLU(),
#                 nn.Linear(nph, nph), nn.ReLU(),
#                 nn.Linear(nph, 1), ReHU(args.rehu_factor)
#             )
#         V = MakePSD(seq, nx, eps=args.eps, d=1.0)
#     elif args.proj_fn == "EndPSICNN":
#         V = nn.Sequential(
#             nn.Linear(nx, nph, bias=False), nn.LeakyReLU(),
#             nn.Linear(nph, nx, bias=False), nn.LeakyReLU(),
#             PosDefICNN([nx, nph, nph, 1], eps=args.eps, negative_slope=0.3)
#         )
#     elif args.proj_fn == "NN":
#         V = nn.Sequential(
#             nn.Linear(nx, nph,), nn.ReLU(),
#             nn.Linear(nph, nph), nn.ReLU(),
#             nn.Linear(nph, 1)
#         )

#     model = Dynamics(fhat, V, alpha=args.alpha, scale_fx=args.scale_fx)

#     return model 

# if __name__ == "__main__":
#     n = 2 
#     batch_size = 5
#     args = {
#         "proj_fn": "ICNN",
#         "nx": 2*n, 
#         "nh": 64,
#         "nph": 64,
#         "alpha": 0.01,
#         "eps": 0.01,
#         "rehu_factor": 0.01,
#         "scale_fx": False 
#     }
#     args = Dict2Class(args)
#     model = stable_dyn(args)
#     x = torch.rand((batch_size, 2*n), requires_grad=True)
#     y = model(x)
#     print(y)

# In LyaProj.py

import torch
import torch.nn.functional as F
from torch import nn
# Make sure ICNN.py and utils.py (for Dict2Class) are accessible
from ICNN import ICNN, ReHU, MakePSD, PosDefICNN # Assuming these are all in ICNN.py
from utils import Dict2Class

class Dynamics(nn.Module):
    def __init__(self, fhat, V_module, alpha=0.01, scale_fx: bool = False): # Renamed V to V_module
        super().__init__()
        self.fhat = fhat
        self.V_module = V_module # This is the nn.Module that computes V(x)
        self.alpha = alpha
        self.scale_fx = scale_fx

    def _prepare_tensor_for_grad(self, t):
        """Helper to prepare a tensor for autograd.grad, usable within no_grad context."""
        if not t.requires_grad:
            return t.requires_grad_(True)
        elif t.grad_fn is not None:
            return t.clone().detach().requires_grad_(True)
        return t

    def forward(self, x): # x is the latent state z
        fx = self.fhat(x)
        if self.scale_fx:
            fx = fx / fx.norm(p=2, dim=1, keepdim=True).clamp(min=1.0)

        gV = None
        Vx_val = None # To store the computed V(x_for_V)
        original_grad_state = torch.is_grad_enabled()
        try:
            if not original_grad_state:
                torch.set_grad_enabled(True)

            x_for_V = self._prepare_tensor_for_grad(x)
            Vx_val = self.V_module(x_for_V) # Vx_val shape: (batch, 1) or (batch,)

            # Ensure Vx_val.sum() is used
            if Vx_val.ndim == 1:
                vx_sum_for_grad = Vx_val.sum()
            else: # Assuming (batch, 1)
                vx_sum_for_grad = Vx_val.sum()

            gV = torch.autograd.grad(vx_sum_for_grad, x_for_V, create_graph=True, retain_graph=True)[0]
        finally:
            if not original_grad_state:
                torch.set_grad_enabled(False)
        
        if gV is None or Vx_val is None:
            raise RuntimeError("gV or Vx_val was not computed in LyaProj.Dynamics.")

        # For the alpha term, use Vx_val directly
        # If V_module outputs (batch,1), then Vx_val[:,0] is (batch,) for the alpha term
        if Vx_val.ndim > 1 and Vx_val.shape[1] == 1:
            vx_for_alpha_term = Vx_val[:,0]
        else: # Assuming Vx_val is already (batch,)
            vx_for_alpha_term = Vx_val
            
        # Lyapunov condition term for projection
        lyap_decrease_term = (gV * fx).sum(dim=1) + self.alpha * vx_for_alpha_term
        projection_magnitude = F.relu(lyap_decrease_term) / ((gV**2).sum(dim=1) + 1e-6) # Add epsilon for stability

        rv = fx - gV * projection_magnitude[:, None]
        return rv

def stable_dyn(args):
    # args should be an object with attributes: nx, nh, nph, proj_fn, eps, alpha, rehu_factor
    nx, nh, nph = args.nx, args.nh, args.nph
    
    fhat = nn.Sequential(
        nn.Linear(nx, nh), nn.ReLU(),
        nn.Linear(nh, nh), nn.ReLU(),
        nn.Linear(nh, nx)
    )

    V_network = None # This will be the actual V(x) nn.Module
    if args.proj_fn == "PSICNN":
        V_network = PosDefICNN([nx, nph, nph, 1], eps=args.eps, negative_slope=0.3)
    elif args.proj_fn == "ICNN":
        V_network = ICNN([nx, nph, nph, 1]) # ICNN outputs (batch, 1)
    elif args.proj_fn == "PSD":
        # MakePSD expects f to be the base network, and n to be input dim for f(zeros(1,n))
        base_icnn_for_psd = ICNN([nx, nph, nph, 1])
        V_network = MakePSD(base_icnn_for_psd, nx, eps=args.eps, d=1.0)
    elif args.proj_fn == "PSD-REHU":
        base_icnn_rehu_for_psd = ICNN([nx, nph, nph, 1], activation=ReHU(float(args.rehu_factor)))
        V_network = MakePSD(base_icnn_rehu_for_psd, nx, eps=args.eps, d=1.0)
    elif args.proj_fn == "NN-REHU":
        # Base NN for MakePSD
        base_nn_for_rehu_psd = nn.Sequential(
                nn.Linear(nx, nph), nn.ReLU(),
                nn.Linear(nph, nph), nn.ReLU(),
                nn.Linear(nph, 1), ReHU(args.rehu_factor) # Ensure ReHU output is (batch, 1) or (batch,)
            )
        V_network = MakePSD(base_nn_for_rehu_psd, nx, eps=args.eps, d=1.0)
    # ... (other proj_fn options if you have them) ...
    else:
        raise ValueError(f"Unsupported proj_fn: {args.proj_fn}")

    model = Dynamics(fhat, V_network, alpha=args.alpha, scale_fx=args.scale_fx)
    return model

if __name__ == "__main__":
    # Basic test for LyaProj.Dynamics
    print("Testing LyaProj.Dynamics...")
    batch_size = 5
    latent_dim = 8
    
    args_test = Dict2Class({
        "proj_fn": "ICNN", # Test with ICNN
        "nx": latent_dim, 
        "nh": 64,
        "nph": 64,
        "alpha": 0.01,
        "eps": 0.01, # eps for PosDefICNN or MakePSD
        "rehu_factor": 0.01, # for ReHU activation or MakePSD with ReHU
        "scale_fx": False 
    })
    
    test_model = stable_dyn(args_test)
    test_x = torch.rand((batch_size, latent_dim), requires_grad=False) # Start with False, it will be handled

    # Test in no_grad context
    with torch.no_grad():
        print("Testing LyaProj.Dynamics within torch.no_grad():")
        output_no_grad = test_model(test_x.clone()) # Pass a clone
        print("Output (no_grad):", output_no_grad.shape)

    # Test with grad enabled context
    print("\nTesting LyaProj.Dynamics with gradients enabled:")
    test_x_grad = test_x.clone().requires_grad_(True)
    output_grad = test_model(test_x_grad)
    print("Output (grad enabled):", output_grad.shape)
    
    # Test backward pass if needed (though not strictly for this error)
    # output_grad.sum().backward()
    # print("Backward pass successful.")
    print("LyaProj.Dynamics test complete.")