# SHND Port-Hamiltonian Training Experiments

This repository contains training script and model definitions for reproducing issues with the Port-Hamiltonian Stable Hamiltonian Neural Dynamics model (adapted from https://github.com/ruigangwang7/StableNODE).


## (New) Repository Contents

- `IO_train.py`  
  Training script with logging and preliminary result tabulation.
  
- `IO_SHND.py`  
  Model definitions for four I/O Model variants:
  - Port-Hamiltonian Stable Neural Dynamics (pH_SHND)
  - Encoder → SHND → Decoder (AE_SHND)
  - Encoder → Input-Convex Neural Network (ICNN) → Decoder (LyaProj-LyaProg-NN-ICNN)
  - Encoder → ReHU Positive Definite Lyapunov Network → Decoder (LyaProj-NN-REHU)
  
- `training-val-test-data.[mat|csv|npz]`  
  Dataset for single pendulum system, used for training, validation, and test splits.

- `trained_models/`  
  Contains saved model checkpoints from `IO_train.py`.

## How to Run
1. Clone the repository
2. Install dependencies
3. Run: `python IO_train.py`