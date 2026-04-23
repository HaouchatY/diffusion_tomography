import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import scipy.sparse as sp
import os
import copy
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

SEED = 42
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

from helpers import *

H, W = 120, 40
weights_path = Path("fusion_epochs/fusion_epoch_2193.pth")
assert weights_path.exists(), f"Missing checkpoint: {weights_path}"

num_train_steps = 1000
schedule = build_schedule(num_train_steps=num_train_steps, device=device)

model = MyTinyUNet(in_c=1, out_c=1, n_steps=num_train_steps).to(device)
raw_state = torch.load(weights_path, map_location=device)

if isinstance(raw_state, dict) and "state_dict" in raw_state:
    raw_state = raw_state["state_dict"]

if isinstance(raw_state, dict) and any(k.startswith("network.") for k in raw_state.keys()):
    net_state = {k[len("network."):]: v for k, v in raw_state.items() if k.startswith("network.")}
else:
    net_state = raw_state

missing, unexpected = model.load_state_dict(net_state, strict=False)
print("missing keys:", len(missing))
print("unexpected keys:", len(unexpected))
if missing or unexpected:
    raise RuntimeError("Checkpoint/model mismatch. Check architecture or key prefix handling.")

# Load test samples
samples_idxs = np.arange(0, 100)
test_samples_path = Path("/home/fusiontomo/Repos/diffusion_tomography/test_set_sxr_samples/sxr_samples_with_background_coarse.npy")
test_samples = torch.from_numpy(np.load(test_samples_path)[samples_idxs]).to(device).to(torch.float32)

# Load and normalize SXR operator
A_path = Path("forward_model/dmpx_geometry_matrix.npy")
#Path("forward_model/pilatus_geometry_matrix.npy")
#Path("forward_model/dmpx_geometry_matrix.npy") #Path("forward_model/forward_model_sxr_full_geometry.npy")
A_sparse = np.load(A_path).astype(np.float32)
A_sparse /= A_sparse.max()
A_tomo = torch.from_numpy(A_sparse).to(device)
print(f"A shape {tuple(A_tomo.shape)}")

# Clean synthetic measurements
y_tomo_clean = test_samples.reshape(-1, H * W) @ A_tomo.T

# Configuration grid
noise_levels = [0.5, 0.05, 0.25] # SXR->[0.05, 0.25, 0.5],  DMPX->[0.1, 0.5, 1.0], #pilatus->[0.05, 0.25, 0.5]
lambda_values = [50, 100, 200, 300, 400, 500] #[50] #[100, 200, 300, 400, 500] #[50, 100, 200, 300, 400, 500]
zeta_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]  ##[0.2, 0.4, 0.6, 0.8] #[0.2, 0.4, 0.6, 0.8] #[0.2, 0.4, 0.6, 0.8, 1.0]

num_eval_samples    = 10
num_posterior_samples = 100
num_inference_steps  = 250

csv_dir = Path("metrics_csv")
csv_dir.mkdir(exist_ok=True)

for sigma_y_tomo in noise_levels:

    for zeta in zeta_values:
        for lambda_ in lambda_values:
            
            csv_name = f"metrics_diffusion_dmpx_{sigma_y_tomo}_zeta{zeta}_lambda{lambda_}.csv"
            csv_path = csv_dir / csv_name
            if Path(csv_path).exists():
                print(f"Already exists {csv_path}, skipping...")
                continue

            print(f"-- sigma={sigma_y_tomo}, zeta={zeta}, lambda_={lambda_}")

            rows = []
            for i in range(num_eval_samples):
                np.random.seed(i)
                torch.manual_seed(i)
                y_tomo_noisy = y_tomo_clean[i] + sigma_y_tomo * torch.randn_like(y_tomo_clean[i])
                y_used  = copy.deepcopy(y_tomo_noisy)
                x_gt_i  = copy.deepcopy(test_samples[i])
                sigma_used = sigma_y_tomo

                posterior_tomo, _, _ = sample_posterior_diffpir(
                    model=model,
                    schedule=schedule,
                    A=A_tomo,
                    y=y_used,
                    image_shape=(1, H, W),
                    num_samples=num_posterior_samples,
                    num_inference_steps=num_inference_steps,
                    zeta=zeta,
                    sigma_y=sigma_used,
                    lambda_=lambda_,
                )

                post_mean = posterior_tomo.mean(dim=0).squeeze() 
                post_std  = posterior_tomo.std(dim=0).squeeze()  

                gt_np      = x_gt_i.cpu().numpy()
                mean_np    = post_mean.detach().cpu().numpy()
                data_range = float(gt_np.max() - gt_np.min())

                rows.append({
                    "sample_idx":    int(samples_idxs[i]),
                    "psnr":          psnr(gt_np, mean_np, data_range=data_range),
                    "ssim":          ssim(gt_np, mean_np, data_range=data_range),
                    "mse":           float(((post_mean - x_gt_i) ** 2).mean().item()),
                    "post_std_mean": float(post_std.mean().item()),
                    "sigma_used":    float(sigma_used),
                    "zeta":          zeta,
                    "lambda_":       lambda_,
                })

            df = pd.DataFrame(rows)
            csv_name = f"metrics_diffusion_dmpx_{sigma_y_tomo}_zeta{zeta}_lambda{lambda_}.csv"
            csv_path = csv_dir / csv_name
            df.to_csv(csv_path, index=False)
            print(f"   saved {csv_path}")
            print(df.describe()[["psnr", "ssim", "mse"]])