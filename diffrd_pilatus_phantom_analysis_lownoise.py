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
import reaction_diffusion.routines.tomo_fusion.tools.helpers as tomo_helps
import reaction_diffusion.routines.tomo_fusion.functionals_definition as fct_def

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

# Load and normalize Pilatus forward model operator
A_path = Path("forward_model/pilatus_geometry_matrix.npy")
A_sparse = np.load(A_path).astype(np.float32)
A_sparse /= A_sparse.max()
A_tomo = torch.from_numpy(A_sparse).to(device)
print(f"A shape {tuple(A_tomo.shape)}")

# Load test samples
num_eval_samples = 1000
samples_idxs = np.arange(0, num_eval_samples)
test_samples_path = Path("/home/fusiontomo/Repos/diffusion_tomography/test_set_sxr_samples/sxr_samples_with_background_coarse.npy")
test_samples = torch.from_numpy(np.load(test_samples_path)[samples_idxs]).to(device).to(torch.float32)
trim_values_path = Path("/home/fusiontomo/Repos/diffusion_tomography/test_set_sxr_samples/trimming_values.npy")
trim_vals = torch.from_numpy(np.load(trim_values_path)[samples_idxs]).to(device).to(torch.float32)
psis_path = Path("/home/fusiontomo/Repos/diffusion_tomography/test_set_sxr_samples/psis_coarse.npy")
psis = np.load(psis_path)[samples_idxs]

# Load training samples (needed for the alpha normalization procedure)
train_samples_idxs = np.arange(0, 10000)
train_samples_path = Path("../../data/sxr_data/sxr_profiles_coarse")
train_samples_normalization = []
for sample_id_ in train_samples_idxs:
    path = os.path.join(train_samples_path, f"sxr_sample_{sample_id_}.npy")
    train_samples_normalization.append(np.load(path))
train_samples_normalization = torch.from_numpy(
    np.array(train_samples_normalization, dtype=np.float32)
).to(device)
# Training-sample projections for alpha normalization
y_tomo_train = train_samples_normalization.reshape(-1, H * W) @ A_tomo.T

# Load pre-computed noise realisations and scaling params
noise_realizations = torch.from_numpy(np.load("noise_realizations_pilatus.npy")).to(device).to(torch.float32)  # (1000, n_channels)
scaling_params      = torch.from_numpy(np.load("scaling_params.npy")).to(device).to(torch.float32)             # (1000,)

# Clean synthetic measurements
y_tomo_clean = test_samples.reshape(-1, H * W) @ A_tomo.T  # (1000, n_channels)

# Fixed hyperparameters — clip-dependent
# clip=True:  lambda_=25,  zeta=0.8
# clip=False: lambda_=200, zeta=0.8

noise_levels = [0.07, 0.35, 0.7]
clip_values  = [True] #[False, True] 
procedures   = [False, True]

HPARAMS = {
    0.07:  {"lambda_": 100, "lambda_rd": 1e-5,  "zeta": 0.9},
    0.35:  {"lambda_": 10, "lambda_rd": 1e-4,  "zeta": 0.9},
    0.7:   {"lambda_": 10, "lambda_rd": 1e-4,  "zeta": 0.9},
}

num_posterior_samples = 100
num_inference_steps   = 250

csv_dir = Path("metrics_csv_phantom_analysis_diffrd_correct_rdparams")
csv_dir.mkdir(exist_ok=True)

for sigma_y_tomo in noise_levels:

    # load rd hyperparameters
    # csv_name = f"metrics_csv_phantom_analysis_rd/metrics_rd_pilatus_{sigma_y_tomo}_false.csv"
    # if os.path.exists(csv_name):
    #     df = pd.read_csv(csv_name)
    #     df["noise"] = sigma_y_tomo
    #     anis_params = df['best_anis_param'][:num_eval_samples]
    #     print(anis_params.shape)
    # else:
    #     print(f"File {csv_name} not found.")

    for clip in clip_values:
        lambda_ = HPARAMS[sigma_y_tomo]["lambda_"]
        lambda_rd = HPARAMS[sigma_y_tomo]["lambda_rd"]
        zeta    = HPARAMS[sigma_y_tomo]["zeta"]

        for procedure in procedures:
            proc_str = "true"  if procedure else "false"
            clip_str = "True"  if clip      else "False"
            print(f"\n=== sigma={sigma_y_tomo}, clip={clip}, lambda_={lambda_}, lambda_rd={lambda_rd}, zeta={zeta}, procedure={procedure} ===")

            # load tuned anisotropic parameters
            if sigma_y_tomo == 0.07:
                csv_anis_params = f"metrics_csv_phantom_analysis_rd/metrics_rd_pilatus_rd_alpha_by_camera_{sigma_y_tomo}_{proc_str}.csv"
            else:
                csv_anis_params = f"metrics_csv_phantom_analysis_rd/metrics_rd_pilatus_{sigma_y_tomo}_{proc_str}.csv"
            if os.path.exists(csv_anis_params):
                df = pd.read_csv(csv_anis_params)
                df["noise"] = sigma_y_tomo
                anis_params = df['best_anis_param'][samples_idxs].values
                print(anis_params.shape)
            else:
                raise RuntimeError(f"File {csv_anis_params} not found.")

            csv_name = f"metrics_diffusion_pilatus_{sigma_y_tomo}_clip{clip_str}_{proc_str}.csv"
            csv_path = csv_dir / csv_name
            if csv_path.exists():
                print(f"Already exists {csv_path}, skipping...")
                continue

            rows       = []
            post_means = []
            post_stds  = []

            for i in range(num_eval_samples):
                # Compute noisy tomographic data
                y_tomo_noisy = y_tomo_clean[i] + sigma_y_tomo * noise_realizations[i, :]

                y_used  = copy.deepcopy(y_tomo_noisy)
                x_gt_i  = copy.deepcopy(test_samples[i])

                if procedure:
                    scale = float(scaling_params[i])
                    y_used  = y_used  * scale
                    x_gt_i  = x_gt_i  * scale
                    sigma_est = estimate_noise_std(y_used.detach().cpu().numpy().reshape(1, -1))
                    sigma_est = float(sigma_est) if not hasattr(sigma_est, "item") else sigma_est.item()
                    alpha, _  = compute_normalization_coefficient(
                        y_used.reshape(1, -1), y_tomo_train, normalization_for_comparison="norm"
                    )
                    y_used     = y_used / alpha
                    sigma_used = sigma_est / alpha
                    print(f"  i={i}, scaling={scale:.4f}, alpha={alpha:.4f}, "
                          f"sigma_true={sigma_y_tomo:.4f}, sigma_used={sigma_used:.4f}")
                else:
                    scale      = 1.0
                    alpha      = 1.0
                    sigma_used = sigma_y_tomo

                # Build clipping mask if requested
                if clip:
                    mask_core = tomo_helps.define_core_mask(
                        psi=psis[i],
                        dim_shape=(H, W),
                        trim_values_x=trim_vals[i],
                    )
                    clipping_mask = torch.tensor(mask_core, dtype=torch.float32, device=device)
                else:
                    clipping_mask = None

                # define reaction diffusion operator based on anis_param
                _, g = fct_def.define_loglikelihoodfromdata_and_logprior(
                y_used.cpu().numpy().reshape(1, -1),
                psis[i],
                fwd_matrix=sp.csr_matrix(A_tomo.cpu().numpy()),
                reconstruction_shape=(1, H, W),
                sigma_err=sigma_used,
                reg_fct_type="anisotropic",
                alpha=anis_params[i],
                sampling=0.0125
                )

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
                    clipping_mask=torch.tensor(mask_core, dtype=torch.float32, device=device),
                    reaction_diffusion_op = g,
                    lambda_rd = lambda_rd, 
                )

                post_mean = posterior_tomo.mean(dim=0).squeeze() * alpha
                post_std  = posterior_tomo.std(dim=0).squeeze()  * alpha

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
                    "alpha":         float(alpha),
                    "scaling":       float(scale),
                    "clip":          clip,
                    "lambda_":       lambda_,
                    "lambda_rd":     lambda_rd,
                    "anis_param":    anis_params[i],
                    "zeta":          zeta,
                })

                post_means.append(post_mean.detach().cpu().numpy().flatten())
                post_stds.append(post_std.detach().cpu().numpy().flatten())

            # Convert to arrays of shape (num_eval_samples, H*W)
            post_means = np.array(post_means)
            post_stds  = np.array(post_stds)

            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            print(f"   saved {csv_path}")
            print(df.describe()[["psnr", "ssim", "mse"]])

            np.save(csv_dir / f"post_means_pilatus_{sigma_y_tomo}_clip{clip_str}_{proc_str}.npy", post_means)
            np.save(csv_dir / f"post_stds_pilatus_{sigma_y_tomo}_clip{clip_str}_{proc_str}.npy",  post_stds)