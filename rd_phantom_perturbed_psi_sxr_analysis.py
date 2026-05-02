import math
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import os
import copy
import time
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

import reaction_diffusion.routines.tomo_fusion.tools.helpers as tomo_helps
import reaction_diffusion.routines.tomo_fusion.functionals_definition as fct_def
import reaction_diffusion.routines.tomo_fusion.hyperparameter_tuning as hyper_tune
import reaction_diffusion.routines.tomo_fusion.bayesian_computations as bcomp
import skimage.transform as skimt  

from helpers import *


def run_study(diag, phantom_indices):
    H, W = 120, 40
    reconstruction_shape = (H, W)

    # --------------------------------------------------------------------------
    # Load SXR forward model operator
    # --------------------------------------------------------------------------
    # load forward model
    fwd_model = np.load("/home/fusiontomo/Repos/diffusion_tomography/forward_model/forward_model_sxr_full_geometry.npy")
    fwd_model /= fwd_model.max()
    A_tomo     = fwd_model
    A_tomo_csr = sp.csr_matrix(A_tomo)

    # --------------------------------------------------------------------------
    # Load test samples
    # --------------------------------------------------------------------------
    num_eval_samples = phantom_indices.size # splitting on 20 cpu corse, therefore 1000/20 = 50 samples per run
    chunk_idx = phantom_indices[0] // num_eval_samples
    print(f"Running on chunk {chunk_idx} with phantom indices {phantom_indices}")

    samples_dir  = Path("/home/fusiontomo/Repos/diffusion_tomography/test_set_sxr_samples")
    test_samples = np.load(samples_dir / "sxr_samples_with_background_coarse.npy")[phantom_indices]  # (1000, H, W)
    psis         = np.load(samples_dir / "psis_coarse.npy")[phantom_indices]
    trim_vals    = np.load(samples_dir / "trimming_values.npy")[phantom_indices]

    # --------------------------------------------------------------------------
    # Load noise realisations and scaling params
    # --------------------------------------------------------------------------
    noise_realizations = np.load(f"noise_realizations_sxr.npy")[phantom_indices]  # (1000, n_channels)
    scaling_params     = np.load("scaling_params.npy")[phantom_indices]        # (1000,)

    # Clean synthetic measurements
    y_tomo_clean = test_samples.reshape(-1, H * W) @ A_tomo.T  # (1000, n_channels)

    # --------------------------------------------------------------------------
    # Fixed hyperparameters
    # --------------------------------------------------------------------------
    reg_param        = 1e-1
    alpha_init       = 1e-2
    reg_fct_type     = "anisotropic"
    sampling_param   = 0.0125
    anis_params_grid = np.logspace(-4, 0, 13)
    ula_samples      = int(1e5)

    noise_levels = [0.25]
    procedures   = [False]

    csv_dir = Path(f"metrics_csv_phantom_analysis_perturbed_psi_sxr_rd")
    csv_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------------------------
    # Main loops
    # --------------------------------------------------------------------------
    for sigma_y_tomo in noise_levels:
        for procedure in procedures:
            proc_str = "true" if procedure else "false"
            print(f"\n=== sigma={sigma_y_tomo}, procedure={procedure} ===")

            csv_name = f"metrics_rd_perturbed_psi_sxr_chunk{chunk_idx}_{sigma_y_tomo}_{proc_str}.csv"
            csv_path = csv_dir / csv_name
            # if csv_path.exists():
            #     print(f"Already exists {csv_path}, skipping...")
            #     continue

            rows = []

            for i in range(num_eval_samples):
                print(f"  Processing sample {phantom_indices[i]} (index {i} in chunk), sigma={sigma_y_tomo}, procedure={procedure}...")
                np.random.seed(phantom_indices[i])

                # ------------------------------------------------------------------
                # Build noisy tomo data
                # ------------------------------------------------------------------
                y_tomo_noisy = y_tomo_clean[i] + sigma_y_tomo * noise_realizations[i, :]

                y_used = copy.deepcopy(y_tomo_noisy)
                x_gt_i = copy.deepcopy(test_samples[i].squeeze())  # (H, W)

                # ------------------------------------------------------------------
                # Procedure: scaling + alpha normalization
                # ------------------------------------------------------------------
                scale      = 1.0
                alpha_norm = 1.0
                sigma_used = sigma_y_tomo
  
                # ------------------------------------------------------------------
                # Perturb the magnetic equilibrium
                # ------------------------------------------------------------------
                psi       = psis[i]
                trim_val_ = trim_vals[i]
                upper_trim_bound, lower_trim_bound, hfs_trim_bound, lfs_trim_bound = 6, 5, 3, 2
                np.random.seed(phantom_indices[i]) # for reproducibility of perturbations across runs
                upper_trim = int(upper_trim_bound * np.random.rand())
                lower_trim = int(H - lower_trim_bound * np.random.rand())
                hfs_trim = int(hfs_trim_bound * np.random.rand())
                lfs_trim = int(W - lfs_trim_bound * np.random.rand())
                trimming_vals_perturb = np.array([upper_trim, lower_trim, hfs_trim, lfs_trim])
                psi_perturb = skimt.resize(psi[upper_trim:lower_trim, hfs_trim:lfs_trim], (H,W), anti_aliasing=True,mode='edge')
                # define core for perturbed psi
                xpoint_loc = int(H * (90 - trim_val_[0]) / (trim_val_[1] - trim_val_[0]) )
                mask_core_perturb = tomo_helps.define_core_mask(
                                    psi=psi_perturb, dim_shape=(H, W), xpoint_idx_base_psi=xpoint_loc, trim_values_x=trimming_vals_perturb)

                # ------------------------------------------------------------------
                # Define functionals with noisy tomo data
                # ------------------------------------------------------------------
                f, g = fct_def.define_loglikelihoodfromdata_and_logprior(
                    y_used,
                    psi_perturb,
                    fwd_matrix=A_tomo_csr,
                    reconstruction_shape=(1, H, W),
                    sigma_err=sigma_used,
                    reg_fct_type=reg_fct_type,
                    alpha=alpha_init,
                    sampling=sampling_param,
                    seed=i,
                )

                # ------------------------------------------------------------------
                # Tune anisotropic parameter via CV (blind to ground truth)
                # ------------------------------------------------------------------
                anis_param_data = hyper_tune.anis_param_tuning(
                    f, g,
                    reg_param=reg_param,
                    tuning_techniques=["CV_full"],
                    with_pos_constraint=True,
                    clipping_mask=mask_core_perturb,
                    cv_strategy=["random"],
                    anis_params=anis_params_grid,
                    plot=False,
                )

                best_anis_idx   = np.argmin(anis_param_data["CV_full_random"][1, :])
                best_anis_param = anis_param_data["CV_full_random"][0, best_anis_idx]

                # ------------------------------------------------------------------
                # Redefine regularization functional with tuned alpha
                # ------------------------------------------------------------------

                g = hyper_tune._redefine_anis_param_logprior(g, best_anis_param)

                # ------------------------------------------------------------------
                # Run ULA
                # ------------------------------------------------------------------
                start_time = time.time()
                uq_data = bcomp.run_ula(
                    f, g, reg_param, psi_perturb, trim_val_,
                    with_pos_constraint=True,
                    clip_iterations="core",
                    clipping_mask=mask_core_perturb,
                    compute_stats_wrt_MAP=True,
                    estimate_quantiles=False,
                    estimate_tomo_data_stats=False,
                    estimate_peak_location=False,
                    samples=ula_samples,
                )
                uq_data["time"] = time.time() - start_time

                # ------------------------------------------------------------------
                # Extract quantities and undo normalization
                # ------------------------------------------------------------------
                im_map  = uq_data["im_MAP"]       * alpha_norm  # (H, W)
                im_mean = uq_data["mean"]         * alpha_norm  # (H, W)
                im_std  = np.sqrt(uq_data["var"]) * alpha_norm  # (H, W)

                gt_np      = x_gt_i
                data_range = float(gt_np.max() - gt_np.min())

                psnr_map  = psnr(gt_np, im_map,  data_range=data_range)
                ssim_map  = ssim(gt_np, im_map,  data_range=data_range)
                mse_map   = float(((im_map  - gt_np) ** 2).mean())
                psnr_mean = psnr(gt_np, im_mean, data_range=data_range)
                ssim_mean = ssim(gt_np, im_mean, data_range=data_range)
                mse_mean  = float(((im_mean - gt_np) ** 2).mean())

                rows.append({
                    # identifiers
                    "sample_idx":      int(phantom_indices[i]),
                    # looped quantities
                    "sigma_y_tomo":    float(sigma_y_tomo),
                    "procedure":       procedure,
                    # procedure-derived quantities
                    "scale":           float(scale),
                    "alpha_norm":      float(alpha_norm),
                    "sigma_used":      float(sigma_used),
                    # functional hyperparameters
                    "reg_param":       float(reg_param),
                    "alpha_init":      float(alpha_init),
                    "reg_fct_type":    reg_fct_type,
                    "sampling_param":  float(sampling_param),
                    "best_anis_param": float(best_anis_param),
                    "best_anis_idx":   int(best_anis_idx),
                    "ula_samples":     int(ula_samples),
                    # timing
                    "time_ula":        float(uq_data["time"]),
                    # MAP metrics
                    "psnr_map":        psnr_map,
                    "ssim_map":        ssim_map,
                    "mse_map":         mse_map,
                    # posterior mean metrics
                    "psnr_mean":       psnr_mean,
                    "ssim_mean":       ssim_mean,
                    "mse_mean":        mse_mean,
                    # posterior spread
                    "post_std_mean":   float(im_std.mean()),
                })

                # ------------------------------------------------------------------
                # Save per-sample results as a single dict
                # ------------------------------------------------------------------
                sample_results = {
                    "im_MAP": im_map,   # (H, W) MAP estimate
                    "mean":   im_mean,  # (H, W) posterior mean
                    "std":    im_std,   # (H, W) posterior std
                    "trimming_vals_perturb": trimming_vals_perturb,  # (4,) values of the random trimming applied to the psi for perturbation
                }
                np.save(
                    csv_dir / f"results_{diag}_{sigma_y_tomo}_{proc_str}_{phantom_indices[i]}.npy",
                    sample_results,
                    allow_pickle=True,
                )

            # ----------------------------------------------------------------------
            # Save summary CSV
            # ----------------------------------------------------------------------
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            print(f"   saved {csv_path}")
            print(df.describe()[["psnr_map", "ssim_map", "mse_map", "psnr_mean", "ssim_mean", "mse_mean"]])
    return




if __name__ == '__main__':

    argv = sys.argv
    if len(argv) == 1:
        phantom_indices = np.arange(0, 1000)
        diag = "sxr"
    elif len(argv) == 3:
        phantom_indices = np.arange(int(argv[1]), int(argv[2]))
        diag = "sxr"
        print("Running pipeline on phantoms {}-{}".format(int(argv[1]), int(argv[2])))
    else:
        raise ValueError("Number of passed arguments must be either 1 or 3")
    
    print("Running pipeline on diagnostic {} on phantoms {}-{}".format(diag, int(argv[1]), int(argv[2])))

    # run study
    run_study(diag=diag, phantom_indices=phantom_indices)