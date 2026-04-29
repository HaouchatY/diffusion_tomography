import scipy.sparse as sp
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import skimage.transform as skimt
import sys
from pathlib import Path

import reaction_diffusion.routines.tomo_fusion.tools.helpers as tomo_helps
import reaction_diffusion.routines.tomo_fusion.functionals_definition as fct_def
import reaction_diffusion.routines.tomo_fusion.hyperparameter_tuning as hyper_tune
import reaction_diffusion.routines.tomo_fusion.bayesian_computations as bcomp


def reg_param_tuning_train_phantoms(sigma_err, saving_dir, diagnostic="sxr"):
    # phantom indices to tune regularization parameter 
    indices = np.arange(0, 100)
    samples_dir = Path(f'/home/fusiontomo/Repos/diffusion_tomography/test_set_sxr_samples')

    # Load phantom data
    psis = np.load(os.path.join(samples_dir, 'psis_coarse.npy'))[indices, :]
    sxr_samples = np.load(os.path.join(samples_dir, 'sxr_samples_with_background_coarse.npy'))[indices, :]
    alphas = np.load(os.path.join(samples_dir, 'alpha_random_values.npy'))[indices]
    trim_val = np.load(os.path.join(samples_dir, 'trimming_values.npy'))[indices, :]

    tcv_mask = np.load("/home/fusiontomo/Repos/diffusion_tomography/reaction_diffusion/routines/tcv_mask_1_subpixels_NINO.npy")
    tcv_mask = skimt.resize(tcv_mask, (120, 40), anti_aliasing=False, mode='edge')
    tcv_mask[tcv_mask>0]=1

    # load forward model
    if diagnostic == "sxr":
        fwd_model = np.load("/home/fusiontomo/Repos/diffusion_tomography/forward_model/forward_model_sxr_full_geometry.npy")
        reg_params_tuning = np.logspace(-3, 1, 13)
    elif diagnostic == "dmpx":
        fwd_model = np.load("/home/fusiontomo/Repos/diffusion_tomography/forward_model/dmpx_geometry_matrix.npy")
        reg_params_tuning = np.logspace(-3, 1, 13)
    elif diagnostic == "pilatus":
        fwd_model = np.load("/home/fusiontomo/Repos/diffusion_tomography/forward_model/pilatus_geometry_matrix.npy")
        reg_params_tuning = np.logspace(-3, 1, 13)
    max_fwd_model = np.max(fwd_model)
    fwd_model = sp.csr_matrix(fwd_model)
    normalized_fwd_model = fwd_model / max_fwd_model
    reconstruction_shape = (120,40)

    # error level on data, computed as 0.05 times the average of the tomographic measurements on 1000 training phantoms
    #sigma_err_noise = (3.5e-11 / max_fwd_model)
    sigma_err_noise = 0.25
    # sigma_err_recon = 0.02 # sigma in reconstruction ("normalized" value)
    normalization_to_one_factor = 1 # no normalization

    reg_param_tuning_data = []

    for idx in indices:
        print("Phantom ", idx)
        ground_truth = copy.deepcopy(sxr_samples[idx, :, :].squeeze())
        psi = psis[idx, :, :]
        alpha = alphas[idx]
        trim_val_ = trim_val[idx, :]
        mask_core = tomo_helps.define_core_mask(psi=psi, dim_shape=reconstruction_shape, trim_values_x=trim_val_)

        # anisotropic regularization functional
        reg_fct_type = "anisotropic"
        # Define functionals
        f, g = fct_def.define_loglikelihood_and_logprior(ground_truth=ground_truth, psi=psi, fwd_matrix=normalized_fwd_model,
                                                         sigma_err=sigma_err_noise, reg_fct_type=reg_fct_type,
                                                         alpha=alpha, plot=False,
                                                         seed=idx)
        # normalized_data = f.noisy_tomo_data / ( max_fwd_model)
        # normalization_to_one_factor = np.max(normalized_data) # scaling factor to have data around 1, undone after reconstruction to compute MSE with original ground truth
        # f_recon = fct_def._DataFidelityFunctional(dim_shape=(1,120,40), noisy_tomo_data=( normalized_data / normalization_to_one_factor ),
        #                                           sigma_err=sigma_err_recon, geometry_matrix=normalized_fwd_model)
        # tune hyperparameters
        reg_param_data_idx = hyper_tune.reg_param_tuning(f, g, tuning_techniques=["GT"], ground_truth=ground_truth,
                                                         with_pos_constraint=True, clipping_mask=mask_core,
                                                         cv_strategy="random", map_scaling_factor=normalization_to_one_factor,
                                                         reg_params=reg_params_tuning, plot=False)

        reg_param_tuning_data.append(reg_param_data_idx)

    best_performing_hyper_params = np.zeros(len(reg_param_tuning_data))
    for i in range(best_performing_hyper_params.size):
        mse_argmin = np.argmin(reg_param_tuning_data[i]['GT'][2, :])
        best_performing_hyper_params[i] = reg_param_tuning_data[i]['GT'][0, mse_argmin]

    nb_occurrences = np.zeros(reg_param_tuning_data[0]["GT"].shape[1])
    for i in range(len(reg_param_tuning_data)):
        j = np.argmin(reg_param_tuning_data[i]['GT'][2, :])
        nb_occurrences[j] += 1

    plt.figure()
    plt.plot(nb_occurrences)
    plt.xticks([0,4,8,12,16,20], [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"])
    plt.title(r"Regularization parameter $\lambda$ minimizing MSE (anisotropic)")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("# occurrences")
    plt.show()

    print("Statistics on best performing regularization parameter (anisotropic)\n")
    print("Mean ", np.mean(best_performing_hyper_params))
    print("Median ", np.median(best_performing_hyper_params))
    print("Standard deviation ", np.std(best_performing_hyper_params))

    # Define regularization parameter as average/median best performing value
    reg_param_mean = np.mean(best_performing_hyper_params)
    reg_param_median = np.median(best_performing_hyper_params)

    # ratio of MSE with average regularization parameter vs best MSE
    factors_average_wrt_best = np.zeros(indices.size)
    # ratio of MSE with median regularization parameter vs best MSE
    factors_median_wrt_best = np.zeros(indices.size)
    # ratio of MSE with regularization parameter fixed to 1e-1 vs best MSE
    factors_001_wrt_best = np.zeros(indices.size)

    # compute MSE for average value of regularization parameter
    for i, idx in enumerate(indices):
        ground_truth = copy.deepcopy(sxr_samples[idx, :, :].squeeze())
        psi = psis[idx, :, :]
        alpha = alphas[idx]
        trim_val_ = trim_val[idx, :]
        mask_core = tomo_helps.define_core_mask(psi=psi, dim_shape=reconstruction_shape, trim_values_x=trim_val_)

        # anisotropic regularization functional
        reg_fct_type = "anisotropic"
        # Define functionals
        f, g = fct_def.define_loglikelihood_and_logprior(ground_truth=ground_truth, psi=psi, fwd_matrix=normalized_fwd_model,
                                                         sigma_err=sigma_err_noise, reg_fct_type=reg_fct_type,
                                                         alpha=alpha, plot=False,
                                                         seed=idx)
        # normalized_data = f.noisy_tomo_data / ( max_fwd_model)
        # normalization_to_one_factor = np.max(normalized_data) # scaling factor to have data around 1, undone after reconstruction to compute MSE with original ground truth
        # f_recon = fct_def._DataFidelityFunctional(dim_shape=(1,120,40), noisy_tomo_data=( normalized_data / normalization_to_one_factor ),
        #                                           sigma_err=sigma_err_recon, geometry_matrix=normalized_fwd_model)
        # compute MSE with reg_param fixed to average value
        map = bcomp.compute_MAP(f, g, reg_param_mean, with_pos_constraint=True, clipping_mask=mask_core)
        map *= normalization_to_one_factor
        mse_avg_reg_param = np.mean((map-skimt.resize(ground_truth, f.dim_shape[1:], anti_aliasing=False, mode='edge'))**2)
        # compute MSE with reg_param fixed to median value
        map = bcomp.compute_MAP(f, g, reg_param_median, with_pos_constraint=True, clipping_mask=mask_core)
        map *= normalization_to_one_factor
        mse_median_reg_param = np.mean((map-skimt.resize(ground_truth, f.dim_shape[1:], anti_aliasing=False, mode='edge'))**2)
        # compute MSE with reg_param fixed to 1e-1
        map = bcomp.compute_MAP(f, g, 1e-1, with_pos_constraint=True, clipping_mask=mask_core)
        map *= normalization_to_one_factor
        mse_001_reg_param = np.mean((map-skimt.resize(ground_truth, f.dim_shape[1:], anti_aliasing=False, mode='edge'))**2)

        # compute factors
        factors_average_wrt_best[i] = mse_avg_reg_param / reg_param_tuning_data[i]['GT'][2, np.argmin(reg_param_tuning_data[i]['GT'][2, :])]
        factors_median_wrt_best[i] = mse_median_reg_param / reg_param_tuning_data[i]['GT'][2, np.argmin(reg_param_tuning_data[i]['GT'][2, :])]
        factors_001_wrt_best[i] = mse_001_reg_param / reg_param_tuning_data[i]['GT'][2, np.argmin(reg_param_tuning_data[i]['GT'][2, :])]

    # save all results
    np.save(saving_dir+'tuning_data.npy', np.array(reg_param_tuning_data))
    np.save(saving_dir+'best_hyperparams.npy', best_performing_hyper_params)
    np.save(saving_dir+'nb_occurrences.npy', nb_occurrences)
    np.save(saving_dir+'factors_avg_wrt_best.npy', factors_average_wrt_best)
    np.save(saving_dir + 'factors_median_wrt_best.npy', factors_median_wrt_best)
    np.save(saving_dir+'factors_001_wrt_best.npy', factors_001_wrt_best)
    np.save(saving_dir+'reg_param_mean.npy', reg_param_mean)
    np.save(saving_dir + 'reg_param_median.npy', reg_param_median)
    np.save(saving_dir + 'sigma_level.npy', sigma_level)
    np.save(saving_dir + 'sigma_err.npy', sigma_err)


if __name__ == '__main__':
    # run reg_param tuning routine on training phantoms

    # Noise model N1, noise level 5%
    sigma_level = 0.02
    script_dir = Path(__file__).resolve().parent

    # sxr
    saving_dir = os.path.join(script_dir, 'tuning_data/reg_param_tuning_sxr/')
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)
    # analyze training phantoms
    reg_param_tuning_train_phantoms(sigma_level, saving_dir, diagnostic="sxr")

    # dmpx
    saving_dir = os.path.join(script_dir, 'tuning_data/reg_param_tuning_dmpx/')
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)
    # analyze training phantoms
    reg_param_tuning_train_phantoms(sigma_level, saving_dir, diagnostic="dmpx")

    # pilatus
    saving_dir = os.path.join(script_dir, 'tuning_data/reg_param_tuning_pilatus/')
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)
    reg_param_tuning_train_phantoms(sigma_level, saving_dir, diagnostic="pilatus")