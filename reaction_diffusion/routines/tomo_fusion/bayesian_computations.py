import numpy as np
import os
import pyxu.opt.stop as pyxst
from pyxu.opt.solver import PGD, CG
import pyxu.operator as pyxop
from pyxu.experimental.sampler._sampler import ULA
from pyxu.experimental.sampler.statistics import OnlineMoment, OnlineVariance

from reaction_diffusion.routines.tomo_fusion.reg_param_est import ProxFuncMoreau

import reaction_diffusion.routines.tomo_fusion.tools.helpers as tomo_helps
import reaction_diffusion.routines.tomo_fusion.functionals_definition as fct_def

dirname = os.path.dirname(__file__)

def compute_MAP(f, g, reg_param, with_pos_constraint=False, clipping_mask=None, show_progress=True):
    # define proximable term if needed
    prox_term = None
    if with_pos_constraint:
        if clipping_mask is None:
            # positivity constraint
            prox_term = pyxop.PositiveOrthant(dim_shape=(f.dim_size,))
        else:
            # positivity constraint and clipping to region of interest
            prox_term = fct_def._PositiveClipToROI(f.dim_shape, clipping_mask)
    else:
        if clipping_mask is not None:
            # clipping to region of interest
            prox_term = fct_def._ClipToROI(f.dim_shape, clipping_mask)
    # Define stopping criterion
    stop_crit = pyxst.MaxIter(int(1e4)) | (pyxst.RelError(1e-5, rank=2))
    # define starting point (backprojection)
    back_projection = f.forward_model_linop.T(f.noisy_tomo_data.reshape(-1))
    if (prox_term is not None) or ('Mfi' in g._name):
        # apply projected gradient descent (PGD) algorithm
        obj = f + reg_param * g
        obj.diff_lipschitz = f.diff_lipschitz + reg_param * g.diff_lipschitz
        # compute MAP
        solver = PGD(f=obj, g=prox_term, show_progress=show_progress, verbosity=1000)
        solver.fit(x0=back_projection, stop_crit=stop_crit, acceleration=True, tau=1 / obj.diff_lipschitz, d=3)
    else:
        # apply conjugate gradient (CG) algorithm
        cg_op = (1 / f.sigma_err**2) * f.forward_model_linop.T * f.forward_model_linop + reg_param * g._grad_matrix_based
        # compute MAP
        solver = CG(A=cg_op, show_progress=show_progress, verbosity=1000)
        back_projection = np.expand_dims(back_projection,0)
        solver.fit(x0=back_projection, b=1 / f.sigma_err ** 2 * back_projection, stop_crit=stop_crit)
    # return MAP
    return solver.solution().squeeze()


def run_ula(f, g, reg_param,
            psi, trim_values_x,
            with_pos_constraint=False,
            clip_iterations=None, clipping_mask=None,
            estimate_quantiles=False, quantile_marks=[0.005, 0.025, 0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95, 0.975, 0.995],
            estimate_tomo_data_stats=False,
            estimate_peak_location=False,
            compute_stats_wrt_MAP=False,
            samples=int(1e5), burn_in=int(1e3), thinning_factor=1,
            seed=0, show_progress=False):
    # define ula objective function
    ula_obj = f + reg_param * g
    pos_constraint = pyxop.PositiveOrthant(dim_shape=f.dim_shape) if with_pos_constraint else None
    if with_pos_constraint:
        # add positivity constraint to ula objective function
        ula_obj += ProxFuncMoreau(pos_constraint, mu=1e-3)
    # define tcv and core masks
    mask_tcv = tomo_helps.define_tcv_mask(dim_shape=ula_obj.dim_shape[1:])
    mask_core = tomo_helps.define_core_mask(psi=psi, dim_shape=ula_obj.dim_shape[1:], trim_values_x=trim_values_x)
    if clipping_mask is not None:
        clipping_mask = clipping_mask
    else:
        if clip_iterations == "core":
            clipping_mask = mask_core
        elif clip_iterations == "tcv":
            clipping_mask = mask_tcv
    # initialize ULA sampler
    gamma = 0.98 / ula_obj.diff_lipschitz
    ula = ULA(f=ula_obj, gamma=gamma)
    x0 = np.zeros(ula_obj.dim_shape[1:])
    rng = np.random.default_rng(seed=seed)
    gen_ula = ula.samples(x0=x0, rng=rng)
    # moments wrt mean
    mean_ula = OnlineMoment(order=1)
    var_ula = OnlineVariance()
    mean_prad_ula_core = OnlineMoment(order=1)
    var_prad_ula_core = OnlineVariance()
    mean_prad_ula_tcv = OnlineMoment(order=1)
    var_prad_ula_tcv = OnlineVariance()
    if estimate_tomo_data_stats:
        # tomographic data
        mean_tomo_ula = OnlineMoment(order=1)
        var_tomo_ula = OnlineVariance()
    if estimate_peak_location:
        # peak position
        mean_peak_loc_ula = OnlineMoment(order=1)
        var_peak_loc_ula = OnlineVariance()
    if compute_stats_wrt_MAP:
        # moments wrt MAP
        var_ula_wrtMAP = OnlineMoment(order=2)
        var_prad_ula_wrtMAP_tcv = OnlineMoment(order=2)
        var_prad_ula_wrtMAP_core = OnlineMoment(order=2)
        # compute MAP and radiated power
        im_MAP = compute_MAP(f, g, reg_param, with_pos_constraint=with_pos_constraint, clipping_mask=clipping_mask, show_progress=show_progress)
        prad_MAP_tcv = tomo_helps.compute_radiated_power(im_MAP, mask_tcv, g.sampling, R_lfs=1.1376)
        prad_MAP_core = tomo_helps.compute_radiated_power(im_MAP, mask_core, g.sampling, R_lfs=1.1376)

    # Run ULA
    print("Running {} ULA iterations".format(samples))
    prads_tcv = np.zeros(int(samples/thinning_factor), dtype=np.float32)
    prads_core = np.zeros(int(samples/thinning_factor), dtype=np.float32)

    # initialize data
    data = {}
    #data["tomo_data"], data["noisy_tomo_data"], data["sigma_err"] = f.tomo_data, f.noisy_tomo_data, f.sigma_err
    data["noisy_tomo_data"], data["sigma_err"] = f.noisy_tomo_data, f.sigma_err
    data["reg_param"], data["sampling"] = reg_param, g.sampling
    if 'Anis' in g._name:
        data["alpha"] = g.diffusion_coefficient.alpha
    data["with_pos_constraint"], data["clip_iterations"] = with_pos_constraint, clip_iterations
    data["samples"], data["burn_in"], data["thinning_factor"] = samples, burn_in, thinning_factor
    if estimate_quantiles:
        # retain one sample every 1000 for quantile estimation
        num_pixels = f.dim_size if clipping_mask is None else np.sum(clipping_mask)
        quantile_samples = np.zeros((int(1e3), num_pixels), dtype=np.float16)
        quantile_sample_interval = int(samples/1e3)
        quantile_sample_counter = 0
        data["quantile_marks"] = quantile_marks

    for i in range(burn_in):
        # Burn-in phase
        sample = next(gen_ula)
        if clipping_mask is not None:
            ula.x = clipping_mask * sample
    for i in range(samples):
        sample = next(gen_ula)
        if clipping_mask is not None:
            ula.x = clipping_mask * sample

        if show_progress and (i+1) % int(1e4) == 0:
            print("iteration ", i+1)

        if estimate_quantiles:
            # quantile estimation
            if (i+1) % quantile_sample_interval == 0:
                quantile_samples[quantile_sample_counter, :] = ula.x[clipping_mask].flatten()
                quantile_sample_counter += 1

        if i % thinning_factor == 0:
            # update central moments
            mean, var = mean_ula.update(sample), var_ula.update(sample)
            prad_tcv = tomo_helps.compute_radiated_power(sample, mask_tcv, g.sampling, R_lfs=1.1376)
            prad_core = tomo_helps.compute_radiated_power(sample, mask_core, g.sampling, R_lfs=1.1376)
            prads_tcv[int(i / thinning_factor)] = prad_tcv
            prads_core[int(i / thinning_factor)] = prad_core
            prad_tcv = np.array([prad_tcv])
            prad_core = np.array([prad_core])
            mean_prad_tcv, var_prad_tcv = mean_prad_ula_tcv.update(prad_tcv), var_prad_ula_tcv.update(prad_tcv)
            mean_prad_core, var_prad_core = mean_prad_ula_core.update(prad_core), var_prad_ula_core.update(prad_core)
            if estimate_tomo_data_stats:
                # tomographic data
                mean_tomo = mean_tomo_ula.update(f.forward_model_linop(sample))
                var_tomo = var_tomo_ula.update(f.forward_model_linop(sample))
            if estimate_peak_location:
                # estimate position of peak emissivity
                peak_loc = np.array(np.where(sample == np.max(sample)), dtype=np.float64).flatten()
                mean_peak_loc, var_peak_loc = mean_peak_loc_ula.update(peak_loc), var_peak_loc_ula.update(peak_loc)
            if compute_stats_wrt_MAP:
                # update moments computed wrt MAP
                sample_wrtMAP = sample - im_MAP
                var_wrtMAP = var_ula_wrtMAP.update(sample_wrtMAP)
                prad_wrtMAP_tcv = prad_tcv - prad_MAP_tcv
                prad_wrtMAP_core = prad_core - prad_MAP_core
                var_prad_wrtMAP_tcv = var_prad_ula_wrtMAP_tcv.update(prad_wrtMAP_tcv)
                var_prad_wrtMAP_core = var_prad_ula_wrtMAP_core.update(prad_wrtMAP_core)

    # store data
    data["mean"], data["var"] = mean, var
    data["mean_prad_tcv"], data["var_prad_tcv"] = mean_prad_tcv, var_prad_tcv
    data["mean_prad_core"], data["var_prad_core"] = mean_prad_core, var_prad_core
    data["prads_tcv"], data["prads_core"] = prads_tcv, prads_core
    if estimate_tomo_data_stats:
        data["mean_tomo_data"], data["var_tomo_data"] = mean_tomo, var_tomo
    if estimate_peak_location:
        data["mean_peak_loc"], data["var_peak_loc"] = mean_peak_loc, var_peak_loc
    if compute_stats_wrt_MAP:
        data["im_MAP"] = im_MAP
        data["prad_map_tcv"], data["prad_map_core"] = prad_MAP_tcv, prad_MAP_core
        data["var_wrtMAP"] = var_wrtMAP
        data["var_prad_wrtMAP_tcv"] = var_prad_wrtMAP_tcv
        data["var_prad_wrtMAP_core"] = var_prad_wrtMAP_core
    if estimate_quantiles:
        quantile_samples = np.sort(quantile_samples, axis=0)
        quantile_values = np.zeros((len(quantile_marks), *f.dim_shape[1:]))
        for mark_id, mark in enumerate(quantile_marks):
            sample_index = int(mark * quantile_samples.shape[0])
            quantile_vals_mask = quantile_samples[sample_index, :]
            quantile_emissivity = np.zeros(f.dim_shape[1:])
            if clipping_mask is not None:
                quantile_emissivity[clipping_mask] = quantile_vals_mask
            else:
                quantile_emissivity = quantile_vals_mask.reshape(f.dim_shape[1:])
            quantile_values[mark_id, :] = quantile_emissivity
        data["empirical_quantiles"] = quantile_values

    return data