import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np
import os
import skimage.transform as skimt

import reaction_diffusion.routines.tomo_fusion.tools.helpers as tomo_helps


dirname = os.path.dirname(__file__)

def plot_profile(image,
                 figsize=(2, 3),
                 tcv_plot_clip=False,
                 contour_image=None, levels=15, lcfs_width=0.75, lwidth=0.2,
                 ax=None, colorbar=False,
                 interpolation=None, vmin=None, vmax=None, cmap="viridis", contour_color="w", aspect=None,
                 peak_stats=None,
                 pad_cbar=0, cbar_tick_params=None, cbar_label=None):
    if tcv_plot_clip:
        # define TCV patch for plotting
        tcv_shape_coords = np.load(dirname + "/../tcv_shape_coords.npy")
        tcv_shape_coords[0, 0] = 0.675
        tcv_shape_coords[0, 1] = 1.0015
        tcv_shape_coords[-1, 0] = 0.684
        #Lr, Lz = 0.511 if peak_stats is None else 0.5, 1.5
        Lr, Lz = 0.5,1.5
        h = Lz / image.shape[0]
        zs = np.linspace(0, Lz, round(Lz / h), endpoint=False) + 0.5 * h
        rs = np.linspace(0, Lr, round(Lr / h), endpoint=False) + 0.5 * h
        tcv_shape_coords[:, 0] = tcv_shape_coords[:, 0] * rs.size - 0.5 * h
        tcv_shape_coords[:, 1] = tcv_shape_coords[:, 1] * zs.size - 0.5 * h
        path = Path(tcv_shape_coords.tolist())
        patch = PathPatch(path, facecolor='none')

    handle = plt if ax is None else ax
    if peak_stats is not None:
        handle.plot(peak_stats["true_loc"][1], peak_stats["true_loc"][0], 'r.', markersize=2*peak_stats["markersize"])
        handle.plot(peak_stats["mean"][1], peak_stats["mean"][0], 'k.', markersize=peak_stats["markersize"])
        lower_bound_hor, upper_bound_hor = (
        peak_stats["mean"][1] - peak_stats["nb_stds"] * peak_stats["std"][1],
        peak_stats["mean"][1] + peak_stats["nb_stds"] * peak_stats["std"][1])
        lower_bound_vert, upper_bound_vert = (
        peak_stats["mean"][0] - peak_stats["nb_stds"] * peak_stats["std"][0],
        peak_stats["mean"][0] + peak_stats["nb_stds"] * peak_stats["std"][0])
        print(peak_stats["mean"])
        print(lower_bound_hor,upper_bound_hor,lower_bound_vert,upper_bound_vert  )
        handle.plot(np.array([lower_bound_hor, lower_bound_hor]), np.array([lower_bound_vert, upper_bound_vert]), 'k', linewidth=peak_stats["linewidth"])#, dashes=(1.5, 1.5))
        handle.plot(np.array([upper_bound_hor, upper_bound_hor]), np.array([lower_bound_vert, upper_bound_vert]), 'k', linewidth=peak_stats["linewidth"])#, dashes=(1.5, 1.5))
        handle.plot(np.array([lower_bound_hor, upper_bound_hor]), np.array([lower_bound_vert, lower_bound_vert]), 'k', linewidth=peak_stats["linewidth"])#, dashes=(1.5, 1.5))
        handle.plot(np.array([lower_bound_hor, upper_bound_hor]), np.array([upper_bound_vert, upper_bound_vert]), 'k', linewidth=peak_stats["linewidth"])#, dashes=(1.5, 1.5))


    # plot clipping to tcv shape
    if ax is None:
        plt.figure(figsize=figsize)
        if contour_image is not None:
            c = plt.contour(np.flip(contour_image, 0), origin="lower", levels=levels, antialiased=True, colors=contour_color,
                            negative_linestyles="solid",
                        linewidths=0.1)
            linewidths = lwidth*np.ones(c.levels.size)
            try:
                lcms_level = np.where(c.levels == 0)[0][0]
                
                linewidths[lcms_level] = lcfs_width
            except:
                pass
            
            #c.collections[lcms_level].set_linewidth(0.75)
            c.set_linewidth(linewidths)
        p = plt.imshow(np.flip(image, 0), interpolation=interpolation, vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect)
        plt.xlim([-0.75, int(image.shape[1])+0.75])
        plt.ylim([-0.75, int(image.shape[0])+0.75])
        # if tcv_plot_clip:
        #     plt.gca().add_patch(patch)
        #     p.set_clip_path(patch)
        #     if contour_image is not None:
        #         c.set_clip_path(patch)
        #else:
        #    plt.imshow(image, interpolation=interpolation, vmin=vmin, vmax=vmax, aspect=aspect)
        # plt.axis('off')
        # if colorbar:
        #     plt.colorbar()
    else:
        # plot on given figure axis
        if contour_image is not None:
            c = ax.contour(np.flip(contour_image,0), origin="lower", levels=levels, antialiased=True, colors=contour_color,
                           negative_linestyles="solid",
                       linewidths=0.2)
            linewidths = lwidth*np.ones(c.levels.size)
            
            try:
                lcms_level = np.where(c.levels == 0)[0][0]
                linewidths[lcms_level] = lcfs_width
            except:
                pass

            #c.collections[lcms_level].set_linewidth(0.75)
            c.set_linewidth(linewidths)
        p = ax.imshow(np.flip(image,0), interpolation=interpolation, vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect)
        ax.set_xlim([-0.75, int(image.shape[1])+0.75])
        ax.set_ylim([-0.75, int(image.shape[0])+0.75])
        # if tcv_plot_clip:
        #     ax.add_patch(patch)
        #     p.set_clip_path(patch)
        #     if contour_image is not None:
        #         c.set_clip_path(patch)
        #else:
        #    p = ax.imshow(image, interpolation=interpolation, aspect=aspect, vmin=vmin, vmax=vmax)
        # ax.axis('off')
        # if colorbar:
        #     plt.colorbar(p, ax=ax)
    #plt.autoscale(False)

    if tcv_plot_clip:
        hax = plt.gca() if ax is None else ax
        hax.add_patch(patch)
        p.set_clip_path(patch)
        if contour_image is not None:
           c.set_clip_path(patch)

    handle.axis('off')
    if colorbar:
        if ax is None:
            cbar = plt.colorbar()
        else:
            cbar = plt.colorbar(p, ax=ax, pad=pad_cbar)
        if cbar_tick_params is not None:
            cbar.ax.tick_params(labelsize=cbar_tick_params["labelsize"])
            cbar.ax.set_yticks(cbar_tick_params["yticks"])
            cbar.ax.set_yticklabels(cbar_tick_params["yticklabels"])
            cbar.set_label(label=cbar_label, fontsize=cbar_tick_params["label_labelsize"], labelpad=cbar_tick_params["label_labelsize_pad"])

    return


def plot_ptheta_LoS_tcv(params, markersize=5):
    """
    This function plots, for a given (p, theta) configuration, the LoS configuration
    taking into account the TCV geometry. The shaded areas correspond to lines that fall
    outside the vessel.
    """
    tcv_mask_finesse = round(300)
    pmin = np.min(params[:, 0])
    prange = np.max(params[:, 0]) - np.min(params[:, 0])
    # pmax corresponding to tcv geometry
    tcv_pmax = 0.5646 + 0.25
    PT_intersecting_tcv_mask = np.load(dirname+'/../forward_model/tcv_mask_sinogram.npy')
    plt.figure(figsize=(6,3))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    # plot tcv mask
    plt.imshow(PT_intersecting_tcv_mask, origin="upper", aspect="auto", cmap="gray", alpha=0.5)
    # plot considered LoS configuration
    scaling_factor_p = prange / (2*tcv_pmax)  # 2*pmax/(...)
    scaling_factor_theta = np.max(params[:, 1]) / np.pi
    ps = (params[:, 0]+np.abs(np.min(params[:, 0])))/np.max(params[:, 0]+np.abs(np.min(params[:, 0])))*(tcv_mask_finesse-1)*scaling_factor_p
    # shift ps
    #ps += eps / (2*pmax + 2*eps) * (tcv_mask_finesse-1)
    ps += (tcv_pmax + pmin) / (2 * tcv_pmax) * (tcv_mask_finesse-1)
    thetas = params[:, 1]/np.max(params[:, 1])*(tcv_mask_finesse-1)*scaling_factor_theta
    for i in range(params.shape[0]):
        #a=0
        plt.plot(thetas[i], ps[i], "r", marker=".", markersize=markersize)
    plt.xticks([0, tcv_mask_finesse/2, tcv_mask_finesse-1], [r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=20)
    plt.yticks([0, tcv_mask_finesse/2, tcv_mask_finesse-1], [r'$-p_{max}$', r'$0$', r'$p_{max}$'], fontsize=20)
    #plt.yticks([])
    plt.xlabel(r"$\theta$", fontsize=25)
    plt.ylabel(r"$p$", rotation=0, fontsize=25, labelpad=10)
    plt.title("Subsampled sinogram", fontsize=25, color="k")