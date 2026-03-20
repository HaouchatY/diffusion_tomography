import numpy as np
import skimage.transform as skimt


def define_tcv_mask(dim_shape, Lr=0.511, Lz=1.5):
    """
    Compute boolean mask detemrining whether a pixel falls within vessel or not
    :param arg_shape:
    :param Lr:
    :param Lz:
    :return:
    """
    tcv_mask = np.ones(dim_shape, dtype=bool)
    x_coord_lfs_corners = 0.348
    x_coord_hfs_corners = 0.046
    y_coord_hfs_upper_corners = 1.454
    y_coord_hfs_lower_corners = 0.046
    y_coord_lfs_upper_corners = 1.305
    y_coord_lfs_lower_corners = 0.195
    # define steepness and intercepts for lines representing vessel corners
    m_hfs_upper_corner = (Lz - y_coord_hfs_upper_corners) / x_coord_hfs_corners
    line_hfs_upper_corner = [m_hfs_upper_corner, Lz - m_hfs_upper_corner * x_coord_hfs_corners]
    m_hfs_lower_corner = - m_hfs_upper_corner
    line_hfs_lower_corner = [m_hfs_lower_corner, - m_hfs_lower_corner * x_coord_hfs_corners]
    m_lfs_lower_corner = y_coord_lfs_lower_corners / (Lr - x_coord_lfs_corners)
    line_lfs_lower_corner = [m_lfs_lower_corner, - m_lfs_lower_corner * x_coord_lfs_corners]
    m_lfs_upper_corner = - m_lfs_lower_corner
    line_lfs_upper_corner = [m_lfs_upper_corner, Lz - m_lfs_upper_corner * x_coord_lfs_corners]
    # define pixel sizes
    pixel_x_size = Lr / dim_shape[1]
    pixel_y_size = Lz / dim_shape[0]
    for i in range(dim_shape[0]):
        for j in range(dim_shape[1]):
            x_coord_pixel = j * pixel_x_size + 0.5 * pixel_x_size
            y_coord_pixel = Lz - (i * pixel_y_size + 0.5 * pixel_y_size)

            tol = 1e-2
            if y_coord_pixel > x_coord_pixel * line_hfs_upper_corner[0] + line_hfs_upper_corner[1] + tol:
                tcv_mask[i, j] = False
            elif y_coord_pixel < x_coord_pixel * line_hfs_lower_corner[0] + line_hfs_lower_corner[1] - tol:
                tcv_mask[i, j] = False
            elif y_coord_pixel < x_coord_pixel * line_lfs_lower_corner[0] + line_lfs_lower_corner[1] - tol:
                tcv_mask[i, j] = False
            elif y_coord_pixel > x_coord_pixel * line_lfs_upper_corner[0] + line_lfs_upper_corner[1] + tol:
                tcv_mask[i, j] = False
    return tcv_mask


def define_core_mask(psi, dim_shape, xpoint_idx_base_psi=90, trim_values_x=[0, 120]):
    # Reshape magnetic equilibrium if necessary
    if psi.shape != dim_shape:
        psi = skimt.resize(psi, dim_shape, anti_aliasing=False, mode='edge')
    mask = np.where(psi<0)
    xpoint_loc = int(dim_shape[0] * (xpoint_idx_base_psi - trim_values_x[0]) / (trim_values_x[1] - trim_values_x[0]) )
    mask_idx_above_xpoint_x = mask[0][mask[0] < xpoint_loc]
    mask_idx_above_xpoint_y = mask[1][mask[0] < xpoint_loc]
    mask_core = np.zeros(dim_shape, dtype=bool)
    mask_core[mask_idx_above_xpoint_x, mask_idx_above_xpoint_y] = True
    return mask_core


def compute_radiated_power(emissivity, ROI_map=None, sampling=1.0, R_hfs=0.624, R_lfs=1.135):
    """
    Compute radiated power P (either total or from a subregion)
    :param emissivity:
    :param ROI_map:
    :param sampling:
    :param R_hfs:
    :param R_lfs:
    :return:
    """
    if isinstance(sampling, (np.floating, float)):
        sampling = [sampling, sampling]
    arg_shape = emissivity.shape
    # select region of interest
    emissivity_ROI = emissivity * ROI_map
    r_values = np.linspace(R_hfs, R_lfs, arg_shape[1], endpoint=False) + (sampling[1] / 2)
    integral_over_rows = np.sum(emissivity_ROI, axis=0) * sampling[0]
    P = (2 * np.pi * np.sum(integral_over_rows * r_values)) * sampling[1]
    return P

