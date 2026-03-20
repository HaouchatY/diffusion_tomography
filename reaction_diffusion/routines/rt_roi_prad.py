import numpy as np
import scipy.sparse as sp
import copy
from scipy import ndimage
import src.routines.tomo_fusion.functionals_definition as fct_def
import src.routines.tomo_fusion.bayesian_computations as bcomp
from pyxu.info import ptype as pxt
from pyxu.abc.operator import LinOp
import scipy.io as scio
import warnings
from pyxu_diffops.operator import AnisDiffusionOp
import shapely
from shapely.geometry import Polygon
import os

class _ExplicitLinOpSparseMatrixQuadraticForm(LinOp):
    def __init__(self, dim_shape, mat):
        assert len(mat.shape) == 2, "Matrix `mat` must be a 2-dimensional array"
        super().__init__(dim_shape=dim_shape, codim_shape=dim_shape)
        self.mat = mat
        self.num_pixels = np.prod(dim_shape[-2:])

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        arr = arr.reshape(*arr.shape[:-2], self.num_pixels)
        y = self.mat.dot(arr.T).T
        y = y.reshape(*y.shape[:-1], *self.codim_shape)
        return y

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        arr = arr.reshape(*arr.shape[:-1], *self.codim_shape)
        y = self.mat.T.dot(arr.T).T
        y = y.reshape(*y.shape[:-1], *self.dim_shape[1:])
        return y


class RtPradROI:

    def __init__(self, shot_nb=78182, Nz=120, Nr=41, forward_model="vos", bolo_good_channels=None, downsample=True, is_usn=False):
        # define inversion grid parameters
        tcv_zmin, tcv_zmax = -0.75, 0.75
        tcv_rmin, tcv_rmax = 0.624, 1.1376
        self.dz, self.dr = (tcv_zmax - tcv_zmin) / Nz, (tcv_rmax - tcv_rmin) / Nr
        self.tcv_Zs = np.linspace(tcv_zmin, tcv_zmax, Nz + 1, endpoint=True)[:-1] + self.dz / 2
        self.tcv_Rs = np.linspace(tcv_rmin, tcv_rmax, Nr + 1, endpoint=True)[:-1] + self.dr / 2
        self.Nz, self.Nr = Nz, Nr
        # define meshgrid
        meshgrid_pt = np.meshgrid(np.flip(self.tcv_Zs.flatten()), self.tcv_Rs.flatten(), indexing='ij')
        self.pixel_coords_meshgrid = np.hstack((meshgrid_pt[0].reshape(-1, 1), meshgrid_pt[1].reshape(-1, 1)))
        # whether equilibrium  is upper single null (USN) or not
        self.is_usn = is_usn

        # load and normalize geometry matrix
        dirname = os.path.dirname(__file__)
        self.geometry_matrix = np.load(dirname+'/../tcv_geometry/geometry_matrix_NINO.npy')
        self.geometry_matrix_rescaling_factor = 1 / np.max(self.geometry_matrix)
        self.geometry_matrix *= self.geometry_matrix_rescaling_factor

        # load tcv mask
        self.tcv_mask = np.load(dirname+'/../tcv_geometry/tcv_mask_1_subpixels_NINO.npy')

        # load shot data
        self.shot_nb = shot_nb
        self.data = scio.loadmat(dirname+'/../data/shot_data_'+str(shot_nb)+'.mat')['shot_data'][0, 0]

        # preprocess magnetic and bolometry data
        self.preprocess_magnetic_data()
        self.preprocess_bolo_data(bolo_good_channels=bolo_good_channels, downsample=downsample)

        # initialize to None the linear coefficients
        self.coeffs_prad = None
        self.std_prad = None

    def preprocess_magnetic_data(self):
        # interpolate original fbte equilibria on the prescribed inversion grid
        self.fbte_eqs = np.zeros((self.data['r_xpts_fbte'].shape[1], self.Nz, self.Nr))
        rs_fbte = self.data['fbte_rs'][:, 0]
        zs_fbte = self.data['fbte_zs'][:, 0]
        dr_fbte = np.mean(np.diff(rs_fbte.flatten()))
        dz_fbte = np.mean(np.diff(zs_fbte.flatten()))
        for i in range(self.fbte_eqs.shape[0]):
            # convert inversion grid coordinates to image coordinates
            r_pos_evals = (self.tcv_Rs - rs_fbte.min()) / dr_fbte
            z_pos_evals = -(np.flip(self.tcv_Zs) - zs_fbte.max()) / dz_fbte
            coord_eval_pos = np.zeros((2, self.Nz * self.Nr))
            for nz in range(self.Nz):
                for nr in range(self.Nr):
                    coord_eval_pos[0, nr + nz * self.Nr] = z_pos_evals[nz]
                    coord_eval_pos[1, nr + nz * self.Nr] = r_pos_evals[nr]
            # evaluate original fbte input on the prescribed inversion grid
            self.fbte_eqs[i, :, :] = ndimage.map_coordinates(np.flip(self.data['psi_fbte'][:, :, i].T, axis=0),
                                                             coord_eval_pos, order=2, mode="nearest",
                                                             prefilter=True).reshape(self.Nz, self.Nr)

        # interpolate original liuqe equilibria on the prescribed inversion grid
        self.liuqe_eqs = np.zeros((self.data['r_xpts_liuqe'].shape[1], self.Nz, self.Nr))
        rs_liuqe = self.data['liuqe_rs'][:, 0]
        zs_liuqe = self.data['liuqe_zs'][:, 0]
        dr_liuqe = np.mean(np.diff(rs_liuqe.flatten()))
        dz_liuqe = np.mean(np.diff(zs_liuqe.flatten()))
        for i in range(self.liuqe_eqs.shape[0]):
            # convert inversion grid coordinates to image coordinates
            r_pos_evals = (self.tcv_Rs - rs_liuqe.min()) / dr_liuqe
            z_pos_evals = -(np.flip(self.tcv_Zs) - zs_liuqe.max()) / dz_liuqe
            coord_eval_pos = np.zeros((2, self.Nz * self.Nr))
            for nz in range(self.Nz):
                for nr in range(self.Nr):
                    coord_eval_pos[0, nr + nz * self.Nr] = z_pos_evals[nz]
                    coord_eval_pos[1, nr + nz * self.Nr] = r_pos_evals[nr]
            # evaluate original liuqe input on the prescribed inversion grid
            self.liuqe_eqs[i, :, :] = ndimage.map_coordinates(np.flip(self.data['psi_liuqe'][:, :, i].T, axis=0),
                                                             coord_eval_pos, order=2, mode="nearest",
                                                             prefilter=True).reshape(self.Nz, self.Nr)
        return

    def preprocess_bolo_data(self, bolo_good_channels=None, downsample=True):
        # liuqe data
        liuqe_time = self.data['liuqe_time'].flatten()
        # bolo data
        bolo_data = self.data['bolo_data']
        bolo_time = self.data['bolo_time'].flatten()
        self.bolo_good_channels = self.data['good_channels'].flatten() if bolo_good_channels is None else bolo_good_channels
        self.bolo_good_channels -= 1
        dirname = os.path.dirname(__file__)
        self.etendues = np.load(dirname+'/../tcv_geometry/etendues.npy').flatten()

        if downsample:
            # select data (retain only shot time range)
            idx_min = np.where(np.sign(liuqe_time[0] - bolo_time) < 0)[0][0]
            idx_max = np.where(np.sign(liuqe_time[-1] - bolo_time) < 0)[0][0]
            downsample_rate = int(1e-3 * (idx_max - idx_min) / (bolo_time[idx_max] - bolo_time[idx_min]))
            bolo_time = bolo_time[idx_min:idx_max:downsample_rate]
            bolo_data = bolo_data[idx_min:idx_max:downsample_rate, :]
        # downsample bolo_data and bolo_time
        self.bolo_time = bolo_time
        bolo_data = bolo_data
        # retain only good bolo channels
        # pre-process data to remove radcam's standard pre-processing operations
        bolo_data *= self.etendues.reshape(1, -1) * 4 * np.pi
        # normalize bolo_data with the same normalization used for the geometry matrix
        bolo_data = bolo_data * self.geometry_matrix_rescaling_factor
        self.bolo_data_all_channels = bolo_data
        self.bolo_data = bolo_data[:, self.bolo_good_channels]
        return

    def compute_roi_fbte_based_masks(self, time_range=None, mask_types=["total"]):
        if time_range is None:
            time_range = [self.bolo_time.min(), self.bolo_time.max()]
        fbte_time = self.data['fbte_time'].flatten()
        idx_fbte_min = int(np.where((fbte_time - time_range[0]) >= 0)[0][0])
        self.idx_fbte_min = idx_fbte_min
        idx_fbte_max = int(np.where((fbte_time - time_range[1]) <= 0)[0][-1])
        self.idx_fbte_max = idx_fbte_max
        nb_fbte_eqs = int(idx_fbte_max - idx_fbte_min) + 1
        self.nb_fbte_eqs = nb_fbte_eqs
        self.fbte_masks = np.zeros((len(mask_types), nb_fbte_eqs, self.Nz, self.Nr), dtype=bool)
        # compute masks
        counter = 0
        if "total" in mask_types:
            self.fbte_masks[counter, :, :, :] = [self.tcv_mask] * nb_fbte_eqs
            counter += 1

        if "core" in mask_types:
            for i in range(nb_fbte_eqs):
                # detect all pixels that fall inside the LCFS
                lcfs = Polygon(np.hstack((self.data['fbte_z_contour'][:, idx_fbte_min + i].reshape(-1, 1),
                                          self.data['fbte_r_contour'][:, idx_fbte_min + i].reshape(-1, 1))))
                polygons_tree = shapely.STRtree([lcfs])
                pts = shapely.points(self.pixel_coords_meshgrid)
                idxs_in_lcfs = polygons_tree.query(pts, predicate="intersects")
                mask_in_lcfs = np.zeros((self.Nz, self.Nr), dtype=bool).flatten()
                mask_in_lcfs[idxs_in_lcfs[0, :]] = True
                mask_in_lcfs = mask_in_lcfs.reshape(self.Nz, self.Nr)
                # detect all pixels for which effective plasma radius is below 0.95
                psi = self.fbte_eqs[idx_fbte_min+i, :, :]
                rho_eff = copy.copy(psi)
                rescaling_factor = np.abs(np.min(rho_eff))
                rho_eff += rescaling_factor
                rho_eff /= rescaling_factor
                # core mask
                rho_eff_sqrt = np.sqrt(rho_eff)
                idxs_psi_095 = np.where(rho_eff_sqrt < 0.95)
                mask_095 = np.zeros((self.Nz, self.Nr), dtype=bool)
                mask_095[idxs_psi_095] = True
                # define core mask as where both the above masks are true
                mask_core = (mask_in_lcfs * mask_095).reshape(self.Nz, self.Nr)
                self.fbte_masks[counter, i, :, :] = mask_core
            counter += 1

        if "divertor" in mask_types:
            for i in range(nb_fbte_eqs):
                z_xpt_idx = int(
                    (self.tcv_Zs.max() - self.data['z_xpts_fbte'][0, idx_fbte_min+i] + self.dz/2) / self.dz)
                mask_divertor = np.zeros((self.Nz, self.Nr), dtype=bool)
                if self.is_usn:
                    mask_divertor[:z_xpt_idx+1, :] = True
                else:
                    mask_divertor[z_xpt_idx:, :] = True
                self.fbte_masks[counter, i, :, :] = mask_divertor
            counter += 1

        if "main_chamber" in mask_types:
            for i in range(nb_fbte_eqs):
                z_xpt_idx = int(
                    (self.tcv_Zs.max() - self.data['z_xpts_fbte'][0, idx_fbte_min+i] + self.dz/2) / self.dz)
                mask_main_chamber = np.zeros((self.Nz, self.Nr), dtype=bool)
                if self.is_usn:
                    mask_main_chamber[z_xpt_idx+1:, :] = True
                else:
                    mask_main_chamber[:z_xpt_idx, :] = True
                self.fbte_masks[counter, i, :, :] = mask_main_chamber
            counter += 1

    def assemble_gradient_matrix(self, bc_type="noflux"):

        # nb of pixels in poloidal grid that belong to tcv_mask
        nb_pixels = int(self.tcv_mask.sum())
        # I need to assemble by hand the finite difference matrix
        D = np.zeros((2 * nb_pixels, nb_pixels))
        counter = 0

        for i in range(self.tcv_mask.shape[0]):
            for j in range(self.tcv_mask.shape[1]):
                # pixel (i,j) -> check if in vessel
                if self.tcv_mask[i, j]:
                    # handle vertical derivatives
                    if i < (self.Nz - 1):
                        # if pixel is not last of its column
                        if self.tcv_mask[i + 1, j]:
                            # if pixel below is still in mask
                            D[counter, counter] = - 1 / self.dz
                            # get index of pixel (i+1, j)
                            gap = np.sum(self.tcv_mask[i, j + 1:]) + np.sum(
                                self.tcv_mask[i + 1, : j]) + 1  # j+1 or j? should be j and then +1, like this
                            idx_below = counter + int(gap)
                            D[counter, idx_below] = 1 / self.dz
                        else:
                            # if pixel below is not in mask
                            if bc_type == "noflux":
                                D[counter, counter] = 0
                            else:
                                D[counter, counter] = - 1 / self.dz

                    # handle horizontal derivatives
                    if j < (self.Nr - 1):
                        # if pixel is not last of its row
                        if self.tcv_mask[i, j + 1]:
                            # if pixel to the right is still in mask
                            D[2 * counter, counter] = - 1 / self.dr
                            D[2 * counter, counter + 1] = 1 / self.dr
                        else:
                            # if pixel to the right is not in mask
                            if bc_type == "noflux":
                                D[2 * counter, counter] = 0
                            else:
                                D[2 * counter, counter] =  - 1 / self.dr
                    # update tcv_mask pixel counter
                    counter += 1
        D = sp.csr_matrix(D)
        return D

    def assemble_reg_op_gradient_matrix_noflux_bc(self, g, D):

        # nb of pixels in poloidal grid that belong to tcv_mask
        nb_pixels = int(self.tcv_mask.sum())
        diff_coeff = g.diffusion_coefficient.frozen_coeff.squeeze()

        W_clipped = sp.diags(
            [
                np.hstack((diff_coeff[0, 0, :, :][self.tcv_mask.astype(bool)].flatten(),
                           diff_coeff[1, 1, :, :][self.tcv_mask.astype(bool)].flatten())),
                diff_coeff[0, 1, :, :][self.tcv_mask.astype(bool)].flatten(),
                diff_coeff[1, 0, :, :][self.tcv_mask.astype(bool)].flatten(),
            ],
            offsets=[0, nb_pixels, -nb_pixels],
            format="csr",
        )

        L = D.T @ W_clipped @ D
        return L

    def compute_fbte_based_coefficients(self, sigma_err=1e-2, reg_param=1e-1, anis_param=1e-2, mask_type="fbte", bc_type="noflux"):

        # retain only good bolo channels in geometry matrix
        geometry_matrix = self.geometry_matrix[self.bolo_good_channels, :, :]
        geometry_matrix = geometry_matrix.reshape(geometry_matrix.shape[0], -1)

        # define average sampling step
        sampling = (self.dz+self.dr) / 2

        # nb of pixels in poloidal grid that belong to tcv_mask
        nb_pixels = int(self.tcv_mask.sum())

        # retain only pixels belonging to mask
        geomat_clipped = np.zeros((geometry_matrix.shape[0], nb_pixels))
        for i in range(geomat_clipped.shape[0]):
            geomat_clipped[i, :] = geometry_matrix[i, self.tcv_mask.flatten().astype(bool)]

        D = self.assemble_gradient_matrix(bc_type=bc_type)

        # compute mapping associating each bolo data to the corresponding closest in time fbte eq
        self.fbte_times = self.data['fbte_time'].flatten()
        fbte_idx_bolo_data = np.zeros(self.bolo_time.size, dtype=int)
        for i in range(self.bolo_time.size):
            idx = (np.abs(self.fbte_times - self.bolo_time[i])).argmin()
            fbte_idx_bolo_data[i] = idx

        # initialize matrices to store the coefficients
        self.coeffs_prad = np.zeros((self.fbte_masks.shape[0], self.fbte_masks.shape[1], self.bolo_data.shape[1]))
        self.std_prad = np.zeros((self.fbte_masks.shape[0], self.fbte_masks.shape[1]))

        # for each fbte equilibrium, compute the linear coefficients for estimation of prad from desired region
        for ph_idx in range(self.fbte_masks.shape[1]):
            fbte_idx = self.idx_fbte_min + ph_idx
            print("Computing coefficients for FBTE index ", fbte_idx)

            masks = self.fbte_masks[:, ph_idx, :, :]

            # initialize regularization operator
            g = AnisDiffusionOp(dim_shape=(1, 120, 41),
                                        alpha=anis_param,
                                        diff_method_struct_tens="fd",
                                        freezing_arr=self.fbte_eqs[fbte_idx, :, :],
                                        sampling=sampling,
                                        matrix_based_impl=True)

            L = self.assemble_reg_op_gradient_matrix_noflux_bc(g, D)
            L = L.toarray()

            # compute gram regularized matrix (T^T*T + lambda*G^T*G)
            gram_reg_mat_clipped = (1 / (
                        sigma_err ** 2)) * geomat_clipped.T @ geomat_clipped + reg_param * L

            # solve linear system gram_reg_mat*W=Tmat.T
            inv_gram_Tt_clipped = np.linalg.solve(gram_reg_mat_clipped, geomat_clipped.T)  # (Npixel, Mdetectors)
            # compute linear coefficients
            lin_coeffs = np.zeros((self.fbte_masks.shape[0], geometry_matrix.shape[0]))
            for mask_idx in range(masks.shape[0]):
                # compute weights for toroidal integration
                weight_vec_2piRs = np.array(
                    [2 * np.pi * self.tcv_Rs] * self.Nz).flatten() * self.dr * self.dz * masks[mask_idx, :, :].flatten()
                # retain only pixels belonging to tcv mask
                weight_vec_2piRs_clipped = weight_vec_2piRs[self.tcv_mask.astype(bool).flatten()]
                for i in range(geometry_matrix.shape[0]):
                    lin_coeffs[mask_idx, i] = (1 / (sigma_err ** 2)) * np.dot(inv_gram_Tt_clipped[:, i].flatten(), weight_vec_2piRs_clipped)
                # compute variance associated to the estimate
                prad_var = np.dot(weight_vec_2piRs_clipped, np.linalg.solve(gram_reg_mat_clipped, weight_vec_2piRs_clipped))

                # store computed coefficients
                self.coeffs_prad[mask_idx, ph_idx, :] = lin_coeffs[mask_idx, :].flatten()
                self.std_prad[mask_idx, ph_idx] = np.sqrt(prad_var)

        return self.coeffs_prad, self.std_prad

    def estimate_prad_roi(self, sigma_err=1e-2, reg_param=1e-1, anis_param=1e-2, mask_type=None):

        # compute coefficients if they have not been computed yet
        if self.coeffs_prad is None:
            self.compute_fbte_based_coefficients(sigma_err=sigma_err, reg_param=reg_param, anis_param=anis_param, mask_type=mask_type)

        # compute mapping associating each bolo data to the corresponding closest in time fbte eq
        self.fbte_times = self.data['fbte_time'].flatten()
        fbte_idx_bolo_data = np.zeros(self.bolo_time.size, dtype=int)
        for i in range(self.bolo_time.size):
            idx = (np.abs(self.fbte_times - self.bolo_time[i])).argmin()
            if idx < self.idx_fbte_min:
                fbte_idx_bolo_data[i] = 0
            elif idx > self.idx_fbte_max:
                fbte_idx_bolo_data[i] = -1
            else:
                fbte_idx_bolo_data[i] = idx - self.idx_fbte_min

        # for each bolo-time, compute corresponding prad and std_prad from ROI, using correct set of coefficients
        prads_roi = np.zeros((self.fbte_masks.shape[0], self.bolo_time.size))
        stds_prad_roi = np.zeros((self.fbte_masks.shape[0], self.bolo_time.size))
        for bolo_time_idx in range(self.bolo_time.size):
            for mask_idx in range(self.fbte_masks.shape[0]):
                # select coefficients corresponding to closest in time fbte equilibrium
                coeffs = self.coeffs_prad[mask_idx, fbte_idx_bolo_data[bolo_time_idx], :].flatten()
                # estimate prad
                prads_roi[mask_idx, bolo_time_idx] = np.dot(coeffs, self.bolo_data[bolo_time_idx, :].flatten())
                # estimate std prad, after having computed data scaling factor
                scaling_factor = np.max(self.bolo_data[bolo_time_idx, :])
                stds_prad_roi[mask_idx, bolo_time_idx] = scaling_factor * self.std_prad[mask_idx, fbte_idx_bolo_data[bolo_time_idx]]

        return prads_roi, stds_prad_roi

    def compute_baseline_tomographic_inversion(self, decimation_factor=10,
                                               sigma_err=1e-2, reg_param=1e-1,
                                               anis_param=1e-2, with_positivity_constraint=False,
                                               mask_type="liuqe"):

        # retain only good bolo channels in geometry matrix
        geometry_matrix = self.geometry_matrix[self.bolo_good_channels, :, :]
        geometry_matrix = geometry_matrix.reshape(geometry_matrix.shape[0], -1)

        # define average sampling step
        sampling = (self.dz+self.dr) / 2

        # select inversion indices (after decimation)
        inversion_idxs = np.arange(0, self.bolo_data.shape[0])[::decimation_factor].flatten()
        inversion_times = self.bolo_time[inversion_idxs]

        # initiliaze vectors/tensors to store inversions and radiated powers
        inversions = np.zeros((inversion_idxs.size, self.Nz, self.Nr))
        prads_roi_inversions = np.zeros((self.fbte_masks.shape[0], inversion_idxs.size))
        masks = np.zeros((self.fbte_masks.shape[0], inversion_idxs.size, self.Nz, self.Nr))
        liuqe_equils_invs = np.zeros((inversion_idxs.size, self.Nz, self.Nr))

        # get liuqe info
        liuqe_times = self.data['liuqe_time'].flatten()

        # compute liuqe-informed tomographic inversions
        for idx, inversion_idx in enumerate(inversion_idxs):
            if (idx) % 10 == 0:
                print("processing {}th/{} phantom".format(idx, inversion_idxs.size))
            # find closest time index
            liuqe_time_idx = np.argmin(np.abs((liuqe_times - inversion_times[idx])))

            # get the corresponding liuqe equilibrium
            psi = self.liuqe_eqs[liuqe_time_idx, :, :]
            liuqe_equils_invs[idx, :, :] = psi

            if mask_type == "liuqe":
                # compute masks
                # total mask
                masks[0, idx, :, :] = self.tcv_mask
                # --- COMPUTE CORE MASK -------
                # detect all pixels that fall inside the LCFS
                lcfs = Polygon(np.hstack((self.data['liuqe_z_contour'][:, liuqe_time_idx].reshape(-1, 1),
                                          self.data['liuqe_r_contour'][:, liuqe_time_idx].reshape(-1, 1))))
                polygons_tree = shapely.STRtree([lcfs])
                pts = shapely.points(self.pixel_coords_meshgrid)
                idxs_in_lcfs = polygons_tree.query(pts, predicate="intersects")
                mask_in_lcfs = np.zeros((self.Nz, self.Nr), dtype=bool).flatten()
                mask_in_lcfs[idxs_in_lcfs[0, :]] = True
                mask_in_lcfs = mask_in_lcfs.reshape(self.Nz, self.Nr)
                # detect all pixels for whcih effective plasma radius is below 0.95
                rho_eff = copy.copy(psi)
                rescaling_factor = np.abs(np.min(rho_eff))
                rho_eff += rescaling_factor
                rho_eff /= rescaling_factor
                # core mask
                rho_eff_sqrt = np.sqrt(rho_eff)
                idxs_psi_095 = np.where(rho_eff_sqrt < 0.95)
                mask_095 = np.zeros((self.Nz, self.Nr), dtype=bool)
                mask_095[idxs_psi_095] = True
                # define core mask as wehre both the above masks are true
                mask_core = (mask_in_lcfs * mask_095).reshape(self.Nz, self.Nr)
                masks[1, idx, :, :] = mask_core
                
                z_xpt_idx = int((self.tcv_Zs.max() - self.data['z_xpts_liuqe'][0, liuqe_time_idx] + self.dz / 2) / self.dz)
                # divertor mask
                mask_divertor = np.zeros((self.Nz, self.Nr), dtype=bool)
                if self.is_usn:
                    mask_divertor[:z_xpt_idx+1, :] = True
                else:
                    mask_divertor[z_xpt_idx:, :] = True
                masks[2, idx, :, :] = mask_divertor
                # main chamber mask
                mask_main_chamber = np.zeros((self.Nz, self.Nr), dtype=bool)
                if self.is_usn:
                    mask_main_chamber[z_xpt_idx+1:, :] = True
                else:
                    mask_main_chamber[:z_xpt_idx, :] = True
                masks[3, idx, :, :] = mask_main_chamber

            # get bolo_data and normalize them
            bolo_data_idx = copy.deepcopy(self.bolo_data[inversion_idx, :])
            max_bolo_data = np.max(bolo_data_idx)
            bolo_data_idx /= max_bolo_data
            # initialize data-fidelity and regularization functionals
            f = fct_def._DataFidelityFunctional(dim_shape=(1, 120, 41), noisy_tomo_data=bolo_data_idx.reshape(1,-1),
                                                sigma_err=sigma_err, geometry_matrix=sp.csr_matrix(geometry_matrix),)
            g = AnisDiffusionOp(dim_shape=(1, 120, 41),
                                      alpha=anis_param,
                                      diff_method_struct_tens="fd",
                                      freezing_arr=psi,
                                      sampling=sampling,
                                      matrix_based_impl=True)
            # cmopute MAP
            im_map = bcomp.compute_MAP(f, g, reg_param, with_pos_constraint=with_positivity_constraint, clipping_mask=self.tcv_mask,
                                       show_progress=False)
            # store inversion
            inversions[idx, :, :] = max_bolo_data * im_map.reshape(self.Nz, self.Nr)

            # store radiated power from roi
            for mask_idx in range(masks.shape[0]):
                # compute weights for toroidal integration (ROI)
                weight_vec_2piRs_roi = np.array(
                    [2 * np.pi * self.tcv_Rs] * self.Nz).flatten() * self.dr * self.dz * masks[mask_idx, idx, :, :].flatten()
                prads_roi_inversions[mask_idx, idx] = max_bolo_data * np.dot(im_map.flatten() * masks[mask_idx, idx, :, :].flatten(),
                                                                   weight_vec_2piRs_roi)


        return inversion_times, inversions, prads_roi_inversions, masks, liuqe_equils_invs