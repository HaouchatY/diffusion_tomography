# Script to reconstruct SXR profiles from tomographic measurements

# Step 0: Select hardware
GPU = True

# Step 1: Load forward model
import numpy as np
import scipy.sparse as sp
import pyxu.abc as pxa

# fwd_matrix = sp.load_npz("fwd/sparse_geometry_matrix_1e3LoS_.npz")
fwd_matrix = sp.load_npz("fwd/sparse_geometry_matrix_sxr.npz")

if GPU:
    import cupyx.scipy.sparse as cps
    fwd_matrix = cps.csr_matrix(fwd_matrix)

fwd = pxa.LinOp.from_array(A=fwd_matrix)
fwd.lipschitz = fwd.estimate_lipschitz()

# Step 2: Load test data
import h5py
test_file = "../training/data/preprocessed/test.h5"

if GPU:
    import cupy as xp
else:
    import numpy as xp

with h5py.File(test_file, 'r') as file:
    keys_list = list(file.keys())

    x = xp.array([file[k] for k in keys_list]).squeeze()

arg_shape = (120, 40)
n_samples = 250

# Step 3: Generate noisy tomographic measurements


def add_white_noise(arr, snr_db, seed=0):
    import pyxu.util as pxu
    xp = pxu.get_array_module(arr)

    signal_power = xp.mean(arr ** 2)
    snr_linear = 10 ** (snr_db / 10)
    # Calculate noise power and generate noise
    noise_power = signal_power / snr_linear
    xp.random.seed(seed=seed)
    y = xp.random.randn(*arr.shape) * np.sqrt(noise_power) + arr
    return y

clean_measurements = fwd(xp.array(x.reshape(n_samples, -1)))
y = add_white_noise(clean_measurements, snr_db=50)

print(y.shape)

# Plot one sample
import matplotlib.pyplot as plt
plt.figure(figsize=(40,20))
plt.plot(clean_measurements[0].get(), "b")
plt.plot(y[0].get(), "r")
plt.show()

# Step 4: Reconstruction with traditional mechanisms

y = y[1]
## Backprojection
x_bp = fwd.adjoint(y).reshape(arg_shape).get()

## Pseudoinerse
x_pinv = fwd.pinv(y, damp=1e-4).reshape(arg_shape).get()

## Tikhonov
import pyxu.operator as pxo
loss = pxo.SquaredL2Norm(y.size).asloss(y) * fwd
gradient = pxo.Gradient(arg_shape=arg_shape, gpu=True, sampling=(0.1, 0.2))
reg = pxo.SquaredL2Norm(2 * x[0].size) * gradient

from pyxu.opt.solver import pgd
from pyxu.opt.stop import RelError, MaxIter
lambda_ = 0.01
solver = pgd.PGD(f=loss+lambda_*reg)
x0 = fwd.adjoint(y)
solver.fit(x0=x0, stop_crit=RelError(eps=1e-4))
x_tik = solver.solution().get().reshape(arg_shape)

# Step 5: Reconstruction with traditional mechanisms
from pyxu.operator.interop import from_torch
from training.utils import utils

device = "cuda:0"
fname = 'test/WCRR-CNN'
model = utils.load_model(fname, device=device)
model.eval()
model.to(device)
sn_pm = model.conv_layer.spectral_norm(mode="power_method", n_steps=500)

import torch
from models import utils as model_utils

sigma_torch = torch.tensor(2).to(device).view(1, 1, 1, 1)
# def denoiser_prox(arr, tau):
#     with torch.no_grad():
#         im_denoised, _, _ = model_utils.accelerated_gd_batch(arr.reshape(1, 1, *arg_shape), model, sigma=sigma_torch, ada_restart=True, tol=1e-4)
#     return im_denoised.ravel()
# nn_denoiser = from_torch(
#     apply=None,
#     prox=denoiser_prox,
#     shape=(1, np.prod(arg_shape)),
#     cls=pxa.ProxDiffFunc,
#     dtype="float32",
#     enable_warnings=True,
#     name='WCRR',
# )
# solver = pgd.PGD(f=loss, g=nn_denoiser)

def denoiser_grad(arr):
    with torch.no_grad():
        im_denoised, _, _ = model_utils.accelerated_gd_batch(arr.reshape(1, 1, *arg_shape), model, sigma=sigma_torch, ada_restart=True, tol=1e-4)
    return arr - im_denoised.ravel()

nn_denoiser = from_torch(
    apply=None,
    grad=denoiser_grad,
    shape=(1, np.prod(arg_shape)),
    cls=pxa.ProxDiffFunc,
    dtype="float32",
    enable_warnings=True,
    name='WCRR',
)
nn_denoiser.diff_lipschitz = 2.

solver = pgd.PGD(f=loss+nn_denoiser)

x0 = fwd.adjoint(y)
solver.fit(x0=x0, stop_crit=RelError(eps=1e-4) | MaxIter(1000))
x_pnp = solver.solution().get().reshape(arg_shape)


fig, axs = plt.subplots(1, 5, figsize=(25, 5))
im = axs[0].imshow(x[1].get())
axs[0].set_title("GT")
plt.colorbar(im, ax=axs[0])
im=axs[1].imshow(x_bp)
axs[1].set_title("BP")
plt.colorbar(im, ax=axs[1])
im=axs[2].imshow(x_pinv)
axs[2].set_title("Pinv")
plt.colorbar(im, ax=axs[2])
im=axs[3].imshow(x_tik)
axs[3].set_title("Tikhonov")
plt.colorbar(im, ax=axs[3])
im=axs[4].imshow(x_pnp)
axs[4].set_title("WRCC")
plt.colorbar(im, ax=axs[4])
plt.show()
