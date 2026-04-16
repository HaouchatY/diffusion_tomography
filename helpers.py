import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import scipy.sparse as sp

from reaction_diffusion.routines.tomo_fusion.tools import plotting_fcts as tomo_plots

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
#  Model: same architecture as training
# ------------------------------

def sinusoidal_embedding(n_steps: int, dim: int) -> torch.Tensor:
    positions = torch.arange(n_steps, dtype=torch.float32).unsqueeze(1)
    div_term = torch.pow(10000.0, (2 * torch.arange(dim, dtype=torch.float32)) / dim).unsqueeze(0)
    emb = positions / div_term
    emb[0::2] = torch.sin(emb[0::2])
    emb[1::2] = torch.cos(emb[1::2])
    return emb


class MyConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(1, out_c) if normalize else nn.Identity()
        self.act = nn.SiLU() if activation is None else activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


def MyTinyBlock(in_c, out_c):
    return nn.Sequential(
        MyConv(in_c, out_c),
        MyConv(out_c, out_c),
        MyConv(out_c, out_c),
    )


def MyTinyUp(in_c):
    return nn.Sequential(
        MyConv(in_c, in_c // 2),
        MyConv(in_c // 2, in_c // 4),
        MyConv(in_c // 4, in_c // 4),
    )


class MyTinyUNet(nn.Module):
    def __init__(self, in_c=1, out_c=1, n_steps=1000, time_emb_dim=100):
        super().__init__()

        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.weight.requires_grad_(False)

        self.te1 = self._make_te(time_emb_dim, in_c)
        self.b1 = MyTinyBlock(in_c, 10)
        self.down1 = nn.Conv2d(10, 10, kernel_size=4, stride=2, padding=1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = MyTinyBlock(10, 20)
        self.down2 = nn.Conv2d(20, 20, kernel_size=4, stride=2, padding=1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = MyTinyBlock(20, 40)
        self.down3 = nn.Conv2d(40, 40, kernel_size=4, stride=2, padding=1)

        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyConv(40, 20),
            MyConv(20, 20),
            MyConv(20, 40),
        )

        self.up1 = nn.ConvTranspose2d(40, 40, kernel_size=4, stride=2, padding=1)
        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = MyTinyUp(80)

        self.up2 = nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, padding=1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = MyTinyUp(40)

        self.up3 = nn.ConvTranspose2d(10, 10, kernel_size=4, stride=2, padding=1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = MyTinyBlock(20, 10)

        self.conv_out = nn.Conv2d(10, out_c, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        t = t.view(-1)
        te = self.time_embed(t)
        b = x.size(0)

        h1 = self.b1(x + self.te1(te).view(b, -1, 1, 1))
        h2 = self.b2(self.down1(h1) + self.te2(te).view(b, -1, 1, 1))
        h3 = self.b3(self.down2(h2) + self.te3(te).view(b, -1, 1, 1))

        h_mid = self.b_mid(self.down3(h3) + self.te_mid(te).view(b, -1, 1, 1))

        u1 = torch.cat([h3, self.up1(h_mid)], dim=1)
        u1 = self.b4(u1 + self.te4(te).view(b, -1, 1, 1))

        u2 = torch.cat([h2, self.up2(u1)], dim=1)
        u2 = self.b5(u2 + self.te5(te).view(b, -1, 1, 1))

        u3 = torch.cat([h1, self.up3(u2)], dim=1)
        u3 = self.b_out(u3 + self.te_out(te).view(b, -1, 1, 1))

        return self.conv_out(u3)

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
        )

# ------------------------------
#  Diffusion schedule + samplers (from scratch)
# ------------------------------

def build_schedule(num_train_steps=1000, beta_start=1e-4, beta_end=2e-2, device=device):
    betas = torch.linspace(beta_start, beta_end, num_train_steps, device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {
        "num_train_steps": num_train_steps,
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
    }


def make_timesteps(num_train_steps, num_inference_steps):
    return np.linspace(num_train_steps - 1, 0, num_inference_steps, dtype=np.int64).tolist()


def predict_x0_from_eps(x_t, eps_t, alpha_bar_t):
    return (x_t - torch.sqrt(1 - alpha_bar_t) * eps_t) / torch.sqrt(alpha_bar_t)


def ddim_update(x0_hat, eps_t, t, t_prev, alpha_bar, eta=0.0):
    if t_prev < 0:
        return x0_hat

    a_t = alpha_bar[t]
    a_prev = alpha_bar[t_prev]

    sigma_t = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
    c_t = torch.sqrt(torch.clamp(1 - a_prev - sigma_t**2, min=0.0))
    noise = torch.randn_like(x0_hat) if eta > 0 else torch.zeros_like(x0_hat)

    x_prev = torch.sqrt(a_prev) * x0_hat + c_t * eps_t + sigma_t * noise
    return x_prev


def apply_A(A, x):
    # x: [B,1,H,W], A: [m,n] with n=H*W
    return x.flatten(1) @ A.T


def apply_At(A, v):
    # v: [B,m], A: [m,n]
    return v @ A


@torch.no_grad()
def sample_prior(
    model,
    schedule,
    image_shape=(1, 120, 40),
    num_samples=8,
    num_inference_steps=250,
    eta=0.0,
    clip_range=None,
):
    model.eval()
    x = torch.randn(num_samples, *image_shape, device=device)

    timesteps = make_timesteps(schedule["num_train_steps"], num_inference_steps)
    alpha_bar = schedule["alpha_bar"]

    for i, t in enumerate(tqdm(timesteps, desc="Prior sampling")):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        eps_t = model(x, t_batch)
        x0_hat = predict_x0_from_eps(x, eps_t, alpha_bar[t])

        if clip_range is not None:
            x0_hat = x0_hat.clamp(*clip_range)

        x = ddim_update(x0_hat, eps_t, t, t_prev, alpha_bar, eta=eta)

    return x


@torch.no_grad()
def sample_posterior_dps(
    model,
    schedule,
    A,
    y,
    image_shape=(1, 120, 40),
    num_samples=8,
    num_inference_steps=250,
    eta=0.0,
    sigma_y=0.05,
    guidance_scale=0.2,
    clip_range=None,
    x_init=None,
):
    """
    DPS-style posterior sampling with linear observations y = A x + n.

    Guidance acts on x0 estimate:
        x0 <- x0 - step_t * grad_x0 ||A x0 - y||^2
    then the diffusion transition uses guided x0.
    """
    model.eval()
    alpha_bar = schedule["alpha_bar"]
    diffusion_process = []  # to store intermediate samples for visualization
    if x_init is None:
        x = torch.randn(num_samples, *image_shape, device=device)
    else:
        x = x_init.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.size(0) == 1 and num_samples > 1:
            x = x.repeat(num_samples, 1, 1, 1)

    if y.dim() == 1:
        y = y.unsqueeze(0)

    timesteps = make_timesteps(schedule["num_train_steps"], num_inference_steps)
    residual_history = []

    for i, t in enumerate(tqdm(timesteps, desc="Posterior sampling")):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)

        eps_t = model(x, t_batch)
        x0_hat = predict_x0_from_eps(x, eps_t, alpha_bar[t])

        pred_y = apply_A(A, x0_hat)
        residual = pred_y - y

        grad_x0 = apply_At(A, residual) / (sigma_y**2 * A.shape[0])
        grad_x0 = grad_x0.view_as(x0_hat)

        step_t = guidance_scale * torch.sqrt(1 - alpha_bar[t])
        grad_norm = grad_x0.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1).clamp_min(1e-8)
        x0_guided = x0_hat - step_t * grad_x0 / grad_norm

        if clip_range is not None:
            x0_guided = x0_guided.clamp(*clip_range)

        x = ddim_update(x0_guided, eps_t, t, t_prev, alpha_bar, eta=eta)
        residual_history.append(residual.norm(dim=1).mean().item())
        diffusion_process.append(x0_guided[0].detach().cpu())

    return x, residual_history, diffusion_process

@torch.no_grad()
def sample_posterior_diffpir(
    model,
    schedule,
    A,
    y,
    image_shape=(1, 120, 40),
    num_samples=8,
    num_inference_steps=250,
    zeta=0.0,
    sigma_y=0.05,
    lambda_=0.2,
    clip_range=None,
    x_init=None,
    clipping_mask=None,
    reaction_diffusion_op=None,
):
    """
    DPS-style posterior sampling with linear observations y = A x + n.

    Guidance acts on x0 estimate:
        x0 <- x0 - step_t * grad_x0 ||A x0 - y||^2
    then the diffusion transition uses guided x0.
    """
    zeta = torch.tensor(zeta, device=device)
    model.eval()
    alpha_bar = schedule["alpha_bar"]
    diffusion_process = []  # to store intermediate samples for visualization
    if x_init is None:
        x = torch.randn(num_samples, *image_shape, device=device)
    else:
        x = x_init.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.size(0) == 1 and num_samples > 1:
            x = x.repeat(num_samples, 1, 1, 1)

    if y.dim() == 1:
        y = y.unsqueeze(0)

    timesteps = make_timesteps(schedule["num_train_steps"], num_inference_steps)
    residual_history = []

    if reaction_diffusion_op is not None:
        # Precompute Lipschitz constant for scaling
        L = reaction_diffusion_op.diff_lipschitz
        grad_op_mat = torch.tensor(reaction_diffusion_op._grad_matrix_based.mat.toarray(), device=device, dtype=torch.float32)

    for i, t in enumerate(tqdm(timesteps, desc="Posterior sampling")):
        sigma_t_bar = torch.sqrt((1 - alpha_bar[t]) / alpha_bar[t])
        rho_t = lambda_ * (sigma_y**2 / sigma_t_bar**2)

        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)

        eps_model = model(x, t_batch)
        x0_hat = predict_x0_from_eps(x, eps_model, alpha_bar[t])

        Id_matrix = torch.eye(A.shape[1], device=A.device)
        H = A.T @ A + rho_t * Id_matrix
        b = apply_At(A, y) + rho_t * x0_hat.view(x0_hat.size(0), -1)
        x0_guided = torch.linalg.solve(H, b.T).T.view_as(x0_hat)
        
        if clipping_mask is not None: #todo introduce vanishing
            x0_guided = x0_guided * clipping_mask

        if reaction_diffusion_op is not None: #todo introduce vanishing
            x0_guided = x0_guided - rho_t/L * (grad_op_mat @ x0_guided.view(x0_guided.size(0), -1).T).T.view_as(x0_guided)

        x = ddim_update(x0_guided, eps_model, t, t_prev, alpha_bar, eta=zeta)

        pred_y = apply_A(A, x0_hat)
        residual = pred_y - y
        residual_history.append(residual.norm(dim=1).mean().item())
        diffusion_process.append(x0_guided[0].detach().cpu())

    return x, residual_history, diffusion_process

def show_samples_(samples, title, ncols=4, vmin=None, vmax=None, figsize=(3,3)):
    samples = samples.detach().cpu()
    n = samples.size(0)
    ncols = min(ncols, n)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        axes[i].axis("off")
        if i < n:
            axes[i].imshow(samples[i, 0], cmap="gray", vmin=vmin, vmax=vmax)
            axes[i].set_title(f"sample {i}")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def show_samples(samples, titles=None, ncols=None, title=None, figsize=None,
                 contour_images=None, colorbar=False, vmax=None, vmin=None):
    n = len(titles) if titles is not None else min(10, samples.shape[0])
    if ncols is None:
        ncols = n
    if figsize is None:
        figsize = (1.5 * ncols, 3)
    fig, ax = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        ax = [ax]
    for i in range(min(n, ncols)):
        contour_image = contour_images[i] if contour_images is not None else None
        tomo_plots.plot_profile(np.array(samples[i].squeeze().detach().cpu()), tcv_plot_clip=True,
                                contour_image=contour_image, cmap="viridis", ax=ax[i],
                                colorbar=colorbar, contour_color="w", vmin=vmin, vmax=vmax,
                                aspect=None, lcfs_width=1, lwidth=0.2)
        ax[i].set_title(titles[i] if titles is not None else "Sample {}".format(i))
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()


def compute_closest_y(y, y_tomo_train):
    # Find the closest y_tomo_train to y
    y_norm = y / torch.norm(y, dim=1, keepdim=True)
    y_tomo_train_norm = y_tomo_train / torch.norm(y_tomo_train, dim=1, keepdim=True)
    idx = torch.argmin(torch.cdist(y_norm, y_tomo_train_norm))
    y_closest = y_tomo_train[idx]
    return y_closest

def compute_normalization_coefficient(y, y_tomo_train):
    y_closest = compute_closest_y(y, y_tomo_train).squeeze()
    coeff = torch.dot(y.squeeze(), y_closest) / torch.dot(y_closest, y_closest)
    return coeff.item(), y_closest