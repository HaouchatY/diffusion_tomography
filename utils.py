import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm.notebook import tqdm
import pyxu.runtime as pxrt
import cupy as cp
import pyxu.opt.stop as pxst
import pyxu.abc as pxa
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def show_images(images, title="", figsize=(8, 50)):
    """Shows the provided images as sub-pictures in a square"""
    images = [im.permute(1,2,0).numpy() for im in images]

    # Defining number of rows and columns
    fig = plt.figure(figsize=figsize)
    rows = len(images)//5
    cols = len(images)//20

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        print(r)
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx], cmap="gray")
                plt.axis('off')
                idx += 1
    fig.suptitle(title, fontsize=30)
    plt.tight_layout()
    # Showing the figure
    plt.show()

def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.tensor([[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)])
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding

class MyConv(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyConv, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        return out
    
def MyTinyBlock(size, in_c, out_c):
    return nn.Sequential(MyConv((in_c, size, size), in_c, out_c), 
                         MyConv((out_c, size, size), out_c, out_c), 
                         MyConv((out_c, size, size), out_c, out_c))

def MyTinyUp(size, in_c):
    return nn.Sequential(MyConv((in_c, size, size), in_c, in_c//2), 
                         MyConv((in_c//2, size, size), in_c//2, in_c//4),
                         MyConv((in_c//4, size, size), in_c//4, in_c//4))

class MyTinyUNet(nn.Module):
  # Here is a network with 3 down and 3 up with the tiny block
    def __init__(self, in_c=1, out_c=1, size=32, n_steps=1000, time_emb_dim=100):
        super(MyTinyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = MyTinyBlock(size, in_c, 10)
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)
        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = MyTinyBlock(size//2, 10, 20)
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)
        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = MyTinyBlock(size//4, 20, 40)
        self.down3 = nn.Conv2d(40, 40, 4, 2, 1)

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyConv((40, size//8, size//8), 40, 20),
            MyConv((20, size//8, size//8), 20, 20),
            MyConv((20, size//8, size//8), 20, 40)
        )

        # Second half
        self.up1 = nn.ConvTranspose2d(40, 40, 4, 2, 1)
        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = MyTinyUp(size//4, 80)
        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = MyTinyUp(size//2, 40)
        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = MyTinyBlock(size, 20, 10)
        self.conv_out = nn.Conv2d(10, out_c, 3, 1, 1)

    def forward(self, x, t): # x is (bs, in_c, size, size) t is (bs)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (bs, 10, size/2, size/2)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (bs, 20, size/4, size/4)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (bs, 40, size/8, size/8)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (bs, 40, size/8, size/8)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (bs, 80, size/8, size/8)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (bs, 20, size/8, size/8)
        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (bs, 40, size/4, size/4)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (bs, 10, size/2, size/2)
        out = torch.cat((out1, self.up3(out5)), dim=1)  # (bs, 20, size, size)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (bs, 10, size, size)
        out = self.conv_out(out) # (bs, out_c, size, size)
        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))
    
def sinusoidal_embedding(n, d):
    """Standard 1D sinusoidal time embedding."""
    embedding = torch.tensor([[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)])
    sin_mask = torch.arange(0, n, 2)
    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])
    return embedding

class MyConv(nn.Module):
    """Conv → (optional) GroupNorm → Activation."""
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1,
                 activation=None, normalize=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(1, out_c) if normalize else nn.Identity()
        self.act  = nn.SiLU() if activation is None else activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)

def MyTinyBlock(in_c, out_c):
    """3× (MyConv)."""
    return nn.Sequential(
        MyConv(in_c,  out_c),
        MyConv(out_c, out_c),
        MyConv(out_c, out_c),
    )

def MyTinyUp(in_c):
    """
    3× (MyConv) that progressively halves channels:
      in_c → in_c//2 → in_c//4 → in_c//4
    """
    return nn.Sequential(
        MyConv(in_c,       in_c // 2),
        MyConv(in_c // 2,  in_c // 4),
        MyConv(in_c // 4,  in_c // 4),
    )

class MyTinyUNet(nn.Module):
    """
    A 3‐down / 3‐up U-Net with tiny blocks,
    now agnostic to H×W (must be divisible by 8).
    """
    def __init__(self, in_c=1, out_c=1, n_steps=1000, time_emb_dim=100):
        super().__init__()

        # time embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.weight.requires_grad_(False)

        # ↓↓↓ Down path ↓↓↓
        self.te1 = self._make_te(time_emb_dim, in_c)
        self.b1  = MyTinyBlock(in_c,  10)
        self.down1 = nn.Conv2d(10, 10, kernel_size=4, stride=2, padding=1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2  = MyTinyBlock(10, 20)
        self.down2 = nn.Conv2d(20, 20, kernel_size=4, stride=2, padding=1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3  = MyTinyBlock(20, 40)
        self.down3 = nn.Conv2d(40, 40, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyConv(40, 20),
            MyConv(20, 20),
            MyConv(20, 40),
        )

        # ↑↑↑ Up path ↑↑↑
        self.up1 = nn.ConvTranspose2d(40, 40, kernel_size=4, stride=2, padding=1)
        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4  = MyTinyUp(80)

        self.up2 = nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, padding=1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5  = MyTinyUp(40)

        self.up3 = nn.ConvTranspose2d(10, 10, kernel_size=4, stride=2, padding=1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out  = MyTinyBlock(20, 10)

        self.conv_out = nn.Conv2d(10, out_c, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        """
        x: (B, in_c, H, W)
        t: (B,) integer timesteps
        """
        # embed time
        te = self.time_embed(t)  # (B, time_emb_dim)
        B  = x.size(0)

        # down 1
        h1 = self.b1(x + self.te1(te).view(B, -1, 1, 1))
        h2 = self.b2(self.down1(h1) + self.te2(te).view(B, -1, 1, 1))
        h3 = self.b3(self.down2(h2) + self.te3(te).view(B, -1, 1, 1))

        # bottleneck
        h_mid = self.b_mid(self.down3(h3) + self.te_mid(te).view(B, -1, 1, 1))

        # up 1
        u1 = torch.cat([h3, self.up1(h_mid)], dim=1)
        u1 = self.b4(u1 + self.te4(te).view(B, -1, 1, 1))

        # up 2
        u2 = torch.cat([h2, self.up2(u1)], dim=1)
        u2 = self.b5(u2 + self.te5(te).view(B, -1, 1, 1))

        # up 3
        u3 = torch.cat([h1, self.up3(u2)], dim=1)
        u3 = self.b_out(u3 + self.te_out(te).view(B, -1, 1, 1))

        return self.conv_out(u3)

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
        )
    
class DDPM(nn.Module):
    def __init__(self, network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device) -> None:
        super(DDPM, self).__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.network = network
        self.device = device
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5 # used in add_noise
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5 # used in add_noise and step

    def add_noise(self, x_start, x_noise, timesteps):
        # The forward process
        # x_start and x_noise (bs, n_c, w, d)
        # timesteps (bs)
        s1 = self.sqrt_alphas_cumprod[timesteps] # bs
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps] # bs
        s1 = s1.reshape(-1,1,1,1) # (bs, 1, 1, 1) for broadcasting
        s2 = s2.reshape(-1,1,1,1) # (bs, 1, 1, 1)
        return s1 * x_start + s2 * x_noise

    def reverse(self, x, t):
        # The network return the estimation of the noise we added
        return self.network(x, t)
    
    def step(self, model_output, timestep, sample):
        # one step of sampling
        # timestep (1)
        t = timestep
        coef_epsilon = (1-self.alphas)/self.sqrt_one_minus_alphas_cumprod
        coef_eps_t = coef_epsilon[t].reshape(-1,1,1,1)
        coef_first = 1/self.alphas ** 0.5
        coef_first_t = coef_first[t].reshape(-1,1,1,1)
        pred_prev_sample = coef_first_t*(sample-coef_eps_t*model_output)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output).to(self.device)
            variance = ((self.betas[t] ** 0.5) * noise)
            
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
    



    # def posterior_sample(self, model_output, timestep, sample, linear_operator, y):
    #     # y is the observation done with the linear operator
    #     # one step of sampling
    #     # timestep (1)
    #     t = timestep
    #     coef_epsilon = (1-self.alphas)/self.sqrt_one_minus_alphas_cumprod
    #     coef_eps_t = coef_epsilon[t].reshape(-1,1,1,1)
    #     coef_first = 1/self.alphas ** 0.5
    #     coef_first_t = coef_first[t].reshape(-1,1,1,1)
        
    #     pred_prev_sample = coef_first_t*(sample-coef_eps_t*model_output)

    #     variance = 0
    #     if t > 0:
    #         noise = torch.randn_like(model_output).to(self.device)
    #         variance = ((self.betas[t] ** 0.5) * noise)
            
    #     pred_prev_sample = pred_prev_sample + variance

    #     #return pred_prev_sample

    #     chainrule = False
    #     with pxrt.Precision(pxrt.Width.SINGLE):
    #         # convert y to cupy array of shape (,-1)
            
    #         if chainrule:
    #             v1 = cp.from_dlpack(pred_prev_sample.detach()).reshape(-1)
    #             v2 = cp.array(y).reshape(-1)
    #             Ax = linear_operator.apply(v1)
    #             v3 = (Ax - v2)**2
    #             l2_error = cp.sum(v3)
    #             zeta = 1/torch.sqrt(torch.tensor(l2_error))

    #             f = lambda x : self.step(model_output, timestep, x)
    #             #A^T(y-Ax)
    #             v = linear_operator.adjoint(linear_operator.apply(cp.from_dlpack(pred_prev_sample.detach()).reshape(-1))-cp.array(y).reshape(-1))
    #             v = torch.tensor(v).reshape(pred_prev_sample.shape).to(self.device)
    #             # make pred_prev_sample.shape[0] times stack of v
    #             #v = torch.stack([v for i in range(pred_prev_sample.shape[0])]).to(self.device)
    #             gradx_l2_error = torch.autograd.functional.vjp(f, sample, v=v)[1]

    #         else:
    #             B, C, H, W = pred_prev_sample.shape
    #             Ax = torch.matmul(pred_prev_sample.reshape(B, H*W), linear_operator.T)
    #             #l2_error = torch.sum(((linear_operator@((pred_prev_sample.reshape(B, H*W)).T)).T - y.reshape(-1))**2)
    #             l2_error = torch.sum((Ax - y.reshape(-1))**2)
    #             zeta = 1/torch.sqrt(l2_error)
    #             gradx_l2_error = torch.autograd.grad(outputs=l2_error, inputs=sample)[0]
           
    #     # masking for stochastic gradient 0 or 1
    #     pred_prev_sample -= 2*zeta*gradx_l2_error
    #     try:
    #         print('res', torch.linalg.norm(sample-pred_prev_sample)/torch.linalg.norm(sample))
    #     except:
    #         print('fail')

    #     return pred_prev_sample
    
    
    def posterior_sample(self, model_output, timestep, sample, AtA, Aty, identity, rho_t, zeta=0.3):
        t = timestep
        
        alpha_bar_t      = self.alphas_cumprod[t]
        sqrt_alpha_bar_t = alpha_bar_t ** 0.5
        sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t) ** 0.5

        # Step 1: predict x_0_hat from (x_t, eps_theta)
        x_0_hat = (sample - sqrt_one_minus_alpha_bar_t * model_output) / sqrt_alpha_bar_t

        # Step 2: proximal / data-consistency update → z
        B, C, H, W = x_0_hat.shape
        x_0_flat = x_0_hat.reshape(B, -1)
        lhs = AtA + rho_t * identity
        rhs = Aty.unsqueeze(0) + rho_t * x_0_flat
        z_flat = torch.linalg.solve(lhs, rhs.T).T
        z = z_flat.reshape(B, C, H, W)

        # Step 3: re-noise using the TRUE DDPM posterior variance
        # p(x_{t-1} | x_t, x_0) = N(mu_t, sigma_t^2)
        # mu_t = sqrt(alpha_bar_{t-1})*beta_t/(1-alpha_bar_t) * z
        #       + sqrt(alpha_t)*(1-alpha_bar_{t-1})/(1-alpha_bar_t) * x_t
        if t > 0:
            alpha_bar_t_prev = self.alphas_cumprod[t - 1]
            alpha_t          = self.alphas[t]
            beta_t           = self.betas[t]

            # posterior mean coefficients (same as DDPM paper eq. 7)
            coef_x0 = (alpha_bar_t_prev ** 0.5) * beta_t / (1 - alpha_bar_t)
            coef_xt = (alpha_t ** 0.5) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
            mu_t = coef_x0 * z + coef_xt * sample

            # posterior variance (DDPM eq. 6) — use zeta to interpolate 0→full stochasticity
            sigma_t_sq = beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
            sigma_t    = (zeta * sigma_t_sq) ** 0.5   # zeta=0 → deterministic, zeta=1 → full DDPM

            pred_prev_sample = mu_t + sigma_t * torch.randn_like(sample)
        else:
            # t=0: just return the clean prediction
            pred_prev_sample = z

        return pred_prev_sample

import torchvision

def training_loop(model, dataloader, optimizer, num_epochs, num_timesteps, device, writer=None):
    """Training loop for DDPM"""

    global_step = 0
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            noise = torch.randn(batch.shape).to(device)
            timesteps = torch.randint(0, num_timesteps, (batch.shape[0],)).long().to(device)

            noisy = model.add_noise(batch, noise, timesteps)
            noise_pred = model.reverse(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- TensorBoard Logging (Loss) ---
            if writer is not None:
                writer.add_scalar('Loss/train', loss.detach().item(), global_step)
            # ----------------------------------

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            
            global_step += 1
        
        progress_bar.close()

        # --- NEW: Generate and log images every 2 epochs ---
        if epoch % 2 == 0 and writer is not None:
            # Infer spatial shape (H, W) from the current batch
            img_shape = (batch.shape[2], batch.shape[3])
            
            # Generate 3 samples (Note: your generate_image function sets model.eval() internally)
            generated_samples, _ = generate_image(model, sample_size=3, channel=1, shape=img_shape)
            
            # Stack the list of tensors into a single batched tensor of shape (3, 1, H, W)
            images_tensor = torch.stack(generated_samples)
            
            # Create a clean grid layout. normalize=True maps pixel values smoothly to [0,1] for display
            img_grid = torchvision.utils.make_grid(images_tensor, nrow=3, normalize=True)
            
            # Send the image grid to TensorBoard
            writer.add_image('Generated_Samples', img_grid, global_step)
            
            # Switch the model back to train mode for the next epoch
            model.train()
        # ---------------------------------------------------

        # Save checkpoint every 100 epochs to prevent disk overflow
        torch.save(model.state_dict(), f"fusion_epoch_{epoch}.pth")

def generate_image(ddpm, sample_size, channel, shape, linear_operator=None, y=None, step_size=1.0):
    frames = []
    frames_mid = []
    ddpm.eval()

    timesteps = list(range(ddpm.num_timesteps))[::-1]
    sample = torch.randn(sample_size, channel, shape[0], shape[1]).to(device)

    for i, t in enumerate(tqdm(timesteps)):
        time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)

        if linear_operator is not None and y is not None:
            # Need grad through sample to backprop the likelihood gradient
            sample = sample.detach().requires_grad_(True)

        residual = ddpm.reverse(sample, time_tensor)

        if linear_operator is not None and y is not None:
            # Tweedie estimate of x_0 (in the graph, so gradients flow)
            alpha_bar_t = ddpm.alphas_cumprod[t]
            x_0_hat = (sample - (1 - alpha_bar_t) ** 0.5 * residual) / (alpha_bar_t ** 0.5)

            # Likelihood gradient: ||A x_0_hat - y||^2
            B = sample.shape[0]
            Ax = (linear_operator @ x_0_hat.reshape(B, -1).T)  # shape: [obs, B]
            residual_y = Ax - y.reshape(-1, 1)                  # shape: [obs, B]
            loss = 0.5 * (residual_y ** 2).sum()
            grad = torch.autograd.grad(loss, sample)[0]

            # Adaptive step size: normalize by ||A x_0_hat - y||
            norm = residual_y.norm()

        with torch.no_grad():
            sample_new = ddpm.step(residual, time_tensor[0], sample)

            if linear_operator is not None and y is not None:
                sample_new = sample_new - (step_size / (norm + 1e-8)) * grad

            sample = sample_new

        if t == 500:
            for j in range(sample_size):
                frames_mid.append(sample[j].detach().cpu())

    for j in range(sample_size):
        frames.append(sample[j].detach().cpu())

    return frames, frames_mid

import torch
from tqdm import tqdm

def posterior_generate_image(ddpm, sample_size, channel, shape, linear_operator, y, lambda_val=1.0, sigma_n=0.01, zeta=0.3, initial_guess=None):
    """Generate the image using DiffPIR (Closed-form Proximal Operator)"""
    frames = []
    frames_mid = []
    ddpm.eval()

    timesteps = list(range(ddpm.num_timesteps))[::-1]
    sample = torch.randn(sample_size, channel, shape[0], shape[1]).to(device)
    
    if initial_guess is not None:
        initial_guess = torch.tensor(initial_guess).to(device).float()
        sample = initial_guess.repeat(sample_size, channel, 1, 1)
        
    # --- DiffPIR Precomputations ---
    # Precompute AtA and Aty so we don't do it 1000 times
    AtA = linear_operator.T @ linear_operator # Shape: [4800, 4800]
    Aty = linear_operator.T @ y.reshape(-1)   # Shape: [4800]
    identity = torch.eye(AtA.shape[0], device=device)
    
    # Base rho (lambda * sigma_n^2) from Algorithm 1
    rho_base = lambda_val * (sigma_n ** 2)

    with torch.no_grad(): # DiffPIR doesn't need gradients! Memory is saved.
        for i, t in enumerate(tqdm(timesteps)):
            
            time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
            
            # 1. Predict noise using the U-Net
            residual = ddpm.reverse(sample, time_tensor)
            
            # 2. Calculate dynamic rho_t for this specific timestep
            # Effective noise variance \bar{\sigma}_t^2 = (1 - \bar{\alpha}_t) / \bar{\alpha}_t
            alpha_bar_t = ddpm.alphas_cumprod[t]
            sigma_bar_t_sq = (1 - alpha_bar_t) / alpha_bar_t
            
            # Prevent division by zero at the very last step
            rho_t = rho_base / (sigma_bar_t_sq + 1e-8) 
            
            # 3. DiffPIR Sample Step
            sample = ddpm.posterior_sample(
                model_output=residual, 
                timestep=t, 
                sample=sample, 
                AtA=AtA, 
                Aty=Aty, 
                identity=identity,
                rho_t=rho_t,
                zeta=zeta
            )
            
            if t == 500:
                for j in range(sample_size):
                    frames_mid.append(sample[j].detach().cpu())

    for i in range(sample_size):
        frames.append(sample[i].detach().cpu())
        
    return frames, frames_mid



def rescale(x):
    return (x+1)/2

def show_images_rescale(images, title=""):
    """Shows the provided images as sub-pictures in a square"""
    images = [rescale((im.permute(1,2,0)).numpy()) for im in images]

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                #plt.imshow(images[idx].reshape(pixel, pixel, n_channels), cmap="gray")
                plt.imshow(images[idx], cmap="gray")
                plt.axis('off')
                idx += 1
    fig.suptitle(title, fontsize=30)
    
    # Showing the figure
    plt.show()