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
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load multiple data files with 'x' replaced by indices from 0 to 999
dataset_fusion = []
for x in range(100000):
    dataset_fusion.append(np.load(f'../../data/sxr_data/sxr_profiles_coarse/sxr_sample_{x}.npy'))
dataset_fusion = np.array(dataset_fusion)
plt.imshow(dataset_fusion[125], cmap='gray')
print(dataset_fusion[125].shape)

# test output of nn with random input
model = MyTinyUNet().to(device)
x = torch.randn(2, 1, 32, 32).to(device)
t = torch.randint(0, 1000, (2,)).to(device)
y = model(x, t)
bs = 3
x = torch.randn(bs,1,120,40)
n_steps=1000
timesteps = torch.randint(0, n_steps, (bs,)).long()
unet = MyTinyUNet(in_c =1, out_c =1)
y = unet(x,timesteps)
y.shape
# plt.imshow(y[0,0].detach().cpu().numpy(), cmap="gray")
# plt.show()

num_timesteps = 1000
betas = torch.linspace(0.0001, 0.02, num_timesteps, dtype=torch.float32).to(device)

network = MyTinyUNet(in_c =1, out_c =1)
model = DDPM(network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device)

bs = 5
x = torch.randn(bs,1,120,40).to(device)
timesteps = 10*torch.ones(bs,).long().long().to(device)
y = model.add_noise(x,x,timesteps)
y.shape
y = model.step(x,timesteps[0],x)
y.shape
# for n, p in model.named_parameters():
#     print(n, p.shape)

dataset_fusion_torch = torch.tensor(dataset_fusion).float().reshape(dataset_fusion.shape[0], 1, dataset_fusion.shape[1], dataset_fusion.shape[2])

transform01 = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.5), (0.5))
    ])

dataset = torch.utils.data.TensorDataset(dataset_fusion_torch)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=512, shuffle=True, num_workers=10)

print(len(dataloader))

import torch
from torch.utils.tensorboard import SummaryWriter # <-- Add this import

train_ = True

if train_ == True:
    learning_rate = 1e-3
    num_epochs = 200000 
    num_timesteps = 1000
    
    # Initialize the TensorBoard writer (it creates a 'runs' folder by default)
    writer = SummaryWriter('runs/fusion_ddpm_experiment') 
    
    network = MyTinyUNet().to(device)
    model = DDPM(network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    
    # Pass the writer into your training loop
    training_loop(model, dataloader, optimizer, num_epochs, num_timesteps, device=device, writer=writer)   

    # Close the writer and save the model
    writer.close()
    torch.save(model.state_dict(), 'fusion_final.pth') 

else:
    model.load_state_dict(torch.load("fusion_final.pth", weights_only=True))