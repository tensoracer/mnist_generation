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

from guided.guided_diffusion.unet import EncoderUNetModel

# архитектура для шумного классификатора

class NoisyClassifier(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.noisy_classifier = EncoderUNetModel(
            image_size=32,
            in_channels=1,
            model_channels=32,
            out_channels=10,
            num_res_blocks=2,
            attention_resolutions=(2, 2),
            pool='spatial'
        )
        self.noisy_classifier.out = nn.Sequential(
            nn.Linear(1472, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x_t, t):
        return self.noisy_classifier(x_t, t)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPM(nn.Module):
    def __init__(self, network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device) -> None:
        super(DDPM, self).__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.network = network
        self.device = device
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5 
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5 

    def add_noise(self, x_start, x_noise, timesteps):
        # The forward process
        # x_start and x_noise (bs, c, w, d)
        # timesteps (bs)
        s1 = self.sqrt_alphas_cumprod[timesteps] # bs
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps] # bs
        s1 = s1.reshape(-1,1,1,1) # (bs, 1, 1, 1) for broadcasting
        s2 = s2.reshape(-1,1,1,1) # (bs, 1, 1, 1)
        return s1 * x_start + s2 * x_noise # sqrt_(alphas_cumprod) . x0 + sqrt_(one_minus_alphas_cumprod) . epsilon Algorithm 1 Training

    def reverse(self, x, t, y):
        # The network return the estimation of the noise we added
        return self.network(x, t, y)

    def step(self, unet_output, timestep, sample):
        # one step of sampling
        # timestep.shape = (1)
        t = timestep
        coef_epsilon = (1-self.alphas)/self.sqrt_one_minus_alphas_cumprod
        coef_eps_t = coef_epsilon[t].reshape(-1,1,1,1)
        coef_first = 1/self.alphas ** 0.5
        coef_first_t = coef_first[t].reshape(-1,1,1,1)

        x_t_m_1 = coef_first_t*(sample-coef_eps_t*unet_output) # Algorithm 2 Sampling

        variance = 0
        if t > 0:
            noise = torch.randn_like(unet_output).to(self.device)
            variance = ((self.betas[t] ** 0.5) * noise) # 3.2 Reverse process

        x_t_m_1 = x_t_m_1 + variance

        return x_t_m_1
    

noisy_classifier = NoisyClassifier()
noisy_classifier.load_state_dict(torch.load("noisy_classifier.pth"))
noisy_classifier = noisy_classifier.to(device)
noisy_classifier.eval()
class DDPM_guide(nn.Module):
    def __init__(self, network, num_timesteps, gradient_scale, beta_start=0.0001, beta_end=0.02, device=device):
        super(DDPM_guide, self).__init__()
        self.gradient_scale =  torch.tensor([gradient_scale]).to(device)
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.network = network
        self.device = device
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5 # used in add_noise
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5 # used in add_noise and step


    def reverse(self, x, t, y):
        # The network return the estimation of the noise we added
        return self.network(x, t, y)


    def cond_fn(self, x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = noisy_classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * self.gradient_scale

    
    def step(self, model_output, timestep, gradient, sample):
        # one step of sampling
        # timestep.shape = (1)
        t = timestep
        coef_epsilon = (1-self.alphas)/self.sqrt_one_minus_alphas_cumprod
        coef_eps_t = coef_epsilon[t].reshape(-1,1,1,1)
        coef_first = 1/self.alphas ** 0.5
        coef_first_t = coef_first[t].reshape(-1,1,1,1)

        x_t_m_1 = coef_first_t*(sample-coef_eps_t*model_output) + gradient * self.gradient_scale # Algorithm 2 Sampling

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output).to(self.device)
            variance = ((self.betas[t] ** 0.5) * noise) # 3.2 Reverse process

        x_t_m_1 = x_t_m_1 + variance

        return x_t_m_1


def training_loop_ddpm(model, dataloader, optimizer, num_epochs, num_timesteps, device=device):
    """Training loop for DDPM"""

    global_step = 0
    losses = []

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch_image in enumerate(dataloader):
            batch = batch_image[0].to(device)
            target = batch_image[1].to(device)

            noise = torch.randn(batch.shape).to(device)

            timesteps = torch.randint(0, num_timesteps, (batch.shape[0],)).long().to(device)

            noisy = model.add_noise(batch, noise, timesteps)

            noise_pred = model.reverse(noisy, timesteps, target)

            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())

            if epoch == 25:

                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.detach().item(),
                }, f"ddpm_checkpoint_at_{epoch}.pth")

            progress_bar.set_postfix(**logs)
            global_step += 1

        progress_bar.close()

num_timesteps = 1000
model_ddpm = DDPM(None, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device)
def training_loop_classifier(model, dataloader, optimizer, num_epochs, num_timesteps, model_ddpm, device=device):
    """Training loop for NoisyClassifier"""

    global_step = 0
    losses = []

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch_images in enumerate(dataloader):
            batch = batch_images[0].to(device)
            target = batch_images[1].to(device)

            noise = torch.randn(batch.shape).to(device)
            timesteps = torch.randint(0, num_timesteps, (batch.shape[0],)).long().to(device)

            noisy = model_ddpm.add_noise(batch, noise, timesteps)

            logits = model(noisy, timesteps)

            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())

            if epoch == 25:

                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.detach().item(),
                }, f"classifier_checkpoint_at_{epoch}.pth")

            progress_bar.set_postfix(**logs)
            global_step += 1

        progress_bar.close()



def return_dataloader():
    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5), (0.5)) # Scale data between [-1, 1]
        ])
    dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=2048, shuffle=True, num_workers=2)
    return dataloader



def generate_image(ddpm, sample_size, channel, size):
    """Generate the image from the Gaussian noise"""

    frames = []
    frames_mid = []
    frames_40 = []
    frames_30 = []
    frames_20 = []
    frames_10 = []
    ddpm.eval()
    with torch.no_grad():
        timesteps = list(range(ddpm.num_timesteps))[::-1]
        sample = torch.randn(sample_size, channel, size, size).to(device)
        target = torch.randint(0, 10, (sample_size,)).to(device)

        for i, t in enumerate(tqdm(timesteps)):
            time_tensor = (torch.ones(sample_size) * t).long().to(device)

            residual = ddpm.reverse(sample, time_tensor, target)
            sample = ddpm.step(residual, time_tensor[0], sample)

            if t==500:
                for i in range(sample_size):
                    frames_mid.append(sample[i].detach().cpu())
            
            if t==400:
                for i in range(sample_size):
                    frames_40.append(sample[i].detach().cpu())

            if t==300:
                for i in range(sample_size):
                    frames_30.append(sample[i].detach().cpu())

            if t==200:
                for i in range(sample_size):
                    frames_20.append(sample[i].detach().cpu())

            if t==100:
                for i in range(sample_size):
                    frames_10.append(sample[i].detach().cpu())

        for i in range(sample_size):
            frames.append(sample[i].detach().cpu())
    return frames, frames_mid, frames_10, frames_20, frames_30, frames_40



def generate_image_with_guid(ddim, sample_size, channel, size):
    """Generate the image from the Gaussian noise"""

    frames = []
    ddim.eval()
    with torch.no_grad():
        timesteps = list(range(1, ddim.num_timesteps))[::-1]
        sample = torch.randn(sample_size, channel, size, size).to(device)
        target = torch.randint(0, 10, (sample_size,)).to(device)

        for i, t in enumerate(tqdm(timesteps)):
            time_tensor = (torch.ones(sample_size) * t).long().to(device)

            residual = ddim.reverse(sample, time_tensor, target)

            gradient = ddim.cond_fn(sample, time_tensor, target)

            sample = ddim.step(residual, time_tensor[0], gradient, sample)

        for i in range(sample_size):
            frames.append(sample[i].detach().cpu())
    return frames


def show_images(images, title="", name=""):
    """Shows the provided images as sub-pictures in a square"""
    images = [im.permute(1,2,0).numpy() for im in images]

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
                plt.imshow(images[idx], cmap="gray")
                plt.axis('off')
                idx += 1
    fig.suptitle(title, fontsize=30)

    plt.savefig(f"{name}.png") 
    # Showing the figure
    plt.show()




noisy_classifier = NoisyClassifier()
noisy_classifier.load_state_dict(torch.load("noisy_classifier.pth"))
noisy_classifier = noisy_classifier.to(device)
noisy_classifier.eval()
class DDIM(nn.Module):
    def __init__(self, network, num_timesteps, gradient_scale, beta_start=0.0001, beta_end=0.02, device=device) -> None:
        super(DDIM, self).__init__()
        self.gradient_scale =  torch.tensor([gradient_scale]).to(device)
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.network = network
        self.device = device
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5 # used in add_noise
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5 # used in add_noise and step


    def reverse(self, x, t, y):
        # The network return the estimation of the noise we added
        return self.network(x, t, y)


    def cond_fn(self, x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = noisy_classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * self.gradient_scale


    def step(self, model_output, timestep, gradient, sample):
        # one step of sampling
        # timestep (1)
        t = timestep

        new_model_output = model_output - self.sqrt_one_minus_alphas_cumprod[t].reshape(-1,1,1,1) * gradient * self.gradient_scale
        frac = self.sqrt_alphas_cumprod[t-1]/self.sqrt_alphas_cumprod[t]

        pred_prev_sample_p1 = frac.reshape(-1,1,1,1) * (sample - self.sqrt_one_minus_alphas_cumprod[t].reshape(-1,1,1,1) * new_model_output)

        pred_prev_sample = pred_prev_sample_p1 + self.sqrt_one_minus_alphas_cumprod[t-1].reshape(-1,1,1,1) * new_model_output

        return pred_prev_sample