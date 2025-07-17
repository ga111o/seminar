import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Select GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SimpleDDPM(nn.Module):
    def __init__(self, image_size=28, channels=3, timesteps=1000,
                 beta_start=1e-4, beta_end=0.02):
        super().__init__()
        # Register buffers to stay on correct device automatically
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_prev = torch.cat([torch.tensor([1.]), alpha_cumprod[:-1]], dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('alpha_prev', alpha_prev)
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha', torch.sqrt(1 - alpha_cumprod))

        self.model = SimpleUNet(image_size, channels, timesteps).to(device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sa = self.sqrt_alpha_cumprod[t].view(-1,1,1,1)
        sb = self.sqrt_one_minus_alpha[t].view(-1,1,1,1)
        return sa * x_start + sb * noise

    def p_losses(self, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred_noise = self.model(x_noisy, t)
        return nn.functional.mse_loss(pred_noise, noise)

    def p_sample(self, x, t):
        beta_t = self.betas[t].view(-1,1,1,1)
        sa_cum = self.sqrt_one_minus_alpha[t].view(-1,1,1,1)
        sr     = (1.0 / torch.sqrt(self.alphas[t])).view(-1,1,1,1)

        eps = self.model(x, t)
        mean = sr * (x - beta_t * eps / sa_cum)
        if t[0] == 0:
            return mean
        var = beta_t * (1 - self.alpha_prev[t]) / (1 - self.alpha_cumprod[t])
        return mean + torch.sqrt(var).view(-1,1,1,1) * torch.randn_like(x)

    def sample(self, batch_size=1):
        shape = (batch_size, 3, 28, 28)
        img = torch.randn(shape, device=device)
        for i in reversed(range(self.betas.size(0))):
            t = torch.full((batch_size,), i, dtype=torch.long, device=device)
            img = self.p_sample(img, t)
        return img

class SimpleUNet(nn.Module):
    def __init__(self, image_size, channels, timesteps):
        super().__init__()
        input_dim  = image_size * image_size * channels
        time_dim   = 128
        hidden_dim = 512

        # Time embedding
        self.time_embed = nn.Embedding(timesteps, time_dim)

        # MLP layers
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        batch = x.size(0)
        x_flat = x.view(batch, -1)
        t_emb  = self.time_embed(t)                    # (batch, time_dim)
        h      = torch.cat([x_flat, t_emb], dim=1)     # (batch, input_dim+time_dim)
        out    = self.net(h)
        return out.view_as(x)

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, image_size=28):
        self.paths = [os.path.join(root_dir, f)
                      for f in os.listdir(root_dir)
                      if f.lower().endswith(('.png','.jpg','.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),                   # [0,1]
            transforms.Normalize(0.5, 0.5)           # [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)

def save_image(image, path):
    # If input is a torch.Tensor, detach and move to CPU
    if hasattr(image, 'detach'):
        image = image.detach().cpu().numpy()

    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2

    # Clip values safely in NumPy
    image = np.clip(image, 0.0, 1.0)

    # Handle batch dimension
    if image.ndim == 4:
        image = image[0]

    # Convert CHW to HWC if necessary
    if image.shape[0] in (1, 3) and image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))

    # Scale to [0,255] and convert to uint8
    image = (image * 255).astype(np.uint8)

    # Save via PIL
    pil = Image.fromarray(image)
    pil.save(path)


def train_ddpm(ddpm, loader, epochs=100, lr=1e-4):
    ddpm.to(device)
    opt = Adam(ddpm.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            t = torch.randint(0, ddpm.betas.size(0), (batch.size(0),), device=device)
            loss = ddpm.p_losses(batch, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{epochs} Loss: {total_loss/len(loader):.4f}")
        
        # Generate sample every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("Generating sample...")
            sample = ddpm.sample(batch_size=1)
            os.makedirs('./generated', exist_ok=True)
            save_image(sample, f'./generated/sample_epoch_{epoch + 1}.png')

def main():
    image_size = 28
    channels = 3
    timesteps = 1000
    epochs = 100
    batch_size = 4
    learning_rate = 1e-4
    
    dataset = ImageFolderDataset('./images', image_size=28)
    loader  = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Loaded {len(dataset)} images")
    
    # Initialize DDPM
    print("Initializing DDPM...")
    ddpm = SimpleDDPM(image_size, channels, timesteps)
    
    # Train model
    print("Training DDPM...")
    train_ddpm(ddpm, loader, epochs, learning_rate)
    
    # Generate final samples
    print("Generating final samples...")
    os.makedirs('./generated', exist_ok=True)
    
    for i in range(5):
        sample = ddpm.sample(batch_size=1)
        save_image(sample, f'./generated/final_sample_{i + 1}.png')
    
    print("Training and generation completed!")
    print("Generated images saved in ./generated/")

if __name__ == "__main__":
    main()
