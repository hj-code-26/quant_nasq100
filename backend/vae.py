"""
변이형 오토인코더 (VAE) — 논문 3.2.3절 / [표 3.2] 구현.

논문 설정을 따르되 실시간 구동을 위해 규모를 축소:
  - 잠재 변수(latent dim) = 15  (논문과 동일, 다중공선성 완화 목적)
  - Conv 필터 크기 3, 스트라이드 2, padding same, ReLU, Adam (논문과 동일)
  - 입력: 2x2 GAF 타일 (1채널 40x40; 논문은 128x128 RGB)
  - 손실: 재구축 오류(BCE) + KL 발산  (식 3.8)
  - 재매개변수 트릭 (식 3.10)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ConvVAE(nn.Module):
    def __init__(self, img: int = 40, latent_dim: int = 15, channels: int = 1):
        super().__init__()
        self.img = img
        self.latent_dim = latent_dim
        self.channels = channels
        # Encoder: 40 -> 20 -> 10 -> 5
        self.enc = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        f = img // 8
        self.flat = 128 * f * f
        self.fc_mu = nn.Linear(self.flat, latent_dim)
        self.fc_logvar = nn.Linear(self.flat, latent_dim)
        # Decoder: 5 -> 10 -> 20 -> 40
        self.fc_dec = nn.Linear(latent_dim, self.flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 3, stride=2, padding=1, output_padding=1),
        )
        self._f = f

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)  # 식 (3.10)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 128, self._f, self._f)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar):
    """식 (3.8): 재구축 오류 + KL 규제항."""
    bce = F.binary_cross_entropy_with_logits(recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


def train_vae(X: np.ndarray, epochs: int = 25, batch_size: int = 64,
              lr: float = 1e-3, latent_dim: int = 15, progress=None,
              seed: int = 42) -> ConvVAE:
    """X: (M, C, H, W) float32 in [0,1]. seed 고정으로 재현성 확보(개선사항)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    model = ConvVAE(img=X.shape[-1], latent_dim=latent_dim,
                    channels=X.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = torch.from_numpy(X)
    n = len(data)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            xb = data[perm[i:i + batch_size]].to(dev)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss = vae_loss(recon, xb, mu, logvar)
            loss.backward()
            opt.step()
            total += loss.item()
        if progress:
            progress(ep + 1, epochs, total / n)
    model.eval()
    return model


@torch.no_grad()
def extract_latents(model: ConvVAE, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """학습된 VAE 인코더로 잠재 변수(mu) 추출 — 논문 3.3절."""
    dev = device()
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(dev)
        mu, _ = model.encode(xb)
        outs.append(mu.cpu().numpy())
    return np.concatenate(outs, axis=0)
