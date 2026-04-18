

from typing import Optional, Dict, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}

def register_model(name: str):
    def deco(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return deco

# --------------------
# Autoencoder (AE)
# --------------------
@register_model("ae")
class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims=(256,128)):
        super().__init__()
        dims = [input_dim, *hidden_dims, latent_dim]
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i == 0 :
                norm = nn.BatchNorm1d(dims[i+1])
                enc_layers.append(norm)
            if i < len(dims) - 2:
                enc_layers.append(nn.ReLU(True))
        self.encoder = nn.Sequential(*enc_layers)

        # ----- Decoder -----
        dec_dims = [latent_dim, *hidden_dims[::-1], input_dim]
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i+1]))
            if i == 0:
                norm = nn.BatchNorm1d(dec_dims[i+1])
                dec_layers.append(norm)
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.ReLU(True))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
    
    @staticmethod
    def loss_function(output, target, **extra):
        x_hat, _z = output
        return nn.functional.mse_loss(x_hat, target, reduction='mean')


# --------------------
# Variational Autoencoder (VAE)
# --------------------
@register_model("vae")
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims=(256,128)):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i == 0 :
                norm = nn.BatchNorm1d(dims[i+1])
                enc_layers.append(norm)
            enc_layers.append(nn.ReLU(True))
        self.encoder = nn.Sequential(*enc_layers)
        last_dim = hidden_dims[-1] if len(hidden_dims) > 0 else input_dim
        self.mu = nn.Linear(last_dim, latent_dim)
        self.logvar = nn.Linear(last_dim, latent_dim)

        # ----- Decoder -----
        dec_hidden = list(hidden_dims[:-1])[::-1]
        dec_dims = [latent_dim, *dec_hidden, input_dim]
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i+1]))
            if i == 0:
                norm = nn.BatchNorm1d(dec_dims[i+1])
                dec_layers.append(norm)
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.ReLU(True))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    @staticmethod
    def loss_function(output, target, **extra):
        beta = float(extra.get("beta", 1.0))
        x_hat, mu, logvar, _z = output
        recon = F.mse_loss(x_hat, target, reduction='mean')
        prior = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        posterior = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
        kld = torch.distributions.kl.kl_divergence(posterior, prior).sum(dim=-1).mean()
        return recon + beta * kld


# --------------------
# LSTM (temporal predictor in latent space)
# --------------------
@register_model("rnn")
class LatentLSTM(nn.Module):
    def __init__(self, latent_dim: int, window: int, hidden_size: int = 100, num_layers: int = 2, proj_ratio: float = 0.5):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.window = int(window)
        self.input_dims = self.window * self.latent_dim
        self.hidden_size = int(hidden_size)
        self.proj_size = max(1, int(self.hidden_size * proj_ratio))

        self.lstm = nn.LSTM(
            input_size=self.input_dims,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            proj_size=self.proj_size,
            batch_first=True,
        )
        self.head = nn.Linear(self.proj_size, self.latent_dim)

    def forward(self, x):
        # x: [B, win, D] or [B, win*D]
        if x.dim() == 3:
            B, T, D = x.shape
            assert T == self.window and D == self.latent_dim, f"Expected [B,{self.window},{self.latent_dim}], got {x.shape}"
            x = x.reshape(B, self.window * self.latent_dim)   # [B, win*D]
        elif x.dim() == 2:
            B, F = x.shape
            assert F == self.window * self.latent_dim, f"Expected [B,{self.window*self.latent_dim}], got {x.shape}"
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")

        out, _ = self.lstm(x)               # [B, proj]
        return self.head(out)              # [B, D]

    @staticmethod
    def loss_function(output, target, **_):
        # output: [B, D], target: [B, D]
        return F.mse_loss(output, target, reduction='mean')


# --------------------
# Transformer (temporal predictor in latent space)
# --------------------
@register_model("transformer")
class LatentTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        win: int = 10,
        n_heads: int = 4,
        depth_enc: int = 2,
        depth_dec: int = 1,
        d_model: int = 256,
        dim_ff: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.win = win

        self.inp = nn.Linear(input_dim, d_model)
        self.in_drop = nn.Dropout(0.05)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth_enc)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=depth_dec)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

        pe = self._build_sinusoidal_pe(max_len=win, d_model=d_model)
        self.register_buffer("pe", pe, persistent=False)

        self.pre_head_ln = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, input_dim)
        )
        self._init_weights()

    @staticmethod
    def _build_sinusoidal_pe(max_len, d_model):
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if isinstance(self.query, nn.Parameter):
            nn.init.xavier_uniform_(self.query)

    def forward(self, x_seq):
        # x_seq: [B, T, D]
        B, T, D = x_seq.shape
        assert D == self.input_dim
        proj = self.inp(x_seq) # [B, T, d_model]
        h = self.in_drop(proj + self.pe[:, :T, :])  # [B, T, d_model]
        memory = self.encoder(h) # [B, T, d_model]
        query = self.query.repeat(B, 1, 1)
        h = self.decoder(tgt=query, memory=memory) # [B, 1, d_model]
        pooled = h.squeeze(1)

        z_next = self.head(self.pre_head_ln(pooled))  # [B, D]
        return z_next

    @staticmethod
    def loss_function(output, target, **_):
        return F.mse_loss(output, target, reduction='mean')

# --------------------
# DNN classifier (PINPOINT head) over residual stacks
# Input shape assumption: [B, M, 3, C]  (M = MC samples, 3 residual types, C channels)
# --------------------
@register_model("pinpoint")
class PINPOINT(nn.Module):
    def __init__(self, 
                 channels: int,
                 num_residuals: int,
                 mc_samples: int = 8, 
                 num_classes: int = 4, 
                 hidden=(128, 64, 32)):
        super().__init__()
        self.mc = mc_samples
        self.c = channels
        self.num_classes = num_classes
        # First block: per-residual extractor
        self.per_residual = nn.Sequential(
            nn.Linear(self.c, hidden[0]), nn.ReLU(True),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(True),
            nn.Linear(hidden[1], hidden[2]), nn.ReLU(True),
        )
        # Second block: combine across residual types and MC samples
        comb_in = hidden[2] * num_residuals  # residual types
        self.combine = nn.Sequential(
            nn.Linear(comb_in, 64), nn.ReLU(True),
            nn.Linear(64, 16), nn.ReLU(True),
            nn.Linear(16, num_classes)  # logits per sample
        )

    def forward(self, x):
        # x: [B, M, R, C]
        B, M, R, C = x.shape
        # per residual type
        x = self.per_residual(x)           # [B, M, R, H]
        x = torch.flatten(x, start_dim=2)  # [B, M, R*H]
        logits = self.combine(x)           # [B, M, num_classes]
        return logits
    
    @torch.no_grad()
    def predict_proba(self, x):
        logits = self.forward(x)                                    # [B, M, num_classes]
        return torch.softmax(logits, dim=-1).mean(dim=1)            # [B, num_classes]
    
    def predict(self, x):
        return self.predict_proba(x).argmax(dim=-1)
    
    @staticmethod
    def loss_function(output, target, **_):
        # output: logits [B, M, K], target: [B] (long)
        B, M, K = output.shape
        logits_flat = output.reshape(B*M, K)
        target_flat = target.repeat_interleave(M)
        return F.cross_entropy(logits_flat, target_flat)
    
    def accuracy(self, logits, target, **_):
        probs = torch.softmax(logits, dim=-1).mean(dim=1)  # [B,K]
        pred = probs.argmax(dim=-1)
        return (pred == target).float().mean()


# --------------------
# DNN baseline classifier for AE residuals only
# Input shape: [B, C]
# --------------------
@register_model("dnn_ae")
class AEDNN(nn.Module):
    def __init__(self, channels: int, num_classes: int = 4, hidden=(128, 64, 32, 16)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, hidden[0]), nn.ReLU(True),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(True),
            nn.Linear(hidden[1], hidden[2]), nn.ReLU(True),
            nn.Linear(hidden[2], hidden[3]), nn.ReLU(True),
            nn.Linear(hidden[3], num_classes)
        )
    def forward(self, x):
        return self.net(x)
    
    @torch.no_grad()
    def predict_proba(self, x):
        logits = self.forward(x)                        # [B, num_classes]
        return torch.softmax(logits, dim=-1)           # [B, num_classes]
    
    def predict(self, x):
        return self.predict_proba(x).argmax(dim=-1)

    @staticmethod
    def loss_function(output, target, **extra):
        # output: logits [B, num_classes]; target: long[B]
        return nn.functional.cross_entropy(output, target)
    
    def accuracy(self, logits, target, **_):
        probs = torch.softmax(logits, dim=-1)  # [B,K]
        pred = probs.argmax(dim=-1)
        return (pred == target).float().mean()
