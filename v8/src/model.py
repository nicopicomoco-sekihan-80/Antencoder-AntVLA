import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    def __init__(
        self,
        action_dim,
        hidden_dim=256,
        latent_dim=64,
        num_layers=2,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        self.to_latent = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        # x: [B, T, D]
        _, h = self.gru(x)

        # Last GRU layer
        h = h[-1]

        z = self.to_latent(h)

        return z


class ActionDecoder(nn.Module):
    def __init__(
        self,
        action_dim,
        hidden_dim=256,
        latent_dim=64,
        num_layers=2,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.from_latent = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
        )

        self.gru = nn.GRU(
            input_size=action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, z, seq_len):
        B = z.size(0)

        h = self.from_latent(z)

        h = h.unsqueeze(0).repeat(
            self.num_layers,
            1,
            1,
        )

        decoder_input = torch.zeros(
            B,
            seq_len,
            self.gru.input_size,
            device=z.device,
            dtype=z.dtype,
        )

        out, _ = self.gru(
            decoder_input,
            h,
        )

        return self.output(out)


class ActionAutoEncoder(nn.Module):
    def __init__(
        self,
        action_dim,
        hidden_dim=256,
        latent_dim=64,
        num_layers=2,
    ):
        super().__init__()

        self.encoder = ActionEncoder(
            action_dim,
            hidden_dim,
            latent_dim,
            num_layers,
        )

        self.decoder = ActionDecoder(
            action_dim,
            hidden_dim,
            latent_dim,
            num_layers,
        )

    def forward(self, actions):
        z = self.encoder(actions)

        reconstructed = self.decoder(
            z,
            actions.size(1),
        )

        return reconstructed, z
