import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "src")

from dataset import BridgeActionDataset
from model import ActionAutoEncoder


ROOT = r"C:\Users\nicop\AntVLA-v8\data\bridge_tfds\bridge_dataset\1.0.0"

SEQ_LEN = 32
STRIDE = 16

# Noneなら、存在するTFRecordから最後まで読む
# まずは実験用に例えば1000などでもよい
MAX_TRAJECTORIES = 1000

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

LATENT_DIMS = [8, 16, 32, 64, 128]


def train(latent_dim):

    dataset = BridgeActionDataset(
        ROOT,
        seq_len=SEQ_LEN,
        stride=STRIDE,
        max_trajectories=MAX_TRAJECTORIES,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    model = ActionAutoEncoder(
        action_dim=7,
        hidden_dim=256,
        latent_dim=latent_dim,
        num_layers=2,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    criterion = nn.MSELoss()

    min_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0
        num_batches = 0
        num_samples = 0

        for actions in loader:

            optimizer.zero_grad()

            reconstructed, z = model(actions)

            loss = criterion(
                reconstructed,
                actions,
            )

            loss.backward()
            optimizer.step()

            batch_size = actions.size(0)

            total_loss += loss.item() * batch_size
            num_samples += batch_size
            num_batches += 1

        if num_samples == 0:
            raise RuntimeError(
                "No valid action windows were found."
            )

        avg_loss = total_loss / num_samples

        min_loss = min(
            min_loss,
            avg_loss,
        )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"latent={latent_dim:3d} | "
                f"epoch={epoch:4d} | "
                f"loss={avg_loss:.8f} | "
                f"windows={num_samples}"
            )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    return (
        avg_loss,
        min_loss,
        params,
    )


def main():

    print("=" * 60)
    print("BridgeData V2 Action AutoEncoder")
    print("Streaming bottleneck dimension comparison")
    print("=" * 60)

    results = []

    for latent_dim in LATENT_DIMS:

        print()
        print(
            f">>> latent_dim = {latent_dim}"
        )

        final_loss, min_loss, params = train(
            latent_dim
        )

        results.append(
            (
                latent_dim,
                final_loss,
                min_loss,
                params,
            )
        )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"{'latent':>8} "
        f"{'final_loss':>14} "
        f"{'min_loss':>14} "
        f"{'params':>12}"
    )

    for (
        latent,
        final_loss,
        min_loss,
        params,
    ) in results:

        print(
            f"{latent:8d} "
            f"{final_loss:14.8f} "
            f"{min_loss:14.8f} "
            f"{params:12d}"
        )


if __name__ == "__main__":
    main()
