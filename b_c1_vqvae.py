"""
# ============================================================
# B-C1: VQ-VAE Training and Codebook Analysis
# ============================================================
# USAGE (Google Colab):
#   !pip install transformers peft accelerate datasets torchvision matplotlib scikit-learn tqdm
#   !python b_c1_vqvae.py --mode train
#   !python b_c1_vqvae.py --mode ablation
#   !python b_c1_vqvae.py --mode analysis
# ============================================================

PSEUDOCODE:
    1. Define VQ-VAE: Encoder (Conv->GroupNorm->ReLU x3), VectorQuantiser (K=256, d=64),
       Decoder (ConvTranspose x2 + Conv + Sigmoid)
    2. Verify ~187K params
    3. Train with L = recon + codebook + beta*commitment for 80 epochs
    4. Support gradient-descent and EMA codebook updates + dead-code restart
    5. Run 4 ablations (Table 2)
    6. Codebook analysis: quantisation gap, usage histogram, cosine similarity heatmap
    7. Token map visualisation
    8. Save best checkpoint as weights/vqvae_best.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from b_c0_data_and_models import (
    set_seed, SEED, generate_dataset, CLASSES, SyntheticImageDataset
)


def get_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="B-C1: VQ-VAE")
    p.add_argument("--mode", choices=["train", "ablation", "analysis"], default="train")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--K", type=int, default=256, help="Codebook size")
    p.add_argument("--d", type=int, default=64, help="Codebook dim")
    p.add_argument("--beta", type=float, default=0.25, help="Commitment loss weight")
    p.add_argument("--use_ema", action="store_true", help="Use EMA codebook update")
    p.add_argument("--ema_gamma", type=float, default=0.99)
    p.add_argument("--dead_threshold", type=int, default=2, help="Dead-code restart threshold")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_path", type=str, default="weights/vqvae_best.pt")
    return p.parse_args()


# ============================================================
# === B-C1.1 === VQ-VAE Architecture
# ============================================================

class Encoder(nn.Module):
    """
    B-C1.1-1: Encoder: Conv(3->32,k=4,s=2) -> Conv(32->64,k=4,s=2) -> Conv(64->64,k=3,s=1).
    Each with GroupNorm + ReLU. Output: z_e in R^{4x4x64}.
    """
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, d, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, d), nn.ReLU(),
        )

    def forward(self, x):
        """Encode image to latent. x: (B,3,16,16) -> (B,64,4,4)."""
        return self.net(x)


class VectorQuantiser(nn.Module):
    """
    B-C1.1-1: Vector quantiser with codebook C in R^{K x d}.
    Supports gradient-descent and EMA updates. Dead-code restart.
    Straight-through estimator for backprop.
    """
    def __init__(self, K=256, d=64, beta=0.25, use_ema=False, ema_gamma=0.99,
                 dead_threshold=2):
        super().__init__()
        self.K = K
        self.d = d
        self.beta = beta
        self.use_ema = use_ema
        self.ema_gamma = ema_gamma
        self.dead_threshold = dead_threshold

        # B-C1.1-1: Init Uniform(-1/K, 1/K)
        self.codebook = nn.Embedding(K, d)
        nn.init.uniform_(self.codebook.weight, -1.0 / K, 1.0 / K)

        if use_ema:
            # EMA tracking buffers
            self.register_buffer('ema_count', torch.zeros(K))
            self.register_buffer('ema_weight', self.codebook.weight.clone())
            self.register_buffer('usage_count', torch.zeros(K))
        else:
            self.register_buffer('usage_count', torch.zeros(K))

    def forward(self, z_e):
        """
        Quantise encoder output z_e.

        Args:
            z_e: (B, d, H, W) encoder output

        Returns:
            z_q: (B, d, H, W) quantised, with straight-through gradient
            indices: (B, H*W) codebook indices
            codebook_loss: scalar
            commitment_loss: scalar
        """
        B, d, H, W = z_e.shape
        # Reshape to (B*H*W, d)
        z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, d)

        # Nearest neighbour lookup
        # dist = ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e^T
        dist = (z_flat ** 2).sum(dim=1, keepdim=True) \
             + (self.codebook.weight ** 2).sum(dim=1) \
             - 2 * z_flat @ self.codebook.weight.t()
        indices = dist.argmin(dim=1)  # (B*H*W,)

        z_q_flat = self.codebook(indices)  # (B*H*W, d)

        # B-C1.1-1: Stop-gradient placement is critical
        # Codebook loss: sg(z_e) as target
        codebook_loss = F.mse_loss(z_flat.detach(), z_q_flat)
        # Commitment loss: sg(z_q) as target
        commitment_loss = F.mse_loss(z_flat, z_q_flat.detach())

        # Straight-through estimator: copy gradient from z_q to z_e
        z_q_st = z_flat + (z_q_flat - z_flat).detach()
        z_q = z_q_st.reshape(B, H, W, d).permute(0, 3, 1, 2)

        # Track usage for dead-code restart
        with torch.no_grad():
            onehot = F.one_hot(indices, self.K).float()
            self.usage_count += onehot.sum(dim=0)

        return z_q, indices.reshape(B, H * W), codebook_loss, commitment_loss

    @torch.no_grad()
    def ema_update(self, z_e, indices):
        """
        B-C1.1-1: EMA codebook update (run after backward, not via loss).

        N_k <- gamma*N_k + (1-gamma)*n_k
        m_k <- gamma*m_k + (1-gamma)*sum(z_e for assigned)
        e_k <- m_k / N_k
        """
        if not self.use_ema:
            return

        B, d, H, W = z_e.shape
        z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, d)
        flat_idx = indices.reshape(-1)
        onehot = F.one_hot(flat_idx, self.K).float()  # (N, K)

        n_k = onehot.sum(dim=0)  # (K,)
        sum_z = onehot.t() @ z_flat  # (K, d)

        self.ema_count = self.ema_gamma * self.ema_count + (1 - self.ema_gamma) * n_k
        self.ema_weight = self.ema_gamma * self.ema_weight + (1 - self.ema_gamma) * sum_z

        # Laplace smoothing to avoid division by zero
        count = self.ema_count.clamp(min=1e-5)
        self.codebook.weight.data = self.ema_weight / count.unsqueeze(1)

    @torch.no_grad()
    def restart_dead_codes(self, z_e):
        """
        B-C1.1-1: Dead-code restart — replace unused codes with random encoder outputs.
        """
        dead_mask = self.usage_count < self.dead_threshold
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, self.d)
        rand_idx = torch.randint(0, z_flat.shape[0], (n_dead,), device=z_e.device)
        self.codebook.weight.data[dead_mask] = z_flat[rand_idx]

        if self.use_ema:
            self.ema_weight[dead_mask] = z_flat[rand_idx]
            self.ema_count[dead_mask] = 1.0

        return n_dead

    def reset_usage(self):
        """Reset usage counts (call at start of each epoch)."""
        self.usage_count.zero_()


class Decoder(nn.Module):
    """
    B-C1.1-1: Decoder: ConvTranspose2d(64->64) -> ConvTranspose2d(64->32) ->
    Conv2d(32->3) + Sigmoid.
    """
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(d, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z_q):
        """Decode quantised latent. z_q: (B,64,4,4) -> (B,3,16,16)."""
        return self.net(z_q)


class VQVAE(nn.Module):
    """
    B-C1.1: Full VQ-VAE model combining encoder, quantiser, decoder.
    """
    def __init__(self, K=256, d=64, beta=0.25, use_ema=False, ema_gamma=0.99,
                 dead_threshold=2):
        super().__init__()
        self.encoder = Encoder(d)
        self.quantiser = VectorQuantiser(K, d, beta, use_ema, ema_gamma, dead_threshold)
        self.decoder = Decoder(d)
        self.beta = beta

    def forward(self, x):
        """
        Full forward pass.

        Returns:
            x_hat: reconstruction
            total_loss: recon + codebook + beta*commitment
            recon_loss, cb_loss, commit_loss: individual components
            indices: codebook indices
        """
        z_e = self.encoder(x)
        z_q, indices, cb_loss, commit_loss = self.quantiser(z_e)
        x_hat = self.decoder(z_q)

        recon_loss = F.mse_loss(x_hat, x)
        total_loss = recon_loss + cb_loss + self.beta * commit_loss

        return x_hat, total_loss, recon_loss, cb_loss, commit_loss, indices, z_e

    def encode_indices(self, x):
        """Encode image to codebook indices only (for tokenisation)."""
        z_e = self.encoder(x)
        _, indices, _, _ = self.quantiser(z_e)
        return indices  # (B, 16)

    def decode_indices(self, indices):
        """Decode from codebook indices to images."""
        z_q = self.quantiser.codebook(indices)  # (B, 16, d)
        B = z_q.shape[0]
        z_q = z_q.reshape(B, 4, 4, -1).permute(0, 3, 1, 2)  # (B, d, 4, 4)
        return self.decoder(z_q)


def count_params(model):
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


# ============================================================
# === B-C1.1 === Codebook perplexity
# ============================================================

def compute_perplexity(indices, K):
    """
    B-C1.1: Compute codebook perplexity = exp(-sum(p_k * log(p_k))).

    Args:
        indices: tensor of codebook indices
        K: codebook size

    Returns:
        perplexity: float
    """
    flat = indices.reshape(-1)
    counts = torch.bincount(flat, minlength=K).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -(probs * probs.log()).sum()
    return entropy.exp().item()


# ============================================================
# === B-C1.2 === Training
# ============================================================

def train_vqvae(model, train_loader, val_loader, args):
    """
    B-C1.2-2: Train VQ-VAE for 80 epochs. Log recon MSE, codebook loss,
    perplexity, dead codes.

    Args:
        model: VQVAE instance
        train_loader: training DataLoader
        val_loader: validation DataLoader
        args: argparse namespace

    Returns:
        model: trained model
        history: dict of training metrics
    """
    device = args.device
    model = model.to(device)

    # If EMA, don't include codebook in optimizer
    if args.use_ema:
        params = [p for n, p in model.named_parameters() if 'codebook' not in n]
    else:
        params = model.parameters()

    optimizer = torch.optim.Adam(params, lr=args.lr)
    history = {'recon': [], 'cb': [], 'commit': [], 'perp': [], 'dead': []}
    best_loss = float('inf')

    for epoch in tqdm(range(args.epochs), desc="VQ-VAE Training"):
        model.train()
        model.quantiser.reset_usage()
        ep_recon, ep_cb, ep_commit, n_batches = 0, 0, 0, 0

        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            x_hat, loss, recon, cb, commit, indices, z_e = model(imgs)

            # NaN check
            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # B-C1.1-1: EMA update after backward (not via loss)
            if args.use_ema:
                model.quantiser.ema_update(z_e.detach(), indices)

            ep_recon += recon.item()
            ep_cb += cb.item()
            ep_commit += commit.item()
            n_batches += 1

        # Dead-code restart at end of epoch
        with torch.no_grad():
            sample_imgs = next(iter(train_loader))[0].to(device)
            z_e_sample = model.encoder(sample_imgs)
            n_dead = model.quantiser.restart_dead_codes(z_e_sample)

        # Validation perplexity
        model.eval()
        all_indices = []
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                idx = model.encode_indices(imgs)
                all_indices.append(idx)
        all_indices = torch.cat(all_indices)
        perp = compute_perplexity(all_indices, args.K)

        avg_recon = ep_recon / max(n_batches, 1)
        history['recon'].append(avg_recon)
        history['cb'].append(ep_cb / max(n_batches, 1))
        history['commit'].append(ep_commit / max(n_batches, 1))
        history['perp'].append(perp)
        history['dead'].append(n_dead)

        if avg_recon < best_loss:
            best_loss = avg_recon
            os.makedirs(os.path.dirname(args.save_path) or "weights", exist_ok=True)
            torch.save(model.state_dict(), args.save_path)

        if (epoch + 1) % 10 == 0:
            print(f"Ep {epoch+1}: recon={avg_recon:.5f}, perp={perp:.1f}, "
                  f"dead={n_dead}")

    print(f"Best recon MSE: {best_loss:.5f}")
    return model, history


# ============================================================
# === B-C1.3 === Codebook Analysis
# ============================================================

def codebook_analysis(model, val_loader, args):
    """
    B-C1.3: Quantisation gap, usage histogram, cosine similarity heatmap,
    token map visualisation.
    """
    device = args.device
    model = model.to(device)
    model.eval()
    os.makedirs("plots", exist_ok=True)

    # B-C1.3-4: Quantisation gap = L_post - L_pre
    total_pre, total_post, n = 0, 0, 0
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            z_e = model.encoder(imgs)
            # Pre-quantisation: decode from z_e directly
            x_pre = model.decoder(z_e)
            l_pre = F.mse_loss(x_pre, imgs)
            # Post-quantisation: decode from z_q
            z_q, indices, _, _ = model.quantiser(z_e)
            x_post = model.decoder(z_q)
            l_post = F.mse_loss(x_post, imgs)
            total_pre += l_pre.item()
            total_post += l_post.item()
            n += 1
    delta = (total_post / n) - (total_pre / n)
    print(f"B-C1.3-4: Quantisation gap Delta = {delta:.4f} (expected ~1.8)")

    # B-C1.3-5: Usage histogram
    all_idx = []
    with torch.no_grad():
        for imgs, _ in val_loader:
            idx = model.encode_indices(imgs.to(device))
            all_idx.append(idx.cpu())
    all_idx = torch.cat(all_idx).reshape(-1)
    counts = torch.bincount(all_idx, minlength=args.K)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(args.K), counts.numpy())
    ax.set_xlabel("Codebook Index")
    ax.set_ylabel("Usage Count")
    ax.set_title("B-C1.3-5: Codebook Usage Histogram")
    plt.savefig("plots/b_c1_usage_histogram.png", dpi=150)
    plt.close()

    # B-C1.3-5: Cosine similarity heatmap
    with torch.no_grad():
        cb = model.quantiser.codebook.weight  # (K, d)
        cb_norm = cb / cb.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sim = cb_norm @ cb_norm.t()  # (K, K)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(sim.cpu().numpy(), cmap='viridis', vmin=-1, vmax=1)
    ax.set_title("B-C1.3-5: Codebook Cosine Similarity")
    plt.savefig("plots/b_c1_cosine_heatmap.png", dpi=150)
    plt.close()

    # B-C1.3-6: Token map visualisation (6 val images, one per class)
    fig, axes = plt.subplots(6, 3, figsize=(9, 18))
    with torch.no_grad():
        for c in range(6):
            # Find first image of class c in val
            mask = val_loader.dataset.labels == c
            img = val_loader.dataset.images[mask][0:1].to(device)
            idx = model.encode_indices(img).cpu().reshape(4, 4)
            recon = model.decode_indices(model.encode_indices(img)).cpu()

            axes[c, 0].imshow(img[0].cpu().permute(1, 2, 0).numpy())
            axes[c, 0].set_title(f"Original ({CLASSES[c]})")
            axes[c, 0].axis('off')

            axes[c, 1].imshow(idx.numpy(), cmap='tab20')
            for y in range(4):
                for x in range(4):
                    axes[c, 1].text(x, y, str(idx[y, x].item()),
                                    ha='center', va='center', fontsize=7)
            axes[c, 1].set_title("4x4 Index Map")

            axes[c, 2].imshow(recon[0].permute(1, 2, 0).numpy())
            axes[c, 2].set_title("Reconstruction")
            axes[c, 2].axis('off')

    plt.suptitle("B-C1.3-6: Token Map Visualisation")
    plt.tight_layout()
    plt.savefig("plots/b_c1_token_maps.png", dpi=150)
    plt.close()
    print("Codebook analysis plots saved.")


# ============================================================
# === B-C1.2-3 === Ablation (Table 2)
# ============================================================

def run_ablation(train_loader, val_loader, device):
    """
    B-C1.2-3: Run Table 2 ablation configs.
    """
    configs = [
        {"name": "Baseline", "K": 256, "beta": 0.25, "use_ema": False},
        {"name": "A (K=128)", "K": 128, "beta": 0.25, "use_ema": False},
        {"name": "B (beta=1)", "K": 256, "beta": 1.00, "use_ema": False},
        {"name": "C (EMA)", "K": 256, "beta": 0.25, "use_ema": True},
    ]
    results = []
    for cfg in configs:
        print(f"\n--- Ablation: {cfg['name']} ---")
        model = VQVAE(K=cfg['K'], d=64, beta=cfg['beta'],
                       use_ema=cfg['use_ema'], ema_gamma=0.99, dead_threshold=2)

        # Create a simple args namespace for this config
        ab_args = argparse.Namespace(
            epochs=80, lr=3e-4, device=device, K=cfg['K'],
            use_ema=cfg['use_ema'], save_path=f"weights/vqvae_{cfg['name'].split()[0].lower()}.pt"
        )
        model, hist = train_vqvae(model, train_loader, val_loader, ab_args)
        results.append({
            'name': cfg['name'], 'final_recon': hist['recon'][-1],
            'final_perp': hist['perp'][-1], 'final_dead': hist['dead'][-1],
        })
        del model
        torch.cuda.empty_cache()

    print(f"\n{'Config':<15} {'Recon MSE':<12} {'Perplexity':<12} {'Dead Codes':<12}")
    print("-" * 51)
    for r in results:
        print(f"{r['name']:<15} {r['final_recon']:<12.5f} {r['final_perp']:<12.1f} "
              f"{r['final_dead']:<12}")
    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = get_args()
    set_seed(args.seed)

    # Load data
    train_imgs, train_lbls, val_imgs, val_lbls = generate_dataset(seed=args.seed)
    train_ds = SyntheticImageDataset(train_imgs, train_lbls)
    val_ds = SyntheticImageDataset(val_imgs, val_lbls)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.mode == "train":
        # === B-C1.1 === Build and verify model
        model = VQVAE(K=args.K, d=args.d, beta=args.beta,
                       use_ema=args.use_ema, ema_gamma=args.ema_gamma,
                       dead_threshold=args.dead_threshold)
        n_params = count_params(model)
        print(f"B-C1.1: VQ-VAE params = {n_params:,} (expected ~187K)")

        # === B-C1.2 === Train
        model, history = train_vqvae(model, train_loader, val_loader, args)

        # === B-C1.3 === Analysis
        model.load_state_dict(torch.load(args.save_path, map_location="cpu"))
        model = model.to(args.device)
        codebook_analysis(model, val_loader, args)

        # B-C1.3-7: Save best (already saved during training)
        print(f"Best checkpoint saved at {args.save_path}")

    elif args.mode == "ablation":
        # === B-C1.2-3 === Table 2 ablation
        run_ablation(train_loader, val_loader, args.device)

    elif args.mode == "analysis":
        # Load saved model and run analysis only
        model = VQVAE(K=args.K, d=args.d, beta=args.beta)
        model.load_state_dict(torch.load(args.save_path, map_location="cpu"))
        model = model.to(args.device)
        codebook_analysis(model, val_loader, args)

    print("\nB-C1 COMPLETE")
