"""
# ============================================================
# B-C2: Vocabulary Expansion + Projector Warm-Up
# ============================================================
# USAGE (Google Colab):
#   !python b_c1_vqvae.py --mode train    # train VQ-VAE first
#   !python b_c2_vocab_expansion.py
# ============================================================

PSEUDOCODE:
    1. Load SmolLM2, resize embeddings to V_txt + 258 (BEFORE LoRA)
    2. Create overlay nn.Embedding(258, 960) with forward hook
    3. Init <image>/<\/image> rows from mean of text embeddings
    4. Init 256 visual token rows to zero
    5. Phase 1: Train projector P: R^64 -> R^960 (Kaiming init)
       - Keep overlay frozen except special-token rows
       - Train until ||P(c_k)|| converges relative to text norms
    6. Phase 2: Embedding transplant
       - e_k <- P(c_k) for all k
       - Mark K visual rows as requires_grad=True
       - Discard P
    7. Verify norm ratio in [0.2, 5], rescale if needed
    8. Save expanded model state
"""

import os
import argparse
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from b_c0_data_and_models import set_seed, SEED, load_lm_model, LM_MODEL_NAME
from b_c1_vqvae import VQVAE


V_TXT = 49152  # original SmolLM2 vocab size
N_VISUAL = 256  # number of visual tokens (codebook entries)
N_SPECIAL = 2   # <image>, </image>
N_NEW = N_VISUAL + N_SPECIAL  # 258


def get_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="B-C2: Vocabulary Expansion")
    p.add_argument("--vqvae_path", type=str, default="weights/vqvae_best.pt")
    p.add_argument("--projector_epochs", type=int, default=5)
    p.add_argument("--projector_lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_path", type=str, default="weights/expanded_model")
    return p.parse_args()


# ============================================================
# === B-C2.1 === Overlay Embedding
# ============================================================

class OverlayEmbedding(nn.Module):
    """
    B-C2.1-1: Overlay embedding that swaps in new rows for token IDs >= V_txt.

    Keeps original wte frozen. Only the 258 overlay rows are trainable.
    Uses a forward hook on the base embedding layer.

    Layout:
        overlay index 0   -> <image>    (ID = V_txt)
        overlay index 1   -> </image>   (ID = V_txt + 1)
        overlay index 2-257 -> visual tokens v_0..v_255 (ID = V_txt + 2 .. V_txt + 257)
    """
    def __init__(self, n_new=N_NEW, d_lm=960):
        """
        Args:
            n_new: number of new tokens (258)
            d_lm: LM hidden dimension (960)
        """
        super().__init__()
        self.n_new = n_new
        self.overlay = nn.Embedding(n_new, d_lm)
        # B-C2.1-2: Visual rows init to zero (will be populated by projector)
        nn.init.zeros_(self.overlay.weight)

    def init_special_from_mean(self, base_embed_weight):
        """
        B-C2.1-2: Initialise <image>/<\/image> rows from mean of text embeddings.

        Args:
            base_embed_weight: (V_txt, d_lm) original embedding weight
        """
        mean_emb = base_embed_weight[:V_TXT].mean(dim=0)
        with torch.no_grad():
            self.overlay.weight[0] = mean_emb.clone()  # <image>
            self.overlay.weight[1] = mean_emb.clone()  # </image>
        print(f"B-C2.1-2: Special tokens init from mean, norm={mean_emb.norm():.4f}")

    def get_hook(self):
        """
        Return a forward hook function for the base embedding layer.

        The hook replaces embeddings for IDs >= V_txt with overlay embeddings.
        """
        overlay = self.overlay

        def hook_fn(module, args, output):
            input_ids = args[0]
            mask = input_ids >= V_TXT
            if mask.any():
                overlay_ids = input_ids[mask] - V_TXT
                # Clamp to valid range
                overlay_ids = overlay_ids.clamp(0, overlay.num_embeddings - 1)
                output = output.clone()
                output[mask] = overlay(overlay_ids).to(output.dtype)
            return output

        return hook_fn


def setup_expanded_model(model, tokenizer, overlay, device):
    """
    B-C2.1-1: Resize token embeddings and register overlay hook.

    IMPORTANT: resize_token_embeddings BEFORE get_peft_model.

    Args:
        model: base LM model
        tokenizer: tokenizer
        overlay: OverlayEmbedding instance
        device: torch device

    Returns:
        model: model with resized embeddings + hook
        hook_handle: hook handle (keep reference to prevent GC)
    """
    # Add special tokens to tokenizer
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['<image>', '</image>']
    })

    # B-C2.1-1: resize BEFORE LoRA
    model.resize_token_embeddings(V_TXT + N_NEW)

    # Freeze original wte
    model.get_input_embeddings().weight.requires_grad_(False)

    # Init special tokens in overlay from mean of text embeddings
    base_weight = model.get_input_embeddings().weight.data
    overlay.init_special_from_mean(base_weight.float())
    overlay = overlay.to(device)

    # Register hook
    hook_handle = model.get_input_embeddings().register_forward_hook(overlay.get_hook())

    print(f"B-C2.1-1: Embeddings resized to {V_TXT + N_NEW}, hook registered")
    return model, hook_handle


# ============================================================
# === B-C2.2 === Two-Phase Projector Warm-Up
# ============================================================

class Projector(nn.Module):
    """
    B-C2.2-4: Projector P: R^64 -> R^960 with Kaiming init.

    Maps VQ-VAE codebook vectors to LM embedding space.
    """
    def __init__(self, d_vq=64, d_lm=960):
        """
        Args:
            d_vq: VQ-VAE codebook dim (64)
            d_lm: LM hidden dim (960)
        """
        super().__init__()
        self.linear = nn.Linear(d_vq, d_lm)
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        """Project codebook vectors to LM space."""
        return self.linear(x)


def projector_warmup(projector, codebook_weight, model, overlay, device,
                     epochs=5, lr=1e-3):
    """
    B-C2.2-4: Phase 1 — Projector pre-training.

    Train P so ||P(c_k)|| converges relative to text embedding norms.
    Keep overlay frozen except <image>/<\/image>.

    Args:
        projector: Projector module
        codebook_weight: (K, 64) frozen codebook vectors
        model: LM model (for getting text embedding norms)
        overlay: OverlayEmbedding (frozen during this phase)
        device: torch device
        epochs: number of warmup epochs
        lr: learning rate

    Returns:
        projector: trained projector
    """
    projector = projector.to(device)
    codebook = codebook_weight.to(device).float()

    # Freeze overlay except special tokens
    for p in overlay.parameters():
        p.requires_grad = False
    # Only train projector + special-token rows
    overlay.overlay.weight.requires_grad = False
    # Make special tokens (indices 0, 1) trainable via separate param
    special_params = nn.Parameter(overlay.overlay.weight[:2].clone().to(device))

    optimizer = torch.optim.Adam(list(projector.parameters()) + [special_params], lr=lr)

    # Target: mean text embedding norm
    with torch.no_grad():
        text_norms = model.get_input_embeddings().weight[:V_TXT].float().norm(dim=1)
        target_norm = text_norms.mean().item()
    print(f"B-C2.2-4: Target text embedding norm = {target_norm:.4f}")

    for epoch in tqdm(range(epochs), desc="Projector warmup"):
        projected = projector(codebook)  # (K, 960)
        proj_norms = projected.norm(dim=1)  # (K,)
        # Loss: make projected norms match text norms
        norm_loss = (proj_norms.mean() - target_norm) ** 2
        # Also add a diversity loss so projections don't collapse
        proj_normed = projected / proj_norms.unsqueeze(1).clamp(min=1e-8)
        sim = proj_normed @ proj_normed.t()
        # Penalise high off-diagonal similarity
        eye = torch.eye(len(codebook), device=device)
        diversity_loss = ((sim - eye) ** 2).mean()

        loss = norm_loss + 0.1 * diversity_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Write back special tokens
        with torch.no_grad():
            overlay.overlay.weight[:2] = special_params.data.to(overlay.overlay.weight.dtype)

        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  Ep {epoch+1}: norm_loss={norm_loss.item():.6f}, "
                  f"mean_proj_norm={proj_norms.mean().item():.4f}")

    return projector


def embedding_transplant(projector, codebook_weight, overlay, device):
    """
    B-C2.2-5: Phase 2 — Embedding transplant.

    e_k <- P(c_k) for k=0..K-1. Mark visual rows as trainable. Discard P.

    Args:
        projector: trained Projector
        codebook_weight: (K, 64) codebook vectors
        overlay: OverlayEmbedding
        device: torch device
    """
    projector.eval()
    with torch.no_grad():
        codebook = codebook_weight.to(device).float()
        projected = projector(codebook)  # (K, 960)
        # Write into overlay rows 2..257 (visual tokens)
        overlay.overlay.weight[2:2 + N_VISUAL] = projected.to(overlay.overlay.weight.dtype)

    # Mark all overlay rows as trainable
    overlay.overlay.weight.requires_grad_(True)

    print(f"B-C2.2-5: Transplanted {N_VISUAL} visual embeddings into overlay")
    print(f"  Visual embedding mean norm: {projected.norm(dim=1).mean().item():.4f}")


def verify_norm_ratio(overlay, model, device):
    """
    B-C2.1-3 / B-C2.2-6: Verify norm ratio (visual/text) in [0.2, 5].
    Rescale if needed.

    Args:
        overlay: OverlayEmbedding
        model: LM model
        device: torch device

    Returns:
        ratio: float norm ratio
    """
    with torch.no_grad():
        vis_norms = overlay.overlay.weight[2:].float().norm(dim=1).mean().item()
        txt_norms = model.get_input_embeddings().weight[:V_TXT].float().norm(dim=1).mean().item()

    ratio = vis_norms / max(txt_norms, 1e-8)
    print(f"B-C2.2-6: Norm ratio (visual/text) = {ratio:.4f}")

    if not (0.2 <= ratio <= 5.0):
        target = 1.0
        scale = target * txt_norms / max(vis_norms, 1e-8)
        with torch.no_grad():
            overlay.overlay.weight[2:] *= scale
        print(f"  Rescaled by {scale:.4f}")
        ratio = target

    return ratio


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = get_args()
    set_seed(args.seed)
    device = args.device

    print("=" * 60)
    print("B-C2: Vocabulary Expansion")
    print("=" * 60)

    # Load VQ-VAE to get codebook vectors, then move to CPU
    vqvae = VQVAE(K=256, d=64)
    vqvae.load_state_dict(torch.load(args.vqvae_path, map_location="cpu"))
    codebook_weight = vqvae.quantiser.codebook.weight.data.clone()
    del vqvae
    torch.cuda.empty_cache()
    print(f"Loaded codebook: {codebook_weight.shape}")

    # Load LM
    lm_model, tokenizer = load_lm_model(device)

    # === B-C2.1 === Setup overlay + resize
    overlay = OverlayEmbedding(n_new=N_NEW, d_lm=960)
    lm_model, hook_handle = setup_expanded_model(lm_model, tokenizer, overlay, device)

    # === B-C2.2-4 === Phase 1: Projector pre-training
    print("\n--- Phase 1: Projector Pre-training ---")
    projector = Projector(d_vq=64, d_lm=960)
    projector = projector_warmup(
        projector, codebook_weight, lm_model, overlay, device,
        epochs=args.projector_epochs, lr=args.projector_lr
    )

    # === B-C2.2-5 === Phase 2: Embedding transplant
    print("\n--- Phase 2: Embedding Transplant ---")
    embedding_transplant(projector, codebook_weight, overlay, device)
    del projector  # discard P
    torch.cuda.empty_cache()

    # === B-C2.2-6 === Verify norm ratio
    verify_norm_ratio(overlay, lm_model, device)

    # Save expanded state
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(overlay.state_dict(), os.path.join(args.save_path, "overlay.pt"))
    tokenizer.save_pretrained(args.save_path)
    print(f"\nSaved overlay + tokenizer to {args.save_path}")

    # Clean up
    del lm_model
    torch.cuda.empty_cache()
    print("\nB-C2 COMPLETE")
