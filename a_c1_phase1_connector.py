"""
# ============================================================
# A-C1: Phase 1 — Connector Initialisation
# ============================================================
# USAGE (Google Colab):
#   !pip install transformers peft accelerate datasets torchvision matplotlib scikit-learn
#   !python a_c0_data_and_models.py   # run data setup first
#   !python a_c1_phase1_connector.py  # then run this
# ============================================================

PSEUDOCODE:
    1. Load data and models from a_c0
    2. Define MLP Connector: Linear(768,960) -> GELU -> Linear(960,960) with Kaiming init
    3. Verify ~1.66M trainable params
    4. For each batch:
        a. Get BOS embedding from LM
        b. Pass CLIP features through connector to get visual tokens V (49 tokens)
        c. Get caption token embeddings from LM
        d. Concatenate: [BOS emb, V1:49, caption embs] -> inputs_embeds
        e. Build labels: [-100] * (1 + 49) + caption_ids
        f. Forward through LM with inputs_embeds, compute loss
        g. Backprop through connector only (LM frozen)
    5. Train 1 epoch, batch 32, Adam lr=3e-4
    6. Measure r_norm = mean(||V_i||) / mean(||T_j||), rescale if outside [0.3, 3]
    7. Generate 5 greedy captions from held-out images
    8. Save connector weights to weights/connector_phaseA1.pt
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from a_c0_data_and_models import (
    set_seed, SEED, CLIP_MODEL_NAME, LM_MODEL_NAME,
    get_cifar10_subsets, build_caption_dataset, load_clip_model,
    load_lm_model, load_alpaca_data, extract_clip_features,
    preprocess_images_for_clip, CaptionDataset, compute_ppl,
    CLIPImageProcessor, CIFAR10_CLASSES
)


# ============================================================
# === A-C1 === MLP Connector Definition
# ============================================================

class MLPConnector(nn.Module):
    """
    MLP connector that maps CLIP patch features to LM embedding space.

    Architecture: Linear(768, 960) -> GELU -> Linear(960, 960)
    Uses Kaiming initialisation for both linear layers.

    A-C1: This is the connector (Theta_C) trained in Phase 1.
    """

    def __init__(self, clip_dim=768, lm_dim=960):
        """
        Args:
            clip_dim: CLIP hidden dimension (768 for ViT-B/32)
            lm_dim: LM hidden dimension (960 for SmolLM2-360M)
        """
        super().__init__()
        self.fc1 = nn.Linear(clip_dim, lm_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(lm_dim, lm_dim)

        # Kaiming initialisation
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        Forward pass through the connector.

        Args:
            x: CLIP patch features, shape (batch, 49, 768)

        Returns:
            Visual tokens in LM space, shape (batch, 49, 960)
        """
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


def count_params(model):
    """Count trainable and total parameters in a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ============================================================
# === A-C1-1 === Build caption sequence with inputs_embeds
# ============================================================

def build_caption_batch(connector, clip_feats, caption_ids, caption_mask,
                        lm_model, device):
    """
    A-C1-1: Build input sequence [BOS emb, V1:49, caption embs] via inputs_embeds.
    Labels: [-100] * (1 + 49) + caption IDs.

    Args:
        connector: MLPConnector
        clip_feats: (B, 49, 768) CLIP patch features
        caption_ids: (B, L) tokenized caption IDs
        caption_mask: (B, L) attention mask for captions
        lm_model: SmolLM2 model
        device: torch device

    Returns:
        inputs_embeds: (B, 1+49+L, 960) concatenated embeddings
        labels: (B, 1+49+L) with -100 for non-caption positions
        attention_mask: (B, 1+49+L) attention mask
    """
    B = clip_feats.shape[0]
    embed_layer = lm_model.get_input_embeddings()

    # 1. BOS embedding
    bos_id = lm_model.config.bos_token_id
    if bos_id is None:
        bos_id = 0  # fallback
    bos_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    bos_emb = embed_layer(bos_ids)  # (B, 1, 960)

    # 2. Visual tokens through connector
    visual_tokens = connector(clip_feats.to(device))  # (B, 49, 960)
    # Cast to same dtype as LM embeddings (float16)
    visual_tokens = visual_tokens.to(bos_emb.dtype)

    # 3. Caption embeddings
    caption_emb = embed_layer(caption_ids.to(device))  # (B, L, 960)

    # 4. Concatenate: [BOS, V1:49, caption]
    inputs_embeds = torch.cat([bos_emb, visual_tokens, caption_emb], dim=1)

    # 5. Labels: [-100] * (1 + 49) + caption_ids
    # -100 means "ignore this position in loss computation"
    ignore_labels = torch.full((B, 1 + 49), -100, dtype=torch.long, device=device)
    # For caption labels, mask out padding positions too
    caption_labels = caption_ids.clone().to(device)
    caption_labels[caption_mask == 0] = -100  # ignore padding
    labels = torch.cat([ignore_labels, caption_labels], dim=1)

    # 6. Attention mask: 1 for all real tokens (BOS + visual + caption non-pad)
    bos_mask = torch.ones(B, 1, dtype=torch.long, device=device)
    visual_mask = torch.ones(B, 49, dtype=torch.long, device=device)
    attention_mask = torch.cat([bos_mask, visual_mask, caption_mask.to(device)], dim=1)

    return inputs_embeds, labels, attention_mask


# ============================================================
# === A-C1-3 === Norm ratio measurement
# ============================================================

def compute_norm_ratio(connector, clip_features, lm_model, tokenizer, device,
                       n_samples=200):
    """
    A-C1-3: Measure r_norm = E[||V_i||_2] / E[||T_j||_2].
    
    V_i = visual token norms (connector outputs)
    T_j = text embedding norms (from LM embedding layer)

    Args:
        connector: trained MLPConnector
        clip_features: (N, 49, 768) CLIP features
        lm_model: SmolLM2 model
        tokenizer: LM tokenizer
        device: torch device
        n_samples: number of samples to use

    Returns:
        r_norm: float ratio of mean visual norm to mean text norm
    """
    connector.eval()
    embed_layer = lm_model.get_input_embeddings()

    # Compute mean visual token norm
    with torch.no_grad():
        sample_feats = clip_features[:n_samples].to(device)
        visual_tokens = connector(sample_feats)  # (n, 49, 960)
        visual_norms = visual_tokens.float().norm(dim=-1)  # (n, 49)
        mean_visual_norm = visual_norms.mean().item()

    # Compute mean text embedding norm
    # Use a sample of common words
    sample_text = "The quick brown fox jumps over the lazy dog and runs across the field"
    text_ids = tokenizer(sample_text, return_tensors="pt")['input_ids'].to(device)
    with torch.no_grad():
        text_embs = embed_layer(text_ids)  # (1, L, 960)
        text_norms = text_embs.float().norm(dim=-1)  # (1, L)
        mean_text_norm = text_norms.mean().item()

    r_norm = mean_visual_norm / mean_text_norm
    return r_norm


def rescale_connector(connector, r_norm, target=1.0):
    """
    A-C1-3: Rescale connector output if r_norm is outside [0.3, 3].

    Multiplies the last layer's weight and bias by (target / r_norm).

    Args:
        connector: MLPConnector to rescale
        r_norm: current norm ratio
        target: desired norm ratio (default 1.0)
    """
    if 0.3 <= r_norm <= 3.0:
        print(f"r_norm = {r_norm:.3f} is within [0.3, 3.0], no rescaling needed.")
        return

    scale = target / r_norm
    print(f"r_norm = {r_norm:.3f} is outside [0.3, 3.0], rescaling by {scale:.3f}")
    with torch.no_grad():
        connector.fc2.weight.mul_(scale)
        connector.fc2.bias.mul_(scale)


# ============================================================
# === A-C1-4 === Greedy caption generation
# ============================================================

def generate_caption(connector, clip_feat, lm_model, tokenizer, device,
                     max_new_tokens=50):
    """
    A-C1-4: Generate a caption from visual tokens only (greedy decoding).

    Sequence starts with [BOS, V1:49] and generates text autoregressively.

    Args:
        connector: trained MLPConnector
        clip_feat: (49, 768) single image CLIP features
        lm_model: SmolLM2 model
        tokenizer: LM tokenizer
        device: torch device
        max_new_tokens: max tokens to generate

    Returns:
        caption: generated text string
    """
    connector.eval()
    lm_model.eval()
    embed_layer = lm_model.get_input_embeddings()

    with torch.no_grad():
        # BOS embedding
        bos_id = lm_model.config.bos_token_id or 0
        bos_emb = embed_layer(torch.tensor([[bos_id]], device=device))  # (1, 1, 960)

        # Visual tokens
        visual = connector(clip_feat.unsqueeze(0).to(device))  # (1, 49, 960)
        visual = visual.to(bos_emb.dtype)

        # Start with [BOS, V1:49]
        current_embeds = torch.cat([bos_emb, visual], dim=1)  # (1, 50, 960)

        generated_ids = []
        for _ in range(max_new_tokens):
            outputs = lm_model(inputs_embeds=current_embeds)
            # Get logits for the last position
            next_logits = outputs.logits[:, -1, :]  # (1, vocab)
            next_id = next_logits.argmax(dim=-1)  # greedy

            # Stop if EOS
            if next_id.item() == tokenizer.eos_token_id:
                break

            generated_ids.append(next_id.item())

            # Get embedding for the new token and append
            next_emb = embed_layer(next_id.unsqueeze(0))  # (1, 1, 960)
            current_embeds = torch.cat([current_embeds, next_emb], dim=1)

    caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return caption


# ============================================================
# === A-C1-2 === Phase 1 Training Loop
# ============================================================

def train_phase1(connector, clip_features_train, captions, lm_model, tokenizer,
                 device, batch_size=32, lr=3e-4, num_epochs=1):
    """
    A-C1-2: Train connector only for 1 epoch. CLIP and LM are frozen.

    Uses Adam optimizer with lr=3e-4, batch size 32.
    Only the connector parameters are updated.

    Args:
        connector: MLPConnector (the only trainable part)
        clip_features_train: (N, 49, 768) pre-extracted CLIP features
        captions: list of caption dicts
        lm_model: frozen SmolLM2 model
        tokenizer: LM tokenizer
        device: torch device
        batch_size: training batch size
        lr: learning rate
        num_epochs: number of training epochs

    Returns:
        connector: trained connector
        losses: list of per-step losses
    """
    # Freeze LM completely
    lm_model.eval()
    for p in lm_model.parameters():
        p.requires_grad = False

    # Only connector is trainable
    connector = connector.to(device)
    connector.train()

    # A-C1-2: Verify ~1.66M trainable params
    trainable, total = count_params(connector)
    print(f"\nConnector trainable params: {trainable:,} (expected ~1,660,800)")
    print(f"Connector total params: {total:,}")

    # Create dataset and dataloader
    dataset = CaptionDataset(clip_features_train, captions, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, drop_last=True)

    # Adam optimizer on connector params only
    optimizer = torch.optim.Adam(connector.parameters(), lr=lr)

    losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_steps = 0

        for step, (clip_feats, caption_ids, caption_mask) in enumerate(dataloader):
            # A-C1-1: Build [BOS, V1:49, caption] sequence
            inputs_embeds, labels, attn_mask = build_caption_batch(
                connector, clip_feats, caption_ids, caption_mask,
                lm_model, device
            )

            # Forward pass through frozen LM
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = lm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    labels=labels,
                )
                loss = outputs.loss

            # Backward pass (only connector gets gradients)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            epoch_loss += loss.item()
            num_steps += 1

            if step % 50 == 0:
                print(f"  Epoch {epoch+1}, Step {step}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / num_steps
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}, Time: {elapsed:.1f}s")

    return connector, losses


# ============================================================
# Main: Run Phase 1
# ============================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(SEED)
    print(f"Device: {device}")

    # --- Load data and models (from A-C0) ---
    print("\n--- Loading data and models ---")
    train_subset, test_subset = get_cifar10_subsets()
    clip_model, clip_processor = load_clip_model(device)
    lm_model, tokenizer = load_lm_model(device)

    # Preprocess and extract CLIP features
    train_pixels, train_labels = preprocess_images_for_clip(train_subset, clip_processor)
    test_pixels, test_labels = preprocess_images_for_clip(test_subset, clip_processor)
    clip_features_train = extract_clip_features(train_pixels, clip_model, device)
    clip_features_test = extract_clip_features(test_pixels, clip_model, device)

    # Build caption dataset
    captions = build_caption_dataset(train_subset)

    # === A-C1-2 === Create and train connector
    print("\n" + "="*60)
    print("A-C1: Phase 1 — Connector Initialisation")
    print("="*60)

    connector = MLPConnector(clip_dim=768, lm_dim=960)
    connector, losses = train_phase1(
        connector, clip_features_train, captions, lm_model, tokenizer,
        device, batch_size=32, lr=3e-4, num_epochs=1
    )

    # === A-C1-3 === Measure and check norm ratio
    print("\n--- Checking norm ratio ---")
    r_norm = compute_norm_ratio(connector, clip_features_train, lm_model,
                                tokenizer, device)
    print(f"r_norm = {r_norm:.3f} (target range: [0.3, 3.0])")
    rescale_connector(connector, r_norm)

    # Recompute after potential rescaling
    r_norm_after = compute_norm_ratio(connector, clip_features_train, lm_model,
                                      tokenizer, device)
    print(f"r_norm after rescaling: {r_norm_after:.3f}")

    # === A-C1-4 === Generate 5 captions from held-out test images
    print("\n--- Generating 5 sample captions from test images ---")
    for i in range(5):
        clip_feat = clip_features_test[i]  # (49, 768)
        true_label = test_labels[i].item()
        true_class = CIFAR10_CLASSES[true_label]
        caption = generate_caption(connector, clip_feat, lm_model, tokenizer, device)
        print(f"  Image {i} (true: {true_class}): {caption}")

    # === A-C1-5 === Save connector weights
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/connector_phaseA1.pt"
    torch.save(connector.state_dict(), save_path)
    print(f"\nConnector weights saved to {save_path}")

    # Also compute PPL to have a baseline comparison
    alpaca_texts = load_alpaca_data()
    ppl_after = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
    print(f"PPL after Phase 1 (LM was frozen, should be same): {ppl_after:.2f}")

    print("\n" + "="*60)
    print("A-C1 Phase 1 COMPLETE")
    print("="*60)
