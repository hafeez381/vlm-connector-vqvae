"""
# ============================================================
# A-C5: Modality Gap Analysis
# ============================================================
# USAGE (Google Colab):
#   !pip install transformers peft accelerate datasets torchvision matplotlib scikit-learn umap-learn
#   # Run all previous phases first
#   !python a_c5_modality_gap.py
# ============================================================

PSEUDOCODE:
    1. Fix 200 test images and 200 test questions (seed 42)
    2. For each phase checkpoint (0, 1, 2, 3):
       a. Load connector (and LoRA if applicable)
       b. Get visual embeddings V_i = connector(CLIP(image_i)) for 200 images
       c. Get text embeddings T_j = LM_embed(question_j) for 200 questions
       d. Normalise to unit sphere
       e. Compute MG = ||mean(V_norm) - mean(T_norm)||
       f. Compute within-modal and cross-modal cosine similarities
    3. Plot MG across phases, identify which phase has largest effect
    4. UMAP visualisation after each phase
    5. Re-run Phase 2 with L_norm = (E[||V_i||] - E[||T_j||])^2 (weight 0.01)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from a_c0_data_and_models import (
    set_seed, SEED, get_cifar10_subsets, build_vqa_dataset,
    load_clip_model, load_lm_model, load_alpaca_data,
    extract_clip_features, preprocess_images_for_clip, compute_ppl,
    VQADataset, AlpacaDataset, CLIPImageProcessor, CLIP_MODEL_NAME
)
from a_c1_phase1_connector import MLPConnector
from a_c2_phase2_sft_replay import (
    apply_lora, build_vqa_batch, evaluate_vqa_accuracy,
    compute_alpaca_loss
)


# ============================================================
# === A-C5-1 === Compute modality gap and cosine similarities
# ============================================================

def get_visual_embeddings(connector, clip_features, device, n_samples=200):
    """
    Get visual embeddings from connector output for n_samples images.

    Args:
        connector: MLPConnector
        clip_features: (N, 49, 768) CLIP features
        device: torch device
        n_samples: number of images to use

    Returns:
        visual_embs: (n_samples, 960) mean-pooled visual embeddings
    """
    connector.eval()
    with torch.no_grad():
        feats = clip_features[:n_samples].to(device)
        visual_tokens = connector(feats)  # (n, 49, 960)
        # Mean pool over the 49 tokens to get one vector per image
        visual_embs = visual_tokens.float().mean(dim=1)  # (n, 960)
    return visual_embs


def get_text_embeddings(lm_model, tokenizer, questions, device, n_samples=200):
    """
    Get text embeddings for questions from LM embedding layer.

    Args:
        lm_model: LM model
        tokenizer: tokenizer
        questions: list of question strings
        device: torch device
        n_samples: number to use

    Returns:
        text_embs: (n_samples, 960) mean-pooled text embeddings
    """
    if hasattr(lm_model, 'get_base_model'):
        base = lm_model.get_base_model()
    else:
        base = lm_model
    embed_layer = base.get_input_embeddings()

    text_embs_list = []
    with torch.no_grad():
        for q in questions[:n_samples]:
            tokens = tokenizer(q, return_tensors="pt")
            emb = embed_layer(tokens['input_ids'].to(device))  # (1, L, 960)
            mean_emb = emb.float().mean(dim=1)  # (1, 960)
            text_embs_list.append(mean_emb)

    text_embs = torch.cat(text_embs_list, dim=0)  # (n, 960)
    return text_embs


def compute_modality_gap(visual_embs, text_embs):
    """
    A-C5-1: Compute modality gap MG = ||mean(V_norm) - mean(T_norm)||.

    MG is computed on the unit sphere, so we L2-normalise first.
    MG is in [0, 2].

    Args:
        visual_embs: (N, D) visual embeddings
        text_embs: (M, D) text embeddings

    Returns:
        mg: float modality gap value
        within_v: float mean cosine similarity within visual
        within_t: float mean cosine similarity within text
        cross: float mean cosine similarity between visual and text
    """
    # L2-normalise to unit sphere
    v_norm = visual_embs / visual_embs.norm(dim=-1, keepdim=True)
    t_norm = text_embs / text_embs.norm(dim=-1, keepdim=True)

    # MG = ||mean(v_norm) - mean(t_norm)||
    v_mean = v_norm.mean(dim=0)
    t_mean = t_norm.mean(dim=0)
    mg = (v_mean - t_mean).norm().item()

    # Within-modal cosine similarities
    # Visual-visual
    v_sim = torch.mm(v_norm, v_norm.t())  # (N, N)
    # Only upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones_like(v_sim), diagonal=1).bool()
    within_v = v_sim[mask].mean().item()

    # Text-text
    t_sim = torch.mm(t_norm, t_norm.t())  # (M, M)
    mask_t = torch.triu(torch.ones_like(t_sim), diagonal=1).bool()
    within_t = t_sim[mask_t].mean().item()

    # Cross-modal: visual-text
    cross_sim = torch.mm(v_norm, t_norm.t())  # (N, M)
    cross = cross_sim.mean().item()

    return mg, within_v, within_t, cross


# ============================================================
# === A-C5-2 === Plot MG across phases
# ============================================================

def plot_mg_across_phases(mg_values, save_path="plots"):
    """
    A-C5-2: Plot MG across phases. Identify which phase has largest effect.

    Args:
        mg_values: dict mapping phase name to MG value
        save_path: directory to save plot
    """
    os.makedirs(save_path, exist_ok=True)

    phases = list(mg_values.keys())
    values = list(mg_values.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(phases, values, color=['gray', 'skyblue', 'orange', 'green'])
    plt.xlabel('Phase')
    plt.ylabel('Modality Gap (MG)')
    plt.title('Modality Gap Across Training Phases')
    plt.ylim(0, 2)

    # Annotate values
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center')

    # Find largest change
    changes = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
    largest_idx = changes.index(max(changes)) + 1
    print(f"Largest MG change: Phase {phases[largest_idx]} "
          f"(delta = {max(changes):.3f})")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'modality_gap.png'), dpi=150)
    plt.show()


# ============================================================
# === A-C5-3 === UMAP visualisation
# ============================================================

def umap_visualisation(visual_embs, text_embs, phase_name, save_path="plots"):
    """
    A-C5-3: UMAP visualisation of visual and text embeddings after a phase.

    Shows visual tokens (blue) and text tokens (red) in 2D.

    Args:
        visual_embs: (N, D) visual embeddings
        text_embs: (M, D) text embeddings
        phase_name: string name of the phase (for title)
        save_path: directory to save plot
    """
    try:
        from umap import UMAP
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return

    os.makedirs(save_path, exist_ok=True)

    # Combine embeddings
    all_embs = torch.cat([visual_embs, text_embs], dim=0).cpu().numpy()
    labels = ['visual'] * len(visual_embs) + ['text'] * len(text_embs)

    # Fit UMAP
    reducer = UMAP(n_components=2, random_state=SEED)
    embedding_2d = reducer.fit_transform(all_embs)

    # Plot
    n_v = len(visual_embs)
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_2d[:n_v, 0], embedding_2d[:n_v, 1],
                c='blue', alpha=0.5, label='Visual', s=20)
    plt.scatter(embedding_2d[n_v:, 0], embedding_2d[n_v:, 1],
                c='red', alpha=0.5, label='Text', s=20)
    plt.legend()
    plt.title(f'UMAP: Visual vs Text Embeddings ({phase_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'umap_{phase_name}.png'), dpi=150)
    plt.show()


# ============================================================
# === A-C5-4 === Phase 2 with L_norm
# ============================================================

def compute_norm_loss(visual_tokens, text_embs):
    """
    A-C5-4: Compute L_norm = (E[||V_i||_2] - E[||T_j||_2])^2.

    This loss encourages visual and text embedding norms to match,
    which can help close the modality gap.

    Args:
        visual_tokens: (B, 49, 960) visual tokens from connector
        text_embs: (B, L, 960) text embeddings

    Returns:
        norm_loss: scalar tensor
    """
    v_norms = visual_tokens.float().norm(dim=-1).mean()  # mean over batch and tokens
    t_norms = text_embs.float().norm(dim=-1).mean()
    norm_loss = (v_norms - t_norms) ** 2
    return norm_loss


def analyse_all_phases(clip_features_test, vqa_val, device):
    """
    A-C5: Run modality gap analysis across all phase checkpoints.

    Loads each checkpoint, computes MG, plots results.

    Args:
        clip_features_test: (M, 49, 768) test CLIP features
        vqa_val: val VQA pairs
        device: torch device
    """
    # Fix 200 test images and 200 test questions (seed 42)
    set_seed(SEED)

    # Get 200 unique questions
    seen_questions = set()
    questions_200 = []
    for pair in vqa_val:
        if pair['question'] not in seen_questions and len(questions_200) < 200:
            questions_200.append(pair['question'])
            seen_questions.add(pair['question'])

    mg_values = {}

    # Phase checkpoints to analyse
    phases = [
        ("Phase 0 (no training)", None, None),
        ("Phase 1", "weights/connector_phaseA1.pt", None),
        ("Phase 2", "weights/connector_phaseA2.pt", "weights/lm_phaseA2"),
        ("Phase 3", "weights/connector_phaseA3.pt", "weights/lm_phaseA3"),
    ]

    for phase_name, conn_path, lora_path in phases:
        print(f"\n--- {phase_name} ---")

        # Load LM
        lm_model, tokenizer = load_lm_model(device)

        # Load LoRA if applicable
        if lora_path and os.path.exists(lora_path):
            from peft import PeftModel
            lm_model = PeftModel.from_pretrained(lm_model, lora_path)

        # Load connector
        connector = MLPConnector(clip_dim=768, lm_dim=960).to(device)
        if conn_path and os.path.exists(conn_path):
            connector.load_state_dict(torch.load(conn_path, map_location='cpu'))

        # A-C5-1: Get embeddings
        visual_embs = get_visual_embeddings(connector, clip_features_test, device)
        text_embs = get_text_embeddings(lm_model, tokenizer, questions_200, device)

        # A-C5-1: Compute MG and similarities
        mg, within_v, within_t, cross = compute_modality_gap(visual_embs, text_embs)
        mg_values[phase_name] = mg

        print(f"  MG = {mg:.4f}")
        print(f"  Within-visual cosine sim = {within_v:.4f}")
        print(f"  Within-text cosine sim = {within_t:.4f}")
        print(f"  Cross-modal cosine sim = {cross:.4f}")

        # A-C5-3: UMAP
        umap_visualisation(visual_embs, text_embs, phase_name.replace(" ", "_"))

        # Clean up
        del lm_model, connector
        torch.cuda.empty_cache()

    # A-C5-2: Plot MG across phases
    plot_mg_across_phases(mg_values)

    return mg_values


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(SEED)

    # --- Load data ---
    train_subset, test_subset = get_cifar10_subsets()
    clip_model, clip_processor = load_clip_model(device)

    test_pixels, test_labels = preprocess_images_for_clip(test_subset, clip_processor)
    clip_features_test = extract_clip_features(test_pixels, clip_model, device)

    vqa_val = build_vqa_dataset(test_subset)

    del clip_model
    torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("A-C5: Modality Gap Analysis")
    print("="*60)

    # === A-C5-1,2,3 === Analyse all phases
    mg_values = analyse_all_phases(clip_features_test, vqa_val, device)

    # === A-C5-4 === Re-run Phase 2 with L_norm (optional, uncomment to run)
    # This requires retraining Phase 2 with the added norm loss.
    # See compute_norm_loss() function above.
    # To use: add norm_loss * 0.01 to the mixed loss in train_phase2.
    print("\nA-C5-4: To re-run Phase 2 with L_norm, modify a_c2 training loop to add:")
    print("  norm_loss = compute_norm_loss(visual_tokens, text_embs)")
    print("  loss = loss_vqa + lambda * loss_lm + 0.01 * norm_loss")

    print("\n" + "="*60)
    print("A-C5 Modality Gap Analysis COMPLETE")
    print("="*60)
