"""
# ============================================================
# A-C6: Ablations (LoRA Rank Sweep)
# ============================================================
# USAGE (Google Colab):
#   !pip install transformers peft accelerate datasets torchvision matplotlib scikit-learn
#   # Run a_c0 and a_c1 first (need Phase 1 connector)
#   !python a_c6_ablations.py
# ============================================================

PSEUDOCODE:
    1. Load data and Phase 1 connector
    2. For each LoRA rank r in {2, 4, 16, 32}:
       a. Load fresh LM + apply LoRA with rank r
       b. Load Phase 1 connector
       c. Run Phase 2 training (shortened: 1 epoch)
       d. Evaluate VQA accuracy and R
    3. Plot accuracy and R vs LoRA rank
    4. Report which rank gives best accuracy-forgetting tradeoff
"""

import os
import torch
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model

from a_c0_data_and_models import (
    set_seed, SEED, get_cifar10_subsets, build_vqa_dataset,
    load_clip_model, load_lm_model, load_alpaca_data,
    extract_clip_features, preprocess_images_for_clip, compute_ppl,
    CLIPImageProcessor, CLIP_MODEL_NAME
)
from a_c1_phase1_connector import MLPConnector
from a_c2_phase2_sft_replay import (
    build_vqa_batch, evaluate_vqa_accuracy, compute_alpaca_loss,
    train_phase2
)


# ============================================================
# === A-C6-3 === LoRA rank sweep: r in {2, 4, 16, 32}
# ============================================================

def apply_lora_with_rank(model, r, alpha=None):
    """
    A-C6-3: Apply LoRA with a specific rank.

    Alpha is set to 2*r (following the convention alpha = 2r).

    Args:
        model: base LM model
        r: LoRA rank
        alpha: LoRA alpha (default: 2*r)

    Returns:
        peft_model: model with LoRA applied
    """
    if alpha is None:
        alpha = 2 * r  # standard convention

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"  LoRA rank={r}, alpha={alpha}: {trainable:,} trainable / "
          f"{total:,} total ({trainable/total*100:.2f}%)")

    return peft_model


def lora_rank_sweep(clip_features_train, clip_features_test, vqa_train,
                    vqa_val, alpaca_texts, device):
    """
    A-C6-3: LoRA rank sweep — compare r in {2, 4, 16, 32}.

    For each rank, run Phase 2 training and evaluate.

    Args:
        clip_features_train: train CLIP features
        clip_features_test: test CLIP features
        vqa_train: train VQA pairs
        vqa_val: val VQA pairs
        alpaca_texts: Alpaca texts for replay
        device: torch device

    Returns:
        results: list of dicts with rank, vqa_acc, R, trainable_params
    """
    ranks = [2, 4, 16, 32]
    results = []

    # Compute PPL_0
    lm_ref, tokenizer = load_lm_model(device)
    ppl_0 = compute_ppl(lm_ref, tokenizer, alpaca_texts[:100], device)
    print(f"PPL_0 = {ppl_0:.2f}")
    del lm_ref
    torch.cuda.empty_cache()

    for r in ranks:
        print(f"\n{'='*50}")
        print(f"LoRA Rank Sweep: r = {r}")
        print(f"{'='*50}")

        # Fresh LM with this rank
        lm_model, tokenizer = load_lm_model(device)
        lm_model = apply_lora_with_rank(lm_model, r)

        # Fresh connector from Phase 1
        connector = MLPConnector(clip_dim=768, lm_dim=960)
        if os.path.exists("weights/connector_phaseA1.pt"):
            connector.load_state_dict(torch.load("weights/connector_phaseA1.pt",
                                                  map_location='cpu'))

        trainable_params = sum(p.numel() for p in lm_model.parameters()
                               if p.requires_grad)

        # Train Phase 2
        connector, lm_model, metrics = train_phase2(
            connector, lm_model, tokenizer,
            clip_features_train, clip_features_test,
            vqa_train, vqa_val, alpaca_texts,
            device, lam=0.2, batch_size=32, lr=5e-4, num_epochs=1,
            grad_accum_steps=4
        )

        # Evaluate
        final_acc = evaluate_vqa_accuracy(
            connector, lm_model, tokenizer, clip_features_test,
            vqa_val, device, n_samples=200
        )
        final_ppl = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
        R = final_ppl / ppl_0

        results.append({
            'rank': r,
            'vqa_acc': final_acc,
            'R': R,
            'ppl': final_ppl,
            'trainable_params': trainable_params,
        })
        print(f"  r={r}: VQA acc={final_acc:.3f}, R={R:.3f}, PPL={final_ppl:.2f}")

        del lm_model, connector
        torch.cuda.empty_cache()

    # Print results table
    print(f"\n{'='*60}")
    print("LoRA Rank Sweep Results")
    print(f"{'='*60}")
    print(f"{'Rank':<8} {'Params':<15} {'VQA acc (%)':<14} {'R':<8} {'PPL':<8}")
    print("-" * 53)
    for r_dict in results:
        print(f"{r_dict['rank']:<8} {r_dict['trainable_params']:<15,} "
              f"{r_dict['vqa_acc']*100:<14.1f} {r_dict['R']:<8.3f} "
              f"{r_dict['ppl']:<8.2f}")

    return results


def plot_rank_sweep(results, save_path="plots"):
    """
    Plot LoRA rank sweep results: accuracy and R vs rank.

    Args:
        results: list of dicts from lora_rank_sweep()
        save_path: directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)

    ranks = [r['rank'] for r in results]
    accs = [r['vqa_acc'] * 100 for r in results]
    Rs = [r['R'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ranks, accs, 'bo-', markersize=8)
    ax1.set_xlabel('LoRA Rank (r)')
    ax1.set_ylabel('VQA Accuracy (%)')
    ax1.set_title('VQA Accuracy vs LoRA Rank')
    ax1.set_xticks(ranks)

    ax2.plot(ranks, Rs, 'ro-', markersize=8)
    ax2.axhline(y=1.0, color='gray', linestyle='--', label='No forgetting')
    ax2.set_xlabel('LoRA Rank (r)')
    ax2.set_ylabel('R (PPL_fine / PPL_0)')
    ax2.set_title('Forgetting Ratio vs LoRA Rank')
    ax2.set_xticks(ranks)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'lora_rank_sweep.png'), dpi=150)
    plt.show()
    print(f"Plot saved to {save_path}/lora_rank_sweep.png")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(SEED)

    # --- Load data ---
    train_subset, test_subset = get_cifar10_subsets()
    clip_model, clip_processor = load_clip_model(device)

    train_pixels, train_labels = preprocess_images_for_clip(train_subset, clip_processor)
    test_pixels, test_labels = preprocess_images_for_clip(test_subset, clip_processor)
    clip_features_train = extract_clip_features(train_pixels, clip_model, device)
    clip_features_test = extract_clip_features(test_pixels, clip_model, device)

    vqa_train = build_vqa_dataset(train_subset)
    vqa_val = build_vqa_dataset(test_subset)
    alpaca_texts = load_alpaca_data()

    del clip_model
    torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("A-C6: Ablations — LoRA Rank Sweep")
    print("="*60)

    # === A-C6-3 === LoRA rank sweep
    results = lora_rank_sweep(
        clip_features_train, clip_features_test,
        vqa_train, vqa_val, alpaca_texts, device
    )

    # Plot results
    plot_rank_sweep(results)

    print("\n" + "="*60)
    print("A-C6 Ablations COMPLETE")
    print("="*60)
