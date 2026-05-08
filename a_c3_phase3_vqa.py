"""
# ============================================================
# A-C3: Phase 3 — VQA Alignment
# ============================================================
# USAGE (Google Colab):
#   !pip install transformers peft accelerate datasets torchvision matplotlib scikit-learn
#   !python a_c0_data_and_models.py
#   !python a_c1_phase1_connector.py
#   !python a_c2_phase2_sft_replay.py
#   !python a_c3_phase3_vqa.py
# ============================================================

PSEUDOCODE:
    1. Load data, models, Phase 2 connector + LoRA checkpoint
    2. Train 1 epoch on 10,000 VQA pairs only (no replay, lambda=0)
    3. Use Adam lr=2e-4
    4. Compare accuracy vs Phase 2
    5. Check if R (forgetting ratio) increases compared to Phase 2
    6. (Optional) Unfreeze last 4 CLIP ViT blocks with lr=1e-5
    7. Save weights/connector_phaseA3.pt
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from peft import PeftModel

from a_c0_data_and_models import (
    set_seed, SEED, get_cifar10_subsets, build_vqa_dataset,
    load_clip_model, load_lm_model, load_alpaca_data,
    extract_clip_features, preprocess_images_for_clip, compute_ppl,
    VQADataset, CLIPImageProcessor, CLIP_MODEL_NAME, CIFAR10_CLASSES
)
from a_c1_phase1_connector import MLPConnector, count_params
from a_c2_phase2_sft_replay import (
    apply_lora, build_vqa_batch, evaluate_vqa_accuracy
)


# ============================================================
# === A-C3 === Phase 3 Training Loop
# ============================================================

def train_phase3(connector, lm_model, tokenizer, clip_features_train,
                 clip_features_test, vqa_train, vqa_val, alpaca_texts,
                 device, batch_size=32, lr=2e-4, num_epochs=1):
    """
    A-C3: Phase 3 — VQA Alignment.
    
    Load Phase 2 checkpoint. 1 epoch on 10,000 VQA pairs.
    No replay (lambda=0), lr=2e-4.
    Trains connector + LoRA parameters.

    Args:
        connector: MLPConnector (from Phase 2)
        lm_model: SmolLM2 with LoRA (from Phase 2)
        tokenizer: LM tokenizer
        clip_features_train: (N, 49, 768) train features
        clip_features_test: (M, 49, 768) test features
        vqa_train: train VQA pairs (10,000 from 5 templates x 2,000 unique subset)
        vqa_val: val VQA pairs
        alpaca_texts: for PPL evaluation
        device: torch device

    Returns:
        connector: trained connector
        lm_model: trained LM
        metrics: dict with losses, vqa_acc, ppl
    """
    connector.train()
    connector = connector.to(device)
    lm_model.train()

    # Combine trainable params: connector + LoRA
    all_params = list(connector.parameters()) + [
        p for p in lm_model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(all_params, lr=lr)

    # VQA dataloader (no replay in Phase 3)
    vqa_dataset = VQADataset(clip_features_train, vqa_train, tokenizer)
    vqa_loader = DataLoader(vqa_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, drop_last=True)

    # A-C2-3 style: GradScaler for fp16
    scaler = torch.amp.GradScaler('cuda')

    metrics = {'losses': [], 'vqa_acc': [], 'ppl': []}
    global_step = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_steps = 0

        for step, vqa_batch in enumerate(vqa_loader):
            clip_feats, q_ids, q_mask, a_ids, a_mask = vqa_batch

            # Forward pass
            with torch.amp.autocast('cuda', dtype=torch.float16):
                inputs_embeds, labels, attn_mask = build_vqa_batch(
                    connector, clip_feats, q_ids, q_mask, a_ids, a_mask,
                    lm_model, tokenizer, device
                )
                outputs = lm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    labels=labels,
                )
                # A-C3: No replay (lambda=0), so just VQA loss
                loss = outputs.loss

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            metrics['losses'].append(loss.item())
            epoch_loss += loss.item()
            num_steps += 1
            global_step += 1

            if global_step % 50 == 0:
                print(f"  Step {global_step}/{len(vqa_loader)}, "
                      f"Loss: {loss.item():.4f}")

            # Eval every 300 steps
            if global_step % 300 == 0:
                acc = evaluate_vqa_accuracy(
                    connector, lm_model, tokenizer, clip_features_test,
                    vqa_val, device, n_samples=200
                )
                ppl = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
                metrics['vqa_acc'].append(acc)
                metrics['ppl'].append(ppl)
                print(f"  [Eval @ step {global_step}] VQA acc: {acc:.3f}, PPL: {ppl:.2f}")
                connector.train()
                lm_model.train()

        avg_loss = epoch_loss / max(num_steps, 1)
        print(f"Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"Phase 3 done. Time: {elapsed:.1f}s")
    return connector, lm_model, metrics


# ============================================================
# === A-C3 (Optional) === Unfreeze last 4 CLIP ViT blocks
# ============================================================

def unfreeze_last_clip_blocks(clip_model, n_blocks=4, unfreeze_lr=1e-5):
    """
    A-C3 (Optional): Unfreeze last 4 CLIP ViT blocks with low lr.

    This allows the vision encoder to adapt slightly to the task.
    Use a separate param group with very low lr (1e-5).

    Args:
        clip_model: CLIP model
        n_blocks: number of final encoder blocks to unfreeze
        unfreeze_lr: learning rate for unfrozen blocks

    Returns:
        unfrozen_params: list of unfrozen parameters (for optimizer)
    """
    # CLIP ViT has encoder.layers — unfreeze last n_blocks
    total_layers = len(clip_model.vision_model.encoder.layers)
    start_layer = total_layers - n_blocks

    unfrozen_params = []
    for i in range(start_layer, total_layers):
        for p in clip_model.vision_model.encoder.layers[i].parameters():
            p.requires_grad = True
            unfrozen_params.append(p)

    print(f"Unfroze last {n_blocks} CLIP ViT blocks "
          f"(layers {start_layer}-{total_layers-1})")
    print(f"Unfrozen params: {sum(p.numel() for p in unfrozen_params):,}")
    return unfrozen_params


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(SEED)

    # --- Load data and models ---
    print("Loading data and models...")
    train_subset, test_subset = get_cifar10_subsets()
    clip_model, clip_processor = load_clip_model(device)
    lm_model, tokenizer = load_lm_model(device)

    train_pixels, train_labels = preprocess_images_for_clip(train_subset, clip_processor)
    test_pixels, test_labels = preprocess_images_for_clip(test_subset, clip_processor)
    clip_features_train = extract_clip_features(train_pixels, clip_model, device)
    clip_features_test = extract_clip_features(test_pixels, clip_model, device)

    vqa_train = build_vqa_dataset(train_subset)
    vqa_val = build_vqa_dataset(test_subset)
    alpaca_texts = load_alpaca_data()

    del clip_model
    torch.cuda.empty_cache()

    # --- Load Phase 2 checkpoint ---
    print("\n" + "="*60)
    print("A-C3: Phase 3 — VQA Alignment")
    print("="*60)

    # Apply LoRA and load Phase 2 weights
    lm_model = apply_lora(lm_model)
    if os.path.exists("weights/lm_phaseA2"):
        lm_model = PeftModel.from_pretrained(lm_model.get_base_model(),
                                              "weights/lm_phaseA2")
        print("Loaded Phase 2 LoRA weights.")
    else:
        print("WARNING: Phase 2 weights not found, using fresh LoRA.")

    connector = MLPConnector(clip_dim=768, lm_dim=960)
    if os.path.exists("weights/connector_phaseA2.pt"):
        connector.load_state_dict(torch.load("weights/connector_phaseA2.pt",
                                              map_location='cpu'))
        print("Loaded Phase 2 connector weights.")
    else:
        print("WARNING: Phase 2 connector not found, using Phase 1.")
        connector.load_state_dict(torch.load("weights/connector_phaseA1.pt",
                                              map_location='cpu'))

    # Compute PPL_0 for R calculation
    ppl_0 = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
    print(f"PPL before Phase 3: {ppl_0:.2f}")

    # --- Phase 2 accuracy (for comparison) ---
    acc_phase2 = evaluate_vqa_accuracy(
        connector, lm_model, tokenizer, clip_features_test,
        vqa_val, device, n_samples=200
    )
    print(f"Phase 2 VQA accuracy: {acc_phase2:.3f}")

    # --- Train Phase 3 ---
    # A-C3: 1 epoch, no replay, lr=2e-4
    connector, lm_model, metrics = train_phase3(
        connector, lm_model, tokenizer,
        clip_features_train, clip_features_test,
        vqa_train, vqa_val, alpaca_texts,
        device, batch_size=32, lr=2e-4, num_epochs=1
    )

    # --- Compare with Phase 2 ---
    acc_phase3 = evaluate_vqa_accuracy(
        connector, lm_model, tokenizer, clip_features_test,
        vqa_val, device, n_samples=200
    )
    ppl_phase3 = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
    R_phase3 = ppl_phase3 / ppl_0

    print(f"\n--- Phase 3 vs Phase 2 ---")
    print(f"VQA accuracy: Phase 2={acc_phase2:.3f}, Phase 3={acc_phase3:.3f}")
    print(f"PPL: {ppl_phase3:.2f}")
    print(f"R (forgetting ratio): {R_phase3:.3f}")
    print(f"Does R increase? {'Yes' if R_phase3 > 1.0 else 'No'} "
          f"(R>1 means forgetting)")

    # === Save Phase 3 weights ===
    os.makedirs("weights", exist_ok=True)
    torch.save(connector.state_dict(), "weights/connector_phaseA3.pt")
    lm_model.save_pretrained("weights/lm_phaseA3")
    print(f"\nPhase 3 weights saved.")

    print("\n" + "="*60)
    print("A-C3 Phase 3 COMPLETE")
    print("="*60)
