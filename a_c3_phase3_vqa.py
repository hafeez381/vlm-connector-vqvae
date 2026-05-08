# ============================================================
# A-C3: Phase 3 — VQA Alignment
# ============================================================
# USAGE (Google Colab):
#   !python a_c3_phase3_vqa.py
#
# PSEUDOCODE:
#   1. Load data, models, Phase 2 connector + LoRA checkpoint
#   2. Train 1 epoch on 10,000 VQA pairs only (no replay, lambda=0)
#   3. Use Adam lr=2e-4
#   4. Compare accuracy vs Phase 2
#   5. Check if R (forgetting ratio) increases compared to Phase 2
#   6. (Optional) Unfreeze last 4 CLIP ViT blocks with lr=1e-5
#   7. Save weights/connector_phaseA3.pt
#
# OOM FIXES (same as A-C2):
#   FIX-1  expandable_segments env var (before torch import)
#   FIX-2  del train_pixels / test_pixels after feature extraction
#   FIX-3  del clip_model BEFORE loading the LM
# ============================================================

# [FIX-1] Must be set BEFORE any torch import
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from peft import PeftModel
from tqdm.auto import tqdm

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
                 device, batch_size=32, lr=2e-4, num_epochs=1,
                 eval_every=300):
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
        vqa_train: train VQA pairs
        vqa_val: val VQA pairs
        alpaca_texts: for PPL evaluation
        device: torch device
        batch_size: training batch size
        lr: learning rate
        num_epochs: number of epochs
        eval_every: evaluate every N steps

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
    vqa_loader = DataLoader(
        VQADataset(clip_features_train, vqa_train, tokenizer),
        batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
    )

    total_steps = len(vqa_loader) * num_epochs

    # OneCycleLR with 10% warmup (same style as Phase 2)
    scheduler = OneCycleLR(optimizer, max_lr=lr,
                           total_steps=total_steps, pct_start=0.1)

    # GradScaler for fp16
    scaler = torch.amp.GradScaler("cuda")

    metrics     = {"losses": [], "vqa_acc": [], "ppl": []}
    global_step = 0
    start_time  = time.time()

    # Zero grad once before the loop (not inside, avoids discarding first accum)
    optimizer.zero_grad()

    epoch_bar = tqdm(range(num_epochs), desc="Epoch", unit="epoch", position=0)

    for epoch in epoch_bar:
        step_bar = tqdm(vqa_loader, desc=f"  Phase3 e{epoch+1}",
                        unit="step", position=1, leave=False,
                        dynamic_ncols=True)

        running_loss = 0.0

        for vqa_batch in step_bar:
            clip_feats, q_ids, q_mask, a_ids, a_mask = vqa_batch

            # A-C3: VQA loss only — no replay (lambda=0)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                inputs_embeds, labels, attn_mask = build_vqa_batch(
                    connector, clip_feats, q_ids, q_mask, a_ids, a_mask,
                    lm_model, tokenizer, device,
                )
                loss = lm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    labels=labels,
                ).loss

            # Guard: skip non-finite loss steps
            if not torch.isfinite(loss):
                step_bar.write(f"  [WARNING] Non-finite loss at step {global_step+1}, skipping.")
                global_step += 1
                continue

            scaler.scale(loss).backward()

            # optimizer.step() THEN scheduler.step() — correct order
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            torch.cuda.empty_cache()

            running_loss = 0.95 * running_loss + 0.05 * loss.item()
            step_bar.set_postfix(
                loss=f"{running_loss:.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            metrics["losses"].append(loss.item())
            global_step += 1

            # Eval every eval_every steps
            if global_step % eval_every == 0:
                step_bar.write(f"\n  [Eval @ step {global_step}/{total_steps}]")
                acc = evaluate_vqa_accuracy(
                    connector, lm_model, tokenizer,
                    clip_features_test, vqa_val, device, n_samples=200,
                )
                ppl = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
                metrics["vqa_acc"].append(acc)
                metrics["ppl"].append(ppl)
                step_bar.write(f"  VQA acc={acc:.3f}  PPL={ppl:.2f}\n")
                connector.train(); lm_model.train()

        elapsed = time.time() - start_time
        epoch_bar.set_postfix(loss=f"{running_loss:.3f}",
                              elapsed=f"{elapsed/60:.1f}min")

    print(f"\nPhase 3 done in {(time.time()-start_time)/60:.1f} min")
    return connector, lm_model, metrics


# # ============================================================
# # === A-C3 (Optional) === Unfreeze last 4 CLIP ViT blocks
# # ============================================================

# def unfreeze_last_clip_blocks(clip_model, n_blocks=4, unfreeze_lr=1e-5):
#     """
#     A-C3 (Optional): Unfreeze last 4 CLIP ViT blocks with low lr.

#     This allows the vision encoder to adapt slightly to the task.
#     Use a separate param group with very low lr (1e-5).

#     Args:
#         clip_model: CLIP model
#         n_blocks: number of final encoder blocks to unfreeze
#         unfreeze_lr: learning rate for unfrozen blocks

#     Returns:
#         unfrozen_params: list of unfrozen parameters (for the optimizer)
#     """
#     total_layers = len(clip_model.vision_model.encoder.layers)
#     start_layer  = total_layers - n_blocks

#     unfrozen_params = []
#     for i in range(start_layer, total_layers):
#         for p in clip_model.vision_model.encoder.layers[i].parameters():
#             p.requires_grad = True
#             unfrozen_params.append(p)

#     print(f"Unfroze last {n_blocks} CLIP ViT blocks "
#           f"(layers {start_layer}-{total_layers-1})")
#     print(f"Unfrozen params: {sum(p.numel() for p in unfrozen_params):,}")
#     return unfrozen_params


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(SEED)

    # --- Load data and models ---
    # [FIX-3] Extract CLIP features first, then delete CLIP before loading LM
    print("Loading data and models...")
    train_subset, test_subset = get_cifar10_subsets()

    clip_model, clip_processor = load_clip_model(device)

    train_pixels, _ = preprocess_images_for_clip(train_subset, clip_processor)
    clip_features_train = extract_clip_features(train_pixels, clip_model, device)
    del train_pixels; torch.cuda.empty_cache()   # [FIX-2]

    test_pixels, _ = preprocess_images_for_clip(test_subset, clip_processor)
    clip_features_test = extract_clip_features(test_pixels, clip_model, device)
    del test_pixels; torch.cuda.empty_cache()    # [FIX-2]

    del clip_model; torch.cuda.empty_cache()     # [FIX-3]

    lm_model, tokenizer = load_lm_model(device)

    vqa_train    = build_vqa_dataset(train_subset)
    vqa_val      = build_vqa_dataset(test_subset)
    alpaca_texts = load_alpaca_data()

    if torch.cuda.is_available():
        print(f"VRAM after data load: "
              f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    # --- Load Phase 2 checkpoint ---
    print("\n" + "="*60)
    print("A-C3: Phase 3 — VQA Alignment")
    print("="*60)

    # Apply LoRA then load Phase 2 LoRA weights on top
    lm_model = apply_lora(lm_model)
    if os.path.exists("weights/lm_phaseA2"):
        lm_model = PeftModel.from_pretrained(lm_model.get_base_model(),
                                             "weights/lm_phaseA2")
        print("Loaded Phase 2 LoRA weights.")
    else:
        print("WARNING: Phase 2 LoRA weights not found, using fresh LoRA.")

    connector = MLPConnector(clip_dim=768, lm_dim=960)
    if os.path.exists("weights/connector_phaseA2.pt"):
        connector.load_state_dict(
            torch.load("weights/connector_phaseA2.pt", map_location="cpu")
        )
        print("Loaded Phase 2 connector weights.")
    else:
        print("WARNING: Phase 2 connector not found, falling back to Phase 1.")
        connector.load_state_dict(
            torch.load("weights/connector_phaseA1.pt", map_location="cpu")
        )

    # PPL before Phase 3 = denominator for R = PPL_phase3 / PPL_phase2
    ppl_before = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
    print(f"PPL before Phase 3: {ppl_before:.2f}")

    # Phase 2 accuracy baseline (for the comparison table)
    print("\nEvaluating Phase 2 accuracy (baseline for comparison)...")
    acc_phase2 = evaluate_vqa_accuracy(
        connector, lm_model, tokenizer,
        clip_features_test, vqa_val, device, n_samples=200,
    )
    print(f"Phase 2 VQA accuracy: {acc_phase2:.3f}")

    # --- Train Phase 3 ---
    # A-C3: 1 epoch, no replay (lambda=0), lr=2e-4
    connector, lm_model, metrics = train_phase3(
        connector, lm_model, tokenizer,
        clip_features_train, clip_features_test,
        vqa_train, vqa_val, alpaca_texts,
        device, batch_size=32, lr=2e-4, num_epochs=1,
    )

    # --- Compare Phase 2 vs Phase 3 ---
    print("\nEvaluating Phase 3 accuracy...")
    acc_phase3 = evaluate_vqa_accuracy(
        connector, lm_model, tokenizer,
        clip_features_test, vqa_val, device, n_samples=200,
    )
    ppl_phase3 = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
    R_phase3   = ppl_phase3 / ppl_before

    print(f"\n--- Phase 3 vs Phase 2 ---")
    print(f"VQA accuracy : Phase 2={acc_phase2:.3f}  Phase 3={acc_phase3:.3f}  "
          f"delta={acc_phase3-acc_phase2:+.3f}")
    print(f"PPL          : {ppl_phase3:.2f}")
    print(f"R (PPL_p3 / PPL_p2): {R_phase3:.3f}  "
          f"({'forgetting increased' if R_phase3 > 1.0 else 'no extra forgetting'})")

    # --- Save Phase 3 weights ---
    os.makedirs("weights", exist_ok=True)
    torch.save(connector.state_dict(), "weights/connector_phaseA3.pt")
    lm_model.save_pretrained("weights/lm_phaseA3")
    print("\nPhase 3 weights saved.")

    print("\n" + "="*60)
    print("A-C3 Phase 3 COMPLETE")
    print("="*60)