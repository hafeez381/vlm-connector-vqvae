"""
# ============================================================
# A-C2: Phase 2 — SFT with Language Replay
# ============================================================
# USAGE (Google Colab):
#   !pip install transformers peft accelerate datasets torchvision matplotlib scikit-learn
#   !python a_c0_data_and_models.py   # setup data first
#   !python a_c1_phase1_connector.py  # train connector first
#   !python a_c2_phase2_sft_replay.py # then run this
# ============================================================

PSEUDOCODE:
    1. Load data, models, and Phase 1 connector weights
    2. Apply LoRA (r=16, alpha=32) to LM's q/k/v/o_proj layers
    3. Verify trainable params < 1% of total
    4. For each training step:
        a. VQA batch: build [BOS emb, V, q embs, a embs, EOS]
           Labels: -100 for prefix (BOS+V+question); answer+EOS get gradient
        b. Alpaca batch: standard text-only LM loss (NO visual tokens!)
        c. L_mixed = L_VQA + lambda * L_LM
        d. Backward with GradScaler, accumulate gradients every 4 steps
    5. Use OneCycleLR with 10% warmup
    6. Eval every 300 steps: 200 VQA pairs accuracy + 100 Alpaca PPL
    7. Run lambda ablation: lambda in {0, 0.05, 0.2, 0.5}
    8. Save checkpoint
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from peft import LoraConfig, get_peft_model

from a_c0_data_and_models import (
    set_seed, SEED, get_cifar10_subsets, build_vqa_dataset,
    load_clip_model, load_lm_model, load_alpaca_data,
    extract_clip_features, preprocess_images_for_clip, compute_ppl,
    VQADataset, AlpacaDataset, CLIPImageProcessor, CLIP_MODEL_NAME,
    CIFAR10_CLASSES
)
from a_c1_phase1_connector import MLPConnector, count_params, compute_norm_ratio


# ============================================================
# === A-C2-1 === Apply LoRA
# ============================================================

def apply_lora(model, r=16, alpha=32):
    """
    A-C2-1: Apply LoRA (r=16, alpha=32) to the LM.
    
    Targets: q_proj, k_proj, v_proj, o_proj.
    Dropout: 0.05 as specified in the assignment.

    Args:
        model: SmolLM2 causal LM
        r: LoRA rank
        alpha: LoRA alpha scaling

    Returns:
        peft_model: model with LoRA adapters applied
    """
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


# ============================================================
# === A-C2-2 === Build VQA sequence
# ============================================================

def build_vqa_batch(connector, clip_feats, q_ids, q_mask, a_ids, a_mask,
                    lm_model, tokenizer, device):
    """
    A-C2-2: Build VQA input: [BOS emb, V, q embs, a embs, EOS].
    Labels: -100 for prefix (BOS + V + question); answer + EOS receive gradient.

    Args:
        connector: trained MLPConnector
        clip_feats: (B, 49, 768) CLIP patch features
        q_ids: (B, Lq) question token IDs
        q_mask: (B, Lq) question attention mask
        a_ids: (B, La) answer token IDs
        a_mask: (B, La) answer attention mask
        lm_model: SmolLM2 model (possibly with LoRA)
        tokenizer: LM tokenizer
        device: torch device

    Returns:
        inputs_embeds: (B, 1+49+Lq+La+1, 960)
        labels: (B, 1+49+Lq+La+1) with -100 prefix, answer+EOS with IDs
        attention_mask: (B, total_len)
    """
    B = clip_feats.shape[0]

    # Get the base model's embedding layer (works with or without LoRA)
    if hasattr(lm_model, 'get_base_model'):
        base = lm_model.get_base_model()
    else:
        base = lm_model
    embed_layer = base.get_input_embeddings()

    # 1. BOS embedding
    bos_id = base.config.bos_token_id or 0
    bos_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    bos_emb = embed_layer(bos_ids)  # (B, 1, 960)

    # 2. Visual tokens through connector
    visual_tokens = connector(clip_feats.to(device))  # (B, 49, 960)
    visual_tokens = visual_tokens.to(bos_emb.dtype)

    # 3. Question embeddings
    q_emb = embed_layer(q_ids.to(device))  # (B, Lq, 960)

    # 4. Answer embeddings
    a_emb = embed_layer(a_ids.to(device))  # (B, La, 960)

    # 5. EOS embedding
    eos_id = tokenizer.eos_token_id
    eos_ids = torch.full((B, 1), eos_id, dtype=torch.long, device=device)
    eos_emb = embed_layer(eos_ids)  # (B, 1, 960)

    # Concatenate: [BOS, V1:49, question, answer, EOS]
    inputs_embeds = torch.cat([bos_emb, visual_tokens, q_emb, a_emb, eos_emb], dim=1)

    # Labels: -100 for BOS + visual + question; answer IDs + EOS for gradient
    prefix_len = 1 + 49 + q_ids.shape[1]  # BOS + 49 visual + question length
    ignore_labels = torch.full((B, prefix_len), -100, dtype=torch.long, device=device)

    # Answer labels (mask padding with -100)
    answer_labels = a_ids.clone().to(device)
    answer_labels[a_mask == 0] = -100

    # EOS label
    eos_labels = torch.full((B, 1), eos_id, dtype=torch.long, device=device)

    labels = torch.cat([ignore_labels, answer_labels, eos_labels], dim=1)

    # Attention mask
    bos_m = torch.ones(B, 1, dtype=torch.long, device=device)
    vis_m = torch.ones(B, 49, dtype=torch.long, device=device)
    eos_m = torch.ones(B, 1, dtype=torch.long, device=device)
    attention_mask = torch.cat([bos_m, vis_m, q_mask.to(device), a_mask.to(device), eos_m], dim=1)

    return inputs_embeds, labels, attention_mask


# ============================================================
# === A-C2-3 === Mixed training step
# ============================================================

def compute_alpaca_loss(lm_model, alpaca_batch, device):
    """
    A-C2-3: Compute language model loss on Alpaca text (NO visual tokens!).

    This is the replay loss L_LM to prevent catastrophic forgetting.
    Never inject visual tokens into Alpaca batches.

    Args:
        lm_model: LM with LoRA
        alpaca_batch: tuple of (input_ids, attention_mask) from AlpacaDataset
        device: torch device

    Returns:
        loss: scalar LM loss on Alpaca text
    """
    input_ids, attn_mask = alpaca_batch
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    # Standard causal LM loss — labels = input_ids
    outputs = lm_model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
    return outputs.loss


def evaluate_vqa_accuracy(connector, lm_model, tokenizer, clip_features,
                          vqa_pairs, device, n_samples=200):
    """
    Evaluate VQA exact-match accuracy on a subset.

    Args:
        connector: MLPConnector
        lm_model: LM model (with LoRA)
        tokenizer: LM tokenizer
        clip_features: (N, 49, 768) CLIP features
        vqa_pairs: list of VQA dicts
        device: torch device
        n_samples: number of pairs to evaluate

    Returns:
        accuracy: float (0 to 1)
    """
    connector.eval()
    lm_model.eval()

    if hasattr(lm_model, 'get_base_model'):
        base = lm_model.get_base_model()
    else:
        base = lm_model
    embed_layer = base.get_input_embeddings()

    correct = 0
    total = min(n_samples, len(vqa_pairs))

    with torch.no_grad():
        for i in range(total):
            pair = vqa_pairs[i]
            img_idx = pair['image_idx']
            clip_feat = clip_features[img_idx]  # (49, 768)

            # Build prompt: [BOS, V, question]
            bos_id = base.config.bos_token_id or 0
            bos_emb = embed_layer(torch.tensor([[bos_id]], device=device))

            visual = connector(clip_feat.unsqueeze(0).to(device))
            visual = visual.to(bos_emb.dtype)

            q_tokens = tokenizer(pair['question'], return_tensors="pt")
            q_emb = embed_layer(q_tokens['input_ids'].to(device))

            inputs_embeds = torch.cat([bos_emb, visual, q_emb], dim=1)

            # Generate answer greedily (short answers, max 10 tokens)
            generated_ids = []
            current_embeds = inputs_embeds
            for _ in range(10):
                outputs = lm_model(inputs_embeds=current_embeds)
                next_logits = outputs.logits[:, -1, :]
                next_id = next_logits.argmax(dim=-1)
                if next_id.item() == tokenizer.eos_token_id:
                    break
                generated_ids.append(next_id.item())
                next_emb = embed_layer(next_id.unsqueeze(0))
                current_embeds = torch.cat([current_embeds, next_emb], dim=1)

            pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
            true = pair['answer'].strip().lower()
            if pred == true:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    connector.train()
    lm_model.train()
    return accuracy


# ============================================================
# === A-C2-2, A-C2-3 === Phase 2 Training Loop
# ============================================================

def train_phase2(connector, lm_model, tokenizer, clip_features_train,
                 clip_features_test, vqa_train, vqa_val, alpaca_texts,
                 device, lam=0.2, batch_size=32, lr=5e-4, num_epochs=1,
                 grad_accum_steps=4, eval_every=300):
    """
    A-C2: Phase 2 training — connector + LoRA + Alpaca replay.

    A-C2-3: L_mixed = L_VQA + lambda * L_LM.
    Uses GradScaler for fp16, gradient accumulation x4, OneCycleLR.

    Args:
        connector: MLPConnector (from Phase 1)
        lm_model: SmolLM2 with LoRA applied
        tokenizer: LM tokenizer
        clip_features_train: (N, 49, 768) train CLIP features
        clip_features_test: (M, 49, 768) test CLIP features
        vqa_train: train VQA pairs
        vqa_val: val VQA pairs
        alpaca_texts: list of Alpaca text strings
        device: torch device
        lam: replay weight lambda
        batch_size: batch size
        lr: learning rate
        num_epochs: number of epochs
        grad_accum_steps: gradient accumulation steps
        eval_every: evaluate every N steps

    Returns:
        connector: trained connector
        lm_model: trained LM with LoRA
        metrics: dict of training metrics
    """
    # Put connector and LoRA params in one optimizer
    connector.train()
    connector = connector.to(device)
    lm_model.train()

    # Combine parameters: connector + LoRA
    all_params = list(connector.parameters()) + [
        p for p in lm_model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(all_params, lr=lr)

    # Create dataloaders
    vqa_dataset = VQADataset(clip_features_train, vqa_train, tokenizer)
    vqa_loader = DataLoader(vqa_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, drop_last=True)

    alpaca_dataset = AlpacaDataset(alpaca_texts, tokenizer)
    alpaca_loader = DataLoader(alpaca_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=0, drop_last=True)

    # Total steps for scheduler
    steps_per_epoch = len(vqa_loader)
    total_steps = steps_per_epoch * num_epochs
    # A-C2-3: OneCycleLR with 10% warmup
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps,
                           pct_start=0.1)

    # A-C2-3: GradScaler for mixed precision
    scaler = torch.amp.GradScaler('cuda')

    alpaca_iter = iter(alpaca_loader)
    metrics = {'losses': [], 'vqa_acc': [], 'ppl': []}
    global_step = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        for step, vqa_batch in enumerate(vqa_loader):
            clip_feats, q_ids, q_mask, a_ids, a_mask = vqa_batch

            # === A-C2-2 === VQA forward pass
            with torch.amp.autocast('cuda', dtype=torch.float16):
                inputs_embeds, labels, attn_mask = build_vqa_batch(
                    connector, clip_feats, q_ids, q_mask, a_ids, a_mask,
                    lm_model, tokenizer, device
                )
                vqa_outputs = lm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    labels=labels,
                )
                loss_vqa = vqa_outputs.loss

                # === A-C2-3 === Alpaca replay loss (if lambda > 0)
                loss_lm = torch.tensor(0.0, device=device)
                if lam > 0:
                    try:
                        alpaca_batch = next(alpaca_iter)
                    except StopIteration:
                        alpaca_iter = iter(alpaca_loader)
                        alpaca_batch = next(alpaca_iter)
                    loss_lm = compute_alpaca_loss(lm_model, alpaca_batch, device)

                # A-C2-3: Mixed loss
                loss = loss_vqa + lam * loss_lm

            # A-C2-3: Scale loss for gradient accumulation
            scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()

            # Step optimizer every grad_accum_steps
            if (global_step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            metrics['losses'].append(loss.item())
            global_step += 1

            # Print progress
            if global_step % 50 == 0:
                print(f"  Step {global_step}/{total_steps}, "
                      f"L_vqa: {loss_vqa.item():.4f}, "
                      f"L_lm: {loss_lm.item():.4f}, "
                      f"L_total: {loss.item():.4f}")

            # Evaluate every 300 steps
            if global_step % eval_every == 0:
                acc = evaluate_vqa_accuracy(
                    connector, lm_model, tokenizer, clip_features_test,
                    vqa_val, device, n_samples=200
                )
                ppl = compute_ppl(
                    lm_model, tokenizer, alpaca_texts[:100], device
                )
                metrics['vqa_acc'].append(acc)
                metrics['ppl'].append(ppl)
                print(f"  [Eval @ step {global_step}] VQA acc: {acc:.3f}, PPL: {ppl:.2f}")
                connector.train()
                lm_model.train()

    elapsed = time.time() - start_time
    print(f"\nPhase 2 done. Time: {elapsed:.1f}s")
    return connector, lm_model, metrics


# ============================================================
# === A-C2-4 === Lambda ablation
# ============================================================

def run_lambda_ablation(clip_features_train, clip_features_test, vqa_train,
                        vqa_val, alpaca_texts, device):
    """
    A-C2-4: Run lambda ablation from Table 1.

    Tests lambda in {0.0, 0.05, 0.2, 0.5} and reports VQA accuracy and R.
    R = PPL_fine / PPL_0 (forgetting ratio).

    Returns:
        results: list of dicts with lambda, vqa_acc, R values
    """
    lambdas = [0.0, 0.05, 0.2, 0.5]
    labels = ["No replay", "Weak", "Baseline", "Strong"]
    results = []

    # Compute PPL_0 first (before any fine-tuning)
    lm_model_ref, tokenizer = load_lm_model(device)
    ppl_0 = compute_ppl(lm_model_ref, tokenizer, alpaca_texts[:100], device)
    print(f"PPL_0 = {ppl_0:.2f}")
    del lm_model_ref
    torch.cuda.empty_cache()

    for lam, label in zip(lambdas, labels):
        print(f"\n{'='*40}")
        print(f"Lambda ablation: {label} (lambda={lam})")
        print(f"{'='*40}")

        # Fresh LM with LoRA for each run
        lm_model, tokenizer = load_lm_model(device)
        lm_model = apply_lora(lm_model)

        # Fresh connector from Phase 1
        connector = MLPConnector(clip_dim=768, lm_dim=960)
        connector.load_state_dict(torch.load("weights/connector_phaseA1.pt",
                                              map_location='cpu'))

        # Train Phase 2 with this lambda
        connector, lm_model, metrics = train_phase2(
            connector, lm_model, tokenizer,
            clip_features_train, clip_features_test,
            vqa_train, vqa_val, alpaca_texts,
            device, lam=lam, num_epochs=1
        )

        # Final evaluation
        final_acc = evaluate_vqa_accuracy(
            connector, lm_model, tokenizer, clip_features_test,
            vqa_val, device, n_samples=200
        )
        final_ppl = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
        R = final_ppl / ppl_0

        results.append({
            'condition': label,
            'lambda': lam,
            'vqa_acc': final_acc,
            'R': R,
            'ppl': final_ppl,
        })
        print(f"  Final: VQA acc={final_acc:.3f}, R={R:.3f}, PPL={final_ppl:.2f}")

        # Clean up
        del lm_model, connector
        torch.cuda.empty_cache()

    # Print table
    print(f"\n{'='*60}")
    print("Table 1: Lambda Ablation Results")
    print(f"{'='*60}")
    print(f"{'Condition':<12} {'Lambda':<8} {'VQA acc (%)':<14} {'R':<8}")
    print("-" * 42)
    for r in results:
        print(f"{r['condition']:<12} {r['lambda']:<8.2f} {r['vqa_acc']*100:<14.1f} {r['R']:<8.3f}")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(SEED)

    # --- Load data ---
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

    # Free CLIP from GPU (not needed during training)
    del clip_model
    torch.cuda.empty_cache()

    # === A-C2-1 === Apply LoRA, load Phase 1 connector
    print("\n" + "="*60)
    print("A-C2: Phase 2 — SFT with Language Replay")
    print("="*60)

    lm_model = apply_lora(lm_model)

    connector = MLPConnector(clip_dim=768, lm_dim=960)
    connector.load_state_dict(torch.load("weights/connector_phaseA1.pt",
                                          map_location='cpu'))
    print("Loaded Phase 1 connector weights.")

    # A-C2-1: Verify trainable < 1%
    lora_trainable = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    lm_total = sum(p.numel() for p in lm_model.parameters())
    connector_params = sum(p.numel() for p in connector.parameters())
    total_trainable = lora_trainable + connector_params
    print(f"LoRA trainable: {lora_trainable:,}")
    print(f"Connector params: {connector_params:,}")
    print(f"Total trainable: {total_trainable:,} / {lm_total:,} = "
          f"{total_trainable/lm_total*100:.2f}%")

    # Compute PPL_0 before fine-tuning
    ppl_0 = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
    print(f"PPL_0: {ppl_0:.2f}")

    # === A-C2-2, A-C2-3 === Train Phase 2 (baseline lambda=0.2)
    connector, lm_model, metrics = train_phase2(
        connector, lm_model, tokenizer,
        clip_features_train, clip_features_test,
        vqa_train, vqa_val, alpaca_texts,
        device, lam=0.2, batch_size=32, lr=5e-4, num_epochs=1,
        grad_accum_steps=4
    )

    # Save Phase 2 checkpoint
    os.makedirs("weights", exist_ok=True)
    torch.save(connector.state_dict(), "weights/connector_phaseA2.pt")
    lm_model.save_pretrained("weights/lm_phaseA2")
    print("Phase 2 weights saved.")

    # === A-C2-4 === Lambda ablation (uncomment to run — takes a while)
    # NOTE: The ablation reloads models fresh for each lambda value.
    # Uncomment the line below to run the full ablation:
    # results = run_lambda_ablation(clip_features_train, clip_features_test,
    #                               vqa_train, vqa_val, alpaca_texts, device)

    print("\n" + "="*60)
    print("A-C2 Phase 2 COMPLETE")
    print("="*60)
