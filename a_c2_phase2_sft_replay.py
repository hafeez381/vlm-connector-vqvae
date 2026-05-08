# ============================================================
# A-C2: Phase 2 - SFT with Language Replay
# ============================================================
# USAGE:
#   Baseline:  !python a_c2_phase2_sft_replay.py
#   Ablation:  !python a_c2_phase2_sft_replay.py --mode ablation
#   Custom:    !python a_c2_phase2_sft_replay.py --batch_size 8 --grad_accum 16
#   Help:      !python a_c2_phase2_sft_replay.py --help
#
# PSEUDOCODE:
#   1. Load data, models, and Phase 1 connector weights
#   2. Apply LoRA (r=16, alpha=32) to LM q/k/v/o_proj layers
#   3. Verify trainable params < 1% of total
#   4. For each training step:
#      a. VQA batch: build [BOS emb, V, q embs, a embs, EOS]
#         Labels: -100 for prefix (BOS+V+question); answer+EOS get gradient
#      b. Alpaca batch: standard text-only LM loss (NO visual tokens!)
#      c. L_mixed = L_VQA + lambda * L_LM
#      d. Backward with GradScaler, accumulate gradients every 4 steps
#   5. Use OneCycleLR with 10% warmup
#   6. Eval every 300 steps: 200 VQA pairs accuracy + 100 Alpaca PPL
#   7. Run lambda ablation: lambda in {0, 0.05, 0.2, 0.5}
#   8. Save checkpoint
#
# ============================================================


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

from a_c0_data_and_models import (
    set_seed, SEED, get_cifar10_subsets, build_vqa_dataset,
    load_clip_model, load_lm_model, load_alpaca_data,
    extract_clip_features, preprocess_images_for_clip, compute_ppl,
    VQADataset, AlpacaDataset, CLIPImageProcessor, CLIP_MODEL_NAME,
    CIFAR10_CLASSES
)
from a_c1_phase1_connector import MLPConnector, count_params, compute_norm_ratio


# ============================================================
# === CLI args
# ============================================================

def get_args():
    # All training hyperparameters are controllable from the command line
    # so you never need to edit source code between runs.
    #
    # Examples:
    #   !python a_c2_phase2_sft_replay.py                          # baseline lam=0.2
    #   !python a_c2_phase2_sft_replay.py --mode ablation          # lambda sweep
    #   !python a_c2_phase2_sft_replay.py --lam 0.5                # different lambda
    #   !python a_c2_phase2_sft_replay.py --batch_size 16 --grad_accum 8  # if OOM
    p = argparse.ArgumentParser(description="A-C2: Phase 2 SFT with Language Replay")
    p.add_argument("--mode", choices=["baseline", "ablation"], default="baseline",
                   help="baseline: train once with --lam, save weights.  "
                        "ablation: sweep lambda={0,0.05,0.2,0.5} from Phase 1 weights.")
    p.add_argument("--lam",        type=float, default=0.2)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--grad_accum", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--epochs",     type=int,   default=1)
    p.add_argument("--eval_every", type=int,   default=300)
    p.add_argument("--connector_in",  default="weights/connector_phaseA1.pt")
    p.add_argument("--connector_out", default="weights/connector_phaseA2.pt")
    p.add_argument("--lm_out",        default="weights/lm_phaseA2")
    return p.parse_args()


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
        r=r, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
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
    base = lm_model.get_base_model() if hasattr(lm_model, "get_base_model") else lm_model
    embed_layer = base.get_input_embeddings()

    # 1. BOS embedding
    bos_id  = base.config.bos_token_id or 0
    bos_emb = embed_layer(torch.full((B, 1), bos_id, dtype=torch.long, device=device))

    # 2. Visual tokens through connector
    vis_tok = connector(clip_feats.to(device)).to(bos_emb.dtype)   # (B, 49, 960)

    # 3. Question / answer embeddings
    q_emb   = embed_layer(q_ids.to(device))                        # (B, Lq, 960)
    a_emb   = embed_layer(a_ids.to(device))                        # (B, La, 960)

    # 4. EOS embedding
    eos_id  = tokenizer.eos_token_id
    eos_emb = embed_layer(torch.full((B, 1), eos_id, dtype=torch.long, device=device))

    # Concatenate: [BOS, V1:49, question, answer, EOS]
    inputs_embeds = torch.cat([bos_emb, vis_tok, q_emb, a_emb, eos_emb], dim=1)

    # Labels: -100 for BOS + visual + question; answer IDs + EOS for gradient
    prefix_len    = 1 + 49 + q_ids.shape[1]
    ignore_labels = torch.full((B, prefix_len), -100, dtype=torch.long, device=device)
    ans_labels    = a_ids.clone().to(device)
    ans_labels[a_mask == 0] = -100
    eos_labels    = torch.full((B, 1), eos_id, dtype=torch.long, device=device)
    labels        = torch.cat([ignore_labels, ans_labels, eos_labels], dim=1)

    # Attention mask: 1 for BOS + visual + question + answer + EOS
    attn = torch.cat([
        torch.ones(B, 1,  dtype=torch.long, device=device),   # BOS
        torch.ones(B, 49, dtype=torch.long, device=device),   # visual
        q_mask.to(device),
        a_mask.to(device),
        torch.ones(B, 1,  dtype=torch.long, device=device),   # EOS
    ], dim=1)

    return inputs_embeds, labels, attn


# ============================================================
# === A-C2-3 === Mixed training step helpers
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
        loss: scalar LM loss on Alpaca text (0.0 if non-finite, to skip bad batches)
    """
    input_ids, attn_mask = alpaca_batch
    # autocast here keeps fp16 consistent and avoids overflow-induced NaN
    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = lm_model(input_ids=input_ids.to(device),
                           attention_mask=attn_mask.to(device),
                           labels=input_ids.to(device))
    loss = outputs.loss
    # Guard: return 0 instead of NaN so it never poisons L_total
    if not torch.isfinite(loss):
        return torch.tensor(0.0, device=device, requires_grad=False)
    return loss


def evaluate_vqa_accuracy(connector, lm_model, tokenizer, clip_features,
                           vqa_pairs, device, n_samples=200):
    """
    A-C2-3 (eval): Evaluate VQA exact-match accuracy on a subset.

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
    connector.eval(); lm_model.eval()
    base        = lm_model.get_base_model() if hasattr(lm_model, "get_base_model") else lm_model
    embed_layer = base.get_input_embeddings()
    correct     = 0
    total       = min(n_samples, len(vqa_pairs))

    eval_bar = tqdm(range(total), desc="  Eval", leave=False, unit="pair")
    with torch.no_grad():
        for i in eval_bar:
            pair   = vqa_pairs[i]
            feat   = clip_features[pair["image_idx"]]
            bos_id = base.config.bos_token_id or 0
            bos    = embed_layer(torch.tensor([[bos_id]], device=device))
            vis    = connector(feat.unsqueeze(0).to(device)).to(bos.dtype)
            q_emb  = embed_layer(
                tokenizer(pair["question"], return_tensors="pt")["input_ids"].to(device)
            )
            cur = torch.cat([bos, vis, q_emb], dim=1)
            gen = []
            for _ in range(10):
                nxt = lm_model(inputs_embeds=cur).logits[:, -1, :].argmax(dim=-1)
                if nxt.item() == tokenizer.eos_token_id:
                    break
                gen.append(nxt.item())
                cur = torch.cat([cur, embed_layer(nxt.unsqueeze(0))], dim=1)
            pred = tokenizer.decode(gen, skip_special_tokens=True).strip().lower()
            if pred == pair["answer"].strip().lower():
                correct += 1
            eval_bar.set_postfix(acc=f"{correct/(i+1):.3f}")

    connector.train(); lm_model.train()
    return correct / total if total > 0 else 0.0


# ============================================================
# === A-C2-2, A-C2-3 === Phase 2 Training Loop
# ============================================================

def train_phase2(connector, lm_model, tokenizer, clip_features_train,
                 clip_features_test, vqa_train, vqa_val, alpaca_texts,
                 device, lam=0.2, batch_size=32, lr=5e-4,
                 num_epochs=1, grad_accum_steps=4, eval_every=300):
    """
    A-C2: Phase 2 training — connector + LoRA + Alpaca replay.

    A-C2-3: L_mixed = L_VQA + lambda * L_LM.
    Uses GradScaler for fp16, gradient accumulation x4, OneCycleLR (10% warmup).

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
    connector.train(); connector = connector.to(device)
    lm_model.train()

    # A-C2-3: Combine connector + LoRA params into one optimizer
    all_params = (list(connector.parameters()) +
                  [p for p in lm_model.parameters() if p.requires_grad])
    optimizer  = torch.optim.Adam(all_params, lr=lr)

    # Create dataloaders
    vqa_loader    = DataLoader(VQADataset(clip_features_train, vqa_train, tokenizer),
                               batch_size=batch_size, shuffle=True,
                               num_workers=0, drop_last=True)
    alpaca_loader = DataLoader(AlpacaDataset(alpaca_texts, tokenizer),
                               batch_size=batch_size, shuffle=True,
                               num_workers=0, drop_last=True)

    total_steps = len(vqa_loader) * num_epochs

    # A-C2-3: OneCycleLR with 10% warmup
    scheduler   = OneCycleLR(optimizer, max_lr=lr,
                              total_steps=total_steps, pct_start=0.1)
    # A-C2-3: GradScaler for mixed precision
    scaler      = torch.amp.GradScaler("cuda")
    alpaca_iter = iter(alpaca_loader)
    metrics     = {"losses": [], "vqa_acc": [], "ppl": []}
    global_step = 0
    start_time  = time.time()
    optimizer.zero_grad()

    epoch_bar = tqdm(range(num_epochs), desc="Epoch", unit="epoch", position=0)

    for epoch in epoch_bar:
        step_bar = tqdm(vqa_loader, desc=f"  Phase2 e{epoch+1}",
                        unit="step", position=1, leave=False,
                        dynamic_ncols=True)

        running_vqa = running_lm = running_total = 0.0

        for vqa_batch in step_bar:
            clip_feats, q_ids, q_mask, a_ids, a_mask = vqa_batch

            # === A-C2-2 === VQA forward pass
            with torch.amp.autocast("cuda", dtype=torch.float16):
                embeds, labels, attn = build_vqa_batch(
                    connector, clip_feats, q_ids, q_mask, a_ids, a_mask,
                    lm_model, tokenizer, device)
                loss_vqa = lm_model(inputs_embeds=embeds,
                                    attention_mask=attn, labels=labels).loss

                # === A-C2-3 === Alpaca replay loss (if lambda > 0)
                loss_lm = torch.tensor(0.0, device=device)
                if lam > 0:
                    try:
                        ab = next(alpaca_iter)
                    except StopIteration:
                        alpaca_iter = iter(alpaca_loader)
                        ab = next(alpaca_iter)
                    loss_lm = compute_alpaca_loss(lm_model, ab, device)

                # Guard: skip step entirely if either loss is non-finite
                if not torch.isfinite(loss_vqa) or not torch.isfinite(loss_lm):
                    step_bar.write(f"  [WARNING] Non-finite loss at step {global_step+1}, skipping.")
                    global_step += 1
                    continue

                # A-C2-3: Mixed loss
                loss = loss_vqa + lam * loss_lm

            # A-C2-3: Scale loss for gradient accumulation
            scaler.scale(loss / grad_accum_steps).backward()

            # Step optimizer every grad_accum_steps
            # NOTE: optimizer.step() MUST come before scheduler.step()
            if (global_step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()           # correct order: after optimizer
                torch.cuda.empty_cache()   # [FIX-5]

            # Exponential moving average for smooth postfix display
            running_vqa   = 0.95 * running_vqa   + 0.05 * loss_vqa.item()
            running_lm    = 0.95 * running_lm    + 0.05 * loss_lm.item()
            running_total = 0.95 * running_total + 0.05 * loss.item()

            step_bar.set_postfix(
                vqa=f"{running_vqa:.3f}",
                lm=f"{running_lm:.3f}",
                tot=f"{running_total:.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            metrics["losses"].append(loss.item())
            global_step += 1

            # A-C2-3: Eval every 300 steps
            if global_step % eval_every == 0:
                step_bar.write(f"\n  [Eval @ step {global_step}/{total_steps}]")
                acc = evaluate_vqa_accuracy(connector, lm_model, tokenizer,
                                            clip_features_test, vqa_val, device)
                ppl = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
                metrics["vqa_acc"].append(acc); metrics["ppl"].append(ppl)
                step_bar.write(f"  VQA acc={acc:.3f}  PPL={ppl:.2f}\n")
                connector.train(); lm_model.train()

        elapsed = time.time() - start_time
        epoch_bar.set_postfix(loss=f"{running_total:.3f}",
                              elapsed=f"{elapsed/60:.1f}min")

    print(f"\nPhase 2 done in {(time.time()-start_time)/60:.1f} min")
    return connector, lm_model, metrics


# ============================================================
# === A-C2-4 === Lambda ablation
# ============================================================

def run_lambda_ablation(clip_features_train, clip_features_test,
                        vqa_train, vqa_val, alpaca_texts, device, args):
    """
    A-C2-4: Run lambda ablation from Table 1.

    Tests lambda in {0.0, 0.05, 0.2, 0.5} and reports VQA accuracy and R.
    R = PPL_fine / PPL_0 (forgetting ratio).

    Each run starts fresh from connector_phaseA1.pt — the already-saved
    baseline (connector_phaseA2.pt) is NOT re-trained.

    Returns:
        results: list of dicts with lambda, vqa_acc, R values
    """
    lambdas = [0.0, 0.05, 0.2, 0.5]
    labels  = ["No replay", "Weak", "Baseline", "Strong"]
    results = []

    # Compute PPL_0 on a fresh un-fine-tuned model
    lm_ref, tok = load_lm_model(device)
    ppl_0 = compute_ppl(lm_ref, tok, alpaca_texts[:100], device)
    print(f"PPL_0 = {ppl_0:.2f}")
    del lm_ref; torch.cuda.empty_cache()

    ablation_bar = tqdm(zip(lambdas, labels), total=len(lambdas),
                        desc="Ablation", unit="run")
    for lam, label in ablation_bar:
        ablation_bar.set_description(f"Ablation [{label}]")
        print(f"\n{'='*40}\nAblation: {label} (lambda={lam})\n{'='*40}")

        # Fresh LM + LoRA for each lambda value
        lm_model, tokenizer = load_lm_model(device)
        lm_model = apply_lora(lm_model)

        # Fresh connector from Phase 1 weights
        connector = MLPConnector(clip_dim=768, lm_dim=960)
        connector.load_state_dict(torch.load(args.connector_in, map_location="cpu"))

        connector, lm_model, _ = train_phase2(
            connector, lm_model, tokenizer,
            clip_features_train, clip_features_test,
            vqa_train, vqa_val, alpaca_texts, device,
            lam=lam, batch_size=args.batch_size, lr=args.lr,
            num_epochs=args.epochs, grad_accum_steps=args.grad_accum,
            eval_every=args.eval_every,
        )

        # Final evaluation
        acc = evaluate_vqa_accuracy(connector, lm_model, tokenizer,
                                    clip_features_test, vqa_val, device)
        ppl = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
        R   = ppl / ppl_0

        results.append({"condition": label, "lambda": lam,
                        "vqa_acc": acc, "R": R, "ppl": ppl})
        ablation_bar.write(f"  [{label}] acc={acc:.3f}  R={R:.3f}  PPL={ppl:.2f}")

        del lm_model, connector; torch.cuda.empty_cache()

    # Print summary table
    print(f"\n{'='*60}\nTable 1: Lambda Ablation Results\n{'='*60}")
    print(f"{'Condition':<12} {'Lambda':<8} {'VQA acc (%)':<14} {'R':<8}")
    print("-"*42)
    for r in results:
        print(f"{r['condition']:<12} {r['lambda']:<8.2f} "
              f"{r['vqa_acc']*100:<14.1f} {r['R']:<8.3f}")
    return results


# ============================================================
# === Shared data loading (used by both modes) ===
# ============================================================

def load_all_data(device):
    """
    Load CLIP features, VQA splits, Alpaca texts, and the LM.

    Memory sequencing (prevents CLIP + LM coexisting on GPU):
      1. Load CLIP, extract all features                [FIX-3]
      2. del train_pixels, test_pixels                  [FIX-2]
      3. del clip_model                                 [FIX-3]
      4. Load LM
    """
    train_subset, test_subset = get_cifar10_subsets()

    # [FIX-3] Extract features first, then free CLIP before loading the LM
    clip_model, clip_processor = load_clip_model(device)

    train_pixels, _ = preprocess_images_for_clip(train_subset, clip_processor)
    clip_features_train = extract_clip_features(train_pixels, clip_model, device)
    del train_pixels; torch.cuda.empty_cache()           # [FIX-2]

    test_pixels, _ = preprocess_images_for_clip(test_subset, clip_processor)
    clip_features_test = extract_clip_features(test_pixels, clip_model, device)
    del test_pixels; torch.cuda.empty_cache()            # [FIX-2]

    del clip_model; torch.cuda.empty_cache()             # [FIX-3]

    lm_model, tokenizer = load_lm_model(device)

    vqa_train    = build_vqa_dataset(train_subset)
    vqa_val      = build_vqa_dataset(test_subset)
    alpaca_texts = load_alpaca_data()

    return (clip_features_train, clip_features_test,
            vqa_train, vqa_val, alpaca_texts,
            lm_model, tokenizer)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    args   = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(SEED)

    print(f"\nMode       : {args.mode}")
    print(f"Lambda     : {args.lam}")
    print(f"Batch size : {args.batch_size}  x  grad_accum {args.grad_accum} "
          f"= effective batch {args.batch_size * args.grad_accum}")
    print(f"LR         : {args.lr}    Epochs: {args.epochs}")

    (clip_features_train, clip_features_test,
     vqa_train, vqa_val, alpaca_texts,
     lm_model, tokenizer) = load_all_data(device)

    if torch.cuda.is_available():
        print(f"\nVRAM after data load: "
              f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    # ------------------------------------------------------------------
    if args.mode == "baseline":
    # ------------------------------------------------------------------
        print("\n" + "="*60)
        print("A-C2: Phase 2 -- SFT with Language Replay")
        print("="*60)

        # === A-C2-1 === Apply LoRA, load Phase 1 connector
        lm_model  = apply_lora(lm_model)
        connector = MLPConnector(clip_dim=768, lm_dim=960)
        connector.load_state_dict(torch.load(args.connector_in, map_location="cpu"))
        print(f"Loaded connector from {args.connector_in}")

        # A-C2-1: Verify trainable < 1%
        lora_p = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
        conn_p = sum(p.numel() for p in connector.parameters())
        total  = sum(p.numel() for p in lm_model.parameters())
        print(f"LoRA trainable:   {lora_p:,}")
        print(f"Connector params: {conn_p:,}")
        print(f"Total trainable:  {lora_p+conn_p:,} / {total:,} "
              f"({(lora_p+conn_p)/total*100:.2f}%)")

        # Compute PPL_0 before fine-tuning
        ppl_0 = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
        print(f"PPL_0: {ppl_0:.2f}")

        # === A-C2-2, A-C2-3 === Train Phase 2
        connector, lm_model, metrics = train_phase2(
            connector, lm_model, tokenizer,
            clip_features_train, clip_features_test,
            vqa_train, vqa_val, alpaca_texts, device,
            lam=args.lam, batch_size=args.batch_size, lr=args.lr,
            num_epochs=args.epochs, grad_accum_steps=args.grad_accum,
            eval_every=args.eval_every,
        )

        # Save Phase 2 checkpoint
        os.makedirs("weights", exist_ok=True)
        torch.save(connector.state_dict(), args.connector_out)
        lm_model.save_pretrained(args.lm_out)
        print(f"\nSaved connector -> {args.connector_out}")
        print(f"Saved LM        -> {args.lm_out}")

    # ------------------------------------------------------------------
    elif args.mode == "ablation":
    # ------------------------------------------------------------------
        print("\n" + "="*60)
        print("A-C2: Lambda Ablation (no baseline re-training)")
        print("="*60)

        # === A-C2-4 === Lambda sweep
        # Ablation loads its own fresh LM per lambda -- free this one first
        del lm_model; torch.cuda.empty_cache()
        run_lambda_ablation(
            clip_features_train, clip_features_test,
            vqa_train, vqa_val, alpaca_texts, device, args,
        )

    print("\n" + "="*60)
    print("A-C2 COMPLETE")
    print("="*60)