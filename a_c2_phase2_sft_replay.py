# ============================================================
# A-C2: Phase 2 - SFT with Language Replay
# ============================================================
# USAGE (Google Colab):
#
#   Baseline run (lambda=0.2, saves weights):
#   !python a_c2_phase2_sft_replay.py
#
#   Ablation only (loads Phase 1 weights, no baseline re-training):
#   !python a_c2_phase2_sft_replay.py --mode ablation
#
#   Baseline with a different lambda:
#   !python a_c2_phase2_sft_replay.py --lam 0.5
#
#   Smaller batch if still OOM:
#   !python a_c2_phase2_sft_replay.py --batch_size 8 --grad_accum 16
#
#   All options:
#   !python a_c2_phase2_sft_replay.py --help
#
# MODES:
#   baseline  (default): train once with --lam, save weights/connector_phaseA2.pt
#   ablation:            sweep lambda in {0, 0.05, 0.2, 0.5} from Phase 1 weights
# ============================================================

# [FIX-1] Must be set BEFORE any torch import
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
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
# === CLI argument parser ===
# ============================================================

def get_args():
    # Parse command-line arguments.
    # All training hyperparameters can be overridden without editing source code.
    #
    # From a Colab cell:
    #   !python a_c2_phase2_sft_replay.py --mode ablation
    #   !python a_c2_phase2_sft_replay.py --lam 0.5 --batch_size 8 --grad_accum 16
    p = argparse.ArgumentParser(description="A-C2: Phase 2 SFT with Language Replay")

    p.add_argument(
        "--mode", choices=["baseline", "ablation"], default="baseline",
        help=(
            "baseline (default): train once with --lam, save weights.  "
            "ablation: sweep lambda={0,0.05,0.2,0.5} from Phase 1 weights -- "
            "does NOT re-run the baseline."
        ),
    )
    p.add_argument("--lam",        type=float, default=0.2,
                   help="Replay lambda for baseline mode (default: 0.2)")
    p.add_argument("--batch_size", type=int,   default=16,
                   help="Per-step batch size (default: 16). Halve further if still OOM.")
    p.add_argument("--grad_accum", type=int,   default=8,
                   help="Gradient accumulation steps (default: 8).  "
                        "Effective batch = batch_size * grad_accum.")
    p.add_argument("--lr",         type=float, default=5e-4,
                   help="Learning rate (default: 5e-4)")
    p.add_argument("--epochs",     type=int,   default=1,
                   help="Training epochs (default: 1)")
    p.add_argument("--eval_every", type=int,   default=300,
                   help="Eval every N steps (default: 300)")
    p.add_argument("--connector_in",  default="weights/connector_phaseA1.pt",
                   help="Phase 1 connector to load")
    p.add_argument("--connector_out", default="weights/connector_phaseA2.pt",
                   help="Where to save the trained connector")
    p.add_argument("--lm_out",        default="weights/lm_phaseA2",
                   help="Where to save the trained LoRA LM")
    return p.parse_args()


# ============================================================
# === A-C2-1 === Apply LoRA
# ============================================================

def apply_lora(model, r=16, alpha=32):
    # A-C2-1: Apply LoRA (r=16, alpha=32) to the LM.
    # Targets: q_proj, k_proj, v_proj, o_proj. Dropout 0.05.
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
    # A-C2-2: Build [BOS emb, V, q embs, a embs, EOS].
    # Labels: -100 for prefix (BOS + V + question); answer+EOS receive gradient.
    B = clip_feats.shape[0]
    base = lm_model.get_base_model() if hasattr(lm_model, "get_base_model") else lm_model
    embed_layer = base.get_input_embeddings()

    bos_id  = base.config.bos_token_id or 0
    bos_emb = embed_layer(torch.full((B, 1), bos_id, dtype=torch.long, device=device))
    vis_tok = connector(clip_feats.to(device)).to(bos_emb.dtype)
    q_emb   = embed_layer(q_ids.to(device))
    a_emb   = embed_layer(a_ids.to(device))
    eos_id  = tokenizer.eos_token_id
    eos_emb = embed_layer(torch.full((B, 1), eos_id, dtype=torch.long, device=device))

    inputs_embeds = torch.cat([bos_emb, vis_tok, q_emb, a_emb, eos_emb], dim=1)

    prefix_len    = 1 + 49 + q_ids.shape[1]
    ignore_labels = torch.full((B, prefix_len), -100, dtype=torch.long, device=device)
    ans_labels    = a_ids.clone().to(device)
    ans_labels[a_mask == 0] = -100
    eos_labels    = torch.full((B, 1), eos_id, dtype=torch.long, device=device)
    labels        = torch.cat([ignore_labels, ans_labels, eos_labels], dim=1)

    attn = torch.cat([
        torch.ones(B, 1,  dtype=torch.long, device=device),
        torch.ones(B, 49, dtype=torch.long, device=device),
        q_mask.to(device),
        a_mask.to(device),
        torch.ones(B, 1,  dtype=torch.long, device=device),
    ], dim=1)

    return inputs_embeds, labels, attn


# ============================================================
# === A-C2-3 === Mixed training helpers
# ============================================================

def compute_alpaca_loss(lm_model, alpaca_batch, device):
    # A-C2-3: LM loss on Alpaca text -- NO visual tokens.
    input_ids, attn_mask = alpaca_batch
    outputs = lm_model(
        input_ids=input_ids.to(device),
        attention_mask=attn_mask.to(device),
        labels=input_ids.to(device),
    )
    return outputs.loss


def evaluate_vqa_accuracy(connector, lm_model, tokenizer, clip_features,
                           vqa_pairs, device, n_samples=200):
    # Greedy exact-match VQA accuracy on n_samples pairs.
    connector.eval(); lm_model.eval()
    base        = lm_model.get_base_model() if hasattr(lm_model, "get_base_model") else lm_model
    embed_layer = base.get_input_embeddings()
    correct     = 0
    total       = min(n_samples, len(vqa_pairs))

    with torch.no_grad():
        for i in range(total):
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

    connector.train(); lm_model.train()
    return correct / total if total > 0 else 0.0


# ============================================================
# === A-C2-2, A-C2-3 === Phase 2 training loop
# ============================================================

def train_phase2(connector, lm_model, tokenizer, clip_features_train,
                 clip_features_test, vqa_train, vqa_val, alpaca_texts,
                 device, lam=0.2, batch_size=16, lr=5e-4,
                 num_epochs=1, grad_accum_steps=8, eval_every=300):
    # A-C2: Phase 2 -- connector + LoRA + Alpaca replay.
    # L_mixed = L_VQA + lambda * L_LM
    # fp16 GradScaler, gradient accumulation, OneCycleLR (10% warmup).
    connector.train(); connector = connector.to(device)
    lm_model.train()

    all_params = (list(connector.parameters()) +
                  [p for p in lm_model.parameters() if p.requires_grad])
    optimizer  = torch.optim.Adam(all_params, lr=lr)

    vqa_loader    = DataLoader(VQADataset(clip_features_train, vqa_train, tokenizer),
                               batch_size=batch_size, shuffle=True,
                               num_workers=0, drop_last=True)
    alpaca_loader = DataLoader(AlpacaDataset(alpaca_texts, tokenizer),
                               batch_size=batch_size, shuffle=True,
                               num_workers=0, drop_last=True)

    total_steps = len(vqa_loader) * num_epochs
    scheduler   = OneCycleLR(optimizer, max_lr=lr,
                              total_steps=total_steps, pct_start=0.1)
    scaler      = torch.amp.GradScaler("cuda")
    alpaca_iter = iter(alpaca_loader)
    metrics     = {"losses": [], "vqa_acc": [], "ppl": []}
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        for vqa_batch in vqa_loader:
            clip_feats, q_ids, q_mask, a_ids, a_mask = vqa_batch

            with torch.amp.autocast("cuda", dtype=torch.float16):
                embeds, labels, attn = build_vqa_batch(
                    connector, clip_feats, q_ids, q_mask, a_ids, a_mask,
                    lm_model, tokenizer, device)
                loss_vqa = lm_model(inputs_embeds=embeds,
                                    attention_mask=attn, labels=labels).loss

                loss_lm = torch.tensor(0.0, device=device)
                if lam > 0:
                    try:
                        ab = next(alpaca_iter)
                    except StopIteration:
                        alpaca_iter = iter(alpaca_loader)
                        ab = next(alpaca_iter)
                    loss_lm = compute_alpaca_loss(lm_model, ab, device)

                loss = loss_vqa + lam * loss_lm

            scaler.scale(loss / grad_accum_steps).backward()

            if (global_step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(); scheduler.step()
                torch.cuda.empty_cache()   # [FIX-5]

            metrics["losses"].append(loss.item())
            global_step += 1

            if global_step % 50 == 0:
                print(f"  Step {global_step}/{total_steps}  "
                      f"L_vqa={loss_vqa.item():.4f}  "
                      f"L_lm={loss_lm.item():.4f}  "
                      f"L_total={loss.item():.4f}")

            if global_step % eval_every == 0:
                acc = evaluate_vqa_accuracy(connector, lm_model, tokenizer,
                                            clip_features_test, vqa_val, device)
                ppl = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
                metrics["vqa_acc"].append(acc); metrics["ppl"].append(ppl)
                print(f"  [Eval @ {global_step}] acc={acc:.3f}  PPL={ppl:.2f}")
                connector.train(); lm_model.train()

    return connector, lm_model, metrics


# ============================================================
# === A-C2-4 === Lambda ablation (standalone -- no baseline re-run)
# ============================================================

def run_lambda_ablation(clip_features_train, clip_features_test,
                        vqa_train, vqa_val, alpaca_texts, device, args):
    # A-C2-4: Sweep lambda in {0, 0.05, 0.2, 0.5}.
    # Each run starts fresh from connector_phaseA1.pt so the already-saved
    # baseline (connector_phaseA2.pt) is never re-trained.
    lambdas = [0.0, 0.05, 0.2, 0.5]
    labels  = ["No replay", "Weak", "Baseline", "Strong"]
    results = []

    lm_ref, tok = load_lm_model(device)
    ppl_0 = compute_ppl(lm_ref, tok, alpaca_texts[:100], device)
    print(f"PPL_0 = {ppl_0:.2f}")
    del lm_ref; torch.cuda.empty_cache()

    for lam, label in zip(lambdas, labels):
        print(f"\n{'='*40}\nLambda ablation: {label} (lambda={lam})\n{'='*40}")

        lm_model, tokenizer = load_lm_model(device)
        lm_model = apply_lora(lm_model)

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

        acc = evaluate_vqa_accuracy(connector, lm_model, tokenizer,
                                    clip_features_test, vqa_val, device)
        ppl = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
        R   = ppl / ppl_0
        results.append({"condition": label, "lambda": lam,
                        "vqa_acc": acc, "R": R, "ppl": ppl})
        print(f"  -> acc={acc:.3f}  R={R:.3f}  PPL={ppl:.2f}")

        del lm_model, connector; torch.cuda.empty_cache()

    print(f"\n{'='*60}\nTable 1: Lambda Ablation\n{'='*60}")
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
    # Load CLIP features, VQA splits, Alpaca, and the LM.
    # CLIP is deleted before the LM is loaded [FIX-3].
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

    if args.mode == "baseline":
        print("\n" + "="*60)
        print("A-C2: Baseline -- SFT with Language Replay")
        print("="*60)

        lm_model  = apply_lora(lm_model)
        connector = MLPConnector(clip_dim=768, lm_dim=960)
        connector.load_state_dict(torch.load(args.connector_in, map_location="cpu"))
        print(f"Loaded connector from {args.connector_in}")

        lora_p = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
        conn_p = sum(p.numel() for p in connector.parameters())
        total  = sum(p.numel() for p in lm_model.parameters())
        print(f"Trainable: {lora_p+conn_p:,} / {total:,} "
              f"({(lora_p+conn_p)/total*100:.2f}%)")

        ppl_0 = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
        print(f"PPL_0: {ppl_0:.2f}")

        connector, lm_model, metrics = train_phase2(
            connector, lm_model, tokenizer,
            clip_features_train, clip_features_test,
            vqa_train, vqa_val, alpaca_texts, device,
            lam=args.lam, batch_size=args.batch_size, lr=args.lr,
            num_epochs=args.epochs, grad_accum_steps=args.grad_accum,
            eval_every=args.eval_every,
        )

        os.makedirs("weights", exist_ok=True)
        torch.save(connector.state_dict(), args.connector_out)
        lm_model.save_pretrained(args.lm_out)
        print(f"\nSaved connector -> {args.connector_out}")
        print(f"Saved LM        -> {args.lm_out}")

    elif args.mode == "ablation":
        print("\n" + "="*60)
        print("A-C2: Lambda Ablation (no baseline re-training)")
        print("="*60)

        # Ablation loads its own fresh LM per lambda -- free this one
        del lm_model; torch.cuda.empty_cache()

        run_lambda_ablation(
            clip_features_train, clip_features_test,
            vqa_train, vqa_val, alpaca_texts,
            device, args,
        )

    print("\n" + "="*60)
    print("A-C2 COMPLETE")
    print("="*60)