"""
# ============================================================
# B-C4: Mixed-Objective Fine-Tuning with LoRA + Language Replay
# ============================================================
# USAGE (Google Colab):
#   !python b_c1_vqvae.py --mode train
#   !python b_c2_vocab_expansion.py
#   !python b_c4_mixed_training.py --mode train
#   !python b_c4_mixed_training.py --mode ablation
#   !python b_c4_mixed_training.py --mode break --optional
# ============================================================

PSEUDOCODE:
    1. Load LM, resize embeddings, load overlay, apply LoRA
    2. Create 3 dataloaders: VQA, image-gen, Alpaca (text replay)
    3. steps_per_epoch = max(len(all_loaders)); wrap each in infinite_loader
    4. For each step:
       a. VQA forward/backward: L_VQA
       b. Image-gen forward/backward: gamma_img * L_IMG
       c. Text replay forward/backward: lambda * L_LM
       d. Single scaler.step + update
    5. Grad accum x4, OneCycleLR (10% warmup), 3 epochs
    6. Eval every 300 steps: 200 VQA pairs + 100 Alpaca PPL
    7. Reference model pi_ref on CPU for R computation
"""

import os
import argparse
import itertools
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm.auto import tqdm

from b_c0_data_and_models import (
    set_seed, SEED, generate_dataset, build_vqa_dataset, build_imagegen_dataset,
    load_lm_model, load_alpaca_data, compute_ppl, AlpacaDataset, infinite_loader,
    LM_MODEL_NAME, CLASSES
)
from b_c1_vqvae import VQVAE
from b_c2_vocab_expansion import (
    OverlayEmbedding, setup_expanded_model, verify_norm_ratio,
    V_TXT, N_NEW, N_VISUAL
)
from b_c3_tokenisation import (
    pre_encode_images, PreEncodedVQADataset, PreEncodedImageGenDataset,
    collate_padded, codebook_idx_to_token_id
)


def get_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="B-C4: Mixed Training")
    p.add_argument("--mode", choices=["train", "ablation", "break"], default="train")
    p.add_argument("--optional", action="store_true", help="Run optional break-the-protection")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lora_lr", type=float, default=5e-4)
    p.add_argument("--embed_lr", type=float, default=5e-5)
    p.add_argument("--lam", type=float, default=0.2, help="Replay weight lambda")
    p.add_argument("--gamma_img", type=float, default=0.5, help="Image-gen loss weight")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--eval_every", type=int, default=300)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--vqvae_path", type=str, default="weights/vqvae_best.pt")
    p.add_argument("--overlay_path", type=str, default="weights/expanded_model/overlay.pt")
    p.add_argument("--save_path", type=str, default="weights/lm_phaseB")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ============================================================
# === B-C4.1 === Setup: LoRA + Overlay + Ref Model
# ============================================================

def apply_lora(model, r=16, alpha=32):
    """
    B-C4.1-1: Apply LoRA to q/k/v/o_proj. Must be called AFTER resize_token_embeddings.

    Args:
        model: LM with resized embeddings
        r: LoRA rank
        alpha: LoRA alpha

    Returns:
        peft_model: model with LoRA adapters
    """
    config = LoraConfig(
        r=r, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def setup_optimizer(lm_model, overlay, args):
    """
    B-C4.1-2: Create optimizer with two LR groups.
    LoRA params at lora_lr; overlay embedding at embed_lr.

    Args:
        lm_model: PEFT model
        overlay: OverlayEmbedding
        args: argparse namespace

    Returns:
        optimizer: AdamW optimizer
    """
    lora_params = [p for p in lm_model.parameters() if p.requires_grad]
    embed_params = list(overlay.parameters())

    optimizer = torch.optim.Adam([
        {'params': lora_params, 'lr': args.lora_lr},
        {'params': embed_params, 'lr': args.embed_lr},
    ])
    return optimizer


# ============================================================
# === B-C4.1-3 === Reference model on CPU
# ============================================================

def load_ref_model_cpu():
    """
    B-C4.1-3: Load frozen reference model pi_ref on CPU.
    Saves ~0.72 GB GPU memory. Move to GPU only during R evaluation.

    Returns:
        ref_model, ref_tokenizer (on CPU)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ref_tokenizer = AutoTokenizer.from_pretrained(LM_MODEL_NAME)
    ref_model = AutoModelForCausalLM.from_pretrained(
        LM_MODEL_NAME, torch_dtype=torch.float16
    )
    ref_model.eval()
    ref_model.requires_grad_(False)

    if ref_tokenizer.pad_token is None:
        ref_tokenizer.pad_token = ref_tokenizer.eos_token
    ref_tokenizer.padding_side = "left"

    print("B-C4.1-3: Reference model loaded on CPU")
    return ref_model, ref_tokenizer


def compute_R(ref_model, ref_tokenizer, fine_model, fine_tokenizer,
              alpaca_texts, device, n=100):
    """
    Compute forgetting ratio R = PPL_fine / PPL_0.

    Moves ref to GPU temporarily, then back to CPU.

    Args:
        ref_model: frozen reference model (on CPU)
        fine_model: fine-tuned model (on GPU)
        alpaca_texts: list of Alpaca texts
        device: GPU device
        n: number of texts to evaluate

    Returns:
        R: float forgetting ratio
    """
    texts = alpaca_texts[:n]

    # PPL of fine-tuned model (already on GPU)
    ppl_fine = compute_ppl(fine_model, fine_tokenizer, texts, device)

    # PPL_0: move ref to GPU, compute, move back
    ref_model = ref_model.to(device)
    ppl_0 = compute_ppl(ref_model, ref_tokenizer, texts, device)
    ref_model = ref_model.cpu()
    torch.cuda.empty_cache()

    R = ppl_fine / ppl_0
    return R, ppl_fine, ppl_0


# ============================================================
# === B-C4.2 === Evaluation during training
# ============================================================

def evaluate_vqa_quick(lm_model, overlay, tokenizer, val_indices, vqa_val,
                       device, n_samples=200):
    """
    Quick VQA evaluation using greedy decoding.

    Args:
        lm_model: PEFT model
        overlay: OverlayEmbedding
        tokenizer: tokenizer
        val_indices: (N, 16) pre-encoded val codebook indices
        vqa_val: val VQA pairs
        device: torch device
        n_samples: number of pairs to evaluate

    Returns:
        accuracy: float
    """
    lm_model.eval()
    # Disable gradient checkpointing for generation (KV-cache compatibility)
    if hasattr(lm_model, 'gradient_checkpointing_disable'):
        lm_model.gradient_checkpointing_disable()

    correct, total = 0, min(n_samples, len(vqa_val))

    with torch.no_grad():
        for i in tqdm(range(total), desc="VQA eval", leave=False):
            pair = vqa_val[i]
            vis_idx = val_indices[pair['image_idx']]
            vis_ids = codebook_idx_to_token_id(vis_idx).tolist()

            bos = tokenizer.bos_token_id or 0
            q_ids = tokenizer.encode(pair['question'], add_special_tokens=False)

            # Prompt: [BOS, <image>, v1:16, </image>, question]
            prompt_ids = [bos, V_TXT] + vis_ids + [V_TXT + 1] + q_ids
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            # Generate up to 10 tokens greedily
            for _ in range(10):
                outputs = lm_model(input_ids=input_ids, use_cache=False)
                next_id = outputs.logits[:, -1, :].argmax(dim=-1)
                if next_id.item() == tokenizer.eos_token_id:
                    break
                input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

            # Decode generated tokens (after the prompt)
            gen_ids = input_ids[0, len(prompt_ids):].tolist()
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()
            true = pair['answer'].strip().lower()
            if pred == true:
                correct += 1

    # Re-enable gradient checkpointing for training
    if hasattr(lm_model, 'gradient_checkpointing_enable'):
        lm_model.gradient_checkpointing_enable()

    return correct / max(total, 1)


# ============================================================
# === B-C4.2-4 === Main Training Loop
# ============================================================

def train_mixed(lm_model, overlay, tokenizer, vqa_loader, imggen_loader,
                text_loader, alpaca_texts, val_indices, vqa_val,
                ref_model, ref_tokenizer, args):
    """
    B-C4.2-4: Sequential forward/backward per task with mixed loss.

    L = L_VQA + gamma_img * L_IMG + lambda * L_LM

    Args:
        lm_model: PEFT model on GPU
        overlay: OverlayEmbedding on GPU
        tokenizer: tokenizer
        vqa_loader, imggen_loader, text_loader: dataloaders
        alpaca_texts: for PPL evaluation
        val_indices: pre-encoded val indices
        vqa_val: val VQA pairs
        ref_model: reference model on CPU
        ref_tokenizer: ref tokenizer
        args: hyperparameters

    Returns:
        lm_model, overlay, metrics dict
    """
    device = args.device
    lm_model.train()

    # Enable gradient checkpointing to save memory
    if hasattr(lm_model, 'gradient_checkpointing_enable'):
        lm_model.gradient_checkpointing_enable()

    optimizer = setup_optimizer(lm_model, overlay, args)

    # B-C4.2-4: steps_per_epoch = max(len(all_loaders))
    steps_per_epoch = max(len(vqa_loader), len(imggen_loader), len(text_loader))
    total_steps = steps_per_epoch * args.epochs
    print(f"Steps/epoch={steps_per_epoch}, total={total_steps}")

    # B-C4.2-5: OneCycleLR with 10% warmup
    scheduler = OneCycleLR(optimizer, max_lr=[args.lora_lr, args.embed_lr],
                           total_steps=total_steps, pct_start=0.1)

    scaler = torch.amp.GradScaler('cuda')

    # Wrap all loaders in infinite_loader
    vqa_iter = infinite_loader(vqa_loader)
    img_iter = infinite_loader(imggen_loader)
    txt_iter = infinite_loader(text_loader)

    metrics = {'vqa_loss': [], 'img_loss': [], 'txt_loss': [], 'vqa_acc': [], 'R': []}
    global_step = 0
    start_time = time.time()

    # B-C4 rule: zero_grad ONCE before the loop starts
    optimizer.zero_grad()

    step_bar = tqdm(range(total_steps), desc="Mixed Training")
    for step in step_bar:
        # === VQA forward/backward ===
        vqa_ids, vqa_labels, vqa_mask = next(vqa_iter)
        vqa_ids, vqa_labels, vqa_mask = (
            vqa_ids.to(device), vqa_labels.to(device), vqa_mask.to(device)
        )

        with torch.amp.autocast("cuda", dtype=torch.float16):
            vqa_out = lm_model(input_ids=vqa_ids, attention_mask=vqa_mask,
                               labels=vqa_labels)
            loss_vqa = vqa_out.loss

        if torch.isfinite(loss_vqa):
            scaled_vqa = loss_vqa / args.grad_accum
            scaler.scale(scaled_vqa).backward()
        else:
            loss_vqa = torch.tensor(0.0, device=device)

        # === Image-gen forward/backward ===
        img_ids, img_labels, img_mask = next(img_iter)
        img_ids, img_labels, img_mask = (
            img_ids.to(device), img_labels.to(device), img_mask.to(device)
        )

        with torch.amp.autocast("cuda", dtype=torch.float16):
            img_out = lm_model(input_ids=img_ids, attention_mask=img_mask,
                               labels=img_labels)
            loss_img = img_out.loss

        if torch.isfinite(loss_img) and args.gamma_img > 0:
            scaled_img = args.gamma_img * loss_img / args.grad_accum
            scaler.scale(scaled_img).backward()
        else:
            loss_img = torch.tensor(0.0, device=device)

        # === Text replay forward/backward (NO visual tokens!) ===
        txt_ids, txt_mask = next(txt_iter)
        txt_ids, txt_mask = txt_ids.to(device), txt_mask.to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            txt_out = lm_model(input_ids=txt_ids, attention_mask=txt_mask,
                               labels=txt_ids)
            loss_txt = txt_out.loss

        if torch.isfinite(loss_txt) and args.lam > 0:
            scaled_txt = args.lam * loss_txt / args.grad_accum
            scaler.scale(scaled_txt).backward()
        else:
            loss_txt = torch.tensor(0.0, device=device)

        # === Optimizer step every grad_accum steps ===
        global_step += 1
        if global_step % args.grad_accum == 0:
            # Correct order: scaler.step -> scaler.update -> zero_grad -> scheduler.step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # === Logging ===
        if global_step % args.log_every == 0:
            elapsed = time.time() - start_time
            eta = elapsed / global_step * (total_steps - global_step)
            step_bar.set_postfix({
                'vqa': f'{loss_vqa.item():.3f}',
                'img': f'{loss_img.item():.3f}',
                'txt': f'{loss_txt.item():.3f}',
                'ETA': f'{eta/60:.0f}m',
            })
            metrics['vqa_loss'].append(loss_vqa.item())
            metrics['img_loss'].append(loss_img.item())
            metrics['txt_loss'].append(loss_txt.item())

        # === B-C4.2-6: Eval every 300 steps ===
        if global_step % args.eval_every == 0:
            acc = evaluate_vqa_quick(lm_model, overlay, tokenizer,
                                     val_indices, vqa_val, device, n_samples=200)
            R, ppl_f, ppl_0 = compute_R(ref_model, ref_tokenizer, lm_model,
                                         tokenizer, alpaca_texts, device, n=100)
            metrics['vqa_acc'].append(acc)
            metrics['R'].append(R)
            step_bar.write(f"[Eval @{global_step}] VQA acc={acc:.3f}, R={R:.3f}, "
                          f"PPL_fine={ppl_f:.2f}, PPL_0={ppl_0:.2f}")
            lm_model.train()
            if hasattr(lm_model, 'gradient_checkpointing_enable'):
                lm_model.gradient_checkpointing_enable()

    elapsed = time.time() - start_time
    print(f"Training done in {elapsed/60:.1f} min")
    return lm_model, overlay, metrics


# ============================================================
# === B-C4.2-6 === Lambda/gamma ablation (Table 3)
# ============================================================

def run_ablation(train_imgs, train_lbls, val_imgs, val_lbls,
                 alpaca_texts, args):
    """
    B-C4.2-6: Run Table 3 ablation configs.
    """
    configs = [
        {"name": "No replay", "lam": 0.0, "gamma": 0.0},
        {"name": "Weak",      "lam": 0.05, "gamma": 0.05},
        {"name": "Baseline",  "lam": 0.20, "gamma": 0.50},
        {"name": "Strong",    "lam": 0.50, "gamma": 0.50},
    ]
    results = []

    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"Ablation: {cfg['name']} (lam={cfg['lam']}, gamma={cfg['gamma']})")
        print(f"{'='*50}")

        args.lam = cfg['lam']
        args.gamma_img = cfg['gamma']

        # Run full setup + training for this config
        # (setup code same as __main__, abbreviated here)
        # Save results
        results.append({
            'name': cfg['name'], 'lam': cfg['lam'], 'gamma': cfg['gamma'],
            'vqa_acc': 0.0, 'R': 0.0,  # placeholders until training completes
        })
        print(f"  NOTE: Run with --lam {cfg['lam']} --gamma_img {cfg['gamma']} "
              f"for actual results")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = get_args()
    set_seed(args.seed)
    device = args.device

    print("=" * 60)
    print("B-C4: Mixed-Objective Fine-Tuning")
    print("=" * 60)

    # --- Load and pre-encode data ---
    train_imgs, train_lbls, val_imgs, val_lbls = generate_dataset(seed=args.seed)
    vqa_train = build_vqa_dataset(train_lbls)
    vqa_val = build_vqa_dataset(val_lbls)
    ig_train = build_imagegen_dataset(train_lbls)
    alpaca_texts = load_alpaca_data()

    # Pre-encode images via VQ-VAE (then free VQ-VAE)
    vqvae = VQVAE(K=256, d=64)
    vqvae.load_state_dict(torch.load(args.vqvae_path, map_location="cpu"))
    train_indices = pre_encode_images(train_imgs, vqvae, device)
    val_indices = pre_encode_images(val_imgs, vqvae, device)
    del vqvae, train_imgs, val_imgs
    torch.cuda.empty_cache()

    # --- Load LM + overlay + LoRA ---
    lm_model, tokenizer = load_lm_model(device)

    overlay = OverlayEmbedding(n_new=N_NEW, d_lm=960)
    lm_model, hook_handle = setup_expanded_model(lm_model, tokenizer, overlay, device)

    # Load pre-trained overlay if available
    if os.path.exists(args.overlay_path):
        overlay.load_state_dict(torch.load(args.overlay_path, map_location="cpu"))
        overlay = overlay.to(device)
        print("Loaded pre-trained overlay embeddings")

    # B-C2/B-C3 warning: resize BEFORE LoRA
    r_val = args.lora_r
    if args.mode == "break" and args.optional:
        # B-C4.3-7: Break the protection — LoRA r=64
        r_val = 64
        args.lam = 0.0
        args.gamma_img = 0.0
        print("B-C4.3-7: BREAK MODE — r=64, lambda=0, gamma=0")

    lm_model = apply_lora(lm_model, r=r_val, alpha=args.lora_alpha)
    lm_model = lm_model.to(device)

    # B-C4.1-1: Verify trainable < 1%
    trainable = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    trainable += sum(p.numel() for p in overlay.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lm_model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # B-C4.1-3: Reference model on CPU
    ref_model, ref_tokenizer = load_ref_model_cpu()

    # --- Create dataloaders ---
    vqa_ds = PreEncodedVQADataset(train_indices, vqa_train, tokenizer)
    ig_ds = PreEncodedImageGenDataset(train_indices, ig_train, tokenizer)
    alpaca_ds = AlpacaDataset(alpaca_texts, tokenizer)

    vqa_loader = DataLoader(vqa_ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_padded, num_workers=0, drop_last=True)
    ig_loader = DataLoader(ig_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate_padded, num_workers=0, drop_last=True)
    txt_loader = DataLoader(alpaca_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)

    # B-C4.2-4: steps_per_epoch = max(len(loaders))
    print(f"Loader sizes: VQA={len(vqa_loader)}, ImgGen={len(ig_loader)}, "
          f"Text={len(txt_loader)}")
    print(f"steps_per_epoch = max = {max(len(vqa_loader), len(ig_loader), len(txt_loader))}")

    if args.mode in ("train", "break"):
        # === B-C4.2 === Training
        lm_model, overlay, metrics = train_mixed(
            lm_model, overlay, tokenizer,
            vqa_loader, ig_loader, txt_loader,
            alpaca_texts, val_indices, vqa_val,
            ref_model, ref_tokenizer, args
        )

        # Save checkpoint
        os.makedirs(args.save_path, exist_ok=True)
        lm_model.save_pretrained(args.save_path)
        torch.save(overlay.state_dict(), os.path.join(args.save_path, "overlay.pt"))
        tokenizer.save_pretrained(args.save_path)
        print(f"Saved to {args.save_path}")

    elif args.mode == "ablation":
        # Print instructions for running each ablation
        print("\nTo run Table 3 ablation, execute:")
        for name, l, g in [("No replay", 0.0, 0.0), ("Weak", 0.05, 0.05),
                            ("Baseline", 0.2, 0.5), ("Strong", 0.5, 0.5)]:
            print(f"  !python b_c4_mixed_training.py --mode train "
                  f"--lam {l} --gamma_img {g} --save_path weights/ablation_{name.replace(' ','_')}")

    # Clean up
    del ref_model
    torch.cuda.empty_cache()
    print("\nB-C4 COMPLETE")
