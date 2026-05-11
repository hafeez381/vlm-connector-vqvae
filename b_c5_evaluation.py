"""
# ============================================================
# B-C5: Evaluation — VQA, Image Generation, Error Analysis
# ============================================================
# USAGE (Google Colab):
#   !python b_c5_evaluation.py --mode vqa
#   !python b_c5_evaluation.py --mode imagegen
#   !python b_c5_evaluation.py --mode logits
#   !python b_c5_evaluation.py --mode qualitative
# ============================================================

PSEUDOCODE:
    1. VQA accuracy: overall, per template, per class + baselines + confusion matrix
    2. Image generation: 12 images (2/class) with image logit mask, decode via VQ-VAE
    3. Logit masking: before/after histograms, temperature sweep T={0.5,1.0,1.5}
    4. Qualitative error analysis: 6 examples with top-5 logits
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from tqdm.auto import tqdm

from b_c0_data_and_models import (
    set_seed, SEED, generate_dataset, build_vqa_dataset, build_imagegen_dataset,
    load_lm_model, load_alpaca_data, compute_ppl, CLASSES, LM_MODEL_NAME
)
from b_c1_vqvae import VQVAE
from b_c2_vocab_expansion import (
    OverlayEmbedding, setup_expanded_model, V_TXT, N_NEW, N_VISUAL
)
from b_c3_tokenisation import (
    pre_encode_images, codebook_idx_to_token_id, IMAGE_START_ID, IMAGE_END_ID
)
from b_c4_mixed_training import apply_lora
from peft import PeftModel


def get_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="B-C5: Evaluation")
    p.add_argument("--mode", choices=["vqa", "imagegen", "logits", "qualitative", "all"],
                   default="all")
    p.add_argument("--vqvae_path", type=str, default="weights/vqvae_best.pt")
    p.add_argument("--lm_path", type=str, default="weights/lm_phaseB")
    p.add_argument("--overlay_path", type=str, default="weights/lm_phaseB/overlay.pt")
    p.add_argument("--n_vqa_eval", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ============================================================
# === B-C5.1 === VQA Accuracy
# ============================================================

def evaluate_vqa_full(lm_model, tokenizer, val_indices, vqa_val, device,
                      n_samples=500):
    """
    B-C5.1-1: Full VQA evaluation — overall, per template, per class.
    Uses KV-cache for fast generation.

    Args:
        lm_model: fine-tuned PEFT model
        tokenizer: tokenizer
        val_indices: (N, 16) pre-encoded codebook indices
        vqa_val: list of VQA dicts
        device: torch device
        n_samples: max pairs to evaluate

    Returns:
        results dict with overall_acc, per_template, per_class, all_preds
    """
    lm_model.eval()
    if hasattr(lm_model, 'gradient_checkpointing_disable'):
        lm_model.gradient_checkpointing_disable()

    template_correct = defaultdict(int)
    template_total = defaultdict(int)
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    all_preds = []
    total_correct = 0
    total = min(n_samples, len(vqa_val))

    with torch.no_grad():
        for i in tqdm(range(total), desc="VQA Evaluation"):
            pair = vqa_val[i]
            vis_idx = val_indices[pair['image_idx']]
            vis_ids = codebook_idx_to_token_id(vis_idx).tolist()

            bos = tokenizer.bos_token_id or 0
            q_ids = tokenizer.encode(pair['question'], add_special_tokens=False)
            prompt_ids = [bos, IMAGE_START_ID] + vis_ids + [IMAGE_END_ID] + q_ids
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            # Greedy decode
            gen_ids = []
            for _ in range(10):
                out = lm_model(input_ids=input_ids, use_cache=False)
                nid = out.logits[:, -1, :].argmax(dim=-1)
                if nid.item() == tokenizer.eos_token_id:
                    break
                gen_ids.append(nid.item())
                input_ids = torch.cat([input_ids, nid.unsqueeze(0)], dim=1)

            pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()
            true = pair['answer'].strip().lower()
            correct = pred == true

            all_preds.append({
                'question': pair['question'], 'true': true, 'pred': pred,
                'correct': correct, 'template': pair['template_idx'],
                'class': pair['class_name'],
            })

            if correct:
                total_correct += 1
                template_correct[pair['template_idx']] += 1
                class_correct[pair['class_name']] += 1
            template_total[pair['template_idx']] += 1
            class_total[pair['class_name']] += 1

    overall = total_correct / max(total, 1)

    tmpl_names = ["What shape?", "Is there a...?", "Geometric?", "Symmetry axes?"]
    per_template = {}
    for t in range(4):
        per_template[tmpl_names[t]] = template_correct[t] / max(template_total[t], 1)

    per_class = {}
    for cls in CLASSES:
        per_class[cls] = class_correct[cls] / max(class_total[cls], 1)

    print(f"\nOverall VQA accuracy: {overall*100:.1f}%")
    print("\nPer-template:")
    for name, acc in per_template.items():
        print(f"  {name:<20} {acc*100:.1f}%")
    print("\nPer-class:")
    for cls, acc in per_class.items():
        print(f"  {cls:<15} {acc*100:.1f}%")

    return {'overall': overall, 'per_template': per_template,
            'per_class': per_class, 'all_preds': all_preds}


def baselines(vqa_val):
    """
    B-C5.1-2: Text-only and majority-class baselines.
    """
    # Majority vote per template
    tmpl_answers = defaultdict(list)
    for p in vqa_val:
        tmpl_answers[p['template_idx']].append(p['answer'].lower())

    majority = {}
    for t, answers in tmpl_answers.items():
        majority[t] = Counter(answers).most_common(1)[0][0]

    correct = sum(1 for p in vqa_val if majority[p['template_idx']] == p['answer'].lower())
    maj_acc = correct / len(vqa_val)
    print(f"\nB-C5.1-2: Majority baseline accuracy: {maj_acc*100:.1f}%")
    print(f"  Majority answers: {majority}")
    return maj_acc


def confusion_matrix_shape(all_preds):
    """
    B-C5.1-3: Confusion matrix for "What shape?" template.
    """
    shape_preds = [p for p in all_preds if p['template'] == 0]
    if not shape_preds:
        print("No 'What shape?' predictions found.")
        return

    true_labels = [p['true'] for p in shape_preds]
    pred_labels = [p['pred'] for p in shape_preds]
    all_labels = sorted(set(true_labels + pred_labels))

    cm = np.zeros((len(CLASSES), len(all_labels)), dtype=int)
    label_to_idx_true = {c: i for i, c in enumerate(CLASSES)}
    label_to_idx_pred = {c: i for i, c in enumerate(all_labels)}

    for t, p in zip(true_labels, pred_labels):
        if t in label_to_idx_true and p in label_to_idx_pred:
            cm[label_to_idx_true[t], label_to_idx_pred[p]] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("B-C5.1-3: Confusion Matrix (What shape?)")
    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("plots/b_c5_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion matrix to plots/b_c5_confusion_matrix.png")


# ============================================================
# === B-C5.2 === Image Generation
# ============================================================

def generate_images(lm_model, tokenizer, vqvae, device, n_per_class=2):
    """
    B-C5.2-4: Generate images autoregressively with image logit mask.
    Decode via frozen VQ-VAE decoder.

    Args:
        lm_model: fine-tuned model
        tokenizer: tokenizer
        vqvae: frozen VQ-VAE (for decoding)
        device: torch device
        n_per_class: images per class

    Returns:
        generated_images: list of (class_name, image_tensor) tuples
    """
    lm_model.eval()
    if hasattr(lm_model, 'gradient_checkpointing_disable'):
        lm_model.gradient_checkpointing_disable()

    vqvae = vqvae.to(device)
    vqvae.eval()

    # B-C5.3-5: Image logit mask — only allow visual token IDs during generation
    text_mask = torch.ones(V_TXT + N_NEW + 10, device=device) * float('-inf')
    # Allow visual tokens (IDs V_TXT+2 to V_TXT+257) and </image> (V_TXT+1)
    for k in range(N_VISUAL):
        text_mask[V_TXT + 2 + k] = 0
    text_mask[V_TXT + 1] = 0  # </image>

    results = []
    with torch.no_grad():
        for cls_name in tqdm(CLASSES, desc="Generating images"):
            for j in range(n_per_class):
                prompt = f"Generate an image of a {cls_name}."
                p_ids = tokenizer.encode(prompt, add_special_tokens=False)
                bos = tokenizer.bos_token_id or 0
                input_ids = [bos] + p_ids + [IMAGE_START_ID]
                input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

                gen_visual_ids = []
                for _ in range(16):  # generate 16 visual tokens
                    out = lm_model(input_ids=input_ids, use_cache=False)
                    logits = out.logits[:, -1, :]
                    # Apply image logit mask
                    masked_logits = logits + text_mask[:logits.shape[-1]]
                    nid = masked_logits.argmax(dim=-1)
                    gen_visual_ids.append(nid.item())
                    input_ids = torch.cat([input_ids, nid.unsqueeze(0)], dim=1)

                # Convert to codebook indices and decode
                cb_indices = torch.tensor(
                    [vid - V_TXT - 2 for vid in gen_visual_ids],
                    dtype=torch.long, device=device
                ).clamp(0, N_VISUAL - 1)
                img = vqvae.decode_indices(cb_indices.unsqueeze(0))
                results.append((cls_name, img[0].cpu()))

    vqvae = vqvae.cpu()
    torch.cuda.empty_cache()

    # Plot
    fig, axes = plt.subplots(len(CLASSES), n_per_class, figsize=(4*n_per_class, 4*len(CLASSES)))
    for i, (cls_name, img) in enumerate(results):
        r, c = i // n_per_class, i % n_per_class
        axes[r, c].imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
        axes[r, c].set_title(f"{cls_name} #{c+1}")
        axes[r, c].axis('off')
    plt.suptitle("B-C5.2-4: Generated Images")
    plt.tight_layout()
    plt.savefig("plots/b_c5_generated_images.png", dpi=150)
    plt.close()
    print("Saved generated images to plots/b_c5_generated_images.png")
    return results


# ============================================================
# === B-C5.3 === Logit Masking and Decoding
# ============================================================

def logit_histograms(lm_model, tokenizer, val_indices, device):
    """
    B-C5.3-5: Show before/after logit histograms for one example.
    """
    lm_model.eval()
    vis_ids = codebook_idx_to_token_id(val_indices[0]).tolist()
    bos = tokenizer.bos_token_id or 0
    prompt = "Generate an image of a spiral."
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([[bos] + p_ids + [IMAGE_START_ID]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = lm_model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, -1, :].cpu().float().numpy()

    # Create mask
    mask = np.full(len(logits), float('-inf'))
    for k in range(N_VISUAL):
        if V_TXT + 2 + k < len(mask):
            mask[V_TXT + 2 + k] = 0
    if V_TXT + 1 < len(mask):
        mask[V_TXT + 1] = 0
    masked = logits + mask

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(logits[np.isfinite(logits)], bins=100, alpha=0.7)
    ax1.set_title("B-C5.3-5: Before Masking")
    ax1.set_xlabel("Logit value")
    ax2.hist(masked[np.isfinite(masked)], bins=100, alpha=0.7, color='orange')
    ax2.set_title("After Image Logit Mask")
    ax2.set_xlabel("Logit value")
    plt.tight_layout()
    plt.savefig("plots/b_c5_logit_histograms.png", dpi=150)
    plt.close()
    print("Saved logit histograms to plots/b_c5_logit_histograms.png")


def temperature_sweep(lm_model, tokenizer, vqvae, device):
    """
    B-C5.3-6: Temperature sweep T in {0.5, 1.0, 1.5}.
    """
    lm_model.eval()
    vqvae = vqvae.to(device)
    vqvae.eval()

    temps = [0.5, 1.0, 1.5]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    text_mask = torch.ones(V_TXT + N_NEW + 10, device=device) * float('-inf')
    for k in range(N_VISUAL):
        text_mask[V_TXT + 2 + k] = 0
    text_mask[V_TXT + 1] = 0

    for ti, T in enumerate(temps):
        bos = tokenizer.bos_token_id or 0
        prompt = "Generate an image of a circle."
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([[bos] + p_ids + [IMAGE_START_ID]],
                                 dtype=torch.long, device=device)

        gen_ids = []
        with torch.no_grad():
            for _ in range(16):
                out = lm_model(input_ids=input_ids, use_cache=False)
                logits = out.logits[:, -1, :] / T
                logits = logits + text_mask[:logits.shape[-1]]
                probs = torch.softmax(logits.float(), dim=-1)
                nid = torch.multinomial(probs, 1).squeeze(-1)
                gen_ids.append(nid.item())
                input_ids = torch.cat([input_ids, nid.unsqueeze(0)], dim=1)

        cb_idx = torch.tensor([g - V_TXT - 2 for g in gen_ids],
                              dtype=torch.long, device=device).clamp(0, N_VISUAL-1)
        img = vqvae.decode_indices(cb_idx.unsqueeze(0))[0].cpu()
        axes[ti].imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
        axes[ti].set_title(f"T={T}")
        axes[ti].axis('off')

    plt.suptitle("B-C5.3-6: Temperature Sweep")
    plt.savefig("plots/b_c5_temperature_sweep.png", dpi=150)
    plt.close()

    vqvae = vqvae.cpu()
    torch.cuda.empty_cache()
    print("B-C5.3-7: Note — 1D raster prediction on 2D grid can cause spatial "
          "coherence issues: adjacent rows may not match because the model "
          "predicts tokens left-to-right, top-to-bottom without explicit 2D "
          "spatial awareness.")


# ============================================================
# === B-C5.4 === Qualitative Error Analysis
# ============================================================

def qualitative_analysis(lm_model, tokenizer, val_indices, vqa_val, device, n=500):
    """
    B-C5.4-8: 6 examples (2 correct VQA, 2 failures, 2 gen images) with top-5 logits.
    """
    lm_model.eval()
    correct_examples = []
    failure_examples = []

    with torch.no_grad():
        for i in tqdm(range(min(n, len(vqa_val))), desc="Collecting examples", leave=False):
            pair = vqa_val[i]
            vis_idx = val_indices[pair['image_idx']]
            vis_ids = codebook_idx_to_token_id(vis_idx).tolist()
            bos = tokenizer.bos_token_id or 0
            q_ids = tokenizer.encode(pair['question'], add_special_tokens=False)
            prompt = [bos, IMAGE_START_ID] + vis_ids + [IMAGE_END_ID] + q_ids
            input_ids = torch.tensor([prompt], dtype=torch.long, device=device)

            out = lm_model(input_ids=input_ids, use_cache=False)
            logits = out.logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
            top5_probs, top5_ids = probs.topk(5)
            top5_tokens = [(tokenizer.decode([tid.item()]).strip(), p.item())
                          for tid, p in zip(top5_ids, top5_probs)]

            pred_id = logits.argmax().item()
            pred = tokenizer.decode([pred_id]).strip().lower()
            true = pair['answer'].strip().lower()

            ex = {'Q': pair['question'], 'true': true, 'pred': pred,
                  'class': pair['class_name'], 'top5': top5_tokens}

            if pred == true and len(correct_examples) < 2:
                correct_examples.append(ex)
            elif pred != true and len(failure_examples) < 2:
                failure_examples.append(ex)

            if len(correct_examples) >= 2 and len(failure_examples) >= 2:
                break

    print("\n" + "=" * 60)
    print("B-C5.4: Qualitative Error Analysis")
    print("=" * 60)
    for label, examples in [("CORRECT", correct_examples), ("FAILURES", failure_examples)]:
        print(f"\n--- {label} ---")
        for ex in examples:
            print(f"  Q: {ex['Q']}, Class: {ex['class']}")
            print(f"  True: {ex['true']}, Pred: {ex['pred']}")
            print(f"  Top-5: {ex['top5']}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = get_args()
    set_seed(args.seed)
    device = args.device

    print("=" * 60)
    print("B-C5: Evaluation")
    print("=" * 60)

    # Load data and pre-encode
    _, _, val_imgs, val_lbls = generate_dataset(seed=args.seed)
    vqa_val = build_vqa_dataset(val_lbls)

    vqvae = VQVAE(K=256, d=64)
    vqvae.load_state_dict(torch.load(args.vqvae_path, map_location="cpu"))
    val_indices = pre_encode_images(val_imgs, vqvae, device)
    vqvae = vqvae.cpu()
    torch.cuda.empty_cache()

    # Load fine-tuned LM
    lm_model, tokenizer = load_lm_model(device)
    overlay = OverlayEmbedding(n_new=N_NEW, d_lm=960)
    lm_model, hook_handle = setup_expanded_model(lm_model, tokenizer, overlay, device)

    if os.path.exists(args.overlay_path):
        overlay.load_state_dict(torch.load(args.overlay_path, map_location="cpu"))
        overlay = overlay.to(device)

    if os.path.exists(args.lm_path):
        # Load PEFT model
        base = lm_model
        lm_model = PeftModel.from_pretrained(base, args.lm_path)
        lm_model = lm_model.to(device)
        print("Loaded fine-tuned LoRA weights")
    else:
        print("WARNING: No fine-tuned weights found, using base model")

    os.makedirs("plots", exist_ok=True)

    run_all = args.mode == "all"

    if run_all or args.mode == "vqa":
        # === B-C5.1 ===
        results = evaluate_vqa_full(lm_model, tokenizer, val_indices, vqa_val,
                                     device, n_samples=args.n_vqa_eval)
        baselines(vqa_val)
        confusion_matrix_shape(results['all_preds'])

    if run_all or args.mode == "imagegen":
        # === B-C5.2 ===
        vqvae_dec = VQVAE(K=256, d=64)
        vqvae_dec.load_state_dict(torch.load(args.vqvae_path, map_location="cpu"))
        generate_images(lm_model, tokenizer, vqvae_dec, device, n_per_class=2)
        del vqvae_dec
        torch.cuda.empty_cache()

    if run_all or args.mode == "logits":
        # === B-C5.3 ===
        logit_histograms(lm_model, tokenizer, val_indices, device)
        vqvae_dec = VQVAE(K=256, d=64)
        vqvae_dec.load_state_dict(torch.load(args.vqvae_path, map_location="cpu"))
        temperature_sweep(lm_model, tokenizer, vqvae_dec, device)
        del vqvae_dec
        torch.cuda.empty_cache()

    if run_all or args.mode == "qualitative":
        # === B-C5.4 ===
        qualitative_analysis(lm_model, tokenizer, val_indices, vqa_val, device)

    print("\nB-C5 COMPLETE")
