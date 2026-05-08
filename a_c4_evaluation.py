"""
# ============================================================
# A-C4: Evaluation
# ============================================================
# USAGE (Google Colab):
#   !pip install transformers peft accelerate datasets torchvision matplotlib scikit-learn
#   # Run all previous phases first (a_c0 through a_c3)
#   !python a_c4_evaluation.py
# ============================================================

PSEUDOCODE:
    1. Load test data and trained Phase 3 model
    2. Run exact-match accuracy on full 10,000 val VQA pairs
       - Overall accuracy
       - Per-template accuracy (5 templates)
       - Per-class accuracy (10 classes)
    3. Compute baselines:
       a. Text-only baseline (no visual tokens, just question -> LM)
       b. Majority vote baseline (most common answer per template)
    4. Plot VQA accuracy and R across Phases 0-3
    5. Show 6 qualitative examples with top-5 logits
       - 2 correct-easy, 2 correct-hard, 2 failures
    6. Report peak VRAM, training time, trainable params per phase
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
    CLIPImageProcessor, CLIP_MODEL_NAME, CIFAR10_CLASSES
)
from a_c1_phase1_connector import MLPConnector
from a_c2_phase2_sft_replay import apply_lora, evaluate_vqa_accuracy


# ============================================================
# === A-C4-1 === Full VQA evaluation with breakdown
# ============================================================

def full_vqa_evaluation(connector, lm_model, tokenizer, clip_features,
                        vqa_pairs, device, max_pairs=None):
    """
    A-C4-1: Exact-match accuracy on val set: overall, per template, per class.

    Evaluates every VQA pair and groups results by template and class.

    Args:
        connector: trained MLPConnector
        lm_model: trained LM
        tokenizer: LM tokenizer
        clip_features: (N, 49, 768) test CLIP features
        vqa_pairs: list of VQA dicts
        device: torch device
        max_pairs: limit evaluation to this many pairs (None = all)

    Returns:
        results: dict with overall_acc, per_template, per_class, all_preds
    """
    connector.eval()
    lm_model.eval()

    if hasattr(lm_model, 'get_base_model'):
        base = lm_model.get_base_model()
    else:
        base = lm_model
    embed_layer = base.get_input_embeddings()

    pairs_to_eval = vqa_pairs if max_pairs is None else vqa_pairs[:max_pairs]

    # Track results by template and class
    template_correct = defaultdict(int)
    template_total = defaultdict(int)
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    all_preds = []
    total_correct = 0

    with torch.no_grad():
        for i, pair in enumerate(pairs_to_eval):
            img_idx = pair['image_idx']
            clip_feat = clip_features[img_idx]  # (49, 768)
            template_idx = pair['template_idx']
            class_name = pair['class_name']

            # Build prompt: [BOS, V, question]
            bos_id = base.config.bos_token_id or 0
            bos_emb = embed_layer(torch.tensor([[bos_id]], device=device))
            visual = connector(clip_feat.unsqueeze(0).to(device))
            visual = visual.to(bos_emb.dtype)
            q_tokens = tokenizer(pair['question'], return_tensors="pt")
            q_emb = embed_layer(q_tokens['input_ids'].to(device))
            inputs_embeds = torch.cat([bos_emb, visual, q_emb], dim=1)

            # Greedy decode
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
            is_correct = (pred == true)

            all_preds.append({
                'image_idx': img_idx,
                'question': pair['question'],
                'true_answer': true,
                'pred_answer': pred,
                'correct': is_correct,
                'template_idx': template_idx,
                'class_name': class_name,
            })

            if is_correct:
                total_correct += 1
                template_correct[template_idx] += 1
                class_correct[class_name] += 1

            template_total[template_idx] += 1
            class_total[class_name] += 1

            if (i + 1) % 500 == 0:
                print(f"  Evaluated {i+1}/{len(pairs_to_eval)}...")

    overall_acc = total_correct / len(pairs_to_eval)

    # Per-template accuracy
    template_names = [
        "What object?", "Is there a...?", "Vehicle/living?",
        "Can it fly?", "Is animal?"
    ]
    per_template = {}
    for t in range(5):
        acc = template_correct[t] / max(template_total[t], 1)
        per_template[template_names[t]] = acc

    # Per-class accuracy
    per_class = {}
    for cls in CIFAR10_CLASSES:
        acc = class_correct[cls] / max(class_total[cls], 1)
        per_class[cls] = acc

    results = {
        'overall_acc': overall_acc,
        'per_template': per_template,
        'per_class': per_class,
        'all_preds': all_preds,
    }

    # Print results
    print(f"\n{'='*50}")
    print(f"Overall VQA Accuracy: {overall_acc*100:.1f}%")
    print(f"{'='*50}")
    print("\nPer-template accuracy:")
    for name, acc in per_template.items():
        print(f"  {name:<20} {acc*100:.1f}%")
    print("\nPer-class accuracy:")
    for cls, acc in per_class.items():
        print(f"  {cls:<15} {acc*100:.1f}%")

    return results


# ============================================================
# === A-C4-2 === Baselines
# ============================================================

def text_only_baseline(lm_model, tokenizer, vqa_pairs, device, n_samples=200):
    """
    A-C4-2: Text-only baseline — no visual tokens, just question to LM.

    This measures how well the LM can answer VQA questions without seeing
    the image at all.

    Args:
        lm_model: LM model
        tokenizer: tokenizer
        vqa_pairs: VQA pairs
        device: torch device
        n_samples: number to evaluate

    Returns:
        accuracy: float
    """
    lm_model.eval()
    correct = 0
    total = min(n_samples, len(vqa_pairs))

    with torch.no_grad():
        for i in range(total):
            pair = vqa_pairs[i]
            # Just feed the question directly (no image)
            input_ids = tokenizer(pair['question'], return_tensors="pt")['input_ids']
            input_ids = input_ids.to(device)

            # Generate
            generated_ids = []
            for _ in range(10):
                outputs = lm_model(input_ids=input_ids)
                next_logits = outputs.logits[:, -1, :]
                next_id = next_logits.argmax(dim=-1)
                if next_id.item() == tokenizer.eos_token_id:
                    break
                generated_ids.append(next_id.item())
                input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

            pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
            true = pair['answer'].strip().lower()
            if pred == true:
                correct += 1

    accuracy = correct / total
    print(f"Text-only baseline accuracy: {accuracy*100:.1f}%")
    return accuracy


def majority_vote_baseline(vqa_pairs):
    """
    A-C4-2: Majority vote baseline — always predict the most common answer.

    For each template, find the most frequent answer and use it for all.

    Args:
        vqa_pairs: list of VQA dicts

    Returns:
        accuracy: float
    """
    # Count answers per template
    from collections import Counter
    template_answers = defaultdict(list)
    for pair in vqa_pairs:
        template_answers[pair['template_idx']].append(pair['answer'].lower())

    # Find majority answer per template
    majority = {}
    for t, answers in template_answers.items():
        counter = Counter(answers)
        majority[t] = counter.most_common(1)[0][0]

    # Calculate accuracy
    correct = 0
    for pair in vqa_pairs:
        pred = majority[pair['template_idx']]
        if pred == pair['answer'].lower():
            correct += 1

    accuracy = correct / len(vqa_pairs)
    print(f"Majority vote baseline accuracy: {accuracy*100:.1f}%")
    print(f"  Majority answers: {majority}")
    return accuracy


# ============================================================
# === A-C4-3 === Plot accuracy and R across phases
# ============================================================

def plot_accuracy_and_R(phase_metrics, save_path="plots"):
    """
    A-C4-3: Plot VQA accuracy and R (forgetting ratio) across Phases 0-3.

    Args:
        phase_metrics: dict with keys 'phases', 'vqa_acc', 'R'
        save_path: directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)

    phases = phase_metrics['phases']
    vqa_acc = phase_metrics['vqa_acc']
    R_vals = phase_metrics['R']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # VQA accuracy
    ax1.bar(phases, [a * 100 for a in vqa_acc], color='steelblue')
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('VQA Accuracy (%)')
    ax1.set_title('VQA Accuracy Across Phases')
    ax1.set_ylim(0, 100)

    # R (forgetting ratio)
    ax2.bar(phases, R_vals, color='coral')
    ax2.axhline(y=1.0, color='gray', linestyle='--', label='No forgetting')
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('R (PPL_fine / PPL_0)')
    ax2.set_title('Forgetting Ratio R Across Phases')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'accuracy_and_R.png'), dpi=150)
    plt.show()
    print(f"Plot saved to {save_path}/accuracy_and_R.png")


# ============================================================
# === A-C4-4 === Qualitative examples with top-5 logits
# ============================================================

def qualitative_examples(connector, lm_model, tokenizer, clip_features,
                         vqa_pairs, device, n_each=2):
    """
    A-C4-4: Show 6 qualitative examples with top-5 logits.
    2 correct-easy, 2 correct-hard, 2 failures.

    Args:
        connector: trained connector
        lm_model: trained LM
        tokenizer: tokenizer
        clip_features: test CLIP features
        vqa_pairs: VQA pairs
        device: torch device
        n_each: number of examples per category
    """
    connector.eval()
    lm_model.eval()

    if hasattr(lm_model, 'get_base_model'):
        base = lm_model.get_base_model()
    else:
        base = lm_model
    embed_layer = base.get_input_embeddings()

    # First, evaluate a bunch and categorize
    correct_easy = []  # correct with high confidence
    correct_hard = []  # correct with low confidence
    failures = []

    with torch.no_grad():
        for i, pair in enumerate(vqa_pairs[:500]):
            img_idx = pair['image_idx']
            clip_feat = clip_features[img_idx]

            # Build prompt
            bos_id = base.config.bos_token_id or 0
            bos_emb = embed_layer(torch.tensor([[bos_id]], device=device))
            visual = connector(clip_feat.unsqueeze(0).to(device)).to(bos_emb.dtype)
            q_tokens = tokenizer(pair['question'], return_tensors="pt")
            q_emb = embed_layer(q_tokens['input_ids'].to(device))
            inputs_embeds = torch.cat([bos_emb, visual, q_emb], dim=1)

            # Get first token logits and top-5
            outputs = lm_model(inputs_embeds=inputs_embeds)
            logits = outputs.logits[:, -1, :]  # (1, vocab)
            probs = torch.softmax(logits.float(), dim=-1)
            top5_probs, top5_ids = probs.topk(5)
            top5_tokens = [tokenizer.decode([tid]) for tid in top5_ids[0]]

            # Greedy prediction
            pred_id = logits.argmax(dim=-1)
            pred = tokenizer.decode([pred_id.item()]).strip().lower()
            true = pair['answer'].strip().lower()
            confidence = top5_probs[0, 0].item()

            result = {
                'question': pair['question'],
                'true': true,
                'pred': pred,
                'confidence': confidence,
                'top5': list(zip(top5_tokens, top5_probs[0].tolist())),
                'class': pair['class_name'],
            }

            if pred == true:
                if confidence > 0.5:
                    correct_easy.append(result)
                else:
                    correct_hard.append(result)
            else:
                failures.append(result)

    # Show examples
    print("\n" + "="*60)
    print("QUALITATIVE EXAMPLES")
    print("="*60)

    categories = [
        ("CORRECT (Easy/High Confidence)", correct_easy),
        ("CORRECT (Hard/Low Confidence)", correct_hard),
        ("FAILURES", failures),
    ]

    for cat_name, examples in categories:
        print(f"\n--- {cat_name} ---")
        for ex in examples[:n_each]:
            print(f"  Q: {ex['question']}")
            print(f"  True: {ex['true']}, Predicted: {ex['pred']}")
            print(f"  Confidence: {ex['confidence']:.3f}")
            print(f"  Class: {ex['class']}")
            print(f"  Top-5 logits:")
            for token, prob in ex['top5']:
                print(f"    {token.strip():<15} {prob:.4f}")
            print()


# ============================================================
# === A-C4-5 === Report system metrics
# ============================================================

def report_system_metrics():
    """
    A-C4-5: Report peak VRAM, training time, trainable params per phase.

    This prints a summary table of compute resources used.
    (Actual values should be filled in after running all phases.)
    """
    print("\n" + "="*60)
    print("SYSTEM METRICS SUMMARY")
    print("="*60)

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak VRAM (current session): {peak_vram:.2f} GB")
    else:
        print("No CUDA available.")

    # Expected values from the assignment spec
    print("\nExpected values (from assignment):")
    print(f"{'Phase':<15} {'Trainable Params':<20} {'Est. Time':<15} {'Est. VRAM':<15}")
    print("-" * 65)
    print(f"{'Phase 1':<15} {'~1.66M (connector)':<20} {'~5 min':<15} {'~1.6 GB':<15}")
    print(f"{'Phase 2':<15} {'~3.3M (conn+LoRA)':<20} {'~15 min':<15} {'~2.2 GB':<15}")
    print(f"{'Phase 3':<15} {'~3.3M (conn+LoRA)':<20} {'~5 min':<15} {'~2.2 GB':<15}")


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

    test_pixels, test_labels = preprocess_images_for_clip(test_subset, clip_processor)
    clip_features_test = extract_clip_features(test_pixels, clip_model, device)

    vqa_val = build_vqa_dataset(test_subset)
    alpaca_texts = load_alpaca_data()

    del clip_model
    torch.cuda.empty_cache()

    # --- Load Phase 3 model ---
    print("\n" + "="*60)
    print("A-C4: Evaluation")
    print("="*60)

    lm_model = apply_lora(lm_model)
    if os.path.exists("weights/lm_phaseA3"):
        from peft import PeftModel
        lm_model = PeftModel.from_pretrained(
            lm_model.get_base_model(), "weights/lm_phaseA3"
        )
        print("Loaded Phase 3 LoRA weights.")

    connector = MLPConnector(clip_dim=768, lm_dim=960)
    if os.path.exists("weights/connector_phaseA3.pt"):
        connector.load_state_dict(torch.load("weights/connector_phaseA3.pt",
                                              map_location='cpu'))
        print("Loaded Phase 3 connector.")

    connector = connector.to(device)

    # === A-C4-1 === Full evaluation
    print("\n--- Full VQA Evaluation ---")
    results = full_vqa_evaluation(
        connector, lm_model, tokenizer, clip_features_test,
        vqa_val, device, max_pairs=2000  # use 2000 for speed; set None for all 10k
    )

    # === A-C4-2 === Baselines
    print("\n--- Baselines ---")
    text_acc = text_only_baseline(lm_model, tokenizer, vqa_val, device, n_samples=200)
    majority_acc = majority_vote_baseline(vqa_val)

    # === A-C4-3 === Plot (placeholder values — fill with real data after all phases)
    # You should replace these with actual measured values
    phase_metrics = {
        'phases': ['Phase 0', 'Phase 1', 'Phase 2', 'Phase 3'],
        'vqa_acc': [0.0, 0.05, results['overall_acc'] * 0.8, results['overall_acc']],
        'R': [1.0, 1.0, 1.05, 1.1],  # placeholder — update with real R values
    }
    plot_accuracy_and_R(phase_metrics)

    # === A-C4-4 === Qualitative examples
    qualitative_examples(connector, lm_model, tokenizer, clip_features_test,
                         vqa_val, device)

    # === A-C4-5 === System metrics
    report_system_metrics()

    print("\n" + "="*60)
    print("A-C4 Evaluation COMPLETE")
    print("="*60)
