"""
# ============================================================
# B-C0: Dataset Generation and Model Loading
# ============================================================
# USAGE (Google Colab):
#   !pip install transformers peft accelerate datasets torchvision matplotlib scikit-learn tqdm umap-learn
#   !python b_c0_data_and_models.py
# ============================================================

PSEUDOCODE:
    1. Generate 16x16 synthetic images for 6 classes (spiral, triangle, circle, cross, checkerboard, gradient)
    2. 1000 per class = 6000 total. 80/20 stratified split -> 4800 train / 1200 val
    3. Build 4 VQA templates x all images -> 19200 train / 4800 val pairs
    4. Build image-gen dataset: 1 prompt per image, 3 cycling templates -> 4800 train / 1200 val
    5. Load SmolLM2-360M in float16, confirm vocab=49152, d_LM=960
    6. Load 1000 Alpaca examples, compute PPL_0
"""

import os
import argparse
import math
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# ============================================================
# Constants
# ============================================================
CLASSES = ['spiral', 'triangle', 'circle', 'cross', 'checkerboard', 'gradient']
IMG_SIZE = 16
LM_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
SEED = 42

# Class properties for VQA
GEOMETRIC = {'triangle', 'circle', 'cross', 'checkerboard'}
NON_GEOMETRIC = {'spiral', 'gradient'}
SYMMETRY_AXES = {
    'spiral': '0', 'triangle': '3', 'circle': 'infinite',
    'cross': '4', 'checkerboard': '4', 'gradient': '1',
}

IMAGEGEN_TEMPLATES = [
    "Generate an image of a {}.",
    "Create a {} pattern.",
    "Draw a {}.",
]


def get_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="B-C0: Data & Model Setup")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_per_class", type=int, default=1000)
    p.add_argument("--n_alpaca", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# === B-C0.1 === Synthetic Image Generation
# ============================================================

def _draw_spiral(img, cx, cy, r, color):
    """Draw a spiral on a 16x16 image."""
    for t in np.linspace(0, 4 * np.pi, 200):
        radius = r * t / (4 * np.pi)
        x = int(cx + radius * np.cos(t))
        y = int(cy + radius * np.sin(t))
        if 0 <= x < IMG_SIZE and 0 <= y < IMG_SIZE:
            img[:, y, x] = color


def _draw_triangle(img, cx, cy, r, color):
    """Draw a filled equilateral triangle."""
    angles = [np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3]
    pts = [(cx + r * np.cos(a), cy - r * np.sin(a)) for a in angles]
    for y in range(IMG_SIZE):
        for x in range(IMG_SIZE):
            # Point-in-triangle test using barycentric coords
            def sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
            d1 = sign((x, y), pts[0], pts[1])
            d2 = sign((x, y), pts[1], pts[2])
            d3 = sign((x, y), pts[2], pts[0])
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            if not (has_neg and has_pos):
                img[:, y, x] = color


def _draw_circle(img, cx, cy, r, color):
    """Draw a filled circle."""
    for y in range(IMG_SIZE):
        for x in range(IMG_SIZE):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                img[:, y, x] = color


def _draw_cross(img, cx, cy, r, color, thickness=2):
    """Draw a cross/plus shape."""
    half_t = max(1, thickness // 2)
    for y in range(IMG_SIZE):
        for x in range(IMG_SIZE):
            in_h = abs(y - cy) <= half_t and abs(x - cx) <= r
            in_v = abs(x - cx) <= half_t and abs(y - cy) <= r
            if in_h or in_v:
                img[:, y, x] = color


def _draw_checkerboard(img, sq_size):
    """Draw a checkerboard pattern."""
    c1 = np.array([0.2, 0.2, 0.8])
    c2 = np.array([0.8, 0.8, 0.2])
    for y in range(IMG_SIZE):
        for x in range(IMG_SIZE):
            if ((x // sq_size) + (y // sq_size)) % 2 == 0:
                img[:, y, x] = torch.tensor(c1, dtype=torch.float32)
            else:
                img[:, y, x] = torch.tensor(c2, dtype=torch.float32)


def _draw_gradient(img, direction, c1, c2):
    """Draw a linear color gradient."""
    for i in range(IMG_SIZE):
        t = i / max(IMG_SIZE - 1, 1)
        color = (1 - t) * np.array(c1) + t * np.array(c2)
        for j in range(IMG_SIZE):
            if direction == 'horizontal':
                img[:, j, i] = torch.tensor(color, dtype=torch.float32)
            else:
                img[:, i, j] = torch.tensor(color, dtype=torch.float32)


def generate_single_image(class_name, rng):
    """
    B-C0.1-1: Generate a single 16x16 synthetic image for a given class.

    Args:
        class_name: one of the 6 class names
        rng: numpy random generator for variation

    Returns:
        img: tensor of shape (3, 16, 16) with values in [0, 1]
    """
    img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
    bg = rng.uniform(0.0, 0.15, size=3)
    img[:] = torch.tensor(bg, dtype=torch.float32).view(3, 1, 1)

    cx, cy = IMG_SIZE // 2 + rng.randint(-1, 2), IMG_SIZE // 2 + rng.randint(-1, 2)
    color = torch.tensor(rng.uniform(0.5, 1.0, size=3), dtype=torch.float32)

    if class_name == 'spiral':
        r = rng.uniform(5, 7)
        _draw_spiral(img, cx, cy, r, color)
    elif class_name == 'triangle':
        r = rng.uniform(4, 6)
        _draw_triangle(img, cx, cy, r, color)
    elif class_name == 'circle':
        r = rng.uniform(3, 6)
        _draw_circle(img, cx, cy, r, color)
    elif class_name == 'cross':
        r = rng.randint(3, 6)
        _draw_cross(img, cx, cy, r, color, thickness=rng.randint(2, 4))
    elif class_name == 'checkerboard':
        sq = rng.choice([2, 4])
        _draw_checkerboard(img, sq)
    elif class_name == 'gradient':
        d = rng.choice(['horizontal', 'vertical'])
        c1 = rng.uniform(0.1, 0.5, size=3).tolist()
        c2 = rng.uniform(0.5, 1.0, size=3).tolist()
        _draw_gradient(img, d, c1, c2)

    return img.clamp(0, 1)


def generate_dataset(n_per_class=1000, seed=42):
    """
    B-C0.1-1: Generate full synthetic dataset with 80/20 stratified split.

    Args:
        n_per_class: images per class (default 1000)
        seed: random seed

    Returns:
        train_images: tensor (4800, 3, 16, 16)
        train_labels: tensor (4800,)
        val_images: tensor (1200, 3, 16, 16)
        val_labels: tensor (1200,)
    """
    rng = np.random.RandomState(seed)
    all_images, all_labels = [], []

    for cls_idx, cls_name in enumerate(tqdm(CLASSES, desc="Generating classes")):
        for _ in range(n_per_class):
            img = generate_single_image(cls_name, rng)
            all_images.append(img)
            all_labels.append(cls_idx)

    all_images = torch.stack(all_images)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    # 80/20 stratified split
    train_imgs, train_lbls, val_imgs, val_lbls = [], [], [], []
    n_train = int(n_per_class * 0.8)
    for c in range(len(CLASSES)):
        mask = all_labels == c
        c_imgs = all_images[mask]
        perm = torch.randperm(len(c_imgs), generator=torch.Generator().manual_seed(seed))
        c_imgs = c_imgs[perm]
        train_imgs.append(c_imgs[:n_train])
        train_lbls.extend([c] * n_train)
        val_imgs.append(c_imgs[n_train:])
        val_lbls.extend([c] * (n_per_class - n_train))

    train_images = torch.cat(train_imgs)
    train_labels = torch.tensor(train_lbls, dtype=torch.long)
    val_images = torch.cat(val_imgs)
    val_labels = torch.tensor(val_lbls, dtype=torch.long)

    print(f"Train: {train_images.shape}, Val: {val_images.shape}")
    return train_images, train_labels, val_images, val_labels


# ============================================================
# === B-C0.2 === VQA and Image-Gen Datasets
# ============================================================

def get_vqa_answer(label, template_idx, image_idx):
    """
    B-C0.2-3: Get question and answer for a VQA template.

    Args:
        label: integer class label (0-5)
        template_idx: VQA template index (0-3)
        image_idx: for deterministic yes/no variation

    Returns:
        question, answer: strings
    """
    cls = CLASSES[label]

    if template_idx == 0:
        return "What shape is in this image?", cls
    elif template_idx == 1:
        if image_idx % 2 == 0:
            return f"Is there a {cls}?", "yes"
        else:
            wrong = [c for c in CLASSES if c != cls]
            wrong_cls = wrong[image_idx % len(wrong)]
            return f"Is there a {wrong_cls}?", "no"
    elif template_idx == 2:
        ans = "geometric" if cls in GEOMETRIC else "non-geometric"
        return "Geometric or non-geometric?", ans
    elif template_idx == 3:
        return "How many axes of symmetry?", SYMMETRY_AXES[cls]


def build_vqa_dataset(labels):
    """
    B-C0.2-3: Build VQA dataset — 4 templates x all images.

    Args:
        labels: tensor of class labels

    Returns:
        vqa_pairs: list of dicts {image_idx, question, answer, template_idx, class_name}
    """
    pairs = []
    for i in tqdm(range(len(labels)), desc="Building VQA"):
        lbl = labels[i].item()
        for t in range(4):
            q, a = get_vqa_answer(lbl, t, i)
            pairs.append({
                'image_idx': i, 'question': q, 'answer': a,
                'template_idx': t, 'class_name': CLASSES[lbl],
            })
    print(f"Built {len(pairs)} VQA pairs")
    return pairs


def build_imagegen_dataset(labels):
    """
    B-C0.2-4: Build image-gen dataset — 1 prompt per image, 3 cycling templates.

    Args:
        labels: tensor of class labels

    Returns:
        imagegen_pairs: list of dicts {image_idx, prompt, class_name}
    """
    pairs = []
    for i in range(len(labels)):
        cls = CLASSES[labels[i].item()]
        tmpl = IMAGEGEN_TEMPLATES[i % len(IMAGEGEN_TEMPLATES)]
        pairs.append({
            'image_idx': i,
            'prompt': tmpl.format(cls),
            'class_name': cls,
        })
    print(f"Built {len(pairs)} image-gen pairs")
    return pairs


# ============================================================
# === B-C0.3 === Model and Alpaca
# ============================================================

def load_lm_model(device='cuda'):
    """
    B-C0.3-5: Load SmolLM2-360M in float16; confirm vocab 49152, d_LM=960.

    Returns:
        model, tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LM_MODEL_NAME, torch_dtype=torch.float16)
    model = model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # B-C0 warning: left padding for decoder-only batched generation
    tokenizer.padding_side = "left"

    d_lm = model.config.hidden_size
    vocab = model.config.vocab_size
    print(f"SmolLM2 loaded: d_LM={d_lm}, vocab={vocab}, dtype={next(model.parameters()).dtype}")
    assert d_lm == 960 and vocab == 49152
    return model, tokenizer


def load_alpaca_data(n_examples=1000):
    """
    B-C0.3-6: Load 1000 Alpaca examples for language replay.

    Returns:
        list of text strings
    """
    from datasets import load_dataset
    set_seed(SEED)
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    indices = np.random.choice(len(ds), size=n_examples, replace=False)
    subset = ds.select(indices)

    texts = []
    for ex in subset:
        t = ex.get('instruction', '')
        if ex.get('input'): t += " " + ex['input']
        if ex.get('output'): t += " " + ex['output']
        texts.append(t.strip())

    print(f"Loaded {len(texts)} Alpaca examples")
    return texts


def compute_ppl(model, tokenizer, texts, device='cuda', max_length=512):
    """
    B-C0.3-6: Compute perplexity on a list of texts.

    Returns:
        ppl: float
    """
    model.eval()
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing PPL", leave=False):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            ids = enc['input_ids'].to(device)
            out = model(input_ids=ids, labels=ids)
            n = ids.shape[1]
            total_nll += out.loss.item() * n
            total_tokens += n
    return math.exp(total_nll / total_tokens)


def infinite_loader(dataloader):
    """Wrap a dataloader to cycle infinitely (for multi-loader training)."""
    for batch in itertools.cycle(dataloader):
        yield batch


# ============================================================
# Simple Dataset classes
# ============================================================

class SyntheticImageDataset(Dataset):
    """Dataset wrapper for synthetic images."""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class AlpacaDataset(Dataset):
    """Dataset for Alpaca language replay."""
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], return_tensors="pt",
                             truncation=True, max_length=self.max_length,
                             padding='max_length')
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = get_args()
    set_seed(args.seed)
    device = args.device

    # === B-C0.1 ===
    print("=" * 60)
    print("B-C0.1: Synthetic Image Generation")
    print("=" * 60)
    train_imgs, train_lbls, val_imgs, val_lbls = generate_dataset(args.n_per_class, args.seed)

    # B-C0.1-2: Verify with 6x5 grid
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(6, 5, figsize=(10, 12))
    for r, cls in enumerate(CLASSES):
        mask = train_lbls == r
        samples = train_imgs[mask][:5]
        for c in range(5):
            axes[r, c].imshow(samples[c].permute(1, 2, 0).numpy())
            axes[r, c].axis('off')
            if c == 0: axes[r, c].set_ylabel(cls)
    plt.suptitle("B-C0.1-2: Synthetic Dataset Verification")
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/b_c0_dataset_grid.png", dpi=150)
    plt.close()
    print("Saved verification grid to plots/b_c0_dataset_grid.png")

    # === B-C0.2 ===
    print("\n" + "=" * 60)
    print("B-C0.2: VQA and Image-Gen Datasets")
    print("=" * 60)
    vqa_train = build_vqa_dataset(train_lbls)
    vqa_val = build_vqa_dataset(val_lbls)
    ig_train = build_imagegen_dataset(train_lbls)
    ig_val = build_imagegen_dataset(val_lbls)
    print(f"VQA: {len(vqa_train)} train, {len(vqa_val)} val")
    print(f"ImageGen: {len(ig_train)} train, {len(ig_val)} val")

    # === B-C0.3 ===
    print("\n" + "=" * 60)
    print("B-C0.3: Model and Alpaca")
    print("=" * 60)
    lm_model, tokenizer = load_lm_model(device)
    alpaca_texts = load_alpaca_data(args.n_alpaca)
    ppl_0 = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
    print(f"PPL_0 = {ppl_0:.2f} (expected ~8)")

    # Clean up GPU
    del lm_model
    torch.cuda.empty_cache()
    print("\nB-C0 COMPLETE")
