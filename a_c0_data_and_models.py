"""
# ============================================================
# A-C0: Data Pipeline and Model Loading
# ============================================================
# USAGE (Google Colab):
#   !pip install transformers peft accelerate datasets torchvision matplotlib scikit-learn umap-learn
#   !python a_c0_data_and_models.py
#
# This script sets up all data and models needed for Part A.
# It is also imported by the other scripts as a utility module.
# ============================================================

PSEUDOCODE:
    1. Load CIFAR-10 train/test from torchvision
    2. Stratified sample: 1000/class train, 200/class test (seed=42)
    3. Preprocess images with CLIPImageProcessor (resize 32->224, CLIP norm)
    4. Verify CLIP normalisation differs from ImageNet defaults
    5. Generate 10,000 captions using 6 rotating templates per class
    6. Generate 50,000 train VQA pairs (5 templates x 10,000 images)
    7. Generate 10,000 val VQA pairs (5 templates x 2,000 images)
    8. Load CLIP model, freeze all params, confirm 50 tokens output
    9. Load SmolLM2-360M in float16, confirm d_LM=960, vocab=49152
    10. Load 1,000 Alpaca examples
    11. Compute baseline PPL_0 on Alpaca
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from transformers import CLIPModel, CLIPImageProcessor, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# Constants

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

VEHICLES = {'airplane', 'automobile', 'ship', 'truck'}
LIVING = {'bird', 'cat', 'deer', 'dog', 'frog', 'horse'}
CAN_FLY = {'airplane', 'bird'}
IS_ANIMAL = {'bird', 'cat', 'deer', 'dog', 'frog', 'horse'}

CAPTION_TEMPLATES = [
    "This is a {}.",
    "A photo of a {}.",
    "An image showing a {}.",
    "A {} is depicted here.",
    "The picture contains a {}.",
    "This image shows a {}.",
]

VQA_TEMPLATES = [
    # (question, answer_function, skill_name)
    ("What object is shown?", "recognition"),
    ("Is there a {}?", "binary"),
    ("Vehicle or living thing?", "abstraction"),
    ("Can it fly?", "attribute"),
    ("Is this an animal?", "category"),
]

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LM_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
SEED = 42


# ============================================================
# === A-C0.1 === CIFAR-10 Preprocessing
# ============================================================

def set_seed(seed=SEED):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cifar10_subsets(data_dir='./data'):
    """
    Load CIFAR-10 and create stratified subsets.

    A-C0.1-1: Stratified subsets (seed 42): 1000/class train, 200/class test.

    Returns:
        train_subset: list of (PIL_image, label) with 10,000 samples (1000/class)
        test_subset: list of (PIL_image, label) with 2,000 samples (200/class)
    """
    set_seed(SEED)

    # Load full CIFAR-10 datasets
    full_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    full_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)

    # Stratified sampling for train: 1000 per class
    train_subset = []
    class_indices = {c: [] for c in range(10)}
    for idx in range(len(full_train)):
        _, label = full_train[idx]
        class_indices[label].append(idx)

    for c in range(10):
        chosen = np.random.choice(class_indices[c], size=1000, replace=False)
        for idx in chosen:
            img, label = full_train[idx]
            train_subset.append((img, label))

    # Stratified sampling for test: 200 per class
    test_subset = []
    class_indices_test = {c: [] for c in range(10)}
    for idx in range(len(full_test)):
        _, label = full_test[idx]
        class_indices_test[label].append(idx)

    for c in range(10):
        chosen = np.random.choice(class_indices_test[c], size=200, replace=False)
        for idx in chosen:
            img, label = full_test[idx]
            test_subset.append((img, label))

    print(f"Train subset: {len(train_subset)} images ({len(train_subset)//10} per class)")
    print(f"Test subset: {len(test_subset)} images ({len(test_subset)//10} per class)")
    return train_subset, test_subset


def verify_clip_normalisation():
    """
    A-C0.1-2: Verify CLIP normalisation mean/std differs from ImageNet defaults.
    
    ImageNet defaults: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    CLIP uses its own normalisation values.
    """
    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_mean = processor.image_mean
    clip_std = processor.image_std

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    print("\n--- CLIP vs ImageNet Normalisation ---")
    print(f"CLIP mean:     {clip_mean}")
    print(f"ImageNet mean: {imagenet_mean}")
    print(f"CLIP std:      {clip_std}")
    print(f"ImageNet std:  {imagenet_std}")
    print(f"Are they different? {clip_mean != imagenet_mean or clip_std != imagenet_std}")


def preprocess_images_for_clip(image_list, clip_processor):
    """
    A-C0.1-1: Preprocess images with CLIPImageProcessor (resizes 32->224, CLIP normalisation).

    Args:
        image_list: list of (PIL_image, label) tuples
        clip_processor: CLIPImageProcessor instance

    Returns:
        pixel_values: tensor of shape (N, 3, 224, 224) - preprocessed images
        labels: tensor of shape (N,) - class labels
    """
    all_pixel_values = []
    all_labels = []

    # Process in batches to avoid memory issues
    batch_size = 256
    for i in range(0, len(image_list), batch_size):
        batch_imgs = [item[0] for item in image_list[i:i+batch_size]]
        batch_labels = [item[1] for item in image_list[i:i+batch_size]]

        processed = clip_processor(images=batch_imgs, return_tensors="pt")
        all_pixel_values.append(processed['pixel_values'])
        all_labels.extend(batch_labels)

    pixel_values = torch.cat(all_pixel_values, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)
    print(f"Preprocessed {len(labels)} images. Shape: {pixel_values.shape}")
    return pixel_values, labels


# ============================================================
# === A-C0.2 === Caption and VQA Datasets
# ============================================================

def build_caption_dataset(train_subset):
    """
    A-C0.2-3: Generate 10,000 captions using six rotating templates per class.

    For image i with class c, use template index (i % 6).

    Args:
        train_subset: list of (PIL_image, label) tuples

    Returns:
        captions: list of dicts with keys {image_idx, class_name, caption}
    """
    captions = []
    for i, (img, label) in enumerate(train_subset):
        class_name = CIFAR10_CLASSES[label]
        template_idx = i % len(CAPTION_TEMPLATES)
        caption = CAPTION_TEMPLATES[template_idx].format(class_name)
        captions.append({
            'image_idx': i,
            'class_name': class_name,
            'caption': caption,
        })
    print(f"Generated {len(captions)} captions")
    print(f"Example: {captions[0]}")
    return captions


def get_vqa_answer(label, template_idx, image_idx):
    """
    Get the answer for a VQA template given the image class.

    A-C0.2-4: Apply all five VQA templates to every image.

    Args:
        label: integer class label (0-9)
        template_idx: which VQA template (0-4)
        image_idx: index of the image (used for yes/no randomisation)

    Returns:
        question: the question string
        answer: the answer string
    """
    class_name = CIFAR10_CLASSES[label]

    if template_idx == 0:
        # "What object is shown?" -> class name (recognition)
        return "What object is shown?", class_name

    elif template_idx == 1:
        # "Is there a {class}?" -> yes/no (binary)
        # For even indices: ask correct class -> yes
        # For odd indices: ask a different class -> no
        if image_idx % 2 == 0:
            return f"Is there a {class_name}?", "yes"
        else:
            # Pick a random wrong class deterministically
            wrong_classes = [c for c in CIFAR10_CLASSES if c != class_name]
            wrong_class = wrong_classes[image_idx % len(wrong_classes)]
            return f"Is there a {wrong_class}?", "no"

    elif template_idx == 2:
        # "Vehicle or living thing?" -> vehicle / living (abstraction)
        if class_name in VEHICLES:
            return "Vehicle or living thing?", "vehicle"
        else:
            return "Vehicle or living thing?", "living"

    elif template_idx == 3:
        # "Can it fly?" -> yes/no (attribute)
        if class_name in CAN_FLY:
            return "Can it fly?", "yes"
        else:
            return "Can it fly?", "no"

    elif template_idx == 4:
        # "Is this an animal?" -> yes/no (category)
        if class_name in IS_ANIMAL:
            return "Is this an animal?", "yes"
        else:
            return "Is this an animal?", "no"


def build_vqa_dataset(subset):
    """
    A-C0.2-4: Generate VQA pairs by applying all 5 templates to every image.

    Args:
        subset: list of (PIL_image, label) tuples

    Returns:
        vqa_pairs: list of dicts {image_idx, question, answer, template_idx, class_name}
    """
    vqa_pairs = []
    for i, (img, label) in enumerate(subset):
        class_name = CIFAR10_CLASSES[label]
        for t in range(5):
            question, answer = get_vqa_answer(label, t, i)
            vqa_pairs.append({
                'image_idx': i,
                'question': question,
                'answer': answer,
                'template_idx': t,
                'class_name': class_name,
            })
    print(f"Generated {len(vqa_pairs)} VQA pairs")
    print(f"Example: {vqa_pairs[0]}")
    return vqa_pairs


# ============================================================
# === A-C0.3 === Model Loading
# ============================================================

def load_clip_model(device='cuda'):
    """
    A-C0.3-5: Load CLIP (frozen); confirm 50 tokens output.

    Returns:
        clip_model: frozen CLIP model
        clip_processor: CLIPImageProcessor
    """
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    clip_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)

    # Freeze all CLIP parameters
    clip_model.requires_grad_(False)
    clip_model = clip_model.to(device)
    clip_model.eval()

    # Verify: pass a dummy image to confirm output shape
    # CLIP ViT-B/32 with 224x224 input: 7x7 = 49 patches + 1 CLS = 50 tokens
    dummy_img = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        vision_out = clip_model.vision_model(pixel_values=dummy_img)
        num_tokens = vision_out.last_hidden_state.shape[1]

    print(f"\n--- CLIP Model Loaded ---")
    print(f"Output tokens: {num_tokens} (expected 50: 1 CLS + 49 patches)")
    print(f"Hidden dim: {vision_out.last_hidden_state.shape[2]} (expected 768)")
    print(f"All params frozen: {all(not p.requires_grad for p in clip_model.parameters())}")
    assert num_tokens == 50, f"Expected 50 tokens, got {num_tokens}"

    return clip_model, clip_processor


def load_lm_model(device='cuda'):
    """
    A-C0.3-6: Load SmolLM2-360M in float16; confirm d_LM=960, vocab=49152.

    Returns:
        lm_model: SmolLM2-360M model in float16
        tokenizer: corresponding tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(LM_MODEL_NAME)
    lm_model = AutoModelForCausalLM.from_pretrained(
        LM_MODEL_NAME,
        torch_dtype=torch.float16,
    )
    lm_model = lm_model.to(device)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Verify model dimensions
    d_lm = lm_model.config.hidden_size
    vocab_size = lm_model.config.vocab_size

    print(f"\n--- SmolLM2 Model Loaded ---")
    print(f"Hidden dim (d_LM): {d_lm} (expected 960)")
    print(f"Vocab size: {vocab_size} (expected 49152)")
    print(f"Dtype: {next(lm_model.parameters()).dtype}")
    assert d_lm == 960, f"Expected d_LM=960, got {d_lm}"
    assert vocab_size == 49152, f"Expected vocab=49152, got {vocab_size}"

    return lm_model, tokenizer


def load_alpaca_data(n_examples=1000):
    """
    A-C0.3-7: Load 1,000 Alpaca examples for language replay.

    Returns:
        alpaca_texts: list of formatted text strings
    """
    set_seed(SEED)
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # Randomly sample 1000 examples
    indices = np.random.choice(len(dataset), size=n_examples, replace=False)
    subset = dataset.select(indices)

    # Format each example as "instruction + input + output"
    alpaca_texts = []
    for example in subset:
        text = ""
        if example['instruction']:
            text += example['instruction']
        if example['input']:
            text += " " + example['input']
        if example['output']:
            text += " " + example['output']
        alpaca_texts.append(text.strip())

    print(f"\n--- Alpaca Data Loaded ---")
    print(f"Number of examples: {len(alpaca_texts)}")
    print(f"Example: {alpaca_texts[0][:100]}...")
    return alpaca_texts


def compute_ppl(model, tokenizer, texts, device='cuda', max_length=512):
    """
    A-C0.3-7: Compute perplexity of the model on given texts.

    PPL = exp(average negative log-likelihood per token)

    Args:
        model: language model
        tokenizer: tokenizer
        texts: list of text strings
        device: 'cuda' or 'cpu'
        max_length: max token length for truncation

    Returns:
        ppl: float perplexity value
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            # Tokenize one example at a time for simplicity
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            input_ids = inputs['input_ids'].to(device)

            # Forward pass with labels = input_ids (teacher forcing)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss  # average cross-entropy over all tokens

            num_tokens = input_ids.shape[1]
            total_nll += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return ppl


def extract_clip_features(pixel_values, clip_model, device='cuda', batch_size=64):
    """
    Extract CLIP patch features for all images (discard CLS token).

    Warning: Always use CLIPImageProcessor first - raw 32x32 gives wrong features.
    Uses last_hidden_state[:,1:,:] to get 49 spatial patches (discard CLS at index 0).

    Args:
        pixel_values: tensor (N, 3, 224, 224) from CLIPImageProcessor
        clip_model: frozen CLIP model
        device: 'cuda' or 'cpu'
        batch_size: batch size for extraction

    Returns:
        features: tensor (N, 49, 768) - patch features without CLS
    """
    clip_model.eval()
    all_features = []

    for i in range(0, len(pixel_values), batch_size):
        batch = pixel_values[i:i+batch_size].to(device)
        with torch.no_grad():
            vision_out = clip_model.vision_model(pixel_values=batch)
            # Discard CLS token at index 0, keep 49 patch tokens
            patches = vision_out.last_hidden_state[:, 1:, :]  # (B, 49, 768)
        all_features.append(patches.cpu())

    features = torch.cat(all_features, dim=0)
    print(f"Extracted CLIP features: {features.shape} (expected: (N, 49, 768))")
    return features


# ============================================================
# === Utility Dataset Classes (used by training scripts) ===
# ============================================================

class CaptionDataset(Dataset):
    """Dataset for Phase 1 caption training."""

    def __init__(self, clip_features, captions, tokenizer, max_caption_len=64):
        """
        Args:
            clip_features: tensor (N, 49, 768) pre-extracted CLIP features
            captions: list of caption dicts from build_caption_dataset()
            tokenizer: LM tokenizer
            max_caption_len: max tokens for caption text
        """
        self.clip_features = clip_features
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_caption_len = max_caption_len

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption_info = self.captions[idx]
        img_idx = caption_info['image_idx']

        # Get pre-extracted CLIP features for this image
        clip_feat = self.clip_features[img_idx]  # (49, 768)

        # Tokenize the caption
        caption_tokens = self.tokenizer(
            caption_info['caption'],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_caption_len,
            padding='max_length',
        )
        caption_ids = caption_tokens['input_ids'].squeeze(0)        # (max_caption_len,)
        caption_mask = caption_tokens['attention_mask'].squeeze(0)  # (max_caption_len,)

        return clip_feat, caption_ids, caption_mask


class VQADataset(Dataset):
    """Dataset for Phase 2/3 VQA training."""

    def __init__(self, clip_features, vqa_pairs, tokenizer, max_q_len=32, max_a_len=16):
        """
        Args:
            clip_features: tensor (N, 49, 768) pre-extracted CLIP features
            vqa_pairs: list of VQA dicts from build_vqa_dataset()
            tokenizer: LM tokenizer
            max_q_len: max tokens for question
            max_a_len: max tokens for answer
        """
        self.clip_features = clip_features
        self.vqa_pairs = vqa_pairs
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len

    def __len__(self):
        return len(self.vqa_pairs)

    def __getitem__(self, idx):
        pair = self.vqa_pairs[idx]
        img_idx = pair['image_idx']

        clip_feat = self.clip_features[img_idx]  # (49, 768)

        # Tokenize question
        q_tokens = self.tokenizer(
            pair['question'],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_q_len,
            padding='max_length',
        )
        q_ids = q_tokens['input_ids'].squeeze(0)
        q_mask = q_tokens['attention_mask'].squeeze(0)

        # Tokenize answer
        a_tokens = self.tokenizer(
            pair['answer'],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_a_len,
            padding='max_length',
        )
        a_ids = a_tokens['input_ids'].squeeze(0)
        a_mask = a_tokens['attention_mask'].squeeze(0)

        return clip_feat, q_ids, q_mask, a_ids, a_mask


class AlpacaDataset(Dataset):
    """Dataset for Alpaca language replay."""

    def __init__(self, texts, tokenizer, max_length=256):
        """
        Args:
            texts: list of alpaca text strings
            tokenizer: LM tokenizer
            max_length: max token length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
        )
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        return input_ids, attention_mask


# ============================================================
# Main: Run all A-C0 checks
# ============================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # === A-C0.1 ===
    print("\n" + "="*60)
    print("A-C0.1: CIFAR-10 Preprocessing")
    print("="*60)
    train_subset, test_subset = get_cifar10_subsets()
    verify_clip_normalisation()

    # Load CLIP processor and preprocess images
    clip_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
    train_pixels, train_labels = preprocess_images_for_clip(train_subset, clip_processor)
    test_pixels, test_labels = preprocess_images_for_clip(test_subset, clip_processor)

    # === A-C0.2 ===
    print("\n" + "="*60)
    print("A-C0.2: Caption and VQA Datasets")
    print("="*60)
    captions = build_caption_dataset(train_subset)
    vqa_train = build_vqa_dataset(train_subset)
    vqa_val = build_vqa_dataset(test_subset)
    print(f"VQA train: {len(vqa_train)} pairs (expected 50,000)")
    print(f"VQA val: {len(vqa_val)} pairs (expected 10,000)")

    # === A-C0.3 ===
    print("\n" + "="*60)
    print("A-C0.3: Model Loading")
    print("="*60)
    clip_model, clip_proc = load_clip_model(device)
    lm_model, tokenizer = load_lm_model(device)
    alpaca_texts = load_alpaca_data()

    # Extract CLIP features and cache
    print("\nExtracting CLIP features (train)...")
    clip_features_train = extract_clip_features(train_pixels, clip_model, device)
    print("Extracting CLIP features (test)...")
    clip_features_test = extract_clip_features(test_pixels, clip_model, device)

    # Compute baseline PPL
    print("\nComputing baseline PPL_0 on Alpaca (this takes a minute)...")
    ppl_0 = compute_ppl(lm_model, tokenizer, alpaca_texts[:100], device)
    print(f"Baseline PPL_0 (on 100 examples): {ppl_0:.2f}")

    print("\n" + "="*60)
    print("A-C0 COMPLETE - All data and models loaded successfully!")
    print("="*60)
