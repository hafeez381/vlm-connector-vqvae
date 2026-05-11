"""
# ============================================================
# B-C3: Tokenisation Pipeline
# ============================================================
# USAGE (Google Colab):
#   !python b_c1_vqvae.py --mode train
#   !python b_c2_vocab_expansion.py
#   !python b_c3_tokenisation.py
# ============================================================

PSEUDOCODE:
    1. Define encode_multimodal: [BOS, <image>, v1:16, </image>, question, answer, EOS]
       Labels: -100 prefix; answer + EOS receive gradient
    2. Define encode_imagegen: [BOS, prompt, <image>, v1:16, </image>, EOS]
       Labels: -100 prefix up to <image>; visual tokens + </image> receive gradient
    3. Pre-encode all samples at dataset init using frozen VQ-VAE
    4. Move VQ-VAE to CPU after encoding
    5. Verify: print token-type sequences for 3 images in both modes
"""

import os
import argparse
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from b_c0_data_and_models import (
    set_seed, SEED, generate_dataset, build_vqa_dataset, build_imagegen_dataset,
    load_lm_model, CLASSES, SyntheticImageDataset
)
from b_c1_vqvae import VQVAE
from b_c2_vocab_expansion import V_TXT, N_VISUAL, N_SPECIAL


def get_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="B-C3: Tokenisation Pipeline")
    p.add_argument("--vqvae_path", type=str, default="weights/vqvae_best.pt")
    p.add_argument("--max_q_len", type=int, default=32)
    p.add_argument("--max_a_len", type=int, default=16)
    p.add_argument("--max_prompt_len", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# Token ID helpers
IMAGE_START_ID = V_TXT       # <image> token ID
IMAGE_END_ID = V_TXT + 1     # </image> token ID


def codebook_idx_to_token_id(codebook_idx):
    """
    B-C3 warning: visual token ID = codebook index + V_txt + 2.

    Args:
        codebook_idx: int or tensor of codebook indices (0..255)

    Returns:
        token_id: int or tensor of LM token IDs
    """
    return codebook_idx + V_TXT + 2


# ============================================================
# === B-C3-1 === encode_multimodal
# ============================================================

def encode_multimodal(visual_indices, question, answer, tokenizer):
    """
    B-C3-1: Encode VQA sample as [BOS, <image>, v1:16, </image>, question, answer, EOS].
    Labels: -100 for prefix (BOS + <image> + visual + </image> + question);
    answer + EOS receive gradient.

    Args:
        visual_indices: tensor of 16 codebook indices (from VQ-VAE)
        question: question string
        answer: answer string
        tokenizer: LM tokenizer

    Returns:
        input_ids: tensor (seq_len,)
        labels: tensor (seq_len,)
        attention_mask: tensor (seq_len,)
    """
    bos_id = tokenizer.bos_token_id or 0
    eos_id = tokenizer.eos_token_id

    # Convert codebook indices to token IDs
    vis_token_ids = codebook_idx_to_token_id(visual_indices).tolist()

    # Tokenize question and answer (no special tokens)
    q_ids = tokenizer.encode(question, add_special_tokens=False)
    a_ids = tokenizer.encode(answer, add_special_tokens=False)

    # Build sequence: [BOS, <image>, v1..v16, </image>, question, answer, EOS]
    input_ids = (
        [bos_id] +
        [IMAGE_START_ID] +
        vis_token_ids +
        [IMAGE_END_ID] +
        q_ids +
        a_ids +
        [eos_id]
    )

    # Labels: -100 for everything up to and including question; answer + EOS get real IDs
    prefix_len = 1 + 1 + 16 + 1 + len(q_ids)  # BOS + <img> + visual + </img> + question
    labels = [-100] * prefix_len + a_ids + [eos_id]

    assert len(input_ids) == len(labels), \
        f"Length mismatch: input_ids={len(input_ids)}, labels={len(labels)}"

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.ones(len(input_ids), dtype=torch.long),
    )


# ============================================================
# === B-C3-2 === encode_imagegen
# ============================================================

def encode_imagegen(visual_indices, prompt, tokenizer):
    """
    B-C3-2: Encode image-gen sample as [BOS, prompt, <image>, v1:16, </image>, EOS].
    Labels: -100 prefix up to <image>; visual tokens + </image> receive gradient.

    Args:
        visual_indices: tensor of 16 codebook indices
        prompt: prompt string
        tokenizer: LM tokenizer

    Returns:
        input_ids: tensor (seq_len,)
        labels: tensor (seq_len,)
        attention_mask: tensor (seq_len,)
    """
    bos_id = tokenizer.bos_token_id or 0
    eos_id = tokenizer.eos_token_id

    vis_token_ids = codebook_idx_to_token_id(visual_indices).tolist()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Build sequence: [BOS, prompt, <image>, v1..v16, </image>, EOS]
    input_ids = (
        [bos_id] +
        prompt_ids +
        [IMAGE_START_ID] +
        vis_token_ids +
        [IMAGE_END_ID] +
        [eos_id]
    )

    # Labels: -100 for BOS + prompt + <image>; visual tokens + </image> get real IDs; EOS = -100
    prefix_len = 1 + len(prompt_ids) + 1  # BOS + prompt + <image>
    labels = (
        [-100] * prefix_len +
        vis_token_ids +
        [IMAGE_END_ID] +
        [-100]  # EOS
    )

    assert len(input_ids) == len(labels)

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.ones(len(input_ids), dtype=torch.long),
    )


# ============================================================
# === B-C3-3 === Pre-encoded Datasets
# ============================================================

def pre_encode_images(images, vqvae, device, batch_size=64):
    """
    B-C3-3: Pre-encode all images to codebook indices using frozen VQ-VAE.
    Move VQ-VAE to CPU after encoding.

    Args:
        images: (N, 3, 16, 16) image tensor
        vqvae: frozen VQ-VAE model
        device: torch device
        batch_size: batch size for encoding

    Returns:
        all_indices: (N, 16) codebook indices tensor
    """
    vqvae = vqvae.to(device)
    vqvae.eval()
    all_indices = []

    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Pre-encoding images"):
            batch = images[i:i+batch_size].to(device)
            indices = vqvae.encode_indices(batch)  # (B, 16)
            all_indices.append(indices.cpu())

    # B-C3-3: Move VQ-VAE to CPU after encoding
    vqvae = vqvae.cpu()
    torch.cuda.empty_cache()

    return torch.cat(all_indices, dim=0)


class PreEncodedVQADataset(Dataset):
    """
    Pre-encoded VQA dataset for LM training.
    All images are pre-encoded to codebook indices at init time.
    """
    def __init__(self, all_indices, vqa_pairs, tokenizer):
        """
        Args:
            all_indices: (N, 16) pre-encoded codebook indices
            vqa_pairs: list of VQA dicts {image_idx, question, answer, ...}
            tokenizer: LM tokenizer
        """
        self.all_indices = all_indices
        self.vqa_pairs = vqa_pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.vqa_pairs)

    def __getitem__(self, idx):
        pair = self.vqa_pairs[idx]
        img_idx = pair['image_idx']
        vis_idx = self.all_indices[img_idx]  # (16,)
        input_ids, labels, attn_mask = encode_multimodal(
            vis_idx, pair['question'], pair['answer'], self.tokenizer
        )
        return input_ids, labels, attn_mask


class PreEncodedImageGenDataset(Dataset):
    """
    Pre-encoded image-gen dataset for LM training.
    """
    def __init__(self, all_indices, imagegen_pairs, tokenizer):
        """
        Args:
            all_indices: (N, 16) pre-encoded codebook indices
            imagegen_pairs: list of imagegen dicts {image_idx, prompt, ...}
            tokenizer: LM tokenizer
        """
        self.all_indices = all_indices
        self.imagegen_pairs = imagegen_pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.imagegen_pairs)

    def __getitem__(self, idx):
        pair = self.imagegen_pairs[idx]
        img_idx = pair['image_idx']
        vis_idx = self.all_indices[img_idx]  # (16,)
        input_ids, labels, attn_mask = encode_imagegen(
            vis_idx, pair['prompt'], self.tokenizer
        )
        return input_ids, labels, attn_mask


def collate_padded(batch):
    """
    Collate function that pads variable-length sequences to the longest in the batch.

    Returns:
        input_ids: (B, max_len)
        labels: (B, max_len) with -100 for padding
        attention_mask: (B, max_len)
    """
    input_ids_list, labels_list, mask_list = zip(*batch)
    max_len = max(ids.shape[0] for ids in input_ids_list)

    padded_ids, padded_labels, padded_mask = [], [], []
    for ids, lbl, msk in zip(input_ids_list, labels_list, mask_list):
        pad_len = max_len - ids.shape[0]
        # Right-pad for training (left-pad is only for generation)
        padded_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)]))
        padded_labels.append(torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)]))
        padded_mask.append(torch.cat([msk, torch.zeros(pad_len, dtype=torch.long)]))

    return (
        torch.stack(padded_ids),
        torch.stack(padded_labels),
        torch.stack(padded_mask),
    )


# ============================================================
# === B-C3-4 === Verification
# ============================================================

def verify_encoding(all_indices, vqa_pairs, imagegen_pairs, tokenizer, n=3):
    """
    B-C3-4: Print token-type sequences for n images in both modes.
    """
    print("\n" + "=" * 60)
    print("B-C3-4: Encoding Verification")
    print("=" * 60)

    for i in range(min(n, len(vqa_pairs))):
        pair = vqa_pairs[i]
        vis_idx = all_indices[pair['image_idx']]
        ids, labels, mask = encode_multimodal(vis_idx, pair['question'], pair['answer'], tokenizer)

        # Build token-type string
        types = []
        for j, tid in enumerate(ids.tolist()):
            if tid == tokenizer.bos_token_id: types.append("BOS")
            elif tid == IMAGE_START_ID: types.append("<img>")
            elif tid == IMAGE_END_ID: types.append("</img>")
            elif tid >= V_TXT + 2: types.append(f"V{tid - V_TXT - 2}")
            elif tid == tokenizer.eos_token_id: types.append("EOS")
            else: types.append("T")

        label_types = ["✓" if l != -100 else "×" for l in labels.tolist()]

        print(f"\n  VQA #{i}: Q='{pair['question']}' A='{pair['answer']}'")
        print(f"  Types:  {' '.join(types[:30])}...")
        print(f"  Labels: {' '.join(label_types[:30])}...")
        print(f"  Length: {len(ids)}")

    for i in range(min(n, len(imagegen_pairs))):
        pair = imagegen_pairs[i]
        vis_idx = all_indices[pair['image_idx']]
        ids, labels, mask = encode_imagegen(vis_idx, pair['prompt'], tokenizer)

        types = []
        for j, tid in enumerate(ids.tolist()):
            if tid == tokenizer.bos_token_id: types.append("BOS")
            elif tid == IMAGE_START_ID: types.append("<img>")
            elif tid == IMAGE_END_ID: types.append("</img>")
            elif tid >= V_TXT + 2: types.append(f"V{tid - V_TXT - 2}")
            elif tid == tokenizer.eos_token_id: types.append("EOS")
            else: types.append("T")

        label_types = ["✓" if l != -100 else "×" for l in labels.tolist()]

        print(f"\n  ImgGen #{i}: prompt='{pair['prompt']}'")
        print(f"  Types:  {' '.join(types[:30])}...")
        print(f"  Labels: {' '.join(label_types[:30])}...")
        print(f"  Length: {len(ids)}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = get_args()
    set_seed(args.seed)
    device = args.device

    print("=" * 60)
    print("B-C3: Tokenisation Pipeline")
    print("=" * 60)

    # Load data
    train_imgs, train_lbls, val_imgs, val_lbls = generate_dataset(seed=args.seed)
    vqa_train = build_vqa_dataset(train_lbls)
    vqa_val = build_vqa_dataset(val_lbls)
    ig_train = build_imagegen_dataset(train_lbls)

    # Load VQ-VAE and pre-encode (then free from GPU)
    vqvae = VQVAE(K=256, d=64)
    vqvae.load_state_dict(torch.load(args.vqvae_path, map_location="cpu"))

    # B-C3-3: Pre-encode all samples
    print("\nPre-encoding train images...")
    train_indices = pre_encode_images(train_imgs, vqvae, device)
    print("Pre-encoding val images...")
    val_indices = pre_encode_images(val_imgs, vqvae, device)

    # VQ-VAE already moved to CPU by pre_encode_images
    del vqvae
    torch.cuda.empty_cache()

    # Load tokenizer
    _, tokenizer = load_lm_model(device)
    # Add special tokens
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['<image>', '</image>']
    })
    # Clean up model (we only needed the tokenizer)
    torch.cuda.empty_cache()

    # B-C3-4: Verify encoding
    verify_encoding(train_indices, vqa_train, ig_train, tokenizer, n=3)

    print("\nB-C3 COMPLETE")
