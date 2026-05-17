import html
import json
import os
import pickle
import re
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


def ndcg_at_k(pred, tgt, k):
    """
    Calculate NDCG at K using NumPy.

    Args:
    - pred: Array of shape [B, K], predicted ranking for each user
    - tgt: Array of shape [B], ground truth relevant item for each user
    - k: int, rank position for NDCG

    Returns:
    - ndcg: NDCG score at K for the batch
    """
    top_k_preds = pred[:, :k]
    relevant_mask = top_k_preds == tgt[:, None]
    dcg_scores = relevant_mask.astype(np.float32) / np.log2(np.arange(2, k + 2))
    dcg = np.sum(dcg_scores, axis=1)
    idcg = 1.0
    ndcg = dcg / idcg
    return np.mean(ndcg)


def recall_at_k(pred, tgt, k):
    """
    Calculate Recall at K using NumPy.

    Args:
    - pred: Array of shape [B, K], predicted ranking for each user
    - tgt: Array of shape [B], ground truth relevant item for each user
    - k: int, rank position for recall

    Returns:
    - recall: Recall score at K for the batch
    """
    top_k_preds = pred[:, :k]
    relevant_mask = np.any(top_k_preds == tgt[:, None], axis=1)
    recall = relevant_mask.astype(np.float32)
    return np.mean(recall)


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device("cpu")
    else:
        return torch.device(
            "cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu"
        )


# https://hf-mirror.com/
# huggingface-cli download --token hf_*** --resume-download meta-llama/Llama-2-7b-hf --local-dir Llama-2-7b-hf
# ./hfd.sh meta-llama/Llama-2-7b-hf --hf_username baiyimeng --hf_token hf_***
def load_plm(model_path="meta-llama/Llama-2-7b-hf"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )
    print("Load Llama-2-7b-hf: ", model_path)
    model = AutoModel.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


def clean_text(raw_text):
    # If the input is a list, process each element in the list
    if isinstance(raw_text, list):
        new_raw_text = []
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r"</?\w+[^>]*>", "", raw)
            raw = re.sub(r'["\n\r]*', "", raw)
            new_raw_text.append(raw.strip())
        cleaned_text = " ".join(new_raw_text)
    else:
        # If the input is a dictionary, convert it to a string without braces
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            # If the input is another type, directly strip leading and trailing whitespace
            cleaned_text = raw_text.strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r"</?\w+[^>]*>", "", cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', "", cleaned_text)
    # Ensure the text ends with a single period
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == ".":
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + "."
    else:
        cleaned_text = cleaned_text[:index] + "."
    if len(cleaned_text) >= 2000:
        cleaned_text = ""
    return cleaned_text


def load_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
