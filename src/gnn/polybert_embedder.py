"""
polyBERT Embedding Extractor — Extract polymer embeddings from pretrained polyBERT.

Uses HuggingFace kuelumbus/polyBERT to encode PSMILES into 768-dim vectors,
then optionally reduces to target_dim via PCA.

Public API:
    extract_polybert_embeddings(smiles_list, batch_size, device) -> np.ndarray [N, 768]
    polybert_pca(embeddings, target_dim, fit_mask) -> np.ndarray [N, target_dim]
"""
import warnings
from typing import List, Optional

import numpy as np

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


MODEL_NAME = "kuelumbus/polyBERT"

# HuggingFace mirror for China mainland servers
HF_MIRRORS = [
    None,                                    # Try official first (or cached)
    "https://hf-mirror.com",                 # China mirror
    "https://huggingface.co",                # Explicit official
]

# Singleton cache
_MODEL = None
_TOKENIZER = None


def _load_model(device: str = "cuda", local_path: str = None):
    """Lazy-load polyBERT model and tokenizer.

    Tries: local_path → offline cache → mirrors → official HuggingFace.
    """
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    if not HAS_TRANSFORMERS:
        raise ImportError("transformers not installed. Run: pip install transformers")

    import os

    # Try local path first
    if local_path and os.path.isdir(local_path):
        print(f"  Loading polyBERT from local: {local_path}")
        _TOKENIZER = AutoTokenizer.from_pretrained(local_path)
        _MODEL = AutoModel.from_pretrained(local_path).to(device).eval()
        print(f"  polyBERT loaded on {device}")
        return _MODEL, _TOKENIZER

    # Try offline mode (use existing cache without network)
    try:
        print(f"  Trying offline cache for {MODEL_NAME}...")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
        print(f"  polyBERT loaded from cache on {device}")
        return _MODEL, _TOKENIZER
    except Exception:
        pass
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    # Try mirrors
    for mirror in HF_MIRRORS:
        try:
            if mirror:
                os.environ["HF_ENDPOINT"] = mirror
                print(f"  Trying mirror: {mirror}")
            else:
                os.environ.pop("HF_ENDPOINT", None)
                print(f"  Trying default HuggingFace...")

            _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
            _MODEL = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
            print(f"  polyBERT loaded on {device}")
            return _MODEL, _TOKENIZER
        except Exception as e:
            print(f"  Failed: {type(e).__name__}")
            _MODEL, _TOKENIZER = None, None

    os.environ.pop("HF_ENDPOINT", None)
    raise RuntimeError(
        "Cannot load polyBERT. Options:\n"
        "  1. Set HF_ENDPOINT=https://hf-mirror.com before running\n"
        "  2. Download model locally and pass local_path\n"
        "  3. Run: huggingface-cli download kuelumbus/polyBERT on a machine with internet"
    )


def _psmiles_format(smiles: str) -> str:
    """Convert standard SMILES with [*] to polyBERT's expected format."""
    # polyBERT expects [*] as polymer endpoints
    import re
    s = smiles.strip()
    s = re.sub(r'(?<!\[)\*(?!\])', '[*]', s)
    return s


def extract_polybert_embeddings(
    smiles_list: List[str],
    batch_size: int = 64,
    device: str = "cuda",
) -> np.ndarray:
    """Extract 768-dim embeddings from polyBERT for a list of SMILES.

    Args:
        smiles_list: List of polymer SMILES (with [*] endpoints).
        batch_size: Batch size for inference.
        device: "cuda" or "cpu".

    Returns:
        np.ndarray of shape [N, 768]. NaN rows for failed SMILES.
    """
    model, tokenizer = _load_model(device)
    n = len(smiles_list)
    embeddings = np.full((n, 768), np.nan)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_smiles = [_psmiles_format(s) for s in smiles_list[start:end]]

        try:
            inputs = tokenizer(
                batch_smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Use [CLS] token embedding (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings[start:end] = cls_embeddings

        except Exception as e:
            warnings.warn(f"  polyBERT batch {start}-{end} failed: {e}")

        if (end) % (batch_size * 10) == 0 or end == n:
            print(f"  polyBERT: {end}/{n} done")

    valid = ~np.any(np.isnan(embeddings), axis=1)
    print(f"  polyBERT: {valid.sum()}/{n} valid embeddings ({100*valid.mean():.1f}%)")
    return embeddings


def polybert_pca(
    embeddings: np.ndarray,
    target_dim: int = 64,
    fit_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reduce polyBERT 768d to target_dim via PCA.

    Args:
        embeddings: [N, 768] array (may contain NaN rows).
        target_dim: Target dimensionality.
        fit_mask: Boolean mask for fitting PCA (e.g., train set only).
                  If None, fit on all non-NaN rows.

    Returns:
        np.ndarray of shape [N, target_dim]. NaN rows preserved.
    """
    from sklearn.decomposition import PCA

    valid = ~np.any(np.isnan(embeddings), axis=1)

    if fit_mask is not None:
        fit_data = embeddings[valid & fit_mask]
    else:
        fit_data = embeddings[valid]

    pca = PCA(n_components=target_dim, random_state=42)
    pca.fit(fit_data)

    result = np.full((len(embeddings), target_dim), np.nan)
    result[valid] = pca.transform(embeddings[valid])

    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA {embeddings.shape[1]}d -> {target_dim}d, explained variance: {explained:.3f}")
    return result
