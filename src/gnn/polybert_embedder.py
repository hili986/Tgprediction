"""
polyBERT Embedding Extractor — Extract polymer embeddings from local polyBERT model.

polyBERT (DeBERTa-v2, hidden_size=600) encodes PSMILES into 600-dim vectors,
then optionally reduces to target_dim via PCA.

Reference: Kuenneth & Ramprasad, Nature Communications 14, 4099 (2023)

Public API:
    extract_polybert_embeddings(smiles_list, model_path, batch_size, device) -> np.ndarray [N, 600]
    polybert_pca(embeddings, target_dim, fit_mask) -> np.ndarray [N, target_dim]
"""
import re
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

POLYBERT_DIM = 600

# Default model path (relative to project root)
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "polybert_model"

# Singleton cache
_MODEL = None
_TOKENIZER = None


def _psmiles_format(smiles: str) -> str:
    """Convert standard SMILES with [*] to PSMILES format."""
    s = smiles.strip()
    s = re.sub(r'(?<!\[)\*(?!\])', '[*]', s)
    return s


def _load_model(model_path: str = None, device: str = "cuda"):
    """Load polyBERT from local directory."""
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    import torch
    from transformers import AutoModel, AutoTokenizer

    path = model_path or str(_DEFAULT_MODEL_PATH)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"polyBERT model not found at {path}.\n"
            "Copy model files to data/polybert_model/ or pass --local-model path."
        )

    print(f"  Loading polyBERT from {path}...")
    _TOKENIZER = AutoTokenizer.from_pretrained(path)
    _MODEL = AutoModel.from_pretrained(path).to(device).eval()
    print(f"  polyBERT loaded on {device} (DeBERTa-v2, {POLYBERT_DIM}d)")
    return _MODEL, _TOKENIZER


def extract_polybert_embeddings(
    smiles_list: List[str],
    model_path: str = None,
    batch_size: int = 64,
    device: str = "cuda",
) -> np.ndarray:
    """Extract 600-dim embeddings from polyBERT for a list of SMILES.

    Uses [CLS] token embedding from the last hidden state.

    Args:
        smiles_list: List of polymer SMILES (with [*] endpoints).
        model_path: Path to local polyBERT model directory.
        batch_size: Batch size for inference.
        device: "cuda" or "cpu".

    Returns:
        np.ndarray of shape [N, 600].
    """
    import torch

    model, tokenizer = _load_model(model_path, device)
    n = len(smiles_list)
    embeddings = np.full((n, POLYBERT_DIM), np.nan)

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

            # [CLS] token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings[start:end] = cls_embeddings

        except Exception as e:
            warnings.warn(f"  polyBERT batch {start}-{end} failed: {e}")

        if end % (batch_size * 10) == 0 or end == n:
            print(f"  polyBERT: {end}/{n} done")

    valid = ~np.any(np.isnan(embeddings), axis=1)
    print(f"  polyBERT: {valid.sum()}/{n} valid ({100*valid.mean():.1f}%)")
    return embeddings


def polybert_pca(
    embeddings: np.ndarray,
    target_dim: int = 64,
    fit_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reduce polyBERT 600d to target_dim via PCA.

    Args:
        embeddings: [N, 600] array (may contain NaN rows).
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
