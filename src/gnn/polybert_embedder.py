"""
polyBERT Embedding Extractor — Extract polymer embeddings from pretrained polyBERT.

Uses sentence-transformers to load kuelumbus/polyBERT, encoding PSMILES into
600-dim vectors, then optionally reduces to target_dim via PCA.

Reference: Kuenneth & Ramprasad, Nature Communications 14, 4099 (2023)

Public API:
    extract_polybert_embeddings(smiles_list, batch_size, device, local_path) -> np.ndarray [N, 600]
    polybert_pca(embeddings, target_dim, fit_mask) -> np.ndarray [N, target_dim]
"""
import re
import warnings
from typing import List, Optional

import numpy as np

POLYBERT_DIM = 600
MODEL_NAME = "kuelumbus/polyBERT"

# Singleton cache
_MODEL = None


def _psmiles_format(smiles: str) -> str:
    """Convert standard SMILES with [*] to polyBERT's expected PSMILES format."""
    s = smiles.strip()
    s = re.sub(r'(?<!\[)\*(?!\])', '[*]', s)
    return s


def _load_model(device: str = "cuda", local_path: str = None):
    """Lazy-load polyBERT model via sentence-transformers."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. Run: pip install sentence-transformers"
        )

    import os

    # Try local path first
    if local_path and os.path.isdir(local_path):
        print(f"  Loading polyBERT from local: {local_path}")
        _MODEL = SentenceTransformer(local_path, device=device)
        print(f"  polyBERT loaded on {device} (local)")
        return _MODEL

    # Try with mirror for China servers
    for endpoint in [None, "https://hf-mirror.com"]:
        try:
            if endpoint:
                os.environ["HF_ENDPOINT"] = endpoint
                print(f"  Trying mirror: {endpoint}")
            else:
                os.environ.pop("HF_ENDPOINT", None)
                print(f"  Loading polyBERT from {MODEL_NAME}...")

            _MODEL = SentenceTransformer(MODEL_NAME, device=device)
            print(f"  polyBERT loaded on {device}")
            return _MODEL
        except Exception as e:
            print(f"  Failed: {type(e).__name__}: {e}")
            _MODEL = None

    os.environ.pop("HF_ENDPOINT", None)
    raise RuntimeError(
        "Cannot load polyBERT. Options:\n"
        "  1. Download on a machine with internet:\n"
        "     python -c \"from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer('kuelumbus/polyBERT').save('polybert_model')\"\n"
        "  2. Upload to server: scp -r polybert_model/ server:~/Tgprediction/data/polybert_model/\n"
        "  3. Run with: python scripts/phase_d_polybert.py --local-model data/polybert_model"
    )


def extract_polybert_embeddings(
    smiles_list: List[str],
    batch_size: int = 64,
    device: str = "cuda",
    local_path: str = None,
) -> np.ndarray:
    """Extract 600-dim embeddings from polyBERT for a list of SMILES.

    Args:
        smiles_list: List of polymer SMILES (with [*] endpoints).
        batch_size: Batch size for inference.
        device: "cuda" or "cpu".
        local_path: Path to locally saved polyBERT model.

    Returns:
        np.ndarray of shape [N, 600]. NaN rows for failed SMILES.
    """
    model = _load_model(device, local_path)
    psmiles = [_psmiles_format(s) for s in smiles_list]

    print(f"  Encoding {len(psmiles)} SMILES...")
    embeddings = model.encode(
        psmiles,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    actual_dim = embeddings.shape[1]
    print(f"  polyBERT output: {embeddings.shape[0]} × {actual_dim}d")

    valid = ~np.any(np.isnan(embeddings), axis=1)
    print(f"  Valid: {valid.sum()}/{len(psmiles)} ({100*valid.mean():.1f}%)")

    return embeddings


def polybert_pca(
    embeddings: np.ndarray,
    target_dim: int = 64,
    fit_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reduce polyBERT embeddings to target_dim via PCA.

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
