"""
Vectorized speaker matching.

Replaces the per-user Python loop in /identify with a single batched cosine
computation. This is mathematically identical to normalizing each pair and
taking cosine similarity, but runs as one matrix op instead of N iterations —
much faster as the enrolled-user count grows, and easy to unit test.

For very large enrollments (10k+), move embeddings into an ANN index
(MongoDB Atlas Vector Search / FAISS) instead of loading them all per request.
"""

from typing import Dict, List, Optional, Tuple

import torch


def _flatten(embedding: torch.Tensor) -> torch.Tensor:
    """Reduce an embedding of shape [.., D] to a 1-D [D] vector."""
    t = embedding
    while t.dim() > 1:
        t = t.squeeze(0) if t.shape[0] == 1 else t.mean(dim=0)
    return t


def rank_matches(
    probe: torch.Tensor,
    enrolled: Dict[str, torch.Tensor],
) -> List[Tuple[str, float]]:
    """Return (user_id, cosine_score) pairs sorted by score, descending.

    Args:
        probe: embedding of the unknown speaker, any shape reducible to [D].
        enrolled: mapping of user_id -> enrolled embedding (any reducible shape).
    """
    if not enrolled:
        return []

    device = probe.device
    probe_vec = _flatten(probe).to(device).float()
    probe_norm = probe_vec / (probe_vec.norm() + 1e-12)

    user_ids: List[str] = []
    rows: List[torch.Tensor] = []
    for user_id, emb in enrolled.items():
        vec = _flatten(emb).to(device).float()
        if vec.shape != probe_norm.shape:
            # Skip mismatched-dimension embeddings rather than crashing.
            continue
        user_ids.append(user_id)
        rows.append(vec)

    if not rows:
        return []

    matrix = torch.stack(rows, dim=0)  # [N, D]
    matrix_norm = matrix / (matrix.norm(dim=1, keepdim=True) + 1e-12)
    scores = matrix_norm @ probe_norm  # [N]

    paired = list(zip(user_ids, scores.tolist()))
    paired.sort(key=lambda x: x[1], reverse=True)
    return paired


def best_match(
    probe: torch.Tensor,
    enrolled: Dict[str, torch.Tensor],
) -> Tuple[Optional[str], float]:
    """Return the single best (user_id, score), or (None, -1.0) if no candidates."""
    ranked = rank_matches(probe, enrolled)
    if not ranked:
        return None, -1.0
    return ranked[0]
