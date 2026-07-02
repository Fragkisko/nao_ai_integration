"""Utilities for hybrid result fusion.

The default strategy is Reciprocal Rank Fusion (RRF), which combines ranked
lists without requiring score normalization across retrievers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _result_key(item: Dict[str, Any]) -> Optional[int]:
    idx = item.get("index")
    if idx is None:
        return None
    try:
        return int(idx)
    except Exception:
        return None


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    fused: Dict[int, Dict[str, Any]] = {}

    for source_idx, results in enumerate(ranked_lists):
        for rank, item in enumerate(results, start=1):
            key = _result_key(item)
            if key is None:
                continue

            score_inc = 1.0 / float(rrf_k + rank)

            if key not in fused:
                fused[key] = {
                    "index": key,
                    "score": 0.0,
                    "meta": item.get("meta"),
                    "dense_rank": None,
                    "sparse_rank": None,
                }

            fused[key]["score"] += score_inc

            if source_idx == 0 and fused[key]["dense_rank"] is None:
                fused[key]["dense_rank"] = rank
            if source_idx == 1 and fused[key]["sparse_rank"] is None:
                fused[key]["sparse_rank"] = rank

            if fused[key].get("meta") is None and item.get("meta") is not None:
                fused[key]["meta"] = item.get("meta")

    return sorted(fused.values(), key=lambda x: x["score"], reverse=True)
