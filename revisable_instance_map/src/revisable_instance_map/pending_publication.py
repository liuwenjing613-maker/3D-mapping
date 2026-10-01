"""Conservative pending protection is a publication bound, not an identity vote."""
import numpy as np
from .surface_evidence import CONFIRMED, UNOBSERVED

REASON_NAMES = ("p0_unobserved", "p0_tentative", "p0_confirmed", "p0_conflict",
                "pending_identity_only", "pending_competition_blocks_publication")


def protect_publication(evidence, unresolved_count_upper, min_votes=2, min_ratio=.67):
    upper = np.asarray(unresolved_count_upper, dtype=np.int32)
    if upper.shape != evidence["state"].shape or np.any(upper < 0):
        raise ValueError("Invalid unresolved surface counts")
    total = evidence["total_frame_votes"].astype(np.int64) + upper
    share = np.divide(evidence["top1_votes"], total, out=np.zeros(len(total), np.float64), where=total > 0)
    allowed = (evidence["top1_votes"] >= min_votes) & (share >= min_ratio)
    labels = np.where(allowed, evidence["top1_instance_id"], -1).astype(np.int32)
    reason = evidence["state"].copy()
    reason[(evidence["total_frame_votes"] == 0) & (upper > 0)] = 4
    reason[(evidence["total_frame_votes"] > 0) & (upper > 0) & ~allowed] = 5
    return {"instance_id": labels, "unresolved_count_upper": upper,
            "pending_identity_coverage": upper > 0, "worst_case_vote_share": share.astype(np.float32),
            "publication_reason": reason,
            "diffusion_protected_mask": (upper > 0) & ~allowed}


def enforce_pending_protection(original_labels, inferred_labels, protected_mask):
    """Future diffusion adapters must call this before publishing inferred labels."""
    original, inferred, mask = np.asarray(original_labels), np.asarray(inferred_labels), np.asarray(protected_mask)
    if original.shape != inferred.shape or mask.shape != original.shape:
        raise ValueError("Protection dimensions differ")
    result = inferred.copy()
    result[mask] = original[mask]
    return result
