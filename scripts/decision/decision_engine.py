"""
AI Sort Bin - Decision Engine

Routes model predictions to physical bins using centralized config.
All taxonomy/routing is loaded from dataset/dataset_config.json.
"""
from scripts.common.config import (
    get_routing_map,
    get_physical_bins,
    get_fallback_bin,
    get_fallback_bin_id,
    get_confidence_thresholds,
    get_threshold_for_label,
    get_unknown_threshold,
    safe_label_to_bin_id,
)


class DecisionEngine:
    """
    Stateless decision engine that routes predictions to bins.

    Uses centralized config - no hardcoded labels or bin mappings.
    """

    def __init__(self):
        """
        Initialize decision engine from centralized config.

        No policy_path needed - all config comes from dataset_config.json.
        """
        # Load config values (cached by config module)
        self.routing_map = get_routing_map()
        self.physical_bins = get_physical_bins()
        self.fallback_bin = get_fallback_bin()
        self.fallback_bin_id = get_fallback_bin_id()
        self.thresholds = get_confidence_thresholds()
        self.unknown_threshold = get_unknown_threshold()

    def decide(self, predictions: list[dict]) -> int:
        """
        Route predictions to a physical bin ID.

        Args:
            predictions: List of dicts with "class" and "confidence" keys
                Example: [{"class": "metal_container", "confidence": 0.85}]

        Returns:
            int: Physical bin ID (hardware index)

        Behavior:
            - Empty predictions -> fallback bin
            - Top prediction below unknown_threshold -> fallback bin
            - Top prediction below class threshold -> fallback bin
            - Unknown/unmapped class -> fallback bin (never raises KeyError)
            - Valid prediction above threshold -> routed bin
        """
        # No predictions -> fallback
        if not predictions:
            return self.fallback_bin_id

        # Sort by confidence (highest first)
        sorted_preds = sorted(
            predictions,
            key=lambda x: x.get("confidence", 0.0),
            reverse=True
        )

        top = sorted_preds[0]
        label = top.get("class", "")
        confidence = top.get("confidence", 0.0)

        # Below unknown threshold -> fallback
        if confidence < self.unknown_threshold:
            return self.fallback_bin_id

        # Get class-specific threshold (or default)
        threshold = get_threshold_for_label(label)

        # Below class threshold -> fallback
        if confidence < threshold:
            return self.fallback_bin_id

        # Route to bin (safe - never raises KeyError)
        bin_id, fallback_reason = safe_label_to_bin_id(label)

        return bin_id

    def decide_with_reason(self, predictions: list[dict]) -> tuple[int, str | None]:
        """
        Route predictions to a bin with explanation.

        Same as decide() but returns reason for fallback routing.

        Args:
            predictions: List of prediction dicts

        Returns:
            tuple[int, str | None]: (bin_id, fallback_reason)
            - fallback_reason is None for successful routing
            - fallback_reason explains why fallback was used
        """
        if not predictions:
            return self.fallback_bin_id, "no_predictions"

        sorted_preds = sorted(
            predictions,
            key=lambda x: x.get("confidence", 0.0),
            reverse=True
        )

        top = sorted_preds[0]
        label = top.get("class", "")
        confidence = top.get("confidence", 0.0)

        if confidence < self.unknown_threshold:
            return self.fallback_bin_id, f"below_unknown_threshold ({confidence:.3f} < {self.unknown_threshold})"

        threshold = get_threshold_for_label(label)
        if confidence < threshold:
            return self.fallback_bin_id, f"below_class_threshold ({confidence:.3f} < {threshold})"

        bin_id, fallback_reason = safe_label_to_bin_id(label)
        return bin_id, fallback_reason
