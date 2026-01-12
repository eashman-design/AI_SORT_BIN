"""
Inference Policy Module - Runtime Inference Contract v1.0

Implements detection filtering, active object selection, and bin routing
for the AI_SORT_BIN waste classification system.

Model: SSD MobileNet V2 (TensorFlow Object Detection API)
"""

import logging
from datetime import datetime
from typing import Any

# Configure module logger
logger = logging.getLogger(__name__)

# Contract-mandated thresholds
CONF_MIN = 0.60
CONF_ROUTE = 0.75

# Fallback bin for all uncertain/error cases
TRASH_BIN = "trash"


def _compute_bbox_area(detection: dict) -> float:
    """
    Compute bounding box area from detection.

    Args:
        detection: dict with 'bbox' key containing (x_min, y_min, x_max, y_max)

    Returns:
        Area as float. Returns 0.0 if bbox is invalid.
    """
    bbox = detection.get("bbox")
    if bbox is None or len(bbox) != 4:
        return 0.0

    x_min, y_min, x_max, y_max = bbox
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    return width * height


def filter_detections(detections: list[dict], conf_min: float) -> list[dict]:
    """
    Filter detections by minimum confidence threshold.

    Contract Rule 1: Discard any detection where confidence < CONF_MIN

    Args:
        detections: List of detection dicts with 'confidence' key
        conf_min: Minimum confidence threshold

    Returns:
        List of detections that meet the confidence threshold
    """
    if not detections:
        return []

    filtered = [
        d for d in detections
        if d.get("confidence", 0.0) >= conf_min
    ]
    return filtered


def select_active_detection(detections: list[dict]) -> dict | None:
    """
    Select the active detection from a list of filtered detections.

    Contract Rule 2:
    - Select the detection with the LARGEST bounding-box area
    - If tie, select higher confidence

    Args:
        detections: List of filtered detection dicts

    Returns:
        Selected detection dict, or None if list is empty
    """
    if not detections:
        return None

    if len(detections) == 1:
        return detections[0]

    # Sort by: (area descending, confidence descending)
    # Python sort is stable, so we sort by secondary key first
    sorted_detections = sorted(
        detections,
        key=lambda d: (_compute_bbox_area(d), d.get("confidence", 0.0)),
        reverse=True
    )

    return sorted_detections[0]


def route_detection(
    detection: dict | None,
    class_to_bin: dict[str, Any],
    conf_route: float
) -> tuple[Any, str | None]:
    """
    Route a detection to the appropriate bin.

    Contract Rules 3 & 4:
    - If detection confidence < CONF_ROUTE → route to TRASH
    - Use class_to_bin lookup table
    - If class is unknown or unmapped → route to TRASH

    Args:
        detection: Selected detection dict, or None
        class_to_bin: Mapping from class_id to bin identifier
        conf_route: Routing confidence threshold

    Returns:
        Tuple of (bin_id, fallback_reason)
        - fallback_reason is None if normal routing
        - fallback_reason is a string explaining why TRASH was selected
    """
    # No detection provided
    if detection is None:
        return TRASH_BIN, "no_detection_provided"

    confidence = detection.get("confidence", 0.0)
    class_id = detection.get("class_id")

    # Contract Rule 3: Confidence gate
    if confidence < conf_route:
        return TRASH_BIN, f"confidence_below_route_threshold ({confidence:.3f} < {conf_route})"

    # Contract Rule 4: Class-to-bin mapping
    if class_id is None:
        return TRASH_BIN, "missing_class_id"

    if class_id not in class_to_bin:
        return TRASH_BIN, f"unmapped_class ({class_id})"

    # Successful routing
    return class_to_bin[class_id], None


def decide_bin(
    detections: list[dict],
    class_to_bin: dict[str, Any],
    conf_min: float = CONF_MIN,
    conf_route: float = CONF_ROUTE
) -> dict:
    """
    Main decision function: process detections and return bin decision.

    Implements the full inference policy pipeline:
    1. Filter detections by CONF_MIN
    2. Select active detection (largest bbox, tie-break by confidence)
    3. Route to bin via CONF_ROUTE gate and class mapping
    4. Handle all fallbacks to TRASH

    Contract Rule 5: Any runtime error → route to TRASH

    Args:
        detections: Raw list of detection dicts from model
        class_to_bin: Mapping from class_id to bin identifier
        conf_min: Minimum confidence for filtering (default: CONF_MIN)
        conf_route: Confidence threshold for routing (default: CONF_ROUTE)

    Returns:
        Decision dict containing:
        - timestamp: ISO format timestamp
        - all_detections: Original detections received
        - filtered_detections: Detections after confidence filtering
        - selected_detection: The chosen active detection (or None)
        - bin_decision: Final bin identifier
        - fallback_reason: Reason for TRASH routing (or None)
    """
    timestamp = datetime.now().isoformat()

    # Initialize result structure
    result = {
        "timestamp": timestamp,
        "all_detections": detections,
        "filtered_detections": [],
        "selected_detection": None,
        "bin_decision": TRASH_BIN,
        "fallback_reason": None
    }

    try:
        # Contract Rule 5: Wrap in try/except for error fallback

        # Handle None or invalid input
        if detections is None:
            result["fallback_reason"] = "null_detections_input"
            _log_decision(result)
            return result

        # Step 1: Filter detections
        filtered = filter_detections(detections, conf_min)
        result["filtered_detections"] = filtered

        # Contract Rule 1: No detections remain → TRASH
        if not filtered:
            result["fallback_reason"] = "no_detections_above_conf_min"
            _log_decision(result)
            return result

        # Step 2: Select active detection
        selected = select_active_detection(filtered)
        result["selected_detection"] = selected

        # Step 3: Route to bin
        bin_decision, fallback_reason = route_detection(
            selected, class_to_bin, conf_route
        )

        result["bin_decision"] = bin_decision
        result["fallback_reason"] = fallback_reason

    except Exception as e:
        # Contract Rule 5: Any runtime error → TRASH
        result["bin_decision"] = TRASH_BIN
        result["fallback_reason"] = f"runtime_error ({type(e).__name__}: {e})"
        logger.exception("Runtime error in decide_bin")

    _log_decision(result)
    return result


def _log_decision(result: dict) -> None:
    """
    Log the inference decision per contract requirements.

    Logs:
    - timestamp
    - all detections
    - selected detection
    - final bin decision
    - reason for trash fallback (if applicable)
    """
    log_message = (
        f"[INFERENCE] "
        f"ts={result['timestamp']} | "
        f"detections={len(result.get('all_detections') or [])} | "
        f"filtered={len(result.get('filtered_detections') or [])} | "
        f"selected={_summarize_detection(result.get('selected_detection'))} | "
        f"bin={result['bin_decision']}"
    )

    if result.get("fallback_reason"):
        log_message += f" | fallback_reason={result['fallback_reason']}"
        logger.warning(log_message)
    else:
        logger.info(log_message)


def _summarize_detection(detection: dict | None) -> str:
    """Create a brief summary string of a detection for logging."""
    if detection is None:
        return "None"

    class_id = detection.get("class_id", "?")
    confidence = detection.get("confidence", 0.0)
    area = _compute_bbox_area(detection)

    return f"{class_id}@{confidence:.2f}(area={area:.0f})"
