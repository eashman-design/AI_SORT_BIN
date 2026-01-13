"""
AI Sort Bin - Centralized Configuration Loader

Single source of truth for taxonomy labels and bin routing.
All scripts should import from here instead of hardcoding labels.

Usage:
    from scripts.common.config import get_config, get_labels, get_routing_map

    config = get_config()
    labels = get_labels()
    routing = get_routing_map()
"""
import json
from pathlib import Path
from functools import lru_cache

# Resolve paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "dataset" / "dataset_config.json"


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    Load and return the dataset configuration.

    Returns:
        dict: Full configuration from dataset_config.json

    Raises:
        ConfigError: If config file is missing or invalid
    """
    if not CONFIG_PATH.exists():
        raise ConfigError(
            f"Configuration file not found: {CONFIG_PATH}\n"
            "Ensure dataset/dataset_config.json exists."
        )

    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {CONFIG_PATH}: {e}")

    # Validate required fields
    required = ["labels", "routing_map", "physical_bins", "fallback_bin"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ConfigError(
            f"Missing required fields in {CONFIG_PATH}: {missing}"
        )

    return config


def get_labels() -> list[str]:
    """
    Get the canonical list of model labels.

    Returns:
        list[str]: Ordered list of class labels the model outputs
    """
    return get_config()["labels"]


def get_label_to_id() -> dict[str, int]:
    """
    Get mapping from label string to numeric ID.

    Returns:
        dict[str, int]: e.g., {"metal_container": 0, "plastic_container": 1, ...}
    """
    return {label: idx for idx, label in enumerate(get_labels())}


def get_id_to_label() -> dict[int, str]:
    """
    Get mapping from numeric ID to label string.

    Returns:
        dict[int, str]: e.g., {0: "metal_container", 1: "plastic_container", ...}
    """
    return {idx: label for idx, label in enumerate(get_labels())}


def get_routing_map() -> dict[str, str]:
    """
    Get mapping from model label to physical bin name.

    Returns:
        dict[str, str]: e.g., {"metal_container": "containers_bin", ...}
    """
    return get_config()["routing_map"]


def get_physical_bins() -> dict[str, int]:
    """
    Get mapping from physical bin name to hardware ID.

    Returns:
        dict[str, int]: e.g., {"containers_bin": 0, "paper_bin": 1, ...}
    """
    return get_config()["physical_bins"]


def get_fallback_bin() -> str:
    """
    Get the fallback bin name for uncertain/unknown predictions.

    Returns:
        str: Physical bin name (e.g., "landfill_bin")
    """
    return get_config()["fallback_bin"]


def get_fallback_bin_id() -> int:
    """
    Get the hardware ID for the fallback bin.

    Returns:
        int: Physical bin ID
    """
    fallback = get_fallback_bin()
    return get_physical_bins()[fallback]


def get_confidence_thresholds() -> dict[str, float]:
    """
    Get per-class confidence thresholds.

    Returns:
        dict[str, float]: Thresholds including "default" key
    """
    return get_config().get("confidence_thresholds", {"default": 0.70})


def get_threshold_for_label(label: str) -> float:
    """
    Get the confidence threshold for a specific label.

    Args:
        label: Model output label

    Returns:
        float: Threshold (uses default if label not specified)
    """
    thresholds = get_confidence_thresholds()
    return thresholds.get(label, thresholds.get("default", 0.70))


def get_unknown_threshold() -> float:
    """
    Get the minimum confidence below which predictions are rejected.

    Returns:
        float: Unknown/reject threshold
    """
    return get_config().get("unknown_threshold", 0.60)


def label_to_bin_id(label: str) -> int:
    """
    Resolve a model label to its physical bin ID.

    Args:
        label: Model output label (e.g., "metal_container")

    Returns:
        int: Physical bin ID

    Raises:
        KeyError: If label is not in routing_map (caller should handle)
    """
    routing = get_routing_map()
    bins = get_physical_bins()
    bin_name = routing[label]  # May raise KeyError
    return bins[bin_name]


def safe_label_to_bin_id(label: str) -> tuple[int, str | None]:
    """
    Safely resolve a model label to its physical bin ID.

    Returns fallback bin if label is unknown/unmapped.

    Args:
        label: Model output label

    Returns:
        tuple[int, str | None]: (bin_id, fallback_reason)
        - fallback_reason is None for normal routing
        - fallback_reason explains why fallback was used
    """
    routing = get_routing_map()
    bins = get_physical_bins()

    if label not in routing:
        return get_fallback_bin_id(), f"unmapped_label ({label})"

    bin_name = routing[label]
    if bin_name not in bins:
        return get_fallback_bin_id(), f"invalid_bin_name ({bin_name})"

    return bins[bin_name], None


def validate_labels_against_dataset(dataset_labels: list[str]) -> tuple[bool, str]:
    """
    Validate that dataset labels match config labels exactly.

    Args:
        dataset_labels: Labels found in the dataset/annotations

    Returns:
        tuple[bool, str]: (is_valid, message)
        - is_valid: True if labels match exactly
        - message: Description of any mismatch
    """
    config_labels = set(get_labels())
    dataset_labels_set = set(dataset_labels)

    missing_in_dataset = config_labels - dataset_labels_set
    extra_in_dataset = dataset_labels_set - config_labels

    if not missing_in_dataset and not extra_in_dataset:
        return True, "Labels match config exactly"

    errors = []
    if missing_in_dataset:
        errors.append(f"Missing in dataset: {sorted(missing_in_dataset)}")
    if extra_in_dataset:
        errors.append(f"Extra in dataset (not in config): {sorted(extra_in_dataset)}")

    return False, "; ".join(errors)


def reload_config() -> None:
    """
    Clear cached config and force reload on next access.

    Use this if config file has been modified during runtime.
    """
    get_config.cache_clear()
