import yaml
from pathlib import Path 

DEFAULT_BINS = {
    "paper": 0,
    "plastic": 1,
    "glass": 2,
    "metal": 3,
    "trash": 4
}

class DecisionEngine:
    def __init__(self, policy_path: str):
        base_dir = Path(__file__).resolve().parent
        policy_file = base_dir / policy_path

        if not policy_file.exists():
            raise FileNotFoundError(
                f"Policy file not found: {policy_file}"
            )

        with open(policy_file, "r") as f:
            self.policy = yaml.safe_load(f) or {}

        # --- REQUIRED FIELDS / DEFAULTS ---
        # Support both YAML keys: "confidence_thresholds" (preferred) or "thresholds" (legacy)
        self.thresholds = self.policy.get("confidence_thresholds") or self.policy.get("thresholds", {})
        self.default_threshold = float(self.policy.get("default_threshold", 0.5))
        self.fallback_bin = self.policy.get("fallback_bin", "trash")
        self.bins = self.policy.get("bins", DEFAULT_BINS)
        

    def decide(self, predictions):
        """
        predictions: list of dicts
        Example:
        [
          {"class": "paper", "confidence": 0.82}
        ]
        Returns: bin_id (int)
        """

        if not predictions:
            return self.bins[self.fallback_bin]

        # Sort by confidence
        predictions = sorted(
            predictions,
            key=lambda x: x["confidence"],
            reverse=True
        )

        top = predictions[0]
        cls = top["class"]
        conf = top["confidence"]

        threshold = float(
            self.thresholds.get(cls, self.default_threshold)
        )

        if threshold is None:
            # Unknown class → trash
            return self.bins[self.fallback_bin]

        if conf < threshold:
            return self.bins[self.fallback_bin]

        return self.bins[cls]
