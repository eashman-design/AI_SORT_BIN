import yaml


class DecisionEngine:
    def __init__(self, policy_path):
        with open(policy_path, "r") as f:
            self.policy = yaml.safe_load(f)

        self.bins = self.policy["bins"]
        self.thresholds = self.policy["confidence_thresholds"]
        self.fallback_bin = self.policy["fallback_bin"]
        self.max_detections = self.policy.get("max_detections", 1)

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

        threshold = self.thresholds.get(cls)

        if threshold is None:
            # Unknown class → trash
            return self.bins[self.fallback_bin]

        if conf < threshold:
            return self.bins[self.fallback_bin]

        return self.bins[cls]
