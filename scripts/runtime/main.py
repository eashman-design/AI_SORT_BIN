"""
AI Sort Bin – Runtime Entry Point

Execution:
    python -m scripts.runtime.main

Must be run from the repository root directory.
"""
import sys
import time

from .state_machine import StateMachine
from scripts.decision.decision_engine import DecisionEngine


def _validate_environment():
    """Validate that the runtime environment is correctly configured."""
    from pathlib import Path

    # Determine repo root from this file's location
    repo_root = Path(__file__).resolve().parents[2]

    # Verify essential directories exist
    required_dirs = [
        repo_root / "scripts" / "decision",
        repo_root / "scripts" / "runtime",
    ]
    for d in required_dirs:
        if not d.is_dir():
            sys.exit(
                f"[ERROR] Required directory not found: {d}\n"
                "Ensure you are running from the repository root:\n"
                "    python -m scripts.runtime.main"
            )



# ---------------- Stub components ---------------- #

class InferenceEngineStub:
    def predict(self, frame):
        """
        Stub inference.
        Replace with real model inference later.
        """
        print("[INFERENCE] Stub prediction")
        # Use a label from the canonical taxonomy
        return [
            {"class": "plastic_container", "confidence": 0.85}
        ]


class ActuatorStub:
    def actuate(self, bin_id):
        """
        Stub actuator.
        Replace with GPIO / servo control later.
        """
        print(f"[ACTUATOR] Actuating bin {bin_id}")


# ---------------- Main runtime ---------------- #

def main():
    _validate_environment()
    print("[SYSTEM] Starting AI Sort Bin runtime")

    inference_engine = InferenceEngineStub()
    decision_engine = DecisionEngine()  # Uses centralized config
    actuator = ActuatorStub()

    state_machine = StateMachine(
        inference_engine=inference_engine,
        decision_engine=decision_engine,
        actuator=actuator,
        config={
            "reset_delay": 0.5
        }
    )

    try:
        while True:
            state_machine.run_once()
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutdown requested by user")

    finally:
        print("[SYSTEM] Runtime stopped cleanly")


if __name__ == "__main__":
    main()
