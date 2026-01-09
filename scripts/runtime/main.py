import time

from state_machine import StateMachine
from scripts.decision.decision_engine import DecisionEngine



# ---------------- Stub components ---------------- #

class InferenceEngineStub:
    def predict(self, frame):
        """
        Stub inference.
        Replace with real model inference later.
        """
        print("[INFERENCE] Stub prediction")
        return [
            {"class": "trash", "confidence": 0.95}
        ]


decision_engine = DecisionEngine(
    policy_path="decision/bin_policy.yaml"
)


class ActuatorStub:
    def actuate(self, bin_id):
        """
        Stub actuator.
        Replace with GPIO / servo control later.
        """
        print(f"[ACTUATOR] Actuating bin {bin_id}")


# ---------------- Main runtime ---------------- #

def main():
    print("[SYSTEM] Starting AI Sort Bin runtime")

    inference_engine = InferenceEngineStub()
    decision_engine = DecisionEngine()
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
