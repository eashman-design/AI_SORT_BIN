from enum import Enum, auto
import time


class State(Enum):
    IDLE = auto()
    DETECT = auto()
    DECIDE = auto()
    ACTUATE = auto()
    RESET = auto()


class StateMachine:
    def __init__(self, inference_engine, decision_engine, actuator, config=None):
        """
        inference_engine: object with method -> predict(frame)
        decision_engine: object with method -> decide(predictions)
        actuator: object with method -> actuate(bin_id)
        config: optional dict for timing, thresholds, etc.
        """
        self.state = State.IDLE
        self.inference_engine = inference_engine
        self.decision_engine = decision_engine
        self.actuator = actuator
        self.config = config or {}

        self.last_predictions = None
        self.selected_bin = None

    def run_once(self):
        """Run exactly one state transition."""
        if self.state == State.IDLE:
            self._idle()

        elif self.state == State.DETECT:
            self._detect()

        elif self.state == State.DECIDE:
            self._decide()

        elif self.state == State.ACTUATE:
            self._actuate()

        elif self.state == State.RESET:
            self._reset()

    # ---------------- State handlers ---------------- #

    def _idle(self):
        print("[STATE] IDLE")
        # Placeholder: always transition for now
        self.state = State.DETECT

    def _detect(self):
        print("[STATE] DETECT")
        # Stub frame (replace with camera frame later)
        fake_frame = None
        self.last_predictions = self.inference_engine.predict(fake_frame)
        self.state = State.DECIDE

    def _decide(self):
        print("[STATE] DECIDE")
        self.selected_bin = self.decision_engine.decide(self.last_predictions)
        self.state = State.ACTUATE

    def _actuate(self):
        print(f"[STATE] ACTUATE → bin {self.selected_bin}")
        self.actuator.actuate(self.selected_bin)
        self.state = State.RESET

    def _reset(self):
        print("[STATE] RESET")
        self.last_predictions = None
        self.selected_bin = None
        time.sleep(self.config.get("reset_delay", 0.5))
        self.state = State.IDLE
