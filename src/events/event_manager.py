from collections import defaultdict

class EventManager:
    def __init__(
        self,
        inactivity_timeout: float = 2.0,
        min_event_duration: float = 0.5,
    ):
        self.inactivity_timeout = inactivity_timeout
        self.min_event_duration = min_event_duration

        self.event_active = False
        self.event_start_time = None
        self.last_motion_time = None

        self.event_id = 0
        self.label_counts = defaultdict(int)

    def update(self, motion_detected: bool, timestamp: float):
        """
        Update event state based on motion signal.

        Args:
            motion_detected (bool)
            timestamp (float): seconds

        Returns:
            "start", "end", or None
        """
        if motion_detected:
            self.last_motion_time = timestamp

            if not self.event_active:
                self._start_event(timestamp)
                return "start"

        else:
            if self.event_active:
                time_since_motion = timestamp - self.last_motion_time
                if time_since_motion >= self.inactivity_timeout:
                    self._end_event(timestamp)
                    return "end"

        return None

    
    def add_detections(self, labels):
        """
        Adds detected class labels (strings) to the current event.
        """
        for label in labels:
            self.label_counts[label] += 1

    def _start_event(self, timestamp: float):
        self.event_active = True
        self.event_start_time = timestamp
        self.event_id += 1
        self.label_counts.clear()

        print(f"[EVENT {self.event_id}] START at {timestamp:.2f}s")

    def _end_event(self, timestamp: float):
        duration = timestamp - self.event_start_time

        if duration >= self.min_event_duration:
            print(
                f"[EVENT {self.event_id}] END at {timestamp:.2f}s "
                f"(duration={duration:.2f}s)"
            )
        else:
            print(
                f"[EVENT {self.event_id}] DISCARDED "
                f"(duration={duration:.2f}s)"
            )

        self.event_active = False
        self.last_motion_time = None

    def get_label_summary(self):
        if not self.label_counts:
            return {
                "primary_label": "unknown",
                "confidence": 0.0,
                "label_distribution": {}
            }

        total = sum(self.label_counts.values())
        primary_label = max(self.label_counts, key=self.label_counts.get)
        confidence = self.label_counts[primary_label] / total

        return {
            "primary_label": primary_label,
            "confidence": round(confidence, 2),
            "label_distribution": dict(self.label_counts)
        }

    def get_event_metadata(self, end_time: float):
        label_summary = self.get_label_summary()

        return {
            "event_id": self.event_id,
            "start_time": self.event_start_time,
            "end_time": end_time,
            "duration": end_time - self.event_start_time,
            **label_summary
        }

    def reset(self):
        self.event_start_time = None
        self.label_counts.clear()