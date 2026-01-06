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
        self.event_labels = set()

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

    
    def add_detections(self, detections):
        for d in detections:
            self.event_labels.add(d.label)

    def _start_event(self, timestamp: float):
        self.event_active = True
        self.event_start_time = timestamp
        self.event_id += 1
        self.event_labels.clear()

        print(f"[EVENT {self.event_id}] START at {timestamp:.2f}s")

    def _end_event(self, timestamp: float):
        duration = timestamp - self.event_start_time

        label_summary = ", ".join(sorted(self.event_labels)) or "unknown"

        if duration >= self.min_event_duration:
            print(
                f"[EVENT {self.event_id}] END at {timestamp:.2f}s "
                f"(duration={duration:.2f}s, labels={label_summary})"
            )
        else:
            print(
                f"[EVENT {self.event_id}] DISCARDED "
                f"(duration={duration:.2f}s)"
            )

        self.event_active = False
        self.event_start_time = None
        self.last_motion_time = None

    def get_event_metadata(self, end_time: float):
        duration = end_time - self.event_start_time

        return {
            "event_id": self.event_id,
            "start_time": round(self.event_start_time, 2),
            "end_time": round(end_time, 2),
            "duration": round(duration, 2),
            "labels": sorted(self.event_labels),
        }