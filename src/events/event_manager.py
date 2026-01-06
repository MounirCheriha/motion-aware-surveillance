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

        self.event_count = 0

    def update(self, motion_detected: bool, timestamp: float):
        """
        Update event state based on motion signal.

        Args:
            motion_detected (bool)
            timestamp (float): seconds
        """
        if motion_detected:
            self.last_motion_time = timestamp

            if not self.event_active:
                self._start_event(timestamp)

        else:
            if self.event_active:
                time_since_motion = timestamp - self.last_motion_time
                if time_since_motion >= self.inactivity_timeout:
                    self._end_event(timestamp)

    def _start_event(self, timestamp: float):
        self.event_active = True
        self.event_start_time = timestamp
        self.event_count += 1

        print(f"[EVENT {self.event_count}] START at {timestamp:.2f}s")

    def _end_event(self, timestamp: float):
        duration = timestamp - self.event_start_time

        if duration >= self.min_event_duration:
            print(
                f"[EVENT {self.event_count}] END at {timestamp:.2f}s "
                f"(duration={duration:.2f}s)"
            )
        else:
            print(
                f"[EVENT {self.event_count}] DISCARDED "
                f"(duration={duration:.2f}s)"
            )

        self.event_active = False
        self.event_start_time = None
        self.last_motion_time = None
