import json
import os


class MetadataWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.events = []

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def add_event(self, event_data: dict):
        self.events.append(event_data)

    def save(self):
        with open(self.output_path, "w") as f:
            json.dump(self.events, f, indent=2)