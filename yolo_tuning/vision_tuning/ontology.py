import json
import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Ontology:
    """Handle ontology prompts and exporting YOLO data.yaml."""

    prompt_to_label: Dict[str, str]
    source_path: str

    @classmethod
    def from_path(cls, ontology_path: str) -> "Ontology":
        if not os.path.exists(ontology_path):
            raise FileNotFoundError(f"Ontology file not found: {ontology_path}")
        with open(ontology_path, "r") as f:
            data = json.load(f)
        return cls(prompt_to_label=data, source_path=ontology_path)

    @property
    def class_names(self) -> List[str]:
        return list(self.prompt_to_label.values())

    @property
    def prompts(self) -> List[str]:
        return list(self.prompt_to_label.keys())

    def to_data_yaml(self, dataset_path: str, yaml_path: str) -> None:
        """Write a YOLO data.yaml file for training."""
        data_yaml = {
            "path": os.path.abspath(dataset_path),
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.class_names),
            "names": self.class_names,
        }
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        import yaml  # Lazy import keeps base dependency light

        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
