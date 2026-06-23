import os 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    lr: float = 0.001
    batch_size: int = 32
    model_name: str = "resnet"

    def __init__(self, json_path: str = "model_config.json"):
        import json
        with open(os.path.join(SCRIPT_DIR, json_path), "r") as f:
            data = json.load(f)

        for k, v in data.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown config key: {k}")