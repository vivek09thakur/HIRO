import json

import h5py
import numpy as np


class IntentModel:
    """NumPy forward pass over the Dense layers saved in a legacy Keras .h5 file."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray, str]]) -> None:
        self.layers = layers

    @classmethod
    def from_h5(cls, path: str) -> "IntentModel":
        with h5py.File(path, "r") as f:
            config = f.attrs["model_config"]
            config = json.loads(config.decode() if isinstance(config, bytes) else config)
            weights = f["model_weights"]
            layers = []
            for layer in config["config"]["layers"]:
                if layer["class_name"] != "Dense":
                    continue
                name = layer["config"]["name"]
                group = weights[name][name]
                layers.append(
                    (
                        np.asarray(group["kernel:0"], dtype=np.float32),
                        np.asarray(group["bias:0"], dtype=np.float32),
                        layer["config"]["activation"],
                    )
                )
        if not layers:
            raise ValueError(f"No Dense layers found in {path}")
        return cls(layers)

    def predict(self, x) -> np.ndarray:
        h = np.asarray(x, dtype=np.float32)
        for kernel, bias, activation in self.layers:
            h = h @ kernel + bias
            if activation == "relu":
                h = np.maximum(h, 0.0)
            elif activation == "softmax":
                h = h - h.max(axis=1, keepdims=True)
                e = np.exp(h)
                h = e / e.sum(axis=1, keepdims=True)
            elif activation != "linear":
                raise ValueError(f"Unsupported activation: {activation}")
        return h
