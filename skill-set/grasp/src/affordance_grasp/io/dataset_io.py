from pathlib import Path
import json

import numpy as np
import cv2


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_rgbd_bundle(path, depth, K, rgb, seg=None):
    path = Path(path)
    payload = {
        "depth": depth.astype(np.float32),
        "K": K.astype(np.float32),
        "rgb": rgb,
    }
    if seg is not None:
        payload["seg"] = seg
    if path.suffix.lower() == ".npz":
        np.savez_compressed(path, **payload)
    else:
        np.save(path, payload, allow_pickle=True)


def load_rgbd_bundle(path):
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    if path.suffix.lower() == ".npz":
        payload = {key: data[key] for key in data.files}
    else:
        if getattr(data, "shape", None) != ():
            raise ValueError(f"{path} is not a dict-style .npy bundle")
        payload = data.item()

    required = {"depth", "K", "rgb"}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
    return payload


def load_rgb_image(path):
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read RGB image from {path}")
    return image


def load_mask(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        if "mask" in data.files:
            return data["mask"]
        if len(data.files) == 1:
            return data[data.files[0]]
        raise ValueError(f"{path} should contain a 'mask' array")

    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to read mask from {path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
