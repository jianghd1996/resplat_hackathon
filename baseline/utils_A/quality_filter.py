"""Replace unusually soft uniform context frames with nearby train frames."""

import numpy as np

from .uniform16 import select as select_uniform16

MIN_RELATIVE_SHARPNESS = 0.8


def select(train_indices, count=16, centers=None, directions=None, image_paths=None):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("quality_filter requires OpenCV in the ReSplat environment") from exc
    chosen = select_uniform16(train_indices, count)
    if len(chosen) == len(train_indices):
        return chosen
    if image_paths is None or len(image_paths) != len(train_indices):
        raise ValueError("quality_filter needs one image path per training view")

    scores = []
    for path in image_paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(path)
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        scores.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
    median = float(np.median(scores))
    eligible = {i for i, score in enumerate(scores)
                if score >= MIN_RELATIVE_SHARPNESS * median}
    selected = set(chosen)
    for old in chosen:
        if old in eligible:
            continue
        available = eligible - selected
        if not available:
            break
        replacement = min(available, key=lambda i: (abs(i - old), i))
        selected.remove(old)
        selected.add(replacement)
    return sorted(selected)
