"""Ordered video-frame extraction with OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def extract_frames_from_video(
    video_path: str | Path,
    *,
    max_frames: int = 60,
    stride: int = 1,
) -> tuple[list[np.ndarray], float]:
    """Extract frames in original temporal order.

    Returns a list of RGB frames and the video's reported frames-per-second value.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1")
    source = Path(video_path)
    if not source.exists():
        raise FileNotFoundError(f"Video file not found: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open the uploaded video.")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    index = 0
    try:
        while len(frames) < max_frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if index % stride == 0:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError("No readable frames were found in the video.")
    return frames, fps
