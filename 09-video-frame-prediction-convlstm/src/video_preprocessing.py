"""Safe handling of uploaded video files and ZIP archives of ordered image frames."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
import zipfile

import numpy as np
from PIL import Image

from .data_preprocessing import prepare_frame_sequence, resize_frame
from .frame_extraction import extract_frames_from_video

ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def preprocess_video_bytes(
    payload: bytes,
    filename: str,
    *,
    input_frames: int = 6,
    target_size: tuple[int, int] = (32, 32),
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float | int]]:
    """Create a model sequence and optional next-frame target from uploaded video bytes."""
    suffix = Path(filename).suffix or ".mp4"
    with NamedTemporaryFile(suffix=suffix, delete=True) as temporary:
        temporary.write(payload)
        temporary.flush()
        frames, fps = extract_frames_from_video(
            temporary.name,
            max_frames=input_frames + 1,
            stride=stride,
        )
    sequence = prepare_frame_sequence(frames, input_frames, target_size, grayscale=True)
    target = None
    if len(frames) > input_frames:
        target = resize_frame(frames[input_frames], target_size, grayscale=True)
    metadata = {"frames_read": len(frames), "reported_fps": fps, "stride": stride}
    return sequence, target, metadata


def preprocess_frame_zip_bytes(
    payload: bytes,
    *,
    input_frames: int = 6,
    target_size: tuple[int, int] = (32, 32),
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Read lexicographically sorted images from a ZIP archive while preserving order."""
    images: list[np.ndarray] = []
    filenames: list[str] = []
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        members = sorted(
            name for name in archive.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() in ALLOWED_IMAGE_SUFFIXES
        )
        if len(members) < input_frames:
            raise ValueError(
                f"ZIP must contain at least {input_frames} supported image files; found {len(members)}."
            )
        for name in members[: input_frames + 1]:
            with archive.open(name) as handle:
                image = Image.open(handle).convert("RGB")
                images.append(np.asarray(image))
                filenames.append(name)
    sequence = prepare_frame_sequence(images, input_frames, target_size, grayscale=True)
    target = resize_frame(images[input_frames], target_size, grayscale=True) if len(images) > input_frames else None
    return sequence, target, filenames
