"""
Misalignment Stress-Test Helpers.

Synthetic pre/post misregistration: integer translations of an image with
zero fill and the schedule of shifts to evaluate. Shifting a boolean array
of ones the same way yields the mask of pixels that still carry imagery.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def shift_image(img_chw: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """
    Translate an image by whole pixels, filling uncovered pixels with zeros.

    Args:
        img_chw: Array whose last two axes are (H, W), e.g. (C, H, W) or (H, W).
        dy: Shift in rows; positive moves content down.
        dx: Shift in columns; positive moves content right.

    Returns:
        Shifted array with the same shape and dtype as the input.
    """
    if int(dy) != dy or int(dx) != dx:
        raise ValueError(f"Shifts must be whole pixels, got dy={dy}, dx={dx}")
    dy, dx = int(dy), int(dx)
    img = np.asarray(img_chw)
    if img.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions, got shape {img.shape}")
    h, w = img.shape[-2:]
    out = np.zeros_like(img)
    if abs(dy) >= h or abs(dx) >= w:
        return out
    src_y = slice(max(0, -dy), h - max(0, dy))
    dst_y = slice(max(0, dy), h - max(0, -dy))
    src_x = slice(max(0, -dx), w - max(0, dx))
    dst_x = slice(max(0, dx), w - max(0, -dx))
    out[..., dst_y, dst_x] = img[..., src_y, src_x]
    return out


def shift_schedule(max_px: int = 40, step: int = 5, directions: int = 8) -> list[tuple[int, int, int]]:
    """
    Shifts for a misalignment sweep: magnitudes 0, step, ..., max_px.

    Every non-zero magnitude is applied in `directions` evenly spaced
    directions starting along +x (angle 0, then counter-clockwise in image
    terms: 90 degrees is -y, i.e. up), rounded to whole pixels.

    Args:
        max_px: Largest shift magnitude in pixels.
        step: Magnitude increment in pixels.
        directions: Number of directions per magnitude (1 = +x only,
            4 = axis-aligned, 8 = axis-aligned and diagonal).

    Returns:
        List of (magnitude, dy, dx), starting with (0, 0, 0).
    """
    if step <= 0 or max_px < 0 or directions < 1:
        raise ValueError(f"Invalid schedule: max_px={max_px}, step={step}, directions={directions}")
    schedule = [(0, 0, 0)]
    angles = 2 * np.pi * np.arange(directions) / directions
    for magnitude in range(step, max_px + 1, step):
        for a in angles:
            dy = int(round(-magnitude * np.sin(a)))
            dx = int(round(magnitude * np.cos(a)))
            schedule.append((magnitude, dy, dx))
    return schedule
