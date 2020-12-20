"""Contains factory functions for creating sequences of detections.
Each function here returns a list of detections for each frame, ie
a list of lists of detections.
"""
import json
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import numpy as np

SAMPLES_DIR = Path(__file__).parent.joinpath("samples")


class Detection(NamedTuple):
    box: np.ndarray
    expected_id: Optional[int] = None

    @staticmethod
    def from_normalized_rect(image_arr: np.ndarray, rect: Dict[str, float]) -> "Detection":
        image_h, image_w, _ = image_arr.shape
        return Detection(
            np.array(
                [
                    rect["x_min"] * image_w,
                    rect["y_min"] * image_h,
                    rect["x_max"] * image_w,
                    rect["y_max"] * image_h,
                ]
            ).astype(int),
            None,
        )


class DetectionsSample(NamedTuple):
    detections: List[List[Detection]]
    expected_track_count: int


def get_sliding_box_detections(image_arr: np.ndarray) -> DetectionsSample:
    detections = [
        [
            Detection(np.array([x, 100, x + 100, 200]), 1),
            Detection(np.array([x, 500, x + 100, 550]), 2),
        ]
        for x in range(0, 1000, 10)
    ]
    start_time = 2
    for n in range(len(detections) // 2):
        detections[start_time + n].append(
            Detection(np.array([10 * n + 10, 300, 10 * n + 40, 400]), 3)
        )
    return DetectionsSample(detections, 3)


def get_circle_detections(image_arr: np.ndarray, acceleration: float = 1.0) -> DetectionsSample:
    detections_length = 300
    angular_velocity = np.pi / 96
    box_size = 100
    img_h, img_w, _ = image_arr.shape
    center_x, center_y = img_w // 2, img_h // 2
    radius = min(img_h, img_w) // 2 - box_size

    detections = []
    for i in range(detections_length):
        top_left_points = [
            (
                center_x + radius * np.sin(np.pi * 2 * j / 3 + i * angular_velocity),
                center_y + radius * np.cos(np.pi * 2 * j / 3 + i * angular_velocity),
            )
            for j in range(3)
        ]
        detections_for_frame = [
            Detection(np.array([x, y, x + box_size, y + box_size]).astype(int), i)
            for i, (x, y) in enumerate(top_left_points)
        ]
        detections.append(detections_for_frame)
        angular_velocity *= acceleration
    return DetectionsSample(detections, 3)


def get_collide_detections(image_arr: np.ndarray) -> DetectionsSample:
    img_h, img_w, _ = image_arr.shape
    detections_length = 90
    box_size = 100
    velocity = img_w // detections_length

    detections = []
    for i in range(detections_length):
        left_position = i * velocity
        top_position = img_h // 2
        detections_for_frame = [
            Detection(
                np.array(
                    [
                        left_position,
                        top_position,
                        left_position + box_size,
                        top_position + box_size,
                    ]
                ),
                1,
            ),
            Detection(
                np.array(
                    [
                        img_w - box_size - (left_position + box_size),
                        top_position,
                        img_w - box_size - left_position,
                        top_position + box_size,
                    ]
                ),
                2,
            ),
        ]
        detections.append(detections_for_frame)
    return DetectionsSample(detections, 2)


def get_saved_detections(image_arr: np.ndarray, sample_num: int) -> DetectionsSample:
    sample_path = SAMPLES_DIR.joinpath(f"sample_{sample_num:03d}.json")
    raw_samples = json.loads(sample_path.read_text())
    detections = [
        [Detection.from_normalized_rect(image_arr, rect) for rect in dets]
        for dets in raw_samples["detections"]
    ]
    i = 0
    while not detections[i]:
        i += 1
    detections = detections[i:]
    return DetectionsSample(detections, raw_samples["expected_num_tracks"])


DETECTIONS_FACTORIES = {
    "sliding": get_sliding_box_detections,
    "circle": get_circle_detections,
    "circle-accel": lambda image_arr: get_circle_detections(image_arr, acceleration=1.01),
    "collide": get_collide_detections,
}
samples = [int(p.stem.split("_")[1]) for p in SAMPLES_DIR.glob("sample_*.json")]
for i in samples:

    def _fn(image_arr, n=i):
        return get_saved_detections(image_arr, n)

    DETECTIONS_FACTORIES[f"sample-{i}"] = _fn
