"""A program to show detection examples.
"""
from typing import Callable, Iterable, Tuple, Optional, List

import click
import cv2
import numpy as np

import matplotlib.pyplot as plt

from numpy.random import randn
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from detections import Detection, DETECTIONS_FACTORIES

Color = Tuple[int, int, int]

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
ALL_COLORS = [RED, GREEN, BLUE]

KEY_ESC = 27
BOX_THICKNESS = 2
TRACKER_THICKNESS = 1
IMSHOW_SLEEP_TIME = 50

def get_tracker():

    # KF related
    dt = IMSHOW_SLEEP_TIME/1000 # time step
    R_std = 0.35
    Q_std = 0.04
    M_TO_FT = 1 / 0.3048

    tracker = KalmanFilter(dim_x=4, dim_z=2)
    tracker.F = np.array([[1, 0, dt,  0],
                      [0, 1,  0, dt],
                      [0, 0,  1,  0],
                      [0, 0,  0,  1]])

    tracker.H = np.array([[M_TO_FT, 0, 0, 0],
                      [0, M_TO_FT, 0, 0]])

    tracker.R = np.eye(2) * R_std**2
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
    tracker.Q[0,0] = q[0,0]
    tracker.Q[1,1] = q[0,0]
    tracker.Q[2,2] = q[1,1]
    tracker.Q[3,3] = q[1,1]
    tracker.Q[0,2] = q[0,1]
    tracker.Q[2,0] = q[0,1]
    tracker.Q[1,3] = q[0,1]
    tracker.Q[3,1] = q[0,1]
    tracker.x = np.array([[0, 0, 0, 0]]).T
    tracker.P = np.eye(4) * 500.
    return tracker


def draw_detections_inplace(
    image_arr: np.ndarray,
    detections: Iterable[Detection],
    trackers: List[KalmanFilter],
    *,
    default_color: Color = GREEN,
    get_color_for_expected_id: Optional[Callable[[int], Color]] = None,
) -> np.ndarray:
    for detection in detections:
        
        x_min, y_min, x_max, y_max = detection.box
        x = (x_min + x_max) / 2.0
        y = (y_min + y_max) / 2.0

        tracker = trackers[detection.expected_id]
        tracker.predict()
        tracker.update([x, y])

        # measurement from corrected belief
        m = np.dot(tracker.H, tracker.x)
        #m = tracker.x
        center_coordinates = (int(m[0][0]), int(m[1][0]))
        #print(x, y, m, tracker)
        

        if get_color_for_expected_id and detection.expected_id is not None:
            color = get_color_for_expected_id(detection.expected_id)
        else:
            color = default_color
        cv2.rectangle(
            image_arr,
            (x_min, y_min),
            (x_max, y_max),
            color=color,
            thickness=BOX_THICKNESS,
        )

        # use circle to track the rectangles. For simplicity, use the same radius value.
        
        cv2.circle(
            image_arr, 
            center_coordinates, 
            radius = 70, 
            color = color, 
            thickness = TRACKER_THICKNESS
        )
        
    return image_arr


def show_image_and_wait(
    image_arr: np.ndarray,
    *,
    should_wait: bool = False,
    sleep_time_ms: int = IMSHOW_SLEEP_TIME,
) -> int:
    cv2.imshow("window", image_arr)

    user_input = None
    if should_wait:
        while True:
            user_input = cv2.waitKey()
            if user_input in [ord("q"), KEY_ESC]:
                break
    else:
        user_input = cv2.waitKey(sleep_time_ms)
    return user_input


@click.command(help="Show detection samples")
@click.option(
    "-t",
    "--detection-type",
    help="Type of sample to show",
    type=click.Choice(list(DETECTIONS_FACTORIES)),
    required=True,
)
def main(detection_type):
    background_image = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    current_frame = np.empty_like(background_image)
    detections_sample = DETECTIONS_FACTORIES[detection_type](background_image)

    num = len(detections_sample.detections)
    trackers = []
    for i in range(num):
        trackers.append(get_tracker())

    for frame_index, detections in enumerate(detections_sample.detections):
        current_frame[...] = background_image
        draw_detections_inplace(
            current_frame,
            detections,
            trackers,
            get_color_for_expected_id=lambda i: ALL_COLORS[i % len(ALL_COLORS)],
        )

        user_input = show_image_and_wait(current_frame)
        if user_input == KEY_ESC:
            break


if __name__ == "__main__":
    main()
