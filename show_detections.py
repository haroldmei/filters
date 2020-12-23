"""A program to show detection examples.
"""
from typing import Callable, Iterable, Tuple, Optional, List

import click
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

from numpy.random import randn
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
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

#########################################################################################
# for sliding, colliding (linear dynamic)
def get_linear_tracker():

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

def wrapToPi(a):
    if isinstance(a, list):    # backwards compatibility for lists (distinct from np.array)
        return [(x + np.pi) % (2*np.pi) - np.pi for x in a]
    return (a + np.pi) % (2*np.pi) - np.pi

# for circle type
def FirstOrderKF(R, Q, dt):
    """ Create first order Kalman filter. 
    Specify R and Q as floats."""
    
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.zeros(2)
    kf.P *= np.array([[100, 0], [0, 1]])
    kf.R *= R
    kf.Q = Q_discrete_white_noise(2, dt, Q)
    kf.F = np.array([[1., dt],
                     [0., 1]])
    kf.H = np.array([[1., 0]])
    return kf


# for circle-accel type
def SecondOrderKF(R_std, Q, dt, P=100):
    """ Create second order Kalman filter. 
    Specify R and Q as floats."""
    
    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.x = np.zeros(3)
    kf.P[0, 0] = P
    kf.P[1, 1] = 1
    kf.P[2, 2] = 1
    kf.R *= R_std**2
    kf.Q = Q_discrete_white_noise(3, dt, Q)
    kf.F = np.array([[1., dt, .5*dt*dt],
                     [0., 1.,       dt],
                     [0., 0.,       1.]])
    kf.H = np.array([[1., 0., 0.]])
    return kf

#######################################################################################
# for sample0~7 (nonlinear dynamic, could also consider other filters.)
def f_cv(x, dt):
    """ state transition function for a 
    constant velocity aircraft"""
    
    F = np.array([[1, dt, 0,  0],
                  [0,  1, 0,  0],
                  [0,  0, 1, dt],
                  [0,  0, 0,  1]])
    return F @ x

def h_cv(x):
    return x[[0, 2]]

def get_nonlinear_tracker():

    dt = IMSHOW_SLEEP_TIME/1000 # time step

    sigmas = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=1.)
    ukf = UKF(dim_x=4, dim_z=2, fx=f_cv, hx=h_cv, dt=dt, points=sigmas)
    ukf.x = np.array([0., 0., 0., 0.])
    ukf.R = np.diag([0.09, 0.09]) 
    ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1, var=0.02)
    ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1, var=0.02)
    return ukf

def get_tracker(detection_type):
    if detection_type == 'circle':
        #return FirstOrderKF(1, 0.03, IMSHOW_SLEEP_TIME/1000)
        return get_nonlinear_tracker()
    elif detection_type == 'circle-accel':
        #return SecondOrderKF(6., 0.02, IMSHOW_SLEEP_TIME/1000)
        return get_nonlinear_tracker()
    elif detection_type == 'sliding' or detection_type == 'collide':
        return get_linear_tracker()
    else:
        return get_nonlinear_tracker()

#######################################################################################
def draw_detections_inplace(
    image_arr: np.ndarray,
    detection_type: str,
    detections: Iterable[Detection],
    trackers: List[KalmanFilter],
    *,
    default_color: Color = GREEN,
    get_color_for_expected_id: Optional[Callable[[int], Color]] = None,
) -> np.ndarray:

    #centers = np.array([[(detection.box[0] + detection.box[2])/2.0, (detection.box[1] + detection.box[3])/2.0] for detection in detections])
    #orig = np.average(centers, axis=0)
    #r = np.linalg.norm(centers[0]-orig)
    #print(r, orig, detections)

    for detection, tracker in zip(detections, trackers):
        
        x_min, y_min, x_max, y_max = detection.box

        center_coordinates = (0, 0)
        x = (x_min + x_max) / 2.0
        y = (y_min + y_max) / 2.0

        tracker.predict()
        if detection_type == 'circle-accel':
            tracker.update([x, y])
            m = tracker.x
            center_coordinates = (int(m[0]), int(m[2]))
        elif detection_type == 'sliding' or detection_type == 'collide':
            tracker.update([x, y])
            m = np.dot(tracker.H, tracker.x)
            center_coordinates = (int(m[0][0]), int(m[1][0]))
        else: 
            tracker.update([x, y])
            m = tracker.x
            center_coordinates = (int(m[0]), int(m[2]))
        
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
        trackers.append(get_tracker(detection_type))

    for frame_index, detections in enumerate(detections_sample.detections):
        current_frame[...] = background_image
        random.shuffle(detections)
        draw_detections_inplace(
            current_frame,
            detection_type,
            detections,
            trackers,
            get_color_for_expected_id=lambda i: ALL_COLORS[i % len(ALL_COLORS)],
        )

        user_input = show_image_and_wait(current_frame)
        if user_input == KEY_ESC:
            break


if __name__ == "__main__":
    main()
