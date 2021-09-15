import cv2
import numpy as np
import pybullet as p


def normalize(x, minimum, maximum):
    return np.clip((x - np.mean(x)) / np.std(x), minimum, maximum)


def place_camera_in_front_of_object(bullet_object, cam_dist=0.7, roll=0.0, pitch=-20, yaw=0.0):
    target_point, _ = p.getBasePositionAndOrientation(bullet_object)
    return p.computeViewMatrixFromYawPitchRoll(np.asarray(target_point), cam_dist,
                                               yaw + 90, pitch, roll, 2)


def get_object_mask(img, color_bgr_low, color_bgr_high, k=np.ones((5, 5))):
    assert type(img) is np.ndarray and len(img.shape) == 3 and img.shape[-1] == 3
    assert type(color_bgr_low) in [tuple, list] and len(color_bgr_low) == 3
    assert type(color_bgr_high) in [tuple, list] and len(color_bgr_high) == 3
    assert all([c >= 0] for c in color_bgr_low)
    assert all([c >= 0] for c in color_bgr_high)
    assert all([high > low for low, high in zip(color_bgr_low, color_bgr_high)])

    img_cpy = cv2.medianBlur(img, 3)
    mask = cv2.inRange(img_cpy, np.asarray(color_bgr_low), np.asarray(color_bgr_high))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask[..., np.newaxis]
