import math
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
L = mp_pose.PoseLandmark

# ---------- Geometry ----------
def angle_3pts(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = np.array([ax - bx, ay - by])
    v2 = np.array([cx - bx, cy - by])
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return None
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def angle_to_vertical(a, b):
    ax, ay = a; bx, by = b
    v = np.array([ax - bx, ay - by])
    if np.linalg.norm(v) == 0:
        return None
    vert = np.array([0, 1.0])  # image y grows downward
    cosang = np.dot(v, vert) / (np.linalg.norm(v) * np.linalg.norm(vert))
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def dist_point_to_line(p, a, b):
    px, py = p; ax, ay = a; bx, by = b
    ap = np.array([px - ax, py - ay])
    ab = np.array([bx - ax, by - ay])
    denom = np.linalg.norm(ab)
    if denom == 0:
        return None
    return float(np.linalg.norm(np.cross(ab, ap)) / denom)

# ---------- Landmarks ----------
def lm_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)

def choose_side_for_leg(landmarks):
    left_vis = landmarks[L.LEFT_KNEE.value].visibility + landmarks[L.LEFT_HIP.value].visibility
    right_vis = landmarks[L.RIGHT_KNEE.value].visibility + landmarks[L.RIGHT_HIP.value].visibility
    return 'left' if left_vis >= right_vis else 'right'

def choose_side_for_arm(landmarks):
    left = (landmarks[L.LEFT_SHOULDER.value].visibility +
            landmarks[L.LEFT_ELBOW.value].visibility +
            landmarks[L.LEFT_WRIST.value].visibility)
    right = (landmarks[L.RIGHT_SHOULDER.value].visibility +
             landmarks[L.RIGHT_ELBOW.value].visibility +
             landmarks[L.RIGHT_WRIST.value].visibility)
    return 'left' if left >= right else 'right'

# ---------- Pose factory ----------
def make_pose():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
