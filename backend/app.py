import os
import tempfile
import math
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---- Pose backend (MediaPipe) ----
import mediapipe as mp
mp_pose = mp.solutions.pose

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# -----------------------------
# Geometry helpers
# -----------------------------
def angle_3pts(a, b, c):
    """
    Returns the angle ABC (in degrees) where A,B,C are (x,y) tuples.
    """
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = np.array([ax - bx, ay - by])
    v2 = np.array([cx - bx, cy - by])
    # handle zero-length vectors
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return None
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def angle_to_vertical(a, b):
    """
    Angle (deg) between vector BA and vertical axis (positive y down in image).
    0° = perfectly vertical; larger = leaning more.
    """
    ax, ay = a; bx, by = b
    v = np.array([ax - bx, ay - by])
    if np.linalg.norm(v) == 0:
        return None
    # vertical unit vector in image coords (downwards)
    vert = np.array([0, 1.0])
    cosang = np.dot(v, vert) / (np.linalg.norm(v) * np.linalg.norm(vert))
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def dist_point_to_line(p, a, b):
    """
    Perpendicular distance from point p to line through a-b (in pixels, normalized later).
    """
    px, py = p; ax, ay = a; bx, by = b
    ap = np.array([px - ax, py - ay])
    ab = np.array([bx - ax, by - ay])
    denom = np.linalg.norm(ab)
    if denom == 0:
        return None
    return float(np.linalg.norm(np.cross(ab, ap)) / denom)

# -----------------------------
# Landmark helpers
# -----------------------------
L = mp_pose.PoseLandmark

def lm_xy(landmarks, idx, image_w, image_h):
    lm = landmarks[idx]
    return (lm.x * image_w, lm.y * image_h)

def side_confidence(landmarks):
    """
    Choose left or right side by visibility – returns 'left' or 'right'.
    """
    left_vis = landmarks[L.LEFT_KNEE.value].visibility + landmarks[L.LEFT_HIP.value].visibility
    right_vis = landmarks[L.RIGHT_KNEE.value].visibility + landmarks[L.RIGHT_HIP.value].visibility
    return 'left' if left_vis >= right_vis else 'right'

# -----------------------------
# Squat analysis core
# -----------------------------
def analyze_squat_video(video_path, max_frames=600, stride=3):
    """
    Returns per-video metrics + heuristic scoring.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video."}

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Accumulators
    knee_angles = []
    hip_angles = []
    torso_angles = []     # torso lean relative to vertical
    ankle_dorsi = []      # proxy via angle foot-shin (ankle mobility rough)
    knee_valgus_dev = []  # lateral knee drift vs toe line
    hip_vs_knee_y = []    # to estimate depth (hip below knee)

    frame_count = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % stride != 0:
            continue
        if processed >= max_frames:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(image_rgb)
        processed += 1

        if not res.pose_landmarks:
            continue

        lms = res.pose_landmarks.landmark
        side = side_confidence(lms)
        if side == 'left':
            hip_idx, knee_idx, ankle_idx, shoulder_idx, foot_idx = L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE, L.LEFT_SHOULDER, L.LEFT_FOOT_INDEX
        else:
            hip_idx, knee_idx, ankle_idx, shoulder_idx, foot_idx = L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE, L.RIGHT_SHOULDER, L.RIGHT_FOOT_INDEX

        # Coordinates
        hip    = lm_xy(lms, hip_idx.value,     width, height)
        knee   = lm_xy(lms, knee_idx.value,    width, height)
        ankle  = lm_xy(lms, ankle_idx.value,   width, height)
        shoulder = lm_xy(lms, shoulder_idx.value, width, height)
        toe    = lm_xy(lms, foot_idx.value,    width, height)

        # Basic angles
        k_angle = angle_3pts(hip, knee, ankle)             # knee flexion (~180 = straight; ~90 = deep)
        h_angle = angle_3pts(shoulder, hip, knee)          # hip flexion
        t_angle = angle_to_vertical(shoulder, hip)         # torso lean relative to vertical (0 = upright)
        # Ankle dorsiflexion proxy: angle between shin (knee->ankle) and foot (ankle->toe)
        a_angle = angle_3pts(knee, ankle, toe)

        if all(v is not None for v in [k_angle, h_angle, t_angle, a_angle]):
            knee_angles.append(k_angle)
            hip_angles.append(h_angle)
            torso_angles.append(t_angle)
            ankle_dorsi.append(180 - a_angle)  # higher ~= more dorsiflexion

        # Knee tracking (valgus/varus) – distance of knee from foot line
        # Line from ankle to toe approximates foot direction; measure knee deviation.
        dev = dist_point_to_line(knee, ankle, toe)
        if dev is not None:
            # Normalize by thigh length to be scale invariant
            thigh_len = np.linalg.norm(np.array(hip) - np.array(knee)) + 1e-6
            knee_valgus_dev.append(dev / thigh_len)

        # Depth proxy: hip y below knee y when squatting (image y grows downward)
        hip_vs_knee_y.append(hip[1] - knee[1])

    cap.release()
    pose.close()

    if len(knee_angles) < 3:
        return {"error": "Not enough pose detections to analyze. Try better lighting, a full-body view, and slower reps."}

    # -----------------------------
    # Compute per-metric summaries
    # -----------------------------
    min_knee = float(np.percentile(knee_angles, 10))   # deepest (smallest angle)
    avg_torso = float(np.median(torso_angles))         # typical torso lean
    max_valgus = float(np.percentile(knee_valgus_dev, 90)) if knee_valgus_dev else 0.0
    max_dorsi = float(np.percentile(ankle_dorsi, 90))  # peak dorsiflexion
    depth_ratio = float(np.percentile(hip_vs_knee_y, 10))  # more positive means deeper (hip below knee)

    # -----------------------------
    # Heuristic scoring (0-100)
    # -----------------------------
    # Depth score: 90° knee = very deep; 100–110 decent; penalize >125 (shallow)
    if min_knee <= 95:
        depth_score = 95
    elif min_knee <= 110:
        depth_score = 85
    elif min_knee <= 125:
        depth_score = 70
    else:
        depth_score = 50

    # Torso neutrality: <15° great, 15–25 ok, >35 poor
    if avg_torso <= 15:
        torso_score = 95
    elif avg_torso <= 25:
        torso_score = 80
    elif avg_torso <= 35:
        torso_score = 65
    else:
        torso_score = 50

    # Knee tracking: normalized deviation – <0.15 great, 0.15–0.25 ok, >0.35 poor
    if max_valgus <= 0.15:
        tracking_score = 95
    elif max_valgus <= 0.25:
        tracking_score = 80
    elif max_valgus <= 0.35:
        tracking_score = 65
    else:
        tracking_score = 50

    # Ankle mobility proxy (more dorsiflexion helpful): >30° great; 20–30 ok; <15 limited
    if max_dorsi >= 30:
        ankle_score = 95
    elif max_dorsi >= 20:
        ankle_score = 80
    elif max_dorsi >= 15:
        ankle_score = 65
    else:
        ankle_score = 50

    detailed_breakdown = {
        "depth": {
            "score": int(depth_score),
            "feedback": f"Deepest knee angle ≈ {min_knee:.0f}°. Aim ~90–110° for most bodyweight back/air squats."
        },
        "torso_alignment": {
            "score": int(torso_score),
            "feedback": f"Median torso lean ≈ {avg_torso:.0f}°. Keep ribs down and spine neutral; brace before descent."
        },
        "knee_tracking": {
            "score": int(tracking_score),
            "feedback": f"Peak knee drift (normalized) ≈ {max_valgus:.2f}. Track knees over toes; avoid knees collapsing inward."
        },
        "ankle_mobility": {
            "score": int(ankle_score),
            "feedback": f"Est. peak dorsiflexion ≈ {max_dorsi:.0f}°. More dorsiflexion allows upright torso and deeper depth."
        }
    }

    # Overall score (weighted)
    overall_score = int(round(
        0.35 * depth_score +
        0.30 * torso_score +
        0.25 * tracking_score +
        0.10 * ankle_score
    ))

    # What’s right
    whats_right = []
    if depth_score >= 80:    whats_right.append("Good squat depth.")
    if torso_score >= 80:    whats_right.append("Solid torso control and bracing.")
    if tracking_score >= 80: whats_right.append("Knees tracking well over toes.")
    if ankle_score >= 80:    whats_right.append("Adequate ankle mobility for this variation.")

    # Corrections needed
    corrections_needed = []

    if tracking_score < 80:
        severity = "critical" if tracking_score < 60 else "warning"
        corrections_needed.append({
            "issue": "Knee valgus (knees caving in)",
            "severity": severity,
            "feedback": "Your knees show lateral drift relative to the foot line during descent.",
            "correction_instruction": "Screw feet into the floor (tripod foot), push knees out to track over 2nd–3rd toe, and slow the eccentric. Try a mini-band just above the knees for 2–3 warm-up sets."
        })

    if depth_score < 80:
        severity = "warning"
        corrections_needed.append({
            "issue": "Shallow depth",
            "severity": severity,
            "feedback": f"Deepest knee angle about {min_knee:.0f}°, suggesting limited depth.",
            "correction_instruction": "Elevate heels on small plates or use weightlifting shoes; sit between the hips while keeping heels down. Tempo squats (3–0–3) can build control."
        })

    if torso_score < 80:
        severity = "warning" if torso_score >= 60 else "critical"
        corrections_needed.append({
            "issue": "Excessive torso lean",
            "severity": severity,
            "feedback": f"Median torso lean ≈ {avg_torso:.0f}°, which may stress the lower back.",
            "correction_instruction": "Brace: big breath into the belly/obliques before each rep; keep chest and hips rising together. Try goblet squats to groove an upright pattern."
        })

    if ankle_score < 80:
        severity = "info"
        corrections_needed.append({
            "issue": "Limited ankle dorsiflexion",
            "severity": severity,
            "feedback": f"Estimated dorsiflexion ≈ {max_dorsi:.0f}°, which may limit depth and upright posture.",
            "correction_instruction": "Mobilize ankles: wall dorsiflexion rocks (2×10/side), calf raises with slow eccentrics, and 1–2 cm heel elevation in early blocks."
        })

    improvement_tips = [
        "Film from ~45° front and far enough to see feet–hips–shoulders.",
        "Brace: inhale, lock ribs down, then squat; exhale at the top.",
        "Drive evenly through tripod foot (big toe, little toe, heel).",
        "Use a slow 2–3s descent to control knee tracking.",
        "Warm up with 2×10 bodyweight tempo squats and ankle rocks."
    ]

    return {
        "overall_score": overall_score,
        "whats_right": whats_right,
        "corrections_needed": corrections_needed,
        "detailed_breakdown": detailed_breakdown,
        "improvement_tips": improvement_tips
    }

# -----------------------------
# Flask route
# -----------------------------
@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "Missing 'video' file field"}), 400

    exercise_type = request.form.get("exercise_type", "squat")
    file = request.files["video"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Basic file guard
    allowed = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format '{ext}'. Use one of {allowed}"}), 400

    # Save to temp and analyze
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if exercise_type != "squat":
            # For hackathon MVP, only squat is implemented.
            result = {"error": f"Exercise '{exercise_type}' not supported yet. Try 'squat'."}
        else:
            result = analyze_squat_video(tmp_path)

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result), 200

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=5000, debug=True)
