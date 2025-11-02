import cv2
import numpy as np
from utils import (
    make_pose, angle_3pts, angle_to_vertical, dist_point_to_line,
    lm_xy, choose_side_for_leg
)
import mediapipe as mp
L = mp.solutions.pose.PoseLandmark

def analyze_squat_video(video_path, max_frames=600, stride=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video."}

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

    pose = make_pose()

    knee_angles, hip_angles, torso_angles, ankle_dorsi = [], [], [], []
    knee_valgus_dev, hip_vs_knee_y = [], []

    processed = 0; frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if frame_idx % stride != 0: continue
        if processed >= max_frames: break

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks: continue
        processed += 1

        lms = res.pose_landmarks.landmark
        side = choose_side_for_leg(lms)

        if side == 'left':
            hip_i, knee_i, ankle_i, sh_i, toe_i = L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE, L.LEFT_SHOULDER, L.LEFT_FOOT_INDEX
        else:
            hip_i, knee_i, ankle_i, sh_i, toe_i = L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE, L.RIGHT_SHOULDER, L.RIGHT_FOOT_INDEX

        hip     = lm_xy(lms, hip_i.value, w, h)
        knee    = lm_xy(lms, knee_i.value, w, h)
        ankle   = lm_xy(lms, ankle_i.value, w, h)
        shoulder= lm_xy(lms, sh_i.value,  w, h)
        toe     = lm_xy(lms, toe_i.value, w, h)

        k_ang = angle_3pts(hip, knee, ankle)
        h_ang = angle_3pts(shoulder, hip, knee)
        t_ang = angle_to_vertical(shoulder, hip)
        a_ang = angle_3pts(knee, ankle, toe)  # dorsiflex proxy

        if all(v is not None for v in [k_ang, h_ang, t_ang, a_ang]):
            knee_angles.append(k_ang)
            hip_angles.append(h_ang)
            torso_angles.append(t_ang)
            ankle_dorsi.append(180 - a_ang)

        dev = dist_point_to_line(knee, ankle, toe)
        if dev is not None:
            thigh_len = np.linalg.norm(np.array(hip) - np.array(knee)) + 1e-6
            knee_valgus_dev.append(dev / thigh_len)
        hip_vs_knee_y.append(hip[1] - knee[1])

    cap.release(); pose.close()

    if len(knee_angles) < 3:
        return {"error": "Not enough pose detections to analyze. Ensure full-body in frame and decent lighting."}

    min_knee  = float(np.percentile(knee_angles, 10))
    avg_torso = float(np.median(torso_angles))
    max_valg  = float(np.percentile(knee_valgus_dev, 90)) if knee_valgus_dev else 0.0
    max_dorsi = float(np.percentile(ankle_dorsi, 90))

    # --- scoring heuristics ---
    depth_score = 95 if min_knee <= 80 else 85 if min_knee <= 90 else 70 if min_knee <= 100 else 50
    torso_score = 95 if avg_torso >= 175 else 80 if avg_torso >= 170 else 65 if avg_torso >= 165 else 50
    track_score = 95 if max_valg >= 0.95 else 80 if max_valg >= 0.90 else 65 if max_valg >= 0.85 else 50
    ankle_score = 95 if max_dorsi >= 30 else 80 if max_dorsi >= 20 else 65 if max_dorsi >= 15 else 50

    detailed = {
        "depth": {"score": int(depth_score), "feedback": f"Deepest knee angle ≈ {min_knee:.0f}°."},
        "torso_alignment": {"score": int(torso_score), "feedback": f"Median torso lean ≈ {avg_torso:.0f}°."},
        "knee_tracking": {"score": int(track_score), "feedback": f"Knee drift (norm) ≈ {max_valg:.2f}."},
        "ankle_mobility": {"score": int(ankle_score), "feedback": f"Peak dorsiflexion proxy ≈ {max_dorsi:.0f}°."}
    }

    overall = int(round(0.35*depth_score + 0.30*torso_score + 0.25*track_score + 0.10*ankle_score))

    whats_right = []
    if depth_score >= 80: whats_right.append("Good squat depth.")
    if torso_score >= 80: whats_right.append("Solid torso control.")
    if track_score >= 80: whats_right.append("Knees tracking well.")
    if ankle_score >= 80: whats_right.append("Adequate ankle mobility.")

    corrections = []
    if track_score < 80:
        corrections.append({
            "issue": "Knee valgus",
            "severity": "critical" if track_score < 60 else "warning",
            "feedback": "Knees show lateral drift vs toe line.",
            "correction_instruction": "Screw feet into floor, push knees over 2nd–3rd toe; add mini-band warm-ups."
        })
    if depth_score < 80:
        corrections.append({
            "issue": "Shallow depth",
            "severity": "warning",
            "feedback": f"Deepest knee angle {min_knee:.0f}° suggests limited depth.",
            "correction_instruction": "Try light heel elevation and tempo squats (3–0–3) to build control."
        })
    if torso_score < 80:
        corrections.append({
            "issue": "Excessive torso lean",
            "severity": "critical" if torso_score < 60 else "warning",
            "feedback": f"Torsion/lean ≈ {avg_torso:.0f}° may stress lower back.",
            "correction_instruction": "Brace and keep chest/hips rising together; try goblet squats."
        })

    tips = [
        "Film from ~45° front, full body in frame.",
        "Brace before descent; exhale on top.",
        "Tripod foot pressure; slow 2–3s eccentric."
    ]

    return {
        "overall_score": overall,
        "whats_right": whats_right,
        "corrections_needed": corrections,
        "detailed_breakdown": detailed,
        "improvement_tips": tips
    }
