import cv2
import numpy as np
from utils import (
    make_pose, angle_3pts, dist_point_to_line, lm_xy, choose_side_for_arm
)
import mediapipe as mp
L = mp.solutions.pose.PoseLandmark

def analyze_pushup_video(video_path, max_frames=600, stride=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video."}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

    pose = make_pose()

    elbow_angles, body_dev, neck_tilt, hand_offset = [], [], [], []
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
        side = choose_side_for_arm(lms)

        if side == 'left':
            sh, el, wr, hip, an, ear = L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST, L.LEFT_HIP, L.LEFT_ANKLE, L.LEFT_EAR
        else:
            sh, el, wr, hip, an, ear = L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST, L.RIGHT_HIP, L.RIGHT_ANKLE, L.RIGHT_EAR

        shoulder = lm_xy(lms, sh.value, w, h)
        elbow    = lm_xy(lms, el.value, w, h)
        wrist    = lm_xy(lms, wr.value, w, h)
        hip_pt   = lm_xy(lms, hip.value, w, h)
        ankle    = lm_xy(lms, an.value, w, h)
        ear_pt   = lm_xy(lms, ear.value, w, h)

        e_ang = angle_3pts(shoulder, elbow, wrist)  # elbow depth
        dev   = dist_point_to_line(hip_pt, shoulder, ankle)
        sa_len = np.linalg.norm(np.array(shoulder) - np.array(ankle)) + 1e-6
        dev_norm = dev / sa_len if dev is not None else None

        # neck tilt vs horizontal
        v = np.array([ear_pt[0] - shoulder[0], ear_pt[1] - shoulder[1]])
        if np.linalg.norm(v) > 0:
            horiz = np.array([1.0, 0.0])
            cosang = np.dot(v, horiz) / (np.linalg.norm(v) * np.linalg.norm(horiz))
            cosang = np.clip(cosang, -1.0, 1.0)
            neck_deg = float(abs(np.degrees(np.arccos(cosang))))
        else:
            neck_deg = None

        upper_arm_len = np.linalg.norm(np.array(shoulder) - np.array(elbow)) + 1e-6
        hand_off = abs(wrist[0] - shoulder[0]) / upper_arm_len

        if e_ang is not None: elbow_angles.append(e_ang)
        if dev_norm is not None: body_dev.append(dev_norm)
        if neck_deg is not None: neck_tilt.append(neck_deg)
        hand_offset.append(hand_off)

    cap.release(); pose.close()

    if len(elbow_angles) < 3:
        return {"error": "Not enough pose detections for push-up. Use a side view and good lighting."}

    min_elbow   = float(np.percentile(elbow_angles, 10))
    med_dev     = float(np.median(body_dev)) if body_dev else 0.0
    med_neck    = float(np.median(neck_tilt)) if neck_tilt else 0.0
    med_hand    = float(np.median(hand_offset)) if hand_offset else 0.0

    elbow_score = 95 if min_elbow <= 70 else 85 if min_elbow <= 90 else 70 if min_elbow <= 110 else 55
    body_score  = 95 if med_dev <= 0.04 else 82 if med_dev <= 0.07 else 68 if med_dev <= 0.12 else 50
    neck_score  = 95 if med_neck <= 10 else 82 if med_neck <= 20 else 68 if med_neck <= 30 else 55
    hand_score  = 95 if med_hand <= 0.5 else 82 if med_hand <= 0.8 else 68 if med_hand <= 1.1 else 55

    detailed = {
        "elbow_depth":   {"score": int(elbow_score), "feedback": f"Bottom elbow angle ≈ {min_elbow:.0f}°."},
        "body_line":     {"score": int(body_score),  "feedback": f"Hip deviation (norm) ≈ {med_dev:.2f}."},
        "neck_alignment":{"score": int(neck_score),  "feedback": f"Neck tilt ≈ {med_neck:.0f}°."},
        "hand_placement":{"score": int(hand_score),  "feedback": f"Hand offset ≈ {med_hand:.2f}× upper-arm length."}
    }

    overall = int(round(0.35*elbow_score + 0.35*body_score + 0.15*neck_score + 0.15*hand_score))

    whats_right = []
    if elbow_score >= 80: whats_right.append("Solid push-up depth.")
    if body_score  >= 80: whats_right.append("Strong plank line.")
    if neck_score  >= 80: whats_right.append("Neutral head/neck.")
    if hand_score  >= 80: whats_right.append("Good hand stacking.")

    corrections = []
    if elbow_score < 80:
        corrections.append({
            "issue": "Shallow depth",
            "severity": "warning" if elbow_score >= 60 else "critical",
            "feedback": f"Bottom elbow angle ~{min_elbow:.0f}° indicates limited depth.",
            "correction_instruction": "Use incline push-ups to keep full ROM without losing body line. Slow 2–3s descent."
        })
    if body_score < 80:
        corrections.append({
            "issue": "Hip sag/pike",
            "severity": "critical" if body_score < 60 else "warning",
            "feedback": "Hips not aligned with shoulders/ankles.",
            "correction_instruction": "Squeeze glutes/quads and keep ribs down; reduce reps if the line breaks."
        })
    if neck_score < 80:
        corrections.append({
            "issue": "Neck not neutral",
            "severity": "info",
            "feedback": "Head position suggests craning or dropping.",
            "correction_instruction": "Gaze 30–50 cm ahead; keep the back of your head long."
        })
    if hand_score < 80:
        corrections.append({
            "issue": "Hands not under shoulders",
            "severity": "warning",
            "feedback": "Hands appear too far forward/back or width off.",
            "correction_instruction": "Stack wrists under shoulders; screw hands into floor."
        })

    tips = [
        "Film from the side; include wrists to ankles.",
        "Brace like a plank (glutes + quads on).",
        "Use tempo (3s down, 1s up) for control."
    ]

    return {
        "overall_score": overall,
        "whats_right": whats_right,
        "corrections_needed": corrections,
        "detailed_breakdown": detailed,
        "improvement_tips": tips
    }
