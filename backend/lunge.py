# lunge.py
import cv2
import numpy as np

from utils import (
    L,
    lm_xy,
    angle_3pts,
    angle_to_vertical,
    dist_point_to_line,
    make_pose,
)

def _side_indices(side: str):
    """Return (shoulder, hip, knee, ankle, foot_index) PoseLandmark indices for a side."""
    if side == "left":
        return (L.LEFT_SHOULDER, L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE, L.LEFT_FOOT_INDEX)
    return (L.RIGHT_SHOULDER, L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE, L.RIGHT_FOOT_INDEX)

def analyze_lunge_video(video_path: str, max_frames: int = 600, stride: int = 3):
    """
    Analyze a lunge video and return frontend-ready JSON:
      - overall_score (0–100)
      - whats_right [str]
      - corrections_needed [{issue,severity,feedback,correction_instruction}]
      - detailed_breakdown {metric:{score,feedback}}
      - improvement_tips [str]
    Heuristics focus on front-leg depth, knee tracking, shin & torso angle, step width, and stability.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video."}

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

    pose = make_pose()

    # Accumulators
    front_knee_min_angles = []   # smallest = deeper
    shin_angles = []             # front shin vs vertical
    torso_angles = []            # torso vs vertical
    knee_track_dev = []          # lateral knee drift vs foot line (normalized)
    step_width_ratios = []       # feet horizontal spacing / pelvis width
    stride_len_ratios = []       # feet vertical spacing / leg length
    knee_x_offsets = []          # x-jitter for stability

    frame_i = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if frame_i % stride != 0:
            continue
        if processed >= max_frames:
            break
        processed += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            continue

        lms = res.pose_landmarks.landmark

        # Landmarks for each side
        L_sh, L_hip, L_knee, L_ank, L_foot = _side_indices("left")
        R_sh, R_hip, R_knee, R_ank, R_foot = _side_indices("right")

        Lhip  = lm_xy(lms, L_hip.value,   width, height)
        Lknee = lm_xy(lms, L_knee.value,  width, height)
        Lank  = lm_xy(lms, L_ank.value,   width, height)
        Ltoe  = lm_xy(lms, L_foot.value,  width, height)
        Lsho  = lm_xy(lms, L_sh.value,    width, height)

        Rhip  = lm_xy(lms, R_hip.value,   width, height)
        Rknee = lm_xy(lms, R_knee.value,  width, height)
        Rank  = lm_xy(lms, R_ank.value,   width, height)
        Rtoe  = lm_xy(lms, R_foot.value,  width, height)
        Rsho  = lm_xy(lms, R_sh.value,    width, height)

        # Compute both knee angles to identify front leg (more flexed = front)
        L_knee_angle = angle_3pts(Lhip, Lknee, Lank)
        R_knee_angle = angle_3pts(Rhip, Rknee, Rank)
        if L_knee_angle is None or R_knee_angle is None:
            continue

        front = "left" if L_knee_angle < R_knee_angle else "right"
        if front == "left":
            f_hip, f_knee, f_ank, f_toe, f_sho = Lhip, Lknee, Lank, Ltoe, Lsho
            b_hip, b_knee, b_ank, b_toe, b_sho = Rhip, Rknee, Rank, Rtoe, Rsho
            pelvis_width = np.linalg.norm(np.array(Rhip) - np.array(Lhip)) + 1e-6
        else:
            f_hip, f_knee, f_ank, f_toe, f_sho = Rhip, Rknee, Rank, Rtoe, Rsho
            b_hip, b_knee, b_ank, b_toe, b_sho = Lhip, Lknee, Lank, Ltoe, Lsho
            pelvis_width = np.linalg.norm(np.array(Lhip) - np.array(Rhip)) + 1e-6

        # --- Metrics for this frame ---

        # 1) Front knee angle (depth)
        fk_angle = angle_3pts(f_hip, f_knee, f_ank)
        if fk_angle is not None:
            front_knee_min_angles.append(fk_angle)

        # 2) Front shin angle vs vertical (knee->ankle)
        shin_ang = angle_to_vertical(f_knee, f_ank)
        if shin_ang is not None:
            shin_angles.append(shin_ang)

        # 3) Torso angle vs vertical (shoulder->hip)
        torso_ang = angle_to_vertical(f_sho, f_hip)
        if torso_ang is not None:
            torso_angles.append(torso_ang)

        # 4) Knee tracking deviation
        dev = dist_point_to_line(f_knee, f_ank, f_toe)
        if dev is not None:
            thigh_len = np.linalg.norm(np.array(f_hip) - np.array(f_knee)) + 1e-6
            knee_track_dev.append(dev / thigh_len)

        # 5) Step width ratio (feet horizontal distance / pelvis width)
        feet_width = abs(Ltoe[0] - Rtoe[0])
        step_width_ratios.append(float(feet_width / pelvis_width))

        # 6) Stride length ratio (feet vertical distance / leg length)
        feet_len = abs(Ltoe[1] - Rtoe[1])
        leg_len = np.linalg.norm(np.array(f_hip) - np.array(f_ank)) + 1e-6
        stride_len_ratios.append(float(feet_len / leg_len))

        # 7) Stability via knee x-jitter
        knee_x_offsets.append(f_knee[0])

    cap.release()
    pose.close()

    # Require enough frames to be meaningful
    if len(front_knee_min_angles) < 3:
        return {"error": "Not enough pose detections to analyze lunges. Try full-body framing, good lighting, and slower reps."}

    # --- Aggregate stats across sampled frames ---
    min_front_knee = float(np.percentile(front_knee_min_angles, 10))     # deeper = smaller
    med_shin       = float(np.median(shin_angles)) if shin_angles else 0.0
    med_torso      = float(np.median(torso_angles)) if torso_angles else 0.0
    max_knee_dev   = float(np.percentile(knee_track_dev, 90)) if knee_track_dev else 0.0
    med_step_w     = float(np.median(step_width_ratios)) if step_width_ratios else 0.0
    med_stride_l   = float(np.median(stride_len_ratios)) if stride_len_ratios else 0.0
    wobble_px_std  = float(np.std(knee_x_offsets)) if len(knee_x_offsets) >= 5 else 0.0

    # Normalize wobble by image width
    wobble_norm = wobble_px_std / (width + 1e-6)

    # --- Heuristic scoring (0–100) ---
    # Depth (front knee): ~90° ideal; 95–110 good; >125 shallow
    if min_front_knee <= 95:   depth_score = 95
    elif min_front_knee <= 110: depth_score = 85
    elif min_front_knee <= 125: depth_score = 70
    else:                       depth_score = 50

    # Shin vs vertical: <=15° great; 15–25 ok; 25–35 warn; >35 poor
    if med_shin <= 15:        shin_score = 95
    elif med_shin <= 25:      shin_score = 80
    elif med_shin <= 35:      shin_score = 65
    else:                     shin_score = 50

    # Torso vs vertical: <=15° great; 15–25 ok; 25–35 warn; >35 poor
    if med_torso <= 15:       torso_score = 95
    elif med_torso <= 25:     torso_score = 80
    elif med_torso <= 35:     torso_score = 65
    else:                     torso_score = 50

    # Knee tracking deviation (normalized): <=0.15 great; <=0.25 ok; <=0.35 warn; >0.35 poor
    if max_knee_dev <= 0.15:  tracking_score = 95
    elif max_knee_dev <= 0.25: tracking_score = 80
    elif max_knee_dev <= 0.35: tracking_score = 65
    else:                      tracking_score = 50

    # Step width (feet width / pelvis width): target ~0.6–1.2; broader band 0.4–1.6
    if 0.6 <= med_step_w <= 1.2: step_width_score = 95
    elif 0.4 <= med_step_w <= 1.6: step_width_score = 80
    else:                          step_width_score = 60

    # Stability via wobble
    if wobble_norm <= 0.01:     stability_score = 95
    elif wobble_norm <= 0.02:   stability_score = 80
    elif wobble_norm <= 0.03:   stability_score = 65
    else:                       stability_score = 50

    overall_score = int(round(
        0.30 * depth_score +
        0.20 * tracking_score +
        0.15 * shin_score +
        0.15 * torso_score +
        0.10 * step_width_score +
        0.10 * stability_score
    ))

    # --- Detailed breakdown for UI ---
    detailed_breakdown = {
        "front_knee_depth": {
            "score": int(depth_score),
            "feedback": f"Deepest front-knee angle ≈ {min_front_knee:.0f}°. Aim ~90–110°."
        },
        "knee_tracking": {
            "score": int(tracking_score),
            "feedback": f"Peak lateral knee drift (normalized) ≈ {max_knee_dev:.2f}. Track knee over 2nd–3rd toe."
        },
        "shin_angle": {
            "score": int(shin_score),
            "feedback": f"Median shin angle vs vertical ≈ {med_shin:.0f}°. Keep tibia more upright if you feel knee stress."
        },
        "torso_alignment": {
            "score": int(torso_score),
            "feedback": f"Median torso lean ≈ {med_torso:.0f}°. Brace and keep ribs stacked over hips."
        },
        "step_width": {
            "score": int(step_width_score),
            "feedback": f"Step width ratio ≈ {med_step_w:.2f} (feet width / pelvis width). Avoid tightrope stance."
        },
        "stability": {
            "score": int(stability_score),
            "feedback": f"Knee path wobble (normalized) ≈ {wobble_norm:.3f}. Slow the descent; focus on tripod foot."
        }
    }

    # --- What went well ---
    whats_right = []
    if depth_score >= 80:      whats_right.append("Good lunge depth on the front leg.")
    if tracking_score >= 80:   whats_right.append("Front knee tracks well over the toes.")
    if step_width_score >= 80: whats_right.append("Solid step width for balance.")
    if stability_score >= 80:  whats_right.append("Stable knee path through reps.")
    if torso_score >= 80:      whats_right.append("Upright torso and good bracing.")
    if shin_score >= 80:       whats_right.append("Appropriate shin angle.")

    # --- Corrections needed ---
    corrections_needed = []
    if tracking_score < 80:
        corrections_needed.append({
            "issue": "Front knee valgus/varus",
            "severity": "critical" if tracking_score < 60 else "warning",
            "feedback": "The front knee drifts laterally relative to the foot line.",
            "correction_instruction": "Press the front foot evenly (tripod) and guide the knee over the 2nd–3rd toe. Slow tempo to build control."
        })
    if depth_score < 80:
        corrections_needed.append({
            "issue": "Shallow front-knee depth",
            "severity": "warning",
            "feedback": f"Deepest front-knee angle ≈ {min_front_knee:.0f}°, suggesting limited range.",
            "correction_instruction": "Take a slightly longer stride, drop the back knee more vertically, and keep the front heel rooted. Try bodyweight tempo lunges (3–0–3)."
        })
    if shin_score < 80:
        corrections_needed.append({
            "issue": "Excessive shin angle",
            "severity": "warning",
            "feedback": f"Shin angle vs vertical ≈ {med_shin:.0f}°.",
            "correction_instruction": "Scoot the front foot forward a touch and descend more vertically. Keep the knee stacked over the mid-foot."
        })
    if torso_score < 80:
        corrections_needed.append({
            "issue": "Torso leaning forward",
            "severity": "warning" if torso_score >= 60 else "critical",
            "feedback": f"Torso lean ≈ {med_torso:.0f}° may indicate poor bracing or stride setup.",
            "correction_instruction": "Big breath into the belly/obliques before each rep; keep ribs stacked over hips. Goblet reverse lunges help groove posture."
        })
    if step_width_score < 80:
        corrections_needed.append({
            "issue": "Tightrope stance",
            "severity": "info",
            "feedback": f"Step width ratio ≈ {med_step_w:.2f}, which may reduce balance.",
            "correction_instruction": "Set the feet hip-width apart like two rails, not a single line. Maintain that width during the step."
        })
    if stability_score < 80:
        corrections_needed.append({
            "issue": "Knee wobble",
            "severity": "warning",
            "feedback": "Notable side-to-side front-knee movement frame-to-frame.",
            "correction_instruction": "Slow the eccentric (2–3s), focus the knee toward the 2nd–3rd toe, and use light support (fingertips on a rack) while learning."
        })

    improvement_tips = [
        "Film at ~45° front angle with the entire body in frame.",
        "Brace before each rep: inhale, ribs down, descend vertically; exhale at the top.",
        "Keep the front heel heavy; think 'down not forward'.",
        "Use a slow 2–3s descent to control tracking and stability.",
        "Practice stationary split squats to build balance before dynamic lunges."
    ]

    return {
        "overall_score": int(overall_score),
        "whats_right": whats_right,
        "corrections_needed": corrections_needed,
        "detailed_breakdown": detailed_breakdown,
        "improvement_tips": improvement_tips
    }
