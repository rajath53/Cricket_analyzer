# video_processor.py

import cv2
import os
import time
import math
import numpy as np
import pandas as pd

from cricket_pose import (
    PoseAnalyzer,
    draw_stylized_skeleton,
    build_trail_from_history,
    safe_get,
)


# ---------------------------- CONFIG ----------------------------
DEFAULT_FPS_FALLBACK = 25.0
MAX_TRAIL_LEN = 18

# Mediapipe landmark indices
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
# ----------------------------------------------------------------


def get_joint_color(name):
    n = name.lower()
    if "left" in n or "l_" in n or "lw" in n:
        return (255, 120, 0)  # left side = blueish
    if "right" in n or "r_" in n or "rw" in n:
        return (60, 60, 255)  # right side = red
    return (240, 240, 240)


def draw_metrics_panel(frame, metrics):
    panel = frame.copy()
    h, w = panel.shape[:2]

    # dim background
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.55, panel, 0.45, 0, panel)

    x, y = 12, 30
    cv2.putText(panel, "Cricket Analysis - Metrics",
                (x, y), cv2.FONT_HERSHEY_DUPLEX,
                0.8, (235, 235, 235), 2)

    y += 36

    # ordered display
    pref = [
        "role", "release_time_s", "last_bounce", "length",
        "speed_kmph", "speed_px_s", "runup_kmph", "runup_px_s",
        "trunk_flexion_deg", "left_elbow_angle", "right_elbow_angle",
        "left_knee_angle", "right_knee_angle",
        "left_shoulder_wrist_angle", "right_shoulder_wrist_angle",
        "hip_v_px_s"
    ]

    shown = set()
    for key in pref:
        if key in metrics:
            color = get_joint_color(key)
            cv2.putText(panel, f"{key}: {metrics[key]}",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
            y += 28
            shown.add(key)

    # remaining metrics
    for k, v in metrics.items():
        if k in shown:
            continue
        color = get_joint_color(k)
        cv2.putText(panel, f"{k}: {v}",
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)
        y += 26
        if y > h - 30:
            break

    return panel


# ---------------------------- MAIN PROCESSOR ----------------------------
def process_video(input_path, slow_factor=1.0):

    analyzer = PoseAnalyzer(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = in_fps if in_fps and in_fps > 0 else DEFAULT_FPS_FALLBACK

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360

    base = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"{base}_processed.mp4")

    out_w = width * 2
    out_h = height
    writer_fps = max(1.0, fps * slow_factor)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, writer_fps, (out_w, out_h))

    rows = []
    hip_history = []
    trail_hist = {"LW": [], "RW": []}

    frame_idx = 0
    t0 = time.time()

    try:
        while True:

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            t = frame_idx / fps

            # -------- Pose detection --------
            overlay_raw, metrics, landmarks = analyzer.analyze_frame(frame)

            # -------- Wrist trails --------
            if landmarks:
                lw = safe_get(landmarks, LEFT_WRIST)
                rw = safe_get(landmarks, RIGHT_WRIST)

                if lw:
                    trail_hist["LW"].append((lw[0], lw[1], t))
                if rw:
                    trail_hist["RW"].append((rw[0], rw[1], t))

                for k in trail_hist:
                    if len(trail_hist[k]) > MAX_TRAIL_LEN:
                        trail_hist[k] = trail_hist[k][-MAX_TRAIL_LEN:]

            trail_draw_dict = {
                k: build_trail_from_history(v, MAX_TRAIL_LEN)
                for k, v in trail_hist.items()
                if build_trail_from_history(v)
            }

            # -------- Split-screen visuals --------
            left_vis = frame.copy()
            right_vis = overlay_raw.copy()

            if landmarks:
                try:
                    left_vis = draw_stylized_skeleton(left_vis, landmarks, trail_draw_dict)
                except:
                    left_vis = frame.copy()

                try:
                    right_vis = draw_stylized_skeleton(right_vis, landmarks, trail_draw_dict)
                except:
                    right_vis = overlay_raw.copy()

            # -------- Metrics logic --------
            display_metrics = dict(metrics) if metrics else {}

            hip_v_px_s = None
            if len(hip_history) >= 1:
                # store hip_x
                if RIGHT_HIP < len(landmarks):
                    hip_x = float(landmarks[RIGHT_HIP][0])
                elif LEFT_HIP < len(landmarks):
                    hip_x = float(landmarks[LEFT_HIP][0])
                else:
                    hip_x = None

                if hip_x is not None:
                    hip_history.append((hip_x, t))

            if len(hip_history) >= 2:
                (x1, t1), (x2, t2) = hip_history[-2], hip_history[-1]
                dt = max(1e-3, t2 - t1)
                hip_v_px_s = (x2 - x1) / dt
                display_metrics["hip_v_px_s"] = f"{hip_v_px_s:.1f}"

            # runup peak
            if len(hip_history) >= 10:
                spd = []
                for i in range(1, len(hip_history)):
                    (x1, t1), (x2, t2) = hip_history[i - 1], hip_history[i]
                    dt = max(1e-3, t2 - t1)
                    spd.append(abs((x2 - x1) / dt))
                if spd:
                    display_metrics["runup_px_s"] = f"{max(spd):.1f}"

            # -------- Draw metrics panel --------
            right_vis = draw_metrics_panel(right_vis, display_metrics)

            # -------- Resize & Grey-screen fix --------
            try:
                left_r = cv2.resize(left_vis, (width, height))
            except:
                left_r = frame.copy()

            try:
                right_r = cv2.resize(right_vis, (width, height))
            except:
                right_r = left_r.copy()

            if right_r.mean() < 8:
                right_r = left_r.copy()

            combined = np.hstack((left_r, right_r))
            writer.write(combined)

            # -------- CSV row --------
            row = {"frame": frame_idx, "time_s": round(t, 4)}
            for k, v in display_metrics.items():
                row[k] = v
            rows.append(row)

            # Debug print
            if frame_idx % int(fps * 5) == 0:
                print(f"[processor] {frame_idx} frames, {time.time() - t0:.1f}s")

    finally:
        cap.release()
        writer.release()
        analyzer.close()
        cv2.destroyAllWindows()

    df = pd.DataFrame(rows)
    csv_path = os.path.join("outputs", f"{base}_framewise.csv")
    df.to_csv(csv_path, index=False)

    return out_path, df, csv_path
