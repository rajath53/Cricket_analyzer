# cricket_pose.py
import math
import cv2
import numpy as np
from PIL import Image
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.utils import ImageReader

# try to import mediapipe, but allow graceful fallback
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False
    mp_pose = None
    mp_draw = None

# ----------------- VISUAL CONFIG -----------------
# Neon trail color: Option B = #39FF14 => RGB (57,255,20) ; OpenCV uses BGR (20,255,57)
NEON_TRAIL_BGR = (20, 255, 57)

# Multi-color limb palette (BGR)
LIMB_COLORS = {
    "left_arm":    (255, 120, 0),   # blue-ish (BGR)
    "right_arm":   (60, 60, 255),   # red
    "left_leg":    (0, 215, 255),   # yellow
    "right_leg":   (255, 0, 180),   # purple
    "torso":       (255, 255, 0),   # cyan
    "head":        (0, 140, 255),   # orange
    "joint_outline": (0, 0, 0),
    "joint_fill":  (240, 240, 240)
}

# MediaPipe landmarks mapping (indices)
# We'll use standard MP indices: https://google.github.io/mediapipe/solutions/pose.html
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Limb pairs grouped for coloring
LIMBS = [
    # left arm (shoulder->elbow->wrist)
    ((LEFT_SHOULDER, LEFT_ELBOW), "left_arm"),
    ((LEFT_ELBOW, LEFT_WRIST), "left_arm"),
    # right arm
    ((RIGHT_SHOULDER, RIGHT_ELBOW), "right_arm"),
    ((RIGHT_ELBOW, RIGHT_WRIST), "right_arm"),
    # shoulders
    ((LEFT_SHOULDER, RIGHT_SHOULDER), "torso"),
    # torso -> hips
    ((LEFT_SHOULDER, LEFT_HIP), "torso"),
    ((RIGHT_SHOULDER, RIGHT_HIP), "torso"),
    ((LEFT_HIP, RIGHT_HIP), "torso"),
    # left leg
    ((LEFT_HIP, LEFT_KNEE), "left_leg"),
    ((LEFT_KNEE, LEFT_ANKLE), "left_leg"),
    # right leg
    ((RIGHT_HIP, RIGHT_KNEE), "right_leg"),
    ((RIGHT_KNEE, RIGHT_ANKLE), "right_leg"),
    # head to shoulders
    ((NOSE, LEFT_SHOULDER), "head"),
    ((NOSE, RIGHT_SHOULDER), "head"),
]

# ----------------- HELPERS -----------------
def safe_get(landmarks, idx):
    if landmarks and idx is not None and idx < len(landmarks):
        return landmarks[idx]
    return None

def angle_between(a, b, c):
    """Angle at b between points a-b-c in degrees"""
    a = np.array(a, dtype=float); b = np.array(b, dtype=float); c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0.0
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return math.degrees(math.acos(cosang))

# ----------------- TRAIL CONSTRUCTION -----------------
def build_trail_from_history(history_points, max_len=18):
    """
    history_points: list of (x, y, t) ordered oldest..newest
    returns trimmed list of (x,y,t)
    """
    if not history_points:
        return []
    return history_points[-max_len:]

def _compute_fade_color(base_bgr, alpha):
    """
    base_bgr: (b,g,r), alpha: 0..1 (1 being newest)
    returns integer BGR tuple scaled for perceived fade (no alpha channel)
    We scale brightness rather than using alpha blending for simplicity/performance.
    """
    b, g, r = base_bgr
    scale = 0.35 + 0.65 * alpha  # min 0.35 for older points
    return (int(b * scale), int(g * scale), int(r * scale))

# ----------------- SKELETON + TRAIL DRAWER -----------------
def draw_stylized_skeleton(img, landmarks_px, trail_history=None, joint_radius=6, limb_thickness=6):
    """
    Draws:
      - smooth neon trails for wrists (trail_history keys: "LW", "RW", each as [(x,y,t),...])
      - colorful limbs (LIMBS mapping)
      - joint circles with outline
    landmarks_px: list of (x,y) pixel coords or None
    trail_history: dict with 'LW' and/or 'RW' -> list of (x,y,t)
    """
    out = img.copy()
    h, w = out.shape[:2]

    # 1) Draw trails first (so limbs and joints draw on top)
    if trail_history:
        # Only handle wrists: 'LW' and 'RW' expected
        for key in ("LW", "RW"):
            hist = trail_history.get(key, [])
            if not hist:
                continue
            pts = build_trail_from_history(hist, max_len=18)
            total = len(pts)
            # draw segments from oldest to newest with increasing brightness and thickness
            for i in range(total - 1):
                (x1, y1, _), (x2, y2, _) = pts[i], pts[i+1]
                # alpha grows towards newest
                alpha = (i + 1) / float(total)
                color = _compute_fade_color(NEON_TRAIL_BGR, alpha)
                thickness = int(2 + 3 * alpha)  # older thin, newer thicker
                cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
            # draw small highlight at newest
            nx, ny, _ = pts[-1]
            cv2.circle(out, (int(nx), int(ny)), max(3, joint_radius+1), NEON_TRAIL_BGR, -1, cv2.LINE_AA)

    # 2) Draw limb connections in grouped colors
    if landmarks_px:
        for (pair, group_name) in LIMBS:
            a_idx, b_idx = pair
            if a_idx < len(landmarks_px) and b_idx < len(landmarks_px):
                a = tuple(map(int, landmarks_px[a_idx]))
                b = tuple(map(int, landmarks_px[b_idx]))
                color = LIMB_COLORS.get(group_name, LIMB_COLORS["torso"])
                cv2.line(out, a, b, color, limb_thickness, cv2.LINE_AA)
                # additionally draw a soft thin highlight (lighter stroke) for nicer aesthetics
                cv2.line(out, a, b, (220,220,220), max(1, limb_thickness//8), cv2.LINE_AA)

        # 3) Draw joints on top
        for i, (x, y) in enumerate(landmarks_px):
            x_i, y_i = int(x), int(y)
            # choose joint fill color slightly based on side:
            if x_i < w / 2:
                fill = LIMB_COLORS.get("left_arm", LIMB_COLORS["joint_fill"])
            else:
                fill = LIMB_COLORS.get("right_arm", LIMB_COLORS["joint_fill"])
            # central joints (hips/shoulders) use torso color
            if i in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP):
                fill = LIMB_COLORS["torso"]
            # head/nose
            if i == NOSE:
                fill = LIMB_COLORS["head"]
            # draw filled circle + outline
            cv2.circle(out, (x_i, y_i), joint_radius, fill, -1, cv2.LINE_AA)
            cv2.circle(out, (x_i, y_i), joint_radius+2, LIMB_COLORS["joint_outline"], 1, cv2.LINE_AA)

    return out

# ----------------- RENDER PREVIEW HELPER -----------------
def render_frame_overlay(frame_bgr, row_or_metrics=None, trail_history=None):
    """
    Render a frame preview in the same visual language.
    frame_bgr: BGR image (numpy)
    row_or_metrics: optional dict with metrics/time to display
    trail_history: optional dict (keys: 'LW','RW') with lists of (x,y,t)
    """
    out = frame_bgr.copy()
    # translucent info box top-left
    overlay_box = out.copy()
    cv2.rectangle(overlay_box, (8, 8), (440, 180), (6, 6, 6), -1)
    cv2.addWeighted(overlay_box, 0.5, out, 0.5, 0, out)
    y = 36
    if row_or_metrics:
        cv2.putText(out, f"t: {row_or_metrics.get('time_s', 'N/A')}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240,240,240), 2, cv2.LINE_AA)
        y += 28
        for k in ("trunk_flexion_deg", "release_angle_deg", "hip_v_px_s"):
            if k in row_or_metrics:
                cv2.putText(out, f"{k}: {row_or_metrics.get(k)}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220,220,220), 2, cv2.LINE_AA)
                y += 24
    # draw trails (if provided) and skeleton with default parameters
    out = draw_stylized_skeleton(out, None if row_or_metrics is None else row_or_metrics.get("landmarks_px", None), trail_history)
    return out

# ----------------- PDF writer for a single frame -----------------
def write_frame_pdf(full_annotated_bgr,
                    metrics_dict,
                    df_full,
                    selected_frame,
                    out_pdf_path,
                    title="Frame Report"):
    """
    full_annotated_bgr : FULL annotated frame from output video (same as Streamlit preview)
    metrics_dict       : Row as dict
    df_full            : Whole dataframe for table page
    selected_frame     : integer
    out_pdf_path       : PDF output
    """

    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.utils import ImageReader
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import io

    # ---------- PAGE 1: Annotated Frame ----------
    c = canvas.Canvas(out_pdf_path, pagesize=landscape(A4))
    page_w, page_h = landscape(A4)

    # frame → RGB
    rgb = full_annotated_bgr[:, :, ::-1]
    pil_img = Image.fromarray(rgb)
    bio = io.BytesIO()
    pil_img.save(bio, format="PNG")
    bio.seek(0)
    img_reader = ImageReader(bio)

    c.setFont("Helvetica-Bold", 20)
    c.drawString(30, page_h - 40, f"{title}")

    # Fit image
    img_w = page_w * 0.95
    img_h = page_h * 0.80
    c.drawImage(img_reader, 20, 50, width=img_w, height=img_h)

    c.showPage()

    # ---------- PAGE 2: Metrics ----------
    c.setFont("Helvetica-Bold", 22)
    c.drawString(40, page_h - 50, "Metrics")

    c.setFont("Helvetica", 14)
    y = page_h - 90
    for k, v in metrics_dict.items():
        c.drawString(60, y, f"{k} : {v}")
        y -= 22
        if y < 60:
            c.showPage()
            c.setFont("Helvetica", 14)
            y = page_h - 60

    c.showPage()

    # ---------- PAGE 3: Table with Highlight ----------
    # Prepare styled table image
    df = df_full.copy()

    def highlight_func(row):
        if row["frame"] == selected_frame:
            return ['background-color: #39FF14'] * len(row)
        return [''] * len(row)

    styled_df = df.style.apply(highlight_func, axis=1)

    # Use matplotlib table backend
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    tbl = pd.plotting.table(
        ax,
        df,
        cellLoc='center',
        loc='center'
    )

    # Apply highlight (manually adjust cell colors)
    for i in range(len(df)):
        if df.iloc[i]["frame"] == selected_frame:
            for j in range(df.shape[1]):
                tbl[(i+1, j)].set_facecolor("#078BDC")

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", dpi=300)
    plt.close(fig)
    buf.seek(0)
    tbl_reader = ImageReader(buf)

    c.setFont("Helvetica-Bold", 20)
    c.drawString(40, page_h - 40, "Framewise Data Table")

    c.drawImage(tbl_reader, 20, 30, width=page_w - 40, height=page_h - 80)
    c.showPage()
    c.save()

# ----------------- PoseAnalyzer wrapper -----------------
try:
    class PoseAnalyzer:
        def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
            self.available = MP_AVAILABLE
            if MP_AVAILABLE:
                self.pose = mp_pose.Pose(static_image_mode=static_image_mode, model_complexity=model_complexity,
                                         min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
            else:
                self.pose = None

        def analyze_frame(self, frame_bgr):
            """
            Returns (overlay_bgr, metrics_dict, landmarks_px_list)
            overlay_bgr: basic mediapipe overlay (we will re-style later)
            metrics: dictionary of numeric metrics (angles etc.)
            landmarks_px: list of (x_px,y_px) pixel tuples
            """
            h, w = frame_bgr.shape[:2]
            overlay = frame_bgr.copy()
            metrics = {}
            landmarks_px = None
            if not self.available:
                return overlay, metrics, None

            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)
            if not results.pose_landmarks:
                return overlay, metrics, None

            # quick draw for a fallback overlay (we re-style in video processing)
            try:
                mp_draw.draw_landmarks(overlay, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            except Exception:
                pass

            # convert to pixel coordinates
            landmarks_px = []
            for lm in results.pose_landmarks.landmark:
                landmarks_px.append((int(lm.x * w), int(lm.y * h)))

            # compute angles & basic metrics
            try:
                # elbow angles
                metrics['left_elbow_angle'] = round(angle_between(landmarks_px[LEFT_SHOULDER], landmarks_px[LEFT_ELBOW], landmarks_px[LEFT_WRIST]), 1)
                metrics['right_elbow_angle'] = round(angle_between(landmarks_px[RIGHT_SHOULDER], landmarks_px[RIGHT_ELBOW], landmarks_px[RIGHT_WRIST]), 1)
                # shoulder->wrist angle (vector)
                def vec_angle(p1, p2):
                    v = np.array(p2) - np.array(p1)
                    ang = math.degrees(math.atan2(-v[1], v[0]))
                    return round(ang, 1)
                metrics['left_shoulder_wrist_angle'] = vec_angle(landmarks_px[LEFT_SHOULDER], landmarks_px[LEFT_WRIST])
                metrics['right_shoulder_wrist_angle'] = vec_angle(landmarks_px[RIGHT_SHOULDER], landmarks_px[RIGHT_WRIST])
                # trunk flexion
                try:
                    sh = ((landmarks_px[LEFT_SHOULDER][0] + landmarks_px[RIGHT_SHOULDER][0]) / 2.0, (landmarks_px[LEFT_SHOULDER][1] + landmarks_px[RIGHT_SHOULDER][1]) / 2.0)
                    hp = ((landmarks_px[LEFT_HIP][0] + landmarks_px[RIGHT_HIP][0]) / 2.0, (landmarks_px[LEFT_HIP][1] + landmarks_px[RIGHT_HIP][1]) / 2.0)
                    v = (sh[0] - hp[0], hp[1] - sh[1])
                    ang = math.degrees(math.atan2(v[1], v[0]))
                    metrics['trunk_flexion_deg'] = round(abs(90.0 - abs(ang)), 2)
                except Exception:
                    metrics['trunk_flexion_deg'] = None
            except Exception:
                pass

            return overlay, metrics, landmarks_px

        def close(self):
            if self.pose:
                self.pose.close()
except Exception:
    # fallback stub
    class PoseAnalyzer:
        def __init__(self, *args, **kwargs):
            self.available = False
        def analyze_frame(self, frame):
            return frame, {}, None
        def close(self):
            pass
