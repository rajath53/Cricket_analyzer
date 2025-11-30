import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tempfile
import os
from fpdf import FPDF
import time
import shutil
from collections import deque 

# --- Configuration and Setup ---
st.set_page_config(layout="wide", page_title="Cricket Biomechanics Analyzer")

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define the number of frames for the trail history
TRAIL_HISTORY_LENGTH = 15 
TRAIL_THICKNESS = 5 
TRAIL_POINT_RADIUS = 5

# --- HIGH VISIBILITY STYLING PARAMETERS ---
SKELETON_THICKNESS = 4
SKELETON_POINT_RADIUS = 6

BRIGHT_YELLOW = (0, 255, 255) # For metrics and highlights (BGR) - SPEED VALUES
WHITE = (255, 255, 255) # For Joints and TITLES (BGR)
BLACK = (0, 0, 0) # For text outline and background for metrics
RED = (0, 0, 255) # New color for Speed Labels (BGR)

# Define a function to draw text with an outline
def draw_text_with_outline(image, text, org, font, font_scale, text_color, outline_color, thickness, outline_thickness_factor=2):
    """Draws text with a thicker outline for better readability against a busy background."""
    # Draw outline
    cv2.putText(image, text, org, font, font_scale, outline_color, thickness + outline_thickness_factor, cv2.LINE_AA)
    # Draw text
    cv2.putText(image, text, org, font, font_scale, text_color, thickness, cv2.LINE_AA)


SKELETON_COLORS = {
    'LEFT_ARM': (0, 255, 0),    
    'RIGHT_ARM': (0, 0, 255),   
    'LEFT_LEG': (255, 255, 0),  
    'RIGHT_LEG': (255, 0, 255), 
    'TORSO': (255, 0, 0),       
    'HEAD': (0, 255, 255),      
}

POSE_SEGMENT_CONNECTIONS = {
    'TORSO': [
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
    ],
    'LEFT_ARM': [
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value),
        (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
    ],
    'RIGHT_ARM': [
        (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value),
        (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value),
    ],
    'LEFT_LEG': [
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
        (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
    ],
    'RIGHT_LEG': [
        (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
        (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value),
    ],
    'HEAD': [
        (mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.LEFT_EAR.value),
        (mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.RIGHT_EAR.value),
        (mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value),
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_EAR.value),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_EAR.value),
    ]
}

# --- Helper Functions (Angle Calculation, Phase Detection) ---

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    if np.array_equal(a, b) or np.array_equal(c, b) or np.linalg.norm(a - b) == 0 or np.linalg.norm(c - b) == 0:
        return 0.0
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_trunk_flex(left_shoulder, left_hip):
    a = np.array(left_shoulder)
    b = np.array(left_hip)
    vertical = np.array([0, 1])
    body = b - a
    if np.linalg.norm(body) == 0:
        return 0.0
    cosine = np.dot(body, vertical) / np.linalg.norm(body)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))

def get_action_phase(landmarks):
    try:
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        arm_elevation_delta = right_shoulder_y - right_wrist_y
        
        if arm_elevation_delta > 0.15:
            return "Delivery/Swing Action (Action in Progress)"
        elif arm_elevation_delta > 0.0:
             return "Pre-Action/High Setup"
        else:
            return "Pre-Action/Follow-Through"
            
    except Exception:
        return "Unknown Phase (No Landmark Data)"

# --- NEW FUNCTION: Simulate Speed ---
def simulate_bowling_speed(frame_num, frame_count):
    """
    Simulates plausible bowling speeds for Bowler, Wrist, and Release.
    Release speed is highest. Speeds scale with frame progress.
    """
    # Normalize frame number from 0 to 1
    progress = frame_num / frame_count
    
    # Base speeds and max fluctuation
    MAX_RELEASE_SPEED = 145 
    MIN_WRIST_SPEED = 60
    MAX_BOWLER_SPEED = 30
    
    # Peak is reached around 70% of the video duration (simulating run-up and delivery)
    peak_factor = min(1.0, progress * 1.5) if progress < 0.7 else min(1.0, (1.0 - progress) * 3.3) 
    
    # Simulate Release Speed (peaks at MAX_RELEASE_SPEED)
    release_speed = (peak_factor * MAX_RELEASE_SPEED) * (0.8 + 0.4 * np.sin(frame_num * 0.1))
    
    # Simulate Wrist Speed (lower, less fluctuation)
    wrist_speed = MIN_WRIST_SPEED + (peak_factor * 20)
    
    # Simulate Bowler Speed (low, representing the body's movement speed)
    bowler_speed = (MAX_BOWLER_SPEED * progress) * (1.0 - peak_factor) + 5
    
    # Ensure speeds are positive and release is max
    release = max(50, release_speed)
    wrist = max(20, wrist_speed)
    bowler = max(5, bowler_speed)
    
    return int(wrist), int(release), int(bowler)

# PDF CREATION (Remains the same)
def create_pdf_report(analysis_df, action_mode):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Cricket Biomechanics Analysis Report", 0, 1, "C")
    pdf.set_font("Arial", "I", 14)
    pdf.cell(200, 8, f"Action Mode: {action_mode}", 0, 1, "C") 
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)

    col_widths = [20, 35, 35, 35, 35, 35]

    pdf.set_fill_color(200, 220, 255)
    pdf.cell(col_widths[0], 7, "Frame", 1, 0, "C", 1)
    pdf.cell(col_widths[1], 7, "Action Phase", 1, 0, "C", 1)
    pdf.cell(col_widths[2], 7, "L Elbow (°)", 1, 0, "C", 1)
    pdf.cell(col_widths[3], 7, "R Elbow (°)", 1, 0, "C", 1)
    pdf.cell(col_widths[4], 7, "R Knee (°)", 1, 0, "C", 1)
    pdf.cell(col_widths[5], 7, "Trunk Flex (°)", 1, 1, "C", 1)

    report_df = analysis_df[['Action Phase', 'Left Elbow Angle', 'Right Elbow Angle', 'Right Knee Angle', 'Trunk Flex Angle']].copy()

    for index, row in report_df.iterrows():
        action_phase = row['Action Phase'] if not pd.isna(row['Action Phase']) else 'N/A'
        left_elbow = row['Left Elbow Angle'] if not pd.isna(row['Left Elbow Angle']) else 0.0
        right_elbow = row['Right Elbow Angle'] if not pd.isna(row['Right Elbow Angle']) else 0.0
        right_knee = row['Right Knee Angle'] if not pd.isna(row['Right Knee Angle']) else 0.0
        trunk_flex = row['Trunk Flex Angle'] if not pd.isna(row['Trunk Flex Angle']) else 0.0

        pdf.cell(col_widths[0], 6, str(index), 1, 0, "C")
        pdf.cell(col_widths[1], 6, action_phase[:15], 1, 0, "L")
        pdf.cell(col_widths[2], 6, f"{left_elbow:.2f}", 1, 0, "C")
        pdf.cell(col_widths[3], 6, f"{right_elbow:.2f}", 1, 0, "C")
        pdf.cell(col_widths[4], 6, f"{right_knee:.2f}", 1, 0, "C")
        pdf.cell(col_widths[5], 6, f"{trunk_flex:.2f}", 1, 1, "C")

    return pdf.output(dest='S').encode('latin1')

# --- State Management Initialization ---
if 'analysis_df' not in st.session_state:
    st.session_state.analysis_df = None
if 'max_frames' not in st.session_state:
    st.session_state.max_frames = 0
if 'annotated_frame_bytes' not in st.session_state:
    st.session_state.annotated_frame_bytes = {}
if 'action_mode' not in st.session_state:
    st.session_state.action_mode = 'Right_Handed_Batsman' 
if 'original_file_name' not in st.session_state:
    st.session_state.original_file_name = 'input_video.mp4'
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False 
if 'output_fps' not in st.session_state:
    st.session_state.output_fps = 30.0
if 'current_frame_index' not in st.session_state:
    st.session_state.current_frame_index = 0
if 'output_video_path' not in st.session_state:
    st.session_state.output_video_path = None
if 'trail_history' not in st.session_state:
    st.session_state.trail_history = deque(maxlen=TRAIL_HISTORY_LENGTH) 
if 'max_bowling_elbow_angle' not in st.session_state:
    st.session_state.max_bowling_elbow_angle = 0.0
if 'frame_count_total' not in st.session_state: # Store total frames
    st.session_state.frame_count_total = 0 


# --- Interactive Frame Player Function (remains the same) ---
def play_frame_simulation():
    """Renders frames sequentially to simulate video playback."""
    
    if not st.session_state.annotated_frame_bytes:
        st.warning("No analyzed frames available to play.")
        st.session_state.is_playing = False
        return

    st.session_state.is_playing = True
    
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    
    frame_bytes_dict = st.session_state.annotated_frame_bytes
    output_fps = st.session_state.output_fps
    delay = 1.0 / output_fps
    frame_keys = sorted(frame_bytes_dict.keys())
    
    status_placeholder.info("Playback in progress... Press 'Stop' to pause.")

    start_index = st.session_state.current_frame_index 
    
    for i in range(start_index, len(frame_keys)):
        if not st.session_state.is_playing:
            status_placeholder.info(f"Playback paused at frame {st.session_state.current_frame_index}.")
            break
            
        frame_num = frame_keys[i]
        st.session_state.current_frame_index = frame_num 
        frame_bytes = frame_bytes_dict[frame_num]
        
        frame_array = np.frombuffer(frame_bytes, np.uint8)
        combined_frame_bgr = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if combined_frame_bgr is not None:
            combined_frame_rgb = cv2.cvtColor(combined_frame_bgr, cv2.COLOR_BGR2RGB)
            
            video_placeholder.image(
                combined_frame_rgb,
                caption=f"Frame {frame_num} / {len(frame_keys) - 1} - Playback FPS: {output_fps:.2f}",
                use_container_width=True 
            )
            time.sleep(delay)
            
        if i == len(frame_keys) - 1:
            st.session_state.is_playing = False
            st.session_state.current_frame_index = 0 
            status_placeholder.success("Playback Simulation Complete.")
    
    st.rerun()


# --- Main Streamlit App ---
st.title("🏏 Frame-by-Frame Cricket Analysis")
st.markdown("Upload a video and run the biomechanical analysis.")

st.sidebar.header("Upload Video")
uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov"])

# --- Action Mode Selector (UPDATED) ---
ACTION_MODES = [
    'Right_Handed_Batsman', 
    'Left_Handed_Batsman',
    'Right_Arm_Medium_Pacer',
    'Right_Arm_Fast_Bowler',
    'Right_Arm_Leg_Spinner',
    'Right_Arm_Off_Spinner', 
    'Left_Arm_Medium_Pacer', 
    'Left_Arm_Fast_Bowler',
    'Left_Arm_Orthodox'
]

st.sidebar.header("Select Player Role")
selected_mode = st.sidebar.selectbox(
    "Choose the exact player role for analysis:",
    options=ACTION_MODES,
    index=ACTION_MODES.index(st.session_state.action_mode) if st.session_state.action_mode in ACTION_MODES else 0
)
st.session_state.action_mode = selected_mode


# Playback speed control
st.sidebar.header("Video Processing Speed")
speed_multiplier = st.sidebar.slider(
    "Playback Slow-Motion Multiplier", 
    min_value=1.0, 
    max_value=20.0, 
    value=2.0, 
    step=0.1,
    help="Playback FPS = Original FPS / Multiplier. A higher value means slower motion in the simulation."
)


if uploaded_file is not None:
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name
    
    st.session_state.original_file_name = uploaded_file.name

    st.sidebar.success("Video uploaded successfully!")
    st.subheader("Original Video Preview")
    st.video(video_path)
    st.info(f"Analysis will be run in the **{st.session_state.action_mode.replace('_', ' ')}** mode.")


    if st.sidebar.button("⚙️ Run Biomechanics Analysis (Generate Annotated Frames)"):
        
        # Reset data for new analysis
        st.session_state.analysis_df = pd.DataFrame(columns=[
            'Left Elbow Angle', 'Right Elbow Angle', 'Right Knee Angle', 'Trunk Flex Angle', 'Action Phase'
        ])
        st.session_state.annotated_frame_bytes = {} 
        st.session_state.is_playing = False
        st.session_state.current_frame_index = 0
        st.session_state.output_video_path = None
        st.session_state.trail_history.clear() 
        st.session_state.max_bowling_elbow_angle = 0.0 # Reset max angle
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video file for processing.")
            st.stop() 

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # Save total frames for speed simulation
        st.session_state.frame_count_total = frame_count 

        output_fps = original_fps / speed_multiplier
        st.session_state.output_fps = output_fps

        st.info(f"Original FPS: {original_fps:.2f}. Playback FPS for simulation: {output_fps:.2f} (Speed Multiplier: {speed_multiplier:.1f})")

        st.session_state.max_frames = frame_count - 1

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- VIDEO WRITER SETUP: NEW SIZE (2W x 2H) for 3-Panel Layout ---
        output_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_temp_path = output_temp_file.name
        output_temp_file.close()
        
        st.session_state.output_video_path = output_temp_path
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        # Output frame size is now 2x Width and 2x Height
        output_video_size = (frame_width * 2, frame_height * 2) 
        video_writer = cv2.VideoWriter(output_temp_path, fourcc, original_fps / speed_multiplier, output_video_size)
        
        if not video_writer.isOpened():
            st.warning("Could not initialize video writer. Annotated video download may fail, but simulation will still work.")
            video_writer = None

        # --- ANALYSIS LOOP: PROCESS AND SAVE TO MEMORY AND DISK ---
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            try:
                frame_num = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # --- Frame Setup ---
                    frame_metrics = {}
                    action_phase = "No Pose Detected" 
                    # Use a slightly less visible color for the top panel metrics for video background consistency
                    metric_color_l, metric_color_r, metric_color_k, metric_color_t = WHITE, WHITE, WHITE, WHITE
                    highlight_joint = None
                    current_landmarks_norm = {} 
                    
                    # Initialize bowling arm parameters
                    bowling_arm_points = [] 
                    bowling_elbow_angle_key = None 

                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    results = pose.process(image_rgb)
                    
                    # --- Determine Arm/Role ---
                    mode = st.session_state.action_mode
                    is_bowler = 'Arm' in mode 
                    is_right_arm = 'Right_Arm' in mode
                    is_left_arm = 'Left_Arm' in mode
                    
                    # 1. Top Left Panel (W x H)
                    top_left_panel = frame.copy() 
                    
                    # 2. Top Right Panel (W x H) - Initialize with a copy of the original frame
                    top_right_panel = frame.copy() 
                    
                    # 3. Bottom Panel (2W x H) - Initialize as a copy of the original frame, scaled to 2W x H
                    bottom_panel_width = frame_width * 2
                    bottom_panel_height = frame_height
                    
                    # Rescale the current frame to 2W x H for the bottom panel background
                    bottom_panel_bg = cv2.resize(frame, (bottom_panel_width, bottom_panel_height), interpolation=cv2.INTER_LINEAR)
                    bottom_panel = bottom_panel_bg.copy()


                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        action_phase = get_action_phase(landmarks)

                        # Normalized coordinates extraction for metrics and drawing
                        # Note: MediaPipe landmarks are normalized [0, 1]
                        points_norm = {
                            'L_S': [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                            'L_E': [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
                            'L_W': [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y],
                            'R_S': [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
                            'R_E': [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y],
                            'R_W': [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y],
                            'R_H': [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
                            'R_K': [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y],
                            'R_A': [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y],
                            'L_H': [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
                            'L_K': [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],
                            'L_A': [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y],
                        }
                        
                        # Store points for trail history 
                        current_landmarks_norm = {k: points_norm[k] for k in ['R_S', 'R_E', 'R_W', 'L_S', 'L_E', 'L_W']}

                        # Metric calculations
                        left_elbow_angle = calculate_angle(points_norm['L_S'], points_norm['L_E'], points_norm['L_W'])
                        right_elbow_angle = calculate_angle(points_norm['R_S'], points_norm['R_E'], points_norm['R_W'])
                        right_knee_angle = calculate_angle(points_norm['R_H'], points_norm['R_K'], points_norm['R_A'])
                        trunk_flex = calculate_trunk_flex(points_norm['L_S'], points_norm['L_H'])

                        frame_metrics['Left Elbow Angle'] = left_elbow_angle
                        frame_metrics['Right Elbow Angle'] = right_elbow_angle
                        frame_metrics['Right Knee Angle'] = right_knee_angle
                        frame_metrics['Trunk Flex Angle'] = trunk_flex
                        frame_metrics['Action Phase'] = action_phase

                        # Metric coloring logic (using BRIGHT_YELLOW for focus)
                        wrist_speed, release_speed, bowler_speed = 0, 0, 0
                        if is_bowler:
                            # SIMULATE SPEED HERE
                            wrist_speed, release_speed, bowler_speed = simulate_bowling_speed(frame_num, st.session_state.frame_count_total)

                            if is_right_arm:
                                metric_color_r, metric_color_t, highlight_joint = BRIGHT_YELLOW, BRIGHT_YELLOW, points_norm['R_E']
                                bowling_elbow_angle_key = 'Right Elbow Angle'
                                bowling_arm_points = ['R_S', 'R_E', 'R_W']
                            else: # Left Arm
                                metric_color_l, metric_color_t, highlight_joint = BRIGHT_YELLOW, BRIGHT_YELLOW, points_norm['L_E']
                                bowling_elbow_angle_key = 'Left Elbow Angle'
                                bowling_arm_points = ['L_S', 'L_E', 'L_W']
                                
                            # Max Angle Tracking (only for bowlers)
                            current_elbow_angle = frame_metrics.get(bowling_elbow_angle_key, 0.0)
                            if current_elbow_angle > st.session_state.max_bowling_elbow_angle:
                                st.session_state.max_bowling_elbow_angle = current_elbow_angle
                                
                        elif 'Right_Handed_Batsman' in mode:
                            metric_color_l, metric_color_k, highlight_joint = BRIGHT_YELLOW, BRIGHT_YELLOW, points_norm['R_K']
                        elif 'Left_Handed_Batsman' in mode:
                            metric_color_r, metric_color_k, highlight_joint = BRIGHT_YELLOW, BRIGHT_YELLOW, points_norm['L_K']
                        else:
                            metric_color_t, highlight_joint = BRIGHT_YELLOW, points_norm['R_K']
                            
                        # --- 1. Top Left Panel (Original + Metrics) ---
                        if highlight_joint:
                            hj_draw = (int(highlight_joint[0] * frame_width), int(highlight_joint[1] * frame_height))
                            cv2.circle(top_left_panel, hj_draw, 12, BRIGHT_YELLOW, -1)
                        
                        # Apply White/Black outline to Title for readability
                        draw_text_with_outline(top_left_panel, "1. Original Frame + Matrices", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, BLACK, 1) 
                        draw_text_with_outline(top_left_panel, f"Mode: {st.session_state.action_mode.replace('_', ' ')}", (frame_width // 2 - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), BLACK, 4) 
                        draw_text_with_outline(top_left_panel, action_phase, (frame_width // 2 - 250, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), BLACK, 4)
                        
                        # Apply outline to metric text
                        draw_text_with_outline(top_left_panel, f"L Elbow: {left_elbow_angle:.2f} deg", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, metric_color_l, BLACK, 2)
                        draw_text_with_outline(top_left_panel, f"R Elbow: {right_elbow_angle:.2f} deg", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, metric_color_r, BLACK, 2)
                        draw_text_with_outline(top_left_panel, f"R Knee: {right_knee_angle:.2f} deg", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, metric_color_k, BLACK, 2)
                        draw_text_with_outline(top_left_panel, f"Trunk Flex: {trunk_flex:.2f} deg", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, metric_color_t, BLACK, 2)
                        
                        
                        # --- 2. Top Right Panel (Full Skeleton Overlay) ---
                        for segment_name, connections in POSE_SEGMENT_CONNECTIONS.items():
                            color = SKELETON_COLORS.get(segment_name, WHITE) 
                            for connection in connections:
                                start_idx, end_idx = connection
                                start_landmark = landmarks[start_idx]
                                end_landmark = landmarks[end_idx]
                                start_point = (int(start_landmark.x * frame_width), int(start_landmark.y * frame_height))
                                end_point = (int(end_landmark.x * frame_width), int(end_landmark.y * frame_height))
                                cv2.line(top_right_panel, start_point, end_point, color, SKELETON_THICKNESS)

                        for idx, landmark in enumerate(landmarks):
                            if landmark.visibility > 0.5:
                                center = (int(landmark.x * frame_width), int(landmark.y * frame_height))
                                cv2.circle(top_right_panel, center, SKELETON_POINT_RADIUS, WHITE, -1) 
                                cv2.circle(top_right_panel, center, SKELETON_POINT_RADIUS + 1, (0, 0, 0), 1) 

                        # Apply White/Black outline to Title for readability
                        draw_text_with_outline(top_right_panel, "2. Full Skeleton Overlay", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, BLACK, 1) 


                        # --- 3. Bottom Panel (Full Width Trail + Speed Placeholders on Video Background) ---
                        
                        TRAIL_TEXT_COLOR = BRIGHT_YELLOW 
                        
                        if is_bowler:
                            st.session_state.trail_history.append(current_landmarks_norm)
                            history_list = list(st.session_state.trail_history)
                            
                            # Draw Trail with Red-to-Blue Gradient (Red is newer, Blue is older)
                            for i, points_norm in enumerate(history_list):
                                alpha = (i + 1) / len(history_list) 
                                
                                # Red-to-Blue Gradient calculation (BGR)
                                blue_component = int(255 * (1 - alpha))
                                red_component = int(255 * alpha)
                                trail_color_bgr = (blue_component, 0, red_component) # (B, G, R)

                                # Draw connections and joints for the SELECTED bowling arm (scaled to 2W)
                                for j in range(len(bowling_arm_points) - 1): 
                                    pt1_norm = points_norm.get(bowling_arm_points[j])
                                    pt2_norm = points_norm.get(bowling_arm_points[j+1])
                                    if pt1_norm and pt2_norm:
                                        # Re-scale the normalized coordinate (x*2W)
                                        pt1_draw = (int(pt1_norm[0] * bottom_panel_width), int(pt1_norm[1] * bottom_panel_height))
                                        pt2_draw = (int(pt2_norm[0] * bottom_panel_width), int(pt2_norm[1] * bottom_panel_height))
                                        cv2.line(bottom_panel, pt1_draw, pt2_draw, trail_color_bgr, TRAIL_THICKNESS)
                                
                                for p_name in bowling_arm_points:
                                    pt_norm = points_norm.get(p_name)
                                    if pt_norm:
                                        pt_draw = (int(pt_norm[0] * bottom_panel_width), int(pt_norm[1] * bottom_panel_height))
                                        cv2.circle(bottom_panel, pt_draw, TRAIL_POINT_RADIUS, trail_color_bgr, -1)
                            
                            # --- Analysis Text on Bottom Panel (Mimicking Speed Analysis Layout) ---
                            arm_name = 'Right' if is_right_arm else 'Left'
                            
                            # Apply White/Black outline to Title for readability
                            draw_text_with_outline(bottom_panel, f"3. {arm_name} Arm Motion Analysis", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, BLACK, 1) 
                            
                            TEXT_SCALE = 1.0
                            TEXT_THICKNESS = 3
                            
                            # 1. Wrist/Top Metric (Label in RED, Value in Yellow)
                            draw_text_with_outline(bottom_panel, "Wrist", (bottom_panel_width // 4 + 100, bottom_panel_height // 3), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, RED, BLACK, TEXT_THICKNESS)
                            draw_text_with_outline(bottom_panel, f"{wrist_speed} km/h", (bottom_panel_width // 4 + 100, bottom_panel_height // 3 + 40), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TRAIL_TEXT_COLOR, BLACK, TEXT_THICKNESS)

                            # 2. Release/Bottom-Left Metric (Label in RED, Value in Yellow)
                            draw_text_with_outline(bottom_panel, "Release", (bottom_panel_width // 4 - 200, bottom_panel_height * 2 // 3), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, RED, BLACK, TEXT_THICKNESS)
                            draw_text_with_outline(bottom_panel, f"{release_speed} km/h", (bottom_panel_width // 4 - 200, bottom_panel_height * 2 // 3 + 40), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TRAIL_TEXT_COLOR, BLACK, TEXT_THICKNESS)

                            # 3. Bowler/Bottom-Right Metric (Label in RED, Value in Yellow)
                            draw_text_with_outline(bottom_panel, "Bowler", (bottom_panel_width * 3 // 4 + 50, bottom_panel_height * 2 // 3), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, RED, BLACK, TEXT_THICKNESS)
                            draw_text_with_outline(bottom_panel, f"{bowler_speed} km/h", (bottom_panel_width * 3 // 4 + 50, bottom_panel_height * 2 // 3 + 40), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TRAIL_TEXT_COLOR, BLACK, TEXT_THICKNESS)
                            
                        else:
                            # Not a bowler, put a different, consistent placeholder in the bottom panel
                            st.session_state.trail_history.clear() 
                            # Apply White/Black outline to Title for readability
                            draw_text_with_outline(bottom_panel, "3. Trailing and Speed Analysis (Skipped - Not a Bowler Role)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, BLACK, 1) 
                            draw_text_with_outline(bottom_panel, "Motion Trail is currently reserved for Bowlers.", (bottom_panel_width // 2 - 300, bottom_panel_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TRAIL_TEXT_COLOR, BLACK, 2)
                            
                    else:
                        # No pose detected: draw status on panels
                        draw_text_with_outline(top_left_panel, "No Pose Detected", (frame_width // 2, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, BRIGHT_YELLOW, BLACK, 2)
                        draw_text_with_outline(top_right_panel, "No Pose Detected", (frame_width // 2, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, BRIGHT_YELLOW, BLACK, 2)
                        draw_text_with_outline(bottom_panel, "No Pose Detected", (bottom_panel_width // 2 - 200, bottom_panel_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TRAIL_TEXT_COLOR, BLACK, 2)


                    # --- Final Stacking to create the 2W x 2H frame ---
                    # 1. Stack the two top panels horizontally (Size: 2W x H)
                    top_row = np.hstack((top_left_panel, top_right_panel))
                    # 2. Stack the top row and the bottom panel vertically (Size: 2W x 2H)
                    combined_frame = np.vstack((top_row, bottom_panel))

                    
                    # 1. Save frame for in-memory simulation
                    is_success, buffer = cv2.imencode(".jpg", combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if is_success:
                        st.session_state.annotated_frame_bytes[frame_num] = buffer.tobytes()

                    # 2. Write frame to disk for video download
                    if video_writer is not None:
                        video_writer.write(combined_frame)
                        
                    st.session_state.analysis_df = pd.concat([
                        st.session_state.analysis_df,
                        pd.DataFrame([frame_metrics], index=[frame_num])
                    ])

                    progress = (frame_num + 1) / frame_count
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Processing frame {frame_num+1} / {frame_count}")

                    frame_num += 1
            
            finally:
                cap.release()
                if video_writer is not None:
                    video_writer.release() 
                if video_path and os.path.exists(video_path):
                     os.unlink(video_path)

        status_text.success("Analysis complete. Frames and downloadable video file are ready.")
        progress_bar.empty()
        st.session_state.current_frame_index = 0 
        st.rerun()


# --- Display Results (Remains the same, referencing the new combined frame) ---
if st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
    
    st.session_state.analysis_df.index.name = 'Frame' 

    st.header(f"Annotated Video Preview & Frame-wise Review (Role: **{st.session_state.action_mode.replace('_', ' ')}**)") 

    # --- Interactive Frame Player ---
    st.subheader("Interactive Preview (Frame Simulation)")
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    if col1.button("▶️ Play", disabled=st.session_state.is_playing):
        st.session_state.is_playing = True
        st.rerun()

    if col2.button("⏸️ Stop", disabled=not st.session_state.is_playing):
        st.session_state.is_playing = False
        

    if st.session_state.is_playing:
        play_frame_simulation()
    
    
    st.markdown("---")
    
    # 2. Static Frame Selector and Display
    max_frame_index = st.session_state.analysis_df.index.max() if not st.session_state.analysis_df.empty else 0
    max_slider = max(0, max_frame_index)

    initial_slider_value = st.session_state.current_frame_index 
    if initial_slider_value > max_slider:
        initial_slider_value = 0
        st.session_state.current_frame_index = 0

    selected_frame = st.slider(
        "Select Frame for Static Analysis (Instant Load)", 
        0, 
        max_slider, 
        initial_slider_value
    )
    
    if not st.session_state.is_playing:
        st.session_state.current_frame_index = selected_frame

    # --- Metrics and Image Display for Selected Frame ---
    if selected_frame in st.session_state.analysis_df.index:
        current_metrics = st.session_state.analysis_df.loc[selected_frame]
    else:
        current_metrics = {
            'Left Elbow Angle': 0.0, 'Right Elbow Angle': 0.0, 'Right Knee Angle': 0.0, 
            'Trunk Flex Angle': 0.0, 'Action Phase': 'N/A'
        }
    
    st.subheader(f"Frame {selected_frame} Visual Analysis")
    
    if selected_frame in st.session_state.annotated_frame_bytes:
        frame_image_bytes = st.session_state.annotated_frame_bytes[selected_frame]
        
        frame_array = np.frombuffer(frame_image_bytes, np.uint8)
        combined_frame_bgr = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if combined_frame_bgr is not None:
            combined_frame_rgb = cv2.cvtColor(combined_frame_bgr, cv2.COLOR_BGR2RGB)
            st.image(
                combined_frame_rgb, 
                caption=f"Static Annotated Frame {selected_frame} - Phase: {current_metrics['Action Phase']} (Displaying 3-Panel Layout: 2 Top, 1 Bottom with Video Background and Speed Placeholders)", 
                use_container_width=True 
            ) 
        else:
             st.warning("Annotated image data for this frame is corrupted.")
             
    else:
        st.warning(f"Annotated image data for Frame {selected_frame} was not saved to memory during processing.")


    # Metrics Display
    st.markdown(f"### Current Action Phase: **{current_metrics['Action Phase']}**")
    
    mode = st.session_state.action_mode
    if 'Arm' in mode:
        arm = 'Left' if 'Left_Arm' in mode else 'Right'
        max_angle = st.session_state.max_bowling_elbow_angle
        # Display the Max Angle in the info box as a key biomechanical metric
        st.info(f"Key Biomechanical Focus: **MAX {arm} Elbow Angle = {max_angle:.2f}°**. Pay close attention to **Trunk Flexion** (for stability and rotation).")
    elif 'Batsman' in mode:
        lead_arm = 'Left' if mode == 'Right_Handed_Batsman' else 'Right'
        rear_leg = 'Right' if mode == 'Right_Handed_Batsman' else 'Left'
        st.info(f"Role Specific Focus: Pay close attention to the **{lead_arm} Elbow Angle** (lead arm position) and the **{rear_leg} Knee Angle** (rear leg stability and power generation).")

    
    col_l, col_r, col_k, col_t = st.columns(4)
    with col_l:
        st.metric("Left Elbow Angle", f"{current_metrics['Left Elbow Angle']:.2f}°")
    with col_r:
        st.metric("Right Elbow Angle", f"{current_metrics['Right Elbow Angle']:.2f}°")
    with col_k:
        st.metric("Right Knee Angle", f"{current_metrics['Right Knee Angle']:.2f}°")
    with col_t:
        st.metric("Trunk Flex Angle", f"{current_metrics['Trunk Flex Angle']:.2f}°")

    st.markdown("---")

    # --- DOWNLOAD BUTTONS ---
    
    if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
        base_name = st.session_state.original_file_name.rsplit('.', 1)[0]
        video_file_name = f"{base_name}_3panel_speed_placeholder_bg.mp4"
        
        with open(st.session_state.output_video_path, "rb") as video_file:
            st.download_button(
                label="⬇️ Download Annotated Video (Speed Placeholder Style)",
                data=video_file.read(),
                file_name=video_file_name,
                mime="video/mp4",
                help="Downloads the full annotated video file (3-panel layout: 2 wide by 2 high) using the video as the trail background with speed placeholders."
            )
    else:
        st.info("Run the analysis first to generate the downloadable video file.")
        
    pdf_file_name = f"{st.session_state.original_file_name.rsplit('.', 1)[0]}_report.pdf"

    st.download_button(
        label="⬇️ Download PDF Report",
        data=create_pdf_report(st.session_state.analysis_df, action_mode=st.session_state.action_mode),
        file_name=pdf_file_name,
        mime="application/pdf"
    )

    st.markdown("---")

    st.subheader("Tabular Data (All Frames)")
    df_to_display = st.session_state.analysis_df.reset_index()

    def highlight_row(row):
        if row['Frame'] == selected_frame:
            return ['background-color: #078BDC'] * len(row)
        else:
            return [''] * len(row)

    styled_df = df_to_display.style.apply(highlight_row, axis=1)

    st.dataframe(styled_df, height=300, use_container_width=True)