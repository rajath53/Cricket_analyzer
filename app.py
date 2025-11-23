import streamlit as st
import os
import time
import io
import cv2
import base64
import pandas as pd

from video_processor import process_video
from cricket_pose import render_frame_overlay, write_frame_pdf

st.set_page_config(page_title="Cricket Analyzer", layout="wide")
st.title("🏏 Pose Analyzer (Framewise + Live Speed Control)")

# -------------------- SESSION STATE --------------------
if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None
if "annotated_video" not in st.session_state:
    st.session_state.annotated_video = None
if "csv_path" not in st.session_state:
    st.session_state.csv_path = None
if "video_bytes_b64" not in st.session_state:
    st.session_state.video_bytes_b64 = None

# -------------------- FILE UPLOAD -----------------------
uploaded = st.file_uploader("Upload a bowling/batting clip", type=["mp4","mov","avi"])
st.markdown("Use the speed selector to change playback rate even after analysis completes.")

# -------------------- SPEED SELECTOR ---------------------
speed_choice = st.selectbox(
    "Annotated playback speed",
    ["Original (1.0x)", "Half speed (0.5x)", "Quarter speed (0.25x)"]
)
speed_map = {"Original (1.0x)": 1.0, "Half speed (0.5x)": 0.5, "Quarter speed (0.25x)": 0.25}
selected_speed = speed_map[speed_choice]

# -------------------- RUN ANALYSIS -----------------------
if uploaded:
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    video_path = os.path.join(uploads_dir, uploaded.name)

    with open(video_path, "wb") as f:
        f.write(uploaded.read())
    st.success("Uploaded successfully")
    st.video(video_path)

    if st.button("Run Analysis"):
        with st.spinner("Running analysis… Please wait…"):
            start = time.time()
            out_video, df, csv_path = process_video(video_path, slow_factor=1.0)
            elapsed = time.time() - start

        st.session_state.analysis_ready = True
        st.session_state.analysis_data = df
        st.session_state.annotated_video = out_video
        st.session_state.csv_path = csv_path

        with open(out_video, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            st.session_state.video_bytes_b64 = b64

        st.success(f"Analysis completed in {elapsed:.1f}s")
        st.rerun()

# -------------------- ANALYSIS PANEL -----------------------
if st.session_state.analysis_ready:
    out_video = st.session_state.annotated_video
    df = st.session_state.analysis_data
    csv_path = st.session_state.csv_path
    b64 = st.session_state.video_bytes_b64

    st.markdown("### Annotated Video")

    video_html = f"""
    <video id="annotated_video" controls style="max-width:100%;height:auto;">
        <source src="data:video/mp4;base64,{b64}" type="video/mp4">
    </video>

    <script>
        const vid = document.getElementById('annotated_video');
        vid.playbackRate = {selected_speed};
    </script>
    """

    st.components.v1.html(video_html, height=450)

    st.components.v1.html(
        f"""
        <script>
            const p = document.getElementById('annotated_video');
            if(p) p.playbackRate = {selected_speed};
        </script>
        """,
        height=0
    )

    # CSV download
    with open(csv_path, "rb") as f:
        st.download_button(
            "Download framewise CSV",
            data=f,
            file_name=os.path.basename(csv_path),
            mime="text/csv"
        )

    # -------------------- FRAMEWISE TABLE -----------------------
    st.markdown("### Framewise Metrics")
    max_frame = int(df["frame"].max())
    sel_frame = st.slider("Select frame to inspect", 1, max_frame, 1)

    # ------------- NEON HIGHLIGHT ----------------
    def highlight_row(row):
        if row["frame"] == sel_frame:
            return ['background-color: #078BDC'] * len(row)
        return [''] * len(row)

    st.dataframe(df.style.apply(highlight_row, axis=1), use_container_width=True)

    # -------------------- FRAME EXTRACTION -----------------------
    cap = cv2.VideoCapture(out_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, sel_frame - 1)
    ret, annotated_frame = cap.read()
    cap.release()

    if ret:
        st.markdown("### Annotated Frame Preview")
        st.image(annotated_frame[:, :, ::-1], caption=f"Annotated frame {sel_frame}", use_column_width=True)

        row = df[df["frame"] == sel_frame].iloc[0].to_dict()

        try:
            preview = render_frame_overlay(annotated_frame.copy(), row)
            st.image(preview[:, :, ::-1], caption="Re-rendered overlay", use_column_width=True)
        except:
            pass

        # -------------------- PDF EXPORT -----------------------
        if st.button("Generate PDF for this frame"):
            pdf_path = os.path.join("reports", f"{os.path.basename(out_video)}_frame_{sel_frame}.pdf")
            os.makedirs("reports", exist_ok=True)

            write_frame_pdf(
                full_annotated_bgr=annotated_frame,
                metrics_dict=row,
                df_full=df,
                selected_frame=sel_frame,
                out_pdf_path=pdf_path,
                title=f"Frame {sel_frame} Report"
            )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download PDF",
                    data=f,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf"
                )

    # -------------------- DOWNLOAD ANNOTATED VIDEO -----------------------
    with open(out_video, "rb") as f:
        st.download_button(
            "Download annotated video",
            data=f,
            file_name=os.path.basename(out_video),
            mime="video/mp4"
        )
