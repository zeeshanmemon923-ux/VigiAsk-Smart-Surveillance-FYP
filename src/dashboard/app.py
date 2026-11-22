"""
Smart CCTV System - Real-Time Monitoring & Search Dashboard

Streamlit-based dashboard providing:
1. Live / Recorded CCTV video stream with real-time CV annotations.
2. Configurable detection, tracking, restricted zones, and activity rules.
3. Real-time alert feed with evidence snapshot gallery.
4. Natural-language query search engine translating English to safe DB filters.
5. System metrics and evaluation reporting.
"""

from datetime import datetime
import json
from pathlib import Path
import sys
import time
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.activity.altercation import AltercationDetector
from src.activity.fall_detection import FallDetector
from src.activity.intrusion import IntrusionDetector, load_zones
from src.activity.loitering import LoiteringDetector
from src.activity.suspicious_object import SuspiciousObjectDetector
from src.alert.alert_manager import AlertManager
from src.detection.weapon_detector import WeaponDetector, draw_weapons
from src.activity.weapon_alert import WeaponEventDetector
from src.config import CONFIG
from src.database.db_manager import DatabaseManager
from src.detection.person_detector import ObjectDetector
from src.search.nl_search import NaturalLanguageSearchEngine
from src.tracking.tracker import PersonTracker
from src.utils.preprocessing import VideoSource

st.set_page_config(
    page_title="AI Smart CCTV Surveillance System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished surveillance console theme
st.markdown("""
<style>
    .main-header { font-size: 26px; font-weight: 700; color: #1E88E5; margin-bottom: 5px; }
    .sub-header { font-size: 14px; color: #78909C; margin-bottom: 20px; }
    .metric-card { background-color: #1E293B; border-radius: 8px; padding: 15px; border-left: 4px solid #1E88E5; }
    .alert-card { background-color: #2D1A1A; border-radius: 6px; padding: 10px; margin-bottom: 8px; border-left: 4px solid #EF5350; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db():
    return DatabaseManager(CONFIG.database.db_path)


@st.cache_resource
def get_search_engine():
    return NaturalLanguageSearchEngine(get_db())


@st.cache_resource
def get_alert_manager():
    return AlertManager(CONFIG.alert, get_db(), CONFIG.events_log_path)


db = get_db()
search_engine = get_search_engine()
alert_mgr = get_alert_manager()

# Sidebar: Configuration & Controls
st.sidebar.title("🎥 Control Console")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation",
    ["🔴 Live Monitoring", "🔎 Natural-Language Search", "📜 Event History & Evidence", "📊 Performance & Reliability"],
)

st.sidebar.markdown("### ⚙️ Video Source")
video_source_type = st.sidebar.selectbox(
    "Select Source",
        ["Sample Video (data/sample.mp4)", "Live Camera (Webcam 0)", "RTSP CCTV Camera", "Upload Video File"],
)

uploaded_file = None
source_path = "data/sample.mp4"

if video_source_type == "Sample Video (data/sample.mp4)":
    source_path = "data/sample.mp4"
elif video_source_type == "Live Camera (Webcam 0)":
    source_path = 0
elif video_source_type == "RTSP CCTV Camera":
    source_path = st.sidebar.text_input(
        "RTSP URL",
        value=os.environ.get("CCTV_RTSP_URL", "rtsp://user:pass@camera-ip:554/stream1"),
        type="password",
    )
elif video_source_type == "Upload Video File":
    uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        save_path = Path("data") / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        source_path = str(save_path)

st.sidebar.markdown("### 🛡️ Detection Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, float(CONFIG.detection.confidence_threshold), 0.05)
CONFIG.detection.confidence_threshold = conf_thresh

active_classes = st.sidebar.multiselect(
    "Target Classes to Detect",
    options=["person", "backpack", "handbag", "suitcase"],
    default=["person", "backpack", "handbag", "suitcase"],
)

# Map names to COCO IDs
name_to_id = {"person": 0, "backpack": 24, "handbag": 26, "suitcase": 28}
CONFIG.detection.target_classes = [name_to_id[c] for c in active_classes if c in name_to_id]

st.sidebar.markdown("### 🚨 Activity Rules")
enable_intrusion = st.sidebar.checkbox("Restricted-Area Intrusion", value=True)
enable_loitering = st.sidebar.checkbox("Loitering Detection", value=True)
enable_unattended = st.sidebar.checkbox("Unattended Object Detection", value=True)
enable_falls = st.sidebar.checkbox("Observable Fall Detection", value=True)
enable_altercation = st.sidebar.checkbox("High-Energy Interaction Indicator", value=True)
enable_weapon = st.sidebar.checkbox("Weapon Detection (Gun / Knife)", value=True)
sound_alert = st.sidebar.checkbox("Desktop Audio Alert", value=False)
CONFIG.alert.sound_alert = sound_alert


# -------------------------------------------------------------
# TAB 1: LIVE MONITORING
# -------------------------------------------------------------
if app_mode == "🔴 Live Monitoring":
    st.markdown('<div class="main-header">Smart CCTV Real-Time Video Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered multi-object tracking and suspicious activity detection pipeline</div>', unsafe_allow_html=True)

    col_video, col_alerts = st.columns([7, 3])

    with col_alerts:
        st.subheader("🚨 Live Event Stream")
        alert_placeholder = st.empty()

    with col_video:
        video_placeholder = st.empty()
        status_bar = st.empty()

        col_btn1, col_btn2 = st.columns([1, 1])
        start_btn = col_btn1.button("▶️ Start Monitoring", use_container_width=True)
        stop_btn = col_btn2.button("⏹️ Stop Monitoring", use_container_width=True)

    if start_btn:
        detector = ObjectDetector(CONFIG.detection)
        tracker = PersonTracker(CONFIG.detection, CONFIG.tracking)
        intrusion_det = IntrusionDetector(CONFIG.intrusion)
        loitering_det = LoiteringDetector(CONFIG.loitering)
        unattended_det = SuspiciousObjectDetector(CONFIG.suspicious_object)
        fall_det = FallDetector(CONFIG.fall_detection)
        altercation_det = AltercationDetector(CONFIG.altercation)
        weapon_det = WeaponDetector(CONFIG.weapon) if enable_weapon else None
        weapon_ev_det = WeaponEventDetector(CONFIG.weapon)

        zones = intrusion_det.zones

        try:
            with VideoSource(source_path, CONFIG.preprocess) as video:
                frame_count = 0
                fps_window = []

                for frame in video:
                    t_start = time.time()
                    frame_count += 1

                    # Run Multi-Class Tracking
                    tracked_items = tracker.update(
                        frame.image,
                        frame.timestamp_sec,
                        classes=CONFIG.detection.target_classes
                    )

                    people = [item for item in tracked_items if item.class_name == "person"]
                    objects = [item for item in tracked_items if item.class_name != "person"]

                    # 1. Intrusion Detection
                    new_alerts = []
                    if enable_intrusion and zones:
                        intrusions = intrusion_det.update(people, frame.timestamp_sec)
                        for ev in intrusions:
                            al = alert_mgr.process_event(ev.to_dict(), str(source_path), frame.image)
                            if al: new_alerts.append(al)

                    # 2. Loitering Detection
                    if enable_loitering:
                        loiterings = loitering_det.update(people, tracker.histories, frame.timestamp_sec)
                        for ev in loiterings:
                            al = alert_mgr.process_event(ev.to_dict(), str(source_path), frame.image)
                            if al: new_alerts.append(al)

                    # 3. Unattended Objects
                    if enable_unattended:
                        unattended = unattended_det.update(objects, people, frame.timestamp_sec)
                        for ev in unattended:
                            al = alert_mgr.process_event(ev.to_dict(), str(source_path), frame.image)
                            if al: new_alerts.append(al)

                    # 4. Fall Detection
                    if enable_falls:
                        falls = fall_det.update(people, tracker.histories, frame.timestamp_sec)
                        for ev in falls:
                            al = alert_mgr.process_event(ev.to_dict(), str(source_path), frame.image)
                            if al: new_alerts.append(al)

                    # 5. Altercation / Rapid Interaction
                    if enable_altercation:
                        altercations = altercation_det.update(people, tracker.histories, frame.timestamp_sec)
                        for ev in altercations:
                            al = alert_mgr.process_event(ev.to_dict(), str(source_path), frame.image)
                            if al: new_alerts.append(al)

                    tracker.prune_stale(frame.timestamp_sec)
                    # 6. Weapon Detection
                    if weapon_det is not None:
                        w_dets = weapon_det.update(frame.image)
                        if weapon_det.ran_this_frame:
                            for ev in weapon_ev_det.update(w_dets, people, frame.timestamp_sec):
                                evidence = draw_weapons(frame.image.copy(), w_dets)
                                al = alert_mgr.process_event(ev.to_dict(), str(source_path), evidence)
                                if al: new_alerts.append(al)

                    # Render Visual Annotations
                    vis = frame.image.copy()

                    # Draw Zones
                    for zone in zones:
                        pts = zone.polygon
                        for i in range(len(pts)):
                            p1 = tuple(map(int, pts[i]))
                            p2 = tuple(map(int, pts[(i + 1) % len(pts)]))
                            cv2.line(vis, p1, p2, (0, 0, 255), 2)
                        cv2.putText(vis, zone.name, tuple(map(int, pts[0])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Draw Tracks
                    inside_now = {tid for tid, _ in intrusion_det.currently_inside()}

                    for item in tracked_items:
                        x1, y1, x2, y2 = map(int, item.bbox_xyxy)
                        is_intruding = item.track_id in inside_now
                        is_loitering = loitering_det.is_loitering(item.track_id)
                        is_unattended = unattended_det.is_unattended(item.track_id)
                        is_fallen = fall_det.is_fallen(item.track_id)
                        is_fighting = altercation_det.is_in_interaction(item.track_id)

                        if is_intruding or is_fighting:
                            color = (0, 0, 255)      # Red
                        elif is_loitering:
                            color = (0, 165, 255)    # Orange
                        elif is_unattended:
                            color = (255, 0, 255)    # Magenta
                        elif is_fallen:
                            color = (0, 255, 255)    # Yellow
                        else:
                            color = (0, 255, 0)      # Green

                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        label = f"{item.class_name} #{item.track_id} ({item.confidence:.2f})"
                        if is_intruding: label += " [INTRUSION]"
                        if is_loitering: label += " [LOITERING]"
                        if is_unattended: label += " [UNATTENDED]"
                        if is_fallen: label += " [FALL]"
                        if is_fighting: label += " [INTERACTION]"

                        cv2.putText(vis, label, (x1, max(15, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if weapon_det is not None:
                        draw_weapons(vis, weapon_det.latest)

                    # Calculate latency & FPS
                    dt = time.time() - t_start
                    fps_window.append(1.0 / dt if dt > 0 else 30.0)
                    if len(fps_window) > 30: fps_window.pop(0)
                    cur_fps = sum(fps_window) / len(fps_window)

                    # Convert to RGB for Streamlit
                    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(vis_rgb, channels="RGB", use_container_width=True)

                    status_bar.markdown(
                        f"**Source:** `{source_path}` | **Frame:** `{frame.index}` | "
                        f"**FPS:** `{cur_fps:.1f}` | **Tracked Objects:** `{len(tracked_items)}` | "
                        f"**Timestamp:** `{frame.timestamp_sec:.2f}s`"
                    )

                    # Update Alert Feed
                    with col_alerts:
                        recent = list(alert_mgr.recent_alerts)[:8]
                        if recent:
                            feed_html = ""
                            for a in recent:
                                feed_html += f"""
                                <div class="alert-card">
                                    <b>⚠️ {a['event_type'].upper().replace('_', ' ')}</b><br>
                                    <small>Time: {a['datetime_str']} | Track: #{a['track_id']} | Conf: {a['confidence']:.2f}</small>
                                </div>
                                """
                            alert_placeholder.markdown(feed_html, unsafe_allow_html=True)
                        else:
                            alert_placeholder.info("No suspicious events detected.")

        except Exception as e:
            st.error(f"Error accessing video source '{source_path}': {e}")


# -------------------------------------------------------------
# TAB 2: NATURAL-LANGUAGE SEARCH
# -------------------------------------------------------------
elif app_mode == "🔎 Natural-Language Search":
    st.markdown('<div class="main-header">Natural-Language Event Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Search recorded CCTV incidents using everyday English queries</div>', unsafe_allow_html=True)

    query = st.text_input(
        "Enter your query in plain English:",
        placeholder="e.g., 'Show all restricted-area intrusions yesterday' or 'Find loitering over 30 seconds on camera 0'",
    )

    st.markdown("**Sample queries to try:**")
    col_q1, col_q2, col_q3 = st.columns(3)
    if col_q1.button("📍 'Show all intrusions'"):
        query = "Show all restricted-area intrusions"
    if col_q2.button("⏱️ 'Find loitering over 30s'"):
        query = "Find events where a person was loitering for more than 30 seconds"
    if col_q3.button("🎒 'Show unattended objects'"):
        query = "Show all unattended bags and suitcases with confidence over 0.4"

    if query:
        results, parsed_filter = search_engine.search(query)

        st.markdown("### 🛡️ Structured Filter Translation")
        st.info(f"**Natural Query:** *\"{query}\"*\n\n**Converted Safe SQL Parameters:** `{parsed_filter.to_dict()}`")

        st.markdown(f"### 📋 Matching Events ({len(results)} found)")

        if results:
            df = pd.DataFrame(results)
            display_cols = [c for c in ["id", "datetime_str", "event_type", "camera_source", "track_id", "confidence", "duration_seconds", "zone_name"] if c in df.columns]
            st.dataframe(df[display_cols], use_container_width=True)

            st.markdown("### 🖼️ Evidence Gallery for Search Results")
            cols = st.columns(4)
            for idx, r in enumerate(results[:12]):
                col = cols[idx % 4]
                img_path = r.get("evidence_frame_path")
                with col:
                    if img_path and Path(img_path).exists():
                        img = Image.open(img_path)
                        st.image(img, caption=f"Event #{r['id']} - {r['event_type']} ({r['datetime_str']})", use_container_width=True)
                    else:
                        st.warning(f"Event #{r['id']}: Snapshot pending")
        else:
            st.warning("No recorded incidents matched the specified query criteria.")


# -------------------------------------------------------------
# TAB 3: EVENT HISTORY & EVIDENCE GALLERY
# -------------------------------------------------------------
elif app_mode == "📜 Event History & Evidence":
    st.markdown('<div class="main-header">CCTV Incident History & Evidence Records</div>', unsafe_allow_html=True)

    stats = db.get_event_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Recorded Incidents", stats.get("total_events", 0))
    col2.metric("Intrusion Alerts", stats.get("by_type", {}).get("restricted_area_intrusion", 0))
    col3.metric("Loitering Alerts", stats.get("by_type", {}).get("loitering", 0))

    st.markdown("---")
    records = db.get_recent_events(limit=100)
    if records:
        df = pd.DataFrame(records)
        display_cols = [c for c in ["id", "datetime_str", "event_type", "camera_source", "track_id", "confidence", "duration_seconds", "zone_name", "status"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)

        st.subheader("📸 Captured Snapshot Evidence Gallery")
        gallery_cols = st.columns(4)
        for idx, rec in enumerate(records[:16]):
            p = rec.get("evidence_frame_path")
            if p and Path(p).exists():
                with gallery_cols[idx % 4]:
                    st.image(Image.open(p), caption=f"Event #{rec['id']}: {rec['event_type']}\n{rec['datetime_str']}", use_container_width=True)
    else:
        st.info("No recorded CCTV events yet in database. Start live monitoring or test footage to generate alerts.")


# -------------------------------------------------------------
# TAB 4: PERFORMANCE & RELIABILITY
# -------------------------------------------------------------
elif app_mode == "📊 Performance & Reliability":
    st.markdown('<div class="main-header">System Reliability & Scientific Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Objective 3 Benchmark: Precision, Recall, F1, Latency, and Retrieval Accuracy</div>', unsafe_allow_html=True)

    report_path = Path("data/evaluation_report.json")
    chart_path = Path("data/evaluation_chart.png")

    if report_path.exists():
        with open(report_path) as f:
            rep = json.load(f)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{rep.get('precision', 0):.3f}")
        c2.metric("Recall", f"{rep.get('recall', 0):.3f}")
        c3.metric("F1-Score", f"{rep.get('f1_score', 0):.3f}")
        c4.metric("Processing Speed", f"{rep.get('fps', 0):.1f} FPS")

        st.markdown(f"**Mean Latency:** `{rep.get('mean_latency_ms', 0):.1f} ms` | **False-Positive Rate:** `{rep.get('fp_rate', 0):.3f}`")

    if chart_path.exists():
        st.image(str(chart_path), caption="System Performance Measures", use_container_width=True)
