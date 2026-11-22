# AI-Based Smart CCTV Surveillance System

Real-Time Suspicious Activity Detection and Natural-Language Event Search Using Computer Vision.

## System Capabilities

1. **Object Detection:** Multi-class detection using YOLOv8 (person, backpack, handbag, suitcase) with configurable per-class thresholds.
2. **Multi-Object Tracking:** ByteTrack & DeepSORT assigning persistent IDs and tracking trajectories across frames.
3. **Suspicious Activity Detection:**
   - **Restricted-Area Intrusion:** Configurable polygon zones with confirmation frame hysteresis.
   - **Loitering Detection:** Dwell time and spatial wander thresholding.
   - **Unattended / Suspicious Objects:** Stationary luggage or bags left without an owner nearby for > 15 seconds.
   - **Observable Fall Detection:** Aspect-ratio collapse ($W/H > 0.85$) and rapid downward velocity.
   - **High-Energy Physical Interaction Indicator:** Proximity and rapid displacement variance between tracked persons.
4. **Alert System:**
   - Real-time alert generation with deduplication / cooldown (preventing alert spam).
   - Automatic snapshot capture of evidence frames saved to `data/evidence/`.
   - Structured persistence in SQLite database (`data/cctv_events.db`) and JSONL log (`data/events.jsonl`).
   - Desktop audio alert and configurable SMTP email dispatch.
5. **CCTV Monitoring Dashboard:**
   - Streamlit console with live video feed, real-time bounding boxes, intrusion zones, and live alert feed.
   - Evidence gallery with snapshot viewer.
6. **Natural-Language Search:**
   - Semantic query translation (e.g. *"Show all intrusions yesterday"*, *"Find loitering over 30s on camera 0"*) into safe, parameterized SQLite database filters.

---

## Running the Monitoring Dashboard

Launch the web surveillance console:

```bash
py -3.11 -m streamlit run src/dashboard/app.py
```

Then navigate to `http://localhost:8501` in your browser.

---

## Running Full Pipeline Demo (CLI)

```bash
# Draw a restricted zone first (if needed):
py -3.11 tests/zone_editor.py --source data/sample.mp4

# Run full pipeline with live video display:
py -3.11 tests/run_full_pipeline.py --source data/sample.mp4

# Run on live webcam:
py -3.11 tests/run_full_pipeline.py --source 0
```

---

## Running Test Suites

```bash
# 1. Module Unit Tests
py -3.11 tests/test_units.py

# 2. Natural-Language Search & SQL-Injection Safety Tests
py -3.11 tests/test_nl_search.py

# 3. System Robustness & Error Handling Tests
py -3.11 tests/test_robustness.py

# 4. Performance Benchmark & Thesis Evaluation Report
py -3.11 tests/run_evaluation.py --source data/sample.mp4
```

