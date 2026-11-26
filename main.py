import os
import time
import threading
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import cfg
from boundary.boundary_store import load_boundary
from boundary.polygon_draw import draw_polygon_on_frame
from utils.geometry import point_in_polygon, distance_to_polygon_edges, point_on_polygon_edge
from logic.threat_engine import ThreatEngine
from logic.violence_engine import ViolenceEngine
from evidence.capture import EvidenceCapture
from dashboard.app import create_app
from dashboard.events import log_event
from alarm.siren import Siren
from detection.violence_detector import HybridFightDetector

print("✅ AI CATCHER RUNNING (Intrusion=Beep, Fight=WAV)")

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area + 1e-6)

def choose_device():
    if str(cfg.DEVICE).lower() == "cpu":
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"

def open_capture(url: str):
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap

def rotate_frame(frame):
    if cfg.ROTATE == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if cfg.ROTATE == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if cfg.ROTATE == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def auto_crop_roi(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray < cfg.ROI_WHITE_THRESH
    ys, xs = np.where(mask)

    if len(xs) < int(h * w * cfg.ROI_MIN_CONTENT_RATIO):
        return frame, {"x": 0, "y": 0, "w": w, "h": h}

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    pad = cfg.ROI_PAD
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)

    roi = {"x": int(x1), "y": int(y1), "w": int(x2 - x1 + 1), "h": int(y2 - y1 + 1)}
    cropped = frame[roi["y"]:roi["y"] + roi["h"], roi["x"]:roi["x"] + roi["w"]]
    return cropped, roi

def draw_polygon_overlay(frame, poly):
    if len(poly) >= 3:
        cv2.polylines(frame, [np.array(poly, dtype=np.int32)], True, (0, 255, 255), 2)

def start_dashboard(shared):
    app = create_app(shared)
    app.run(host=cfg.DASHBOARD_HOST, port=5000, debug=False, use_reloader=False)

def main():
    device = choose_device()
    print("cwd:", os.getcwd())
    print("cuda:", torch.cuda.is_available(), "device:", device)
    print("Dashboard:", f"http://localhost:3000 (Node.js React/HTML Frontend)")
    print("NOTE: If boundary seems wrong, delete boundary_points.json and redraw.")

    shared = {"latest_frame": None, "latest_status": {"text": "NORMAL", "class": "normal"}, "cfg": cfg}
    
    # Start the Flask app for the video_feed on port 5000
    threading.Thread(target=start_dashboard, args=(shared,), daemon=True).start()

    cap = open_capture(cfg.CAMERA_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {cfg.CAMERA_URL}")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame")

    frame = rotate_frame(frame)

    if cfg.ROI_AUTO_CROP:
        roi_frame, roi = auto_crop_roi(frame)
    else:
        h, w = frame.shape[:2]
        roi_frame, roi = frame, {"x": 0, "y": 0, "w": w, "h": h}

    polygon, saved_roi = load_boundary(cfg.BOUNDARY_FILE)
    if saved_roi is None or saved_roi != roi:
        polygon = []

    if len(polygon) < 3:
        polygon = draw_polygon_on_frame(roi_frame, cfg.BOUNDARY_FILE, roi)

    person_model = YOLO(cfg.YOLO_MODEL)

    fight_detector = HybridFightDetector(
        model_path=os.path.join(cfg.BASE_DIR, "runs", "classify", "runs", "classify", "violence_classifier", "weights", "best.pt"),
        device=device
    )
    print("✅ Hybrid Fight detector loaded (ML + Pose Heuristics)")

    cfg.load_dynamic() # Load whatever the user set via the Control Panel JSON
    
    # Initialize Threat Engine (Layer 3)
    threat = ThreatEngine(cfg.INTRUSION_SECONDS, cfg.WARNING_IMG_COOLDOWN_SEC)
    violence = ViolenceEngine(cfg.FIGHT_HITS_TO_CONFIRM, cfg.FIGHT_HIT_DECAY_SEC)

    siren = Siren(cfg.CRITICAL_SIREN_WAV)

    evidence = EvidenceCapture(cfg.SAVE_DIR, cfg.RECORD_FPS, cfg.INTRUSION_CLIP_PRE_SEC, cfg.INTRUSION_CLIP_POST_SEC)
    threat_evidence = EvidenceCapture(cfg.SAVE_DIR, cfg.RECORD_FPS, cfg.THREAT_CLIP_PRE_SEC, cfg.THREAT_CLIP_POST_SEC)

    # status smoothing (no flicker)
    last_status = "NORMAL"
    last_status_time = time.time()
    priority = {"NORMAL": 0, "WARNING": 1, "INTRUSION": 2, "CRITICAL: FIGHTING": 3, "CRITICAL: FIGHTING & INTRUSION": 4}

    # critical cooldown
    last_critical_time = 0.0
    frame_idx = 0

    tracker_path = "custom_tracker.yaml"  # Custom highly-tolerant configuration to survive ultra-fast action blurring

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            time.sleep(0.5)
            cap = open_capture(cfg.CAMERA_URL)
            continue

        frame = rotate_frame(frame)

        if cfg.ROI_AUTO_CROP:
            roi_frame, roi = auto_crop_roi(frame)
        else:
            h, w = frame.shape[:2]
            roi_frame, roi = frame, {"x": 0, "y": 0, "w": w, "h": h}

        now = time.time()
        frame_idx += 1  # type: ignore
        
        # Continuously reload config to instantly reflect changes made in the Node Dashboard Control Panel
        if frame_idx % 10 == 0:
            cfg.load_dynamic()

        evidence.push_frame(now, roi_frame)
        threat_evidence.push_frame(now, roi_frame)

        # Dynamically scale confidence from slider:
        # User 100% -> Yolo 0.15 (High sensitivity)
        # User 0%   -> Yolo 0.65 (Low sensitivity) 
        user_sens = getattr(cfg, "INTRUSION_SENSITIVITY_PCT", 70)
        current_conf = max(0.15, min(0.65, 0.65 - (user_sens / 100.0) * 0.50))

        results = person_model.track(
            roi_frame,
            persist=True,
            tracker=tracker_path,
            conf=current_conf,
            imgsz=cfg.IMGSZ,
            classes=[0],
            device=device,
            verbose=False
        )

        layer_counts = {1: 0, 2: 0, 3: 0}
        any_intrusion = False
        inside_indices = []
        fight_candidate_indices = []  # includes inside + near (layer 2+) people

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
            confs = boxes.conf.cpu().numpy()
            
            keypoints = None
            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                if hasattr(results[0].keypoints, 'data') and results[0].keypoints.data is not None:
                    keypoints = results[0].keypoints.data.cpu().numpy()
                elif hasattr(results[0].keypoints, 'xy') and results[0].keypoints.xy is not None:
                    keypoints = results[0].keypoints.xy.cpu().numpy()

            # --- GHOST BOX IOU NMS RATIO FILTER ---
            keep_indices: list[int] = []
            xyxy_float = xyxy.astype(float)
            for i in range(len(xyxy_float)):
                keep = True
                for idx_j in range(len(keep_indices)):
                    j = keep_indices[idx_j]
                    if compute_iou(xyxy_float[i], xyxy_float[j]) > 0.50:
                        # Massive overlap, drop the lower confidence one
                        if confs[i] > confs[j]:
                            keep_indices[idx_j] = i # faster than remove+append
                        keep = False
                        break
                if keep:
                    keep_indices.append(i)
                    
            if len(keep_indices) < len(xyxy):
                keep_indices = sorted(keep_indices)
                xyxy = xyxy[keep_indices]
                if ids is not None: 
                    ids = np.take(ids, keep_indices)
                confs = confs[keep_indices]
                if keypoints is not None: keypoints = keypoints[keep_indices]
            # --- END GHOST BOX FILTER ---

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].astype(int)
                track_id = int(ids[i]) if ids is not None else -1  # type: ignore

                # ✅ two-point check (feet + center) to avoid "touch line but outsider"
                px = (x1 + x2) // 2
                py_feet = y2
                py_center = (y1 + y2) // 2

                inside_feet = point_in_polygon((px, py_feet), polygon) or point_on_polygon_edge((px, py_feet), polygon, eps=cfg.EDGE_EPS_PX)
                inside_center = point_in_polygon((px, py_center), polygon) or point_on_polygon_edge((px, py_center), polygon, eps=cfg.EDGE_EPS_PX)

                inside = inside_feet or inside_center
                dist = min(distance_to_polygon_edges((px, py_feet), polygon),
                           distance_to_polygon_edges((px, py_center), polygon))
                near = dist <= cfg.NEAR_DISTANCE_PX
                
                # Keep track of indices that are inside for the fight check later
                if inside:
                    inside_indices.append(i)
                
                # Fight candidates: anyone inside or near the boundary (layer 2+)
                if inside or near:
                    fight_candidate_indices.append(i)

                layer, intrusion_just, warn_img = threat.update(track_id, inside, near, now)
                layer_counts[layer] += 1
                if layer == 3:
                    any_intrusion = True

                # warning evidence
                if warn_img and layer == 2:
                    img_path = evidence.capture_image(roi_frame, prefix=f"L2_track{track_id}")
                    log_event(cfg.EVENTS_FILE, {
                        "layer": 2,
                        "track_id": track_id,
                        "reason": "warning",
                        "image_path": img_path
                    })

                # intrusion evidence + beep
                if intrusion_just and layer == 3:
                    img_path = evidence.capture_image(roi_frame, prefix=f"L3_track{track_id}")
                    vid_path = evidence.start_intrusion_clip(now, (roi_frame.shape[1], roi_frame.shape[0]), prefix=f"L3_track{track_id}")  # type: ignore
                    log_event(cfg.EVENTS_FILE, {
                        "layer": 3,
                        "track_id": track_id,
                        "reason": f"intrusion >= {cfg.INTRUSION_SECONDS}s",
                        "image_path": img_path,
                        "video_path": vid_path
                    })
                    if cfg.INTRUSION_BEEP_ENABLED:
                        siren.beep_intrusion()

                # draw boxes
                color = (0,255,0) if layer == 1 else (0,255,255) if layer == 2 else (0,0,255)
                cv2.rectangle(roi_frame, (x1,y1), (x2,y2), color, 2)
                cv2.circle(roi_frame, (px, py_feet), 4, color, -1)
                cv2.putText(roi_frame, f"ID:{track_id} L{layer} {confs[i]:.2f}",
                            (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # draw keypoints and skeleton if pose model returns them
                if keypoints is not None and len(keypoints) > i:  # type: ignore
                    kpts = keypoints[i]
                    for k in kpts:
                        kx, ky = int(k[0]), int(k[1])
                        if kx > 0 and ky > 0:
                            cv2.circle(roi_frame, (kx, ky), 3, (255, 0, 255), -1)
                    
                    skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
                    for edge in skeleton:
                        if edge[0] < len(kpts) and edge[1] < len(kpts):
                            x_1, y_1 = int(kpts[edge[0]][0]), int(kpts[edge[0]][1])  # type: ignore
                            x_2, y_2 = int(kpts[edge[1]][0]), int(kpts[edge[1]][1])  # type: ignore
                            if x_1 > 0 and y_1 > 0 and x_2 > 0 and y_2 > 0:
                                cv2.line(roi_frame, (x_1, y_1), (x_2, y_2), (255, 100, 255), 2)

        draw_polygon_overlay(roi_frame, polygon)

        # ---- Layer-4 fight classification (always checking) ----
        critical = False
        fight_score = 0.0

        num_fight_candidates = len(fight_candidate_indices)
        if cfg.FIGHT_MODULE_ENABLED and fight_detector is not None and num_fight_candidates >= 2:
             if frame_idx % max(1, cfg.FIGHT_CHECK_EVERY_N_FRAMES) == 0:
                 
                 # Initialize empty arrays if no tracking results exist this frame
                 current_xyxy = xyxy[fight_candidate_indices] if 'xyxy' in locals() and xyxy is not None else np.array([])
                 current_ids = ids[fight_candidate_indices] if 'ids' in locals() and ids is not None else np.array([])
                 
                 # Safely construct keypoints array
                 if keypoints is not None:
                     current_keypoints = np.array([keypoints[idx] for idx in fight_candidate_indices if idx < len(keypoints)])
                 else:
                     current_keypoints = None

                 # Pass the whole ROI frame + the pose tracking data to the Hybrid model
                 fight_score = fight_detector.update_and_predict(roi_frame, current_xyxy, current_ids, now, current_keypoints)
                 
                 # Evaluate using ViolenceEngine to prevent 1-frame tracking glitches
                 critical = violence.update(fight_score, cfg.FIGHT_CONF)
        else:
            # Drop violence if < 2 candidates or disabled
            critical = violence.update(0.0, cfg.FIGHT_CONF)

        if critical and (now - last_critical_time) >= cfg.CRITICAL_COOLDOWN_SEC:
            last_critical_time = now
            img_path = threat_evidence.capture_image(roi_frame, prefix="L4_fight")
            vid_path = threat_evidence.start_intrusion_clip(now, (roi_frame.shape[1], roi_frame.shape[0]), prefix="L4_fight")
            log_event(cfg.EVENTS_FILE, {
                "layer": 4,
                "track_id": -1,
                "reason": f"violence_score={fight_score:.2f}",
                "image_path": img_path,
                "video_path": vid_path
            })
            if cfg.CRITICAL_SIREN_ENABLED:
                siren.play_critical()

        # raw status
        if critical and any_intrusion:
            raw_status = "CRITICAL: FIGHTING & INTRUSION"
        elif critical:
            raw_status = "CRITICAL: FIGHTING"
        elif any_intrusion:
            raw_status = "INTRUSION"
        elif layer_counts[2] > 0:
            raw_status = "WARNING"
        else:
            raw_status = "NORMAL"

        # hysteresis hold
        if raw_status != last_status:
            if priority[raw_status] > priority[last_status]:
                last_status = raw_status
                last_status_time = now
            else:
                if (now - last_status_time) >= cfg.STATUS_HOLD_SEC:
                    last_status = raw_status
                    last_status_time = now

        status_text = last_status
        if "CRITICAL" in status_text:
            status_class = "critical"
        elif status_text == "INTRUSION":
            status_class = "intrusion"
        elif status_text == "WARNING":
            status_class = "warning"
        else:
            status_class = "normal"

        shared["latest_status"] = {"text": status_text, "class": status_class}

        # overlay + dashboard frame share
        cv2.putText(roi_frame, f"{cfg.APP_NAME} {cfg.VERSION}", (10, roi_frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(roi_frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0,255,0) if status_class == "normal" else (0,255,255) if status_class == "warning" else (0,0,255), 3)

        if fight_detector is not None:
            cv2.putText(roi_frame, f"violence={fight_score:.2f} thr={cfg.FIGHT_CONF:.2f}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        shared["latest_frame"] = roi_frame.copy()

        cv2.imshow("AI Catcher - ROI View", roi_frame)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord("q"), ord("Q")):
            break
        if k in (ord("b"), ord("B")):
            polygon = draw_polygon_on_frame(roi_frame, cfg.BOUNDARY_FILE, roi)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()