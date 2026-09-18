import os
import time
import argparse
import logging
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import ultralytics as ulx  


@dataclass
class DrowsyConfig:
    # Đường dẫn model
    model_dir: str
    yolov8_face: str = "yolov8-face.pt"
    eye_keras: str = "eye_detector_gpu.h5"
    mouth_keras: str = "mouth_detector_gpu.h5"
    eye_cascade_xml: str = "haarcascade_eye.xml"
    mouth_cascade_xml: str = "haarcascade_mcs_mouth.xml"

    # Phát hiện mặt (tối ưu FPS)
    detect_every_n_frames: int = 10
    yolo_downscale_width: int = 480

    # PERCLOS
    perclos_window_sec: float = 60.0
    perclos_thr: float = 0.30
    perclos_warmup_sec: float = 20.0

    # Eye
    eye_continuous_closed_sec: float = 1.5
    eye_open_prob_thr: float = 0.70
    eye_yawn_aid_min_sec: float = 0.2

    # Yawn counting
    yawn_count_window_sec: float = 60.0
    yawn_count_thr: int = 2
    yawn_min_duration_sec: float = 1.0

    # Balanced yawn decision
    yawn_sure_open_sec: float = 1.4
    yawn_prob_open_sec: float = 0.8
    perclos_yawn_aid_thr: float = 0.20

    # Mouth gates
    mouth_ratio_thr: float = 0.34
    mouth_prob_thr: float = 0.50

    # Temporal analysis window
    mouth_temporal_win: float = 3.0
    speech_peaks_min: int = 4
    speech_zcr_min_hz: float = 2.3
    yawn_plateau_min_s: float = 0.8
    yawn_zcr_max_hz: float = 1.2
    yawn_area_min: float = 0.22

    # Talking/Yawn hysteresis
    on_min_sec: float = 1.5
    off_min_sec: float = 2.0
    yawn_lock_sec: float = 0.5
    talking_min_hold_sec: float = 0.6

    # EMA smoothing
    ema_alpha: float = 0.3

    # Tiền xử lý Keras
    gray_eye: bool = False
    gray_mouth: bool = False

    # Guard "nhìn xuống"
    gaze_down_ratio: float = 0.85
    eye_closed_debounce_sec: float = 0.30
    gaze_down_min_delta: float = 0.08
    gaze_min_brightness: float = 20.0

    # YOLO threshold (CLI)
    yolo_conf: float = 0.50
    yolo_iou: float = 0.45

    # Recovery settings
    recovery_duration_sec: float = 3.0       # thời gian giữ điều kiện recovery
    recovery_perclos_thr: float = 0.15       # PERCLOS < 15%
    recovery_yawn_cooldown: float = 3.0      # không ngáp trong N giây


def _largest_box(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    areas = [w * h for (_, _, w, h) in boxes]
    return boxes[int(np.argmax(areas))]


def _resize_keep_aspect(img, new_w):
    """Resize theo bề rộng, giữ tỉ lệ. Trả về ảnh và tỉ lệ scale."""
    h, w = img.shape[:2]
    if new_w <= 0 or w <= new_w:
        return img, 1.0
    r = new_w / float(w)
    new_h = int(h * r)
    small = cv2.resize(img, (new_w, new_h))
    return small, r


class DrowsinessDetector:
    def __init__(self, cfg: DrowsyConfig, save_csv_path: str = None):
        self.cfg = cfg
        self.save_csv_path = save_csv_path

        # Đường dẫn
        self.face_model_path = os.path.join(cfg.model_dir, cfg.yolov8_face)
        self.eye_model_path = os.path.join(cfg.model_dir, cfg.eye_keras)
        self.mouth_model_path = os.path.join(cfg.model_dir, cfg.mouth_keras)
        self.eye_cascade_path = os.path.join(cfg.model_dir, cfg.eye_cascade_xml)
        self.mouth_cascade_path = os.path.join(cfg.model_dir, cfg.mouth_cascade_xml)

        # Kiểm tra tồn tại
        if not os.path.exists(self.face_model_path):
            raise FileNotFoundError(f"YOLO face model not found: {self.face_model_path}")
        if not os.path.exists(self.eye_model_path):
            raise FileNotFoundError(f"Keras eye model not found: {self.eye_model_path}")
        if not os.path.exists(self.mouth_model_path):
            raise FileNotFoundError(f"Keras mouth model not found: {self.mouth_model_path}")
        if not os.path.exists(self.eye_cascade_path):
            raise FileNotFoundError(f"Eye cascade not found: {self.eye_cascade_path}")
        if not os.path.exists(self.mouth_cascade_path):
            raise FileNotFoundError(f"Mouth cascade not found: {self.mouth_cascade_path}")

        # model
        self.face_model = YOLO(self.face_model_path, task='detect')
        self.eye_model = tf.keras.models.load_model(self.eye_model_path)
        self.mouth_model = tf.keras.models.load_model(self.mouth_model_path)

        # Warmup
        dummy_ch = 1 if (cfg.gray_eye or cfg.gray_mouth) else 3
        dummy = tf.zeros((1, 64, 64, dummy_ch), dtype=tf.float32)
        _ = self.eye_model(dummy, training=False)
        _ = self.mouth_model(dummy, training=False)

        # cascade
        self.eye_cascade = cv2.CascadeClassifier(self.eye_cascade_path)
        if self.eye_cascade.empty():
            raise FileNotFoundError(f"Failed to load eye cascade: {self.eye_cascade_path}")
        self.mouth_cascade = cv2.CascadeClassifier(self.mouth_cascade_path)
        if self.mouth_cascade.empty():
            raise FileNotFoundError(f"Failed to load mouth cascade: {self.mouth_cascade_path}")

        # Vẽ
        self.colors = {'face': (255, 0, 0), 'eye': (0, 0, 255), 'mouth': (0, 255, 255)}
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Trạng thái
        self.frame_idx = 0
        self.last_face_bbox = None
        self._face_bbox_ema = None

        self.eye_closed_frames = 0
        self.perclos_hist = deque(maxlen=4000)  # (ts, True/False/None)
        self.eye_closed_start_ts = None
        self._eye_closed_debounce_ts = None

        self.yawn_active = False
        self.yawn_start_ts = None
        self.yawn_events = deque()

        self.mouth_ratio_hist = deque(maxlen=1000)
        self.mouth_prob_hist = deque(maxlen=1000)
        self.mouth_open_start_ts = None
        self.mouth_yawn_start_ts = None
        self.last_yawn_set_ts = 0.0
        self.talking_since_ts = None

        self.alert_on = False
        self.alert_cross_ts = None

        self.eye_prob_ema = None

        # Recovery state
        self.recovery_start_ts = None

        # FPS
        self._fps_ts = time.time()
        self._fps_cnt = 0
        self._fps = 0.0
        self._fps_log_window_start = time.time()

        # Fallback ctr
        self._mouth_miss_streak = 0

        # CSV logging
        if self.save_csv_path:
            with open(self.save_csv_path, "w", encoding="utf-8") as f:
                f.write("ts,perclos,eye_status,mouth_status,mouth_ratio,mouth_prob,plateau,zcr,peaks,above,status\n")

        logging.info("DrowsinessDetector initialized.")

    # PERCLOS
    def _update_perclos(self, now_ts, is_eye_closed):
        self.perclos_hist.append((now_ts, is_eye_closed))
        cutoff = now_ts - self.cfg.perclos_window_sec
        while self.perclos_hist and self.perclos_hist[0][0] < cutoff:
            self.perclos_hist.popleft()
        if len(self.perclos_hist) < 2:
            return 0.0, 0.0

        span = self.perclos_hist[-1][0] - self.perclos_hist[0][0]
        closed_time = 0.0
        for i in range(1, len(self.perclos_hist)):
            t0, c0 = self.perclos_hist[i - 1]
            t1, _ = self.perclos_hist[i]
            dt = max(0.0, t1 - t0)
            if c0 is True:
                closed_time += dt
        window = min(self.cfg.perclos_window_sec, span)
        perclos = (closed_time / max(1e-6, window)) if span > 0 else 0.0
        return perclos, span

    def _update_yawn_events(self, now_ts):
        cutoff = now_ts - self.cfg.yawn_count_window_sec
        while self.yawn_events and self.yawn_events[0] < cutoff:
            self.yawn_events.popleft()

    # Làm mượt bbox
    def _smooth_bbox(self, new_box, alpha=0.2):
        if new_box is None:
            return self._face_bbox_ema
        if self._face_bbox_ema is None:
            self._face_bbox_ema = new_box
        else:
            x1, y1, x2, y2 = self._face_bbox_ema
            nx1, ny1, nx2, ny2 = new_box
            sx1 = int((1 - alpha) * x1 + alpha * nx1)
            sy1 = int((1 - alpha) * y1 + alpha * ny1)
            sx2 = int((1 - alpha) * x2 + alpha * nx2)
            sy2 = int((1 - alpha) * y2 + alpha * ny2)
            self._face_bbox_ema = (sx1, sy1, sx2, sy2)
        return self._face_bbox_ema

    # Face/Eyes/Mouth
    def _detect_face(self, frame):
        h, w = frame.shape[:2]
        do_detect = (self.frame_idx % self.cfg.detect_every_n_frames == 0) or (self.last_face_bbox is None)

        if do_detect:
            small, r = _resize_keep_aspect(frame, self.cfg.yolo_downscale_width)
            res = self.face_model(small, conf=self.cfg.yolo_conf, iou=self.cfg.yolo_iou, verbose=False)
            best, best_area = None, 0
            for r0 in res:
                if not hasattr(r0, 'boxes') or r0.boxes is None:
                    continue
                xyxy = r0.boxes.xyxy
                if xyxy is None:
                    continue
                for box in xyxy.cpu().numpy():
                    x1s, y1s, x2s, y2s = box.astype(int)
                    if r != 1.0:
                        x1 = int(x1s / r); y1 = int(y1s / r)
                        x2 = int(x2s / r); y2 = int(y2s / r)
                    else:
                        x1, y1, x2, y2 = x1s, y1s, x2s, y2s
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    if area > best_area:
                        best_area = area
                        best = (x1, y1, x2, y2)
            self.last_face_bbox = best

        if self.last_face_bbox is None:
            return None, None

        box = self._smooth_bbox(self.last_face_bbox, alpha=0.2)
        if box is None:
            return None, None
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            self.last_face_bbox = None
            return None, None
        face_roi = frame[y1:y2, x1:x2]
        return face_roi, (x1, y1, x2, y2)

    def _prep_keras_input(self, crop_bgr, to_gray: bool):
        if to_gray:
            x = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            x = cv2.resize(x, (64, 64)).astype(np.float32) / 255.0
            x = np.expand_dims(x, axis=(0, -1))
        else:
            x = cv2.resize(crop_bgr, (64, 64)).astype(np.float32) / 255.0
            x = np.expand_dims(x, axis=0)
        return tf.convert_to_tensor(x)

    def _detect_eyes(self, face_roi):
        """Trả về: 'Opened' | 'Closed' | 'Unknown' (Unknown không cộng PERCLOS)"""
        status = "Closed"
        if face_roi is None or face_roi.size == 0:
            return status

        eye_region = face_roi[: face_roi.shape[0] // 2, :]
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray_eye, scaleFactor=1.1, minNeighbors=5)
        sel = _largest_box(eyes)
        if sel is not None:
            ex, ey, ew, eh = sel
            eye_crop = eye_region[ey:ey+eh, ex:ex+ew]

            # Model prob (eager)
            eye_inp = self._prep_keras_input(eye_crop, to_gray=self.cfg.gray_eye)
            prob_open = float(self.eye_model(eye_inp, training=False).numpy()[0][0])

            # EMA smoothing
            a = self.cfg.ema_alpha
            self.eye_prob_ema = prob_open if self.eye_prob_ema is None else (a*prob_open + (1-a)*self.eye_prob_ema)

            # -------- GAZE-DOWN GUARD (nhìn xuống bàn phím) --------
            g = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
            h2 = g.shape[0] // 2
            top_mean = float(np.mean(g[:h2, :])) + 1e-6
            bot_mean = float(np.mean(g[h2:, :])) + 1e-6
            rel_delta = (top_mean - bot_mean) / max(1e-6, top_mean)
            # <<< NEW: clamp theo delta & sàn sáng
            gaze_down_like = (top_mean >= self.cfg.gaze_min_brightness and
                              rel_delta >= self.cfg.gaze_down_min_delta and
                              bot_mean < self.cfg.gaze_down_ratio * top_mean)

            # Debounce "Closed" ≥ eye_closed_debounce_sec
            now = time.time()
            closed_now = (self.eye_prob_ema < self.cfg.eye_open_prob_thr)
            if closed_now and self._eye_closed_debounce_ts is None:
                self._eye_closed_debounce_ts = now
            if not closed_now:
                self._eye_closed_debounce_ts = None
            closed_debounced = (closed_now and
                                self._eye_closed_debounce_ts is not None and
                                (now - self._eye_closed_debounce_ts) >= self.cfg.eye_closed_debounce_sec)

            if gaze_down_like and closed_now and not closed_debounced:
                status = "Unknown"   # => không tính PERCLOS
            else:
                status = "Opened" if self.eye_prob_ema >= self.cfg.eye_open_prob_thr \
                         else ("Closed" if closed_debounced else "Unknown")
        return status





    def _detect_mouth(self, face_roi):
        """Return (status, ratio, prob_yawn, open_dur)."""
        status = "Normal"
        ratio = 0.0
        prob_yawn = 0.0
        open_dur = 0.0
        now = time.time()

        if face_roi is None or face_roi.size == 0:
            self.mouth_open_start_ts = None
            self.mouth_yawn_start_ts = None
            return status, ratio, prob_yawn, open_dur

        mouth_region = face_roi[face_roi.shape[0] // 3 :, :]
        gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        mouths = self.mouth_cascade.detectMultiScale(gray_mouth, scaleFactor=1.1, minNeighbors=5)

        sel = _largest_box(mouths)
        if sel is None:
            self._mouth_miss_streak += 1
            if self._mouth_miss_streak >= 5:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray2 = clahe.apply(gray_mouth)
                mouths2 = self.mouth_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=3)
                sel = _largest_box(mouths2)
        else:
            self._mouth_miss_streak = 0

        if sel is not None:
            mx, my, mw, mh = sel
            crop = mouth_region[my:my + mh, mx:mx + mw]

            mouth_inp = self._prep_keras_input(crop, to_gray=self.cfg.gray_mouth)
            prob_yawn = float(self.mouth_model(mouth_inp, training=False).numpy()[0][0])

            ratio = mh / max(1.0, mw)
            mouth_open_now = (ratio >= self.cfg.mouth_ratio_thr) and (prob_yawn >= self.cfg.mouth_prob_thr)

            if mouth_open_now:
                if self.mouth_open_start_ts is None:
                    self.mouth_open_start_ts = now
                open_dur = now - self.mouth_open_start_ts
                status = "Opening"

                if open_dur > self.cfg.yawn_sure_open_sec:
                    status = "Yawn"
                    if self.mouth_yawn_start_ts is None:
                        self.mouth_yawn_start_ts = now
            else:
                if self.mouth_open_start_ts is not None:
                    open_dur = now - self.mouth_open_start_ts
                    if open_dur > self.cfg.yawn_sure_open_sec and self.mouth_yawn_start_ts is not None:
                        status = "Yawn"
                    else:
                        status = "Normal"
                else:
                    status = "Normal"

                self.mouth_open_start_ts = None
                self.mouth_yawn_start_ts = None
        else:
            self.mouth_open_start_ts = None
            self.mouth_yawn_start_ts = None
            status = "Normal"

        return status, ratio, prob_yawn, open_dur

    def _analyze_mouth_temporal(self, now, ratio_thr):
        WIN = float(self.cfg.mouth_temporal_win)
        xs = [(t, r) for (t, r) in self.mouth_ratio_hist if t >= now - WIN]
        if len(xs) < 3:
            return {
                "yawn_like": False,
                "speech_like": False,
                "speech_strong": False,
                "plateau_max": 0.0,
                "zcr": 0.0,
                "peaks": 0,
                "area": 0.0,
                "above_frac": 0.0,
            }

        ts = np.array([t for (t, _) in xs], dtype=np.float32)
        rs = np.array([r for (_, r) in xs], dtype=np.float32)

        above = rs >= ratio_thr
        above_frac = float(np.mean(above))

        plateau_max = 0.0
        area = 0.0
        if above.any():
            area = float(np.trapz(np.maximum(0.0, rs - ratio_thr), ts))
            start = None
            frame_dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.0
            for i, v in enumerate(above):
                if v and start is None:
                    start = ts[i]
                if (not v or i == len(above) - 1) and start is not None:
                    end = ts[i] + (frame_dt if (v and i == len(above) - 1) else 0.0)
                    plateau_max = max(plateau_max, end - start)
                    start = None

        dr = np.diff(rs)
        signs = np.sign(dr + 1e-8)
        zc = np.where(signs[:-1] * signs[1:] < 0)[0]
        duration = max(1e-6, ts[-1] - ts[0])
        zcr = float(len(zc) / duration)

        peaks = 0
        margin = 0.02
        prom = 0.04
        for i in range(1, len(rs) - 1):
            if rs[i] >= ratio_thr + margin and rs[i] > rs[i - 1] and rs[i] > rs[i + 1]:
                local_prom = rs[i] - min(rs[i - 1], rs[i + 1])
                if local_prom >= prom:
                    peaks += 1

        yawn_like = (
            plateau_max >= self.cfg.yawn_plateau_min_s
            and zcr <= self.cfg.yawn_zcr_max_hz
            and area >= self.cfg.yawn_area_min
        )

        speech_like = (peaks >= 3) or (plateau_max <= 0.35) or (zcr >= 2.0)
        speech_strong = (
            peaks >= self.cfg.speech_peaks_min
            and zcr >= self.cfg.speech_zcr_min_hz
            and above_frac < 0.6
        )

        return {
            "yawn_like": bool(yawn_like),
            "speech_like": bool(speech_like),
            "speech_strong": bool(speech_strong),
            "plateau_max": float(plateau_max),
            "zcr": float(zcr),
            "peaks": int(peaks),
            "area": float(area),
            "above_frac": float(above_frac),
        }

    def _update_alert(self, should_alarm, now):
        if not self.alert_on:
            if should_alarm:
                self.alert_cross_ts = self.alert_cross_ts or now
                if (now - self.alert_cross_ts) >= self.cfg.on_min_sec:
                    self.alert_on = True
                    self.alert_cross_ts = None
            else:
                self.alert_cross_ts = None
        else:
            if not should_alarm:
                self.alert_cross_ts = self.alert_cross_ts or now
                if (now - self.alert_cross_ts) >= self.cfg.off_min_sec:
                    self.alert_on = False
                    self.alert_cross_ts = None
            else:
                self.alert_cross_ts = None
        return self.alert_on

    def detect_drowsiness(self, frame):
        now = time.time()
        self.frame_idx += 1

        face_roi, bbox = self._detect_face(frame)

        # No-Update Mode khi mất mặt
        if face_roi is None:
            perclos, _ = self._update_perclos(now, None)
            self._update_yawn_events(now)
            yawn_count = len(self.yawn_events)
            status = "DROWSY" if self.alert_on else "AWAKE"
            cv2.putText(frame, "Face: missing", (10, 30), self.font, 0.7, (0, 255, 255), 2)
            self._draw_overlay(frame, bbox=None, eye_status="-", mouth_status="-",
                               perclos=perclos, yawn_count=yawn_count, status=status,
                               mouth_dbg=None, open_dur=0.0)
            self._update_fps()
            return frame

        (x1, y1, x2, y2) = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['face'], 2)

        # Eyes
        eye_status = self._detect_eyes(face_roi)

        # Mouth
        mouth_status, mouth_ratio, prob_yawn, open_dur = self._detect_mouth(face_roi)
        self.mouth_ratio_hist.append((now, float(mouth_ratio)))
        self.mouth_prob_hist.append((now, float(prob_yawn)))
        mt = self._analyze_mouth_temporal(now, ratio_thr=self.cfg.mouth_ratio_thr)

        # Eye state tracking
        if eye_status == "Closed":
            if self.eye_closed_start_ts is None:
                self.eye_closed_start_ts = now
            self.eye_closed_frames += 1
        else:
            self.eye_closed_start_ts = None
            self.eye_closed_frames = 0

        # PERCLOS dùng None khi Unknown
        is_closed_flag = True if eye_status == "Closed" else (False if eye_status == "Opened" else None)
        perclos, span = self._update_perclos(now, is_closed_flag)

        perclos_ready = (span >= self.cfg.perclos_warmup_sec)
        closed_short = (self.eye_closed_start_ts is not None and
                        (now - self.eye_closed_start_ts) >= self.cfg.eye_yawn_aid_min_sec)
        perclos_aid = (perclos_ready and perclos >= self.cfg.perclos_yawn_aid_thr)

        strong_yawn = (open_dur >= self.cfg.yawn_sure_open_sec) or mt["yawn_like"]
        probable_yawn = (open_dur >= self.cfg.yawn_prob_open_sec) and \
                        (closed_short or perclos_aid) and \
                        (mt["above_frac"] >= 0.6) and (not mt["speech_strong"])

        if mt["speech_strong"]:
            if self.talking_since_ts is None:
                self.talking_since_ts = now
        else:
            self.talking_since_ts = None
        strong_talking = (self.talking_since_ts is not None) and \
                         ((now - self.talking_since_ts) >= self.cfg.talking_min_hold_sec)

        if strong_yawn or probable_yawn:
            mouth_status = "Yawn"
            self.last_yawn_set_ts = now
        elif strong_talking and (now - self.last_yawn_set_ts) >= self.cfg.yawn_lock_sec:
            mouth_status = "Opening"

        # Yawn events
        if mouth_status == "Yawn" and not self.yawn_active:
            self.yawn_active = True
            self.yawn_start_ts = now
        elif mouth_status != "Yawn" and self.yawn_active:
            dur = now - (self.yawn_start_ts or now)
            self.yawn_active = False
            self.yawn_start_ts = None
            if dur >= self.cfg.yawn_min_duration_sec:
                self.yawn_events.append(now)

        self._update_yawn_events(now)
        yawn_count = len(self.yawn_events)

        # Alarm logic
        long_eye_close = (self.eye_closed_start_ts is not None and
                          (now - self.eye_closed_start_ts) >= self.cfg.eye_continuous_closed_sec)
        perclos_alarm = (perclos_ready and perclos >= self.cfg.perclos_thr)
        yawn_alarm = (yawn_count >= self.cfg.yawn_count_thr) or \
                     (self.yawn_active and self.yawn_start_ts is not None and
                      (now - self.yawn_start_ts) >= self.cfg.yawn_min_duration_sec)

        drowsy_raw = long_eye_close or perclos_alarm or yawn_alarm

        # RECOVERY LOGIC TỐI ƯU 
        # 1) Eye: Opened
        # 2) Mouth: không Yawn
        # 3) PERCLOS: < threshold HOẶC chưa warmup
        # 4) Không có ngáp mới trong cooldown period

        eyes_ok = (eye_status == "Opened")
        mouth_ok = (mouth_status != "Yawn")

        # PERCLOS check: nếu chưa warmup thì coi như OK
        perclos_ok = (not perclos_ready) or (perclos < self.cfg.recovery_perclos_thr)

        # Yawn cooldown check
        if self.yawn_events:
            last_yawn_age = now - self.yawn_events[-1]
        else:
            last_yawn_age = float("inf")
        no_recent_yawn = (not self.yawn_active) and (last_yawn_age >= self.cfg.recovery_yawn_cooldown)

        recovery_ready = eyes_ok and mouth_ok and perclos_ok and no_recent_yawn

        if recovery_ready:
            # Bắt đầu đếm thời gian recovery
            if self.recovery_start_ts is None:
                self.recovery_start_ts = now
                logging.debug(
                    "Recovery started | eye=%s mouth=%s perclos=%.3f(ok=%s) yawn_age=%.1fs",
                    eye_status, mouth_status, perclos, perclos_ok, last_yawn_age
                )

            recovery_elapsed = now - self.recovery_start_ts

            # Force AWAKE nếu đủ điều kiện
            if self.alert_on and recovery_elapsed >= self.cfg.recovery_duration_sec:
                self.alert_on = False
                self.alert_cross_ts = None
                drowsy_raw = False
                logging.info(
                    "✓ RECOVERY SUCCESS | %.1fs | eye=Opened mouth=%s perclos=%.3f yawn_age=%.1fs -> AWAKE",
                    recovery_elapsed, mouth_status, perclos, last_yawn_age
                )
        else:
            # Reset nếu điều kiện không còn
            if self.recovery_start_ts is not None:
                lost_elapsed = now - self.recovery_start_ts
                logging.debug(
                    "Recovery lost after %.1fs | eye=%s mouth=%s perclos_ok=%s yawn_ok=%s",
                    lost_elapsed, eye_status, mouth_status, perclos_ok, no_recent_yawn
                )
            self.recovery_start_ts = None
        #KẾT THÚC RECOVERY 

        drowsy = self._update_alert(drowsy_raw, now)
        status = "DROWSY" if drowsy else "AWAKE"

        mouth_dbg = {
            "ratio": mouth_ratio, "prob": prob_yawn,
            "plateau": mt["plateau_max"], "zcr": mt["zcr"], "peaks": mt["peaks"],
            "above": mt["above_frac"], "speech_strong": int(mt["speech_strong"])
        }
        self._draw_overlay(frame, bbox=bbox, eye_status=eye_status, mouth_status=mouth_status,
                           perclos=perclos, yawn_count=yawn_count, status=status,
                           mouth_dbg=mouth_dbg, open_dur=open_dur)

        # CSV row
        if self.save_csv_path:
            with open(self.save_csv_path, "a", encoding="utf-8") as f:
                f.write(f"{now:.3f},{perclos:.4f},{eye_status},{mouth_status},{mouth_ratio:.4f},{prob_yawn:.4f},"
                        f"{mt['plateau_max']:.3f},{mt['zcr']:.3f},{mt['peaks']},{mt['above_frac']:.3f},{status}\n")

        self._update_fps()
        return frame

    def _draw_overlay(self, frame, bbox, eye_status, mouth_status, perclos, yawn_count, status, mouth_dbg=None, open_dur=0.0):
        h, w = frame.shape[:2]
        if bbox is not None:
            x1, y1, x2, y2 = bbox
        else:
            x1, y1, x2, y2 = 20, 40, 20, 40

        status_color = (0, 0, 255) if status == "DROWSY" else (0, 255, 0)
        cv2.putText(frame, f"Eyes: {eye_status}", (x1, max(20, y1 - 30)), self.font, 0.6, self.colors['eye'], 2)
        if mouth_dbg:
            cv2.putText(frame, f"Mouth: {mouth_status} open={open_dur:.1f}s r={mouth_dbg['ratio']:.2f} p={mouth_dbg['prob']:.2f}",
                        (x1, max(20, y1 - 60)), self.font, 0.55, self.colors['mouth'], 2)
            cv2.putText(frame, f"plateau={mouth_dbg['plateau']:.2f}s zcr={mouth_dbg['zcr']:.1f} peaks={mouth_dbg['peaks']} above={mouth_dbg['above']:.2f} talkS={mouth_dbg['speech_strong']}",
                        (x1, min(h - 35, y2 + 45)), self.font, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(frame, f"Mouth: {mouth_status}", (x1, max(20, y1 - 60)), self.font, 0.6, self.colors['mouth'], 2)

        cv2.putText(frame, f"Status: {status}", (x1, max(20, y1 - 90)), self.font, 0.7, status_color, 2)
        cv2.putText(frame, f"PERCLOS: {perclos:.2f} | yawns@60s: {yawn_count}", (x1, min(h - 10, y2 + 20)),
                    self.font, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self._fps:.1f}", (10, h - 10), self.font, 0.6, (255, 255, 255), 2)

    def _update_fps(self):
        self._fps_cnt += 1
        now = time.time()
        if now - self._fps_ts >= 1.0:
            self._fps = self._fps_cnt / (now - self._fps_ts)
            self._fps_cnt = 0
            self._fps_ts = now
        if now - self._fps_log_window_start >= 10.0:
            logging.info("Avg FPS (last 10s): %.2f", self._fps)
            self._fps_log_window_start = now


def setup_logger(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.getenv("MODEL_DIR", "models"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=0, help="Resize width (0=keep)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--detect-interval", type=int, default=None,
                        help="Override detect_every_n_frames (e.g., 8 or 10)")
    parser.add_argument("--yolo-downscale", type=int, default=None,
                        help="Downscale width for YOLO (e.g., 640). 0 = disable")
    parser.add_argument("--gray-eye", action="store_true", help="Convert eye crop to grayscale before model")
    parser.add_argument("--gray-mouth", action="store_true", help="Convert mouth crop to grayscale before model")
    # Guard
    parser.add_argument("--gaze-down-ratio", type=float, default=None,
                        help="Lower-half mean < ratio * upper-half mean => gaze down (default 0.85)")
    parser.add_argument("--eye-closed-debounce", type=float, default=None,
                        help="Seconds to confirm eye-closed (default 0.30)")
    parser.add_argument("--gaze-min-delta", type=float, default=None,
                        help="Min relative delta (top-bottom)/top to consider gaze-down, default 0.08)")
    parser.add_argument("--gaze-min-bright", type=float, default=None,
                        help="Min top brightness to evaluate gaze-down, default 20.0)")
    # YOLO thresholds
    parser.add_argument("--yolo-conf", type=float, default=None, help="YOLO confidence threshold (default 0.50)")
    parser.add_argument("--yolo-iou", type=float, default=None, help="YOLO NMS IoU threshold (default 0.45)")
    # CSV
    parser.add_argument("--save-csv", type=str, default=None, help="Path to save time-series CSV")
    # Recovery CLI arguments
    parser.add_argument("--recovery-duration", type=float, default=None,
                        help="Recovery time in seconds (default 3.0)")
    parser.add_argument("--recovery-perclos", type=float, default=None,
                        help="Max PERCLOS for recovery (default 0.15)")
    parser.add_argument("--recovery-yawn-cooldown", type=float, default=None,
                        help="No yawn in last N seconds (default 3.0)")

    args = parser.parse_args()

    setup_logger(args.log_level)
    cv2.setUseOptimized(True)

    cfg = DrowsyConfig(
        model_dir=args.model_dir,
        gray_eye=args.gray_eye,
        gray_mouth=args.gray_mouth
    )
    if args.detect_interval is not None:
        cfg.detect_every_n_frames = max(1, int(args.detect_interval))
    if args.yolo_downscale is not None:
        cfg.yolo_downscale_width = max(0, int(args.yolo_downscale))
    if args.gaze_down_ratio is not None:
        cfg.gaze_down_ratio = float(args.gaze_down_ratio)
    if args.eye_closed_debounce is not None:
        cfg.eye_closed_debounce_sec = float(args.eye_closed_debounce)
    if args.gaze_min_delta is not None:
        cfg.gaze_down_min_delta = float(args.gaze_min_delta)
    if args.gaze_min_bright is not None:
        cfg.gaze_min_brightness = float(args.gaze_min_bright)
    if args.yolo_conf is not None:
        cfg.yolo_conf = float(args.yolo_conf)
    if args.yolo_iou is not None:
        cfg.yolo_iou = float(args.yolo_iou)

    # Apply recovery settings from CLI
    if args.recovery_duration is not None:
        cfg.recovery_duration_sec = float(args.recovery_duration)
    if args.recovery_perclos is not None:
        cfg.recovery_perclos_thr = float(args.recovery_perclos)
    if args.recovery_yawn_cooldown is not None:
        cfg.recovery_yawn_cooldown = float(args.recovery_yawn_cooldown)

    logging.info(
        "Recovery config: duration=%.1fs perclos<%.2f yawn_cooldown=%.1fs",
        cfg.recovery_duration_sec, cfg.recovery_perclos_thr, cfg.recovery_yawn_cooldown
    )

    # Log versions & device
    gpus = tf.config.list_physical_devices('GPU')
    logging.info("Versions: opencv=%s | tensorflow=%s | ultralytics=%s",
                 cv2.__version__, tf.__version__, getattr(ulx, "__version__", "unknown"))
    logging.info("Device: %s", "GPU" if gpus else "CPU")

    try:
        detector = DrowsinessDetector(cfg, save_csv_path=args.save_csv)
        cap = cv2.VideoCapture(args.camera)
        #cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not cap.isOpened():
            raise ValueError("Could not open camera.")
        logging.info("Press 'q' to quit.")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_cam = cap.get(cv2.CAP_PROP_FPS) or 0.0
        logging.info("Camera opened: %dx%d @ %.1f FPS (reported)", w, h, fps_cam)

        while True:
            ok, frame = cap.read()
            if not ok:
                logging.warning("Frame read error")
                break
            if args.width and frame.shape[1] != args.width:
                r = args.width / frame.shape[1]
                frame = cv2.resize(frame, (args.width, int(frame.shape[0] * r)))
            frame = cv2.flip(frame, 1)
            out = detector.detect_drowsiness(frame)
            cv2.imshow("Drowsiness Detection (precedence-fixed)", out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    except Exception as e:
        logging.exception("Unhandled error: %s", e)
    finally:
        try:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        logging.info("Shutdown.")


if __name__ == "__main__":
    main()

