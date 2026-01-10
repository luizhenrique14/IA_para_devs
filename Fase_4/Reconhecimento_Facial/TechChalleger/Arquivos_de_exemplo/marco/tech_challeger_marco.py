"""
Tech Challenge (IADT) – Fase 4
1) Reconhecimento facial (encoding + comparação + ID)
2) Análise de emoções (DeepFace)
3) Detecção/categorização de atividades por intensidade de movimento
4) Detecção de anomalias de movimento (picos fora do padrão)
5) Geração de vídeo anotado + events.jsonl + report.md

Dependências:
pip install opencv-python numpy mediapipe face_recognition deepface

Observações:
- DeepFace é pesado. Para performance, este script calcula emoção a cada N frames (EMOTION_EVERY_N).
- "Atividade" aqui é inferida por intensidade de movimento (energia entre frames).
"""

import os
import json
import argparse
from collections import defaultdict, Counter

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import face_recognition
from deepface import DeepFace


# -----------------------------
# Utilitários
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def bbox_xyxy_from_mp(rel_box, w, h):
    """MediaPipe retorna bbox relativo; converte para xyxy em pixels."""
    x = int(rel_box.xmin * w)
    y = int(rel_box.ymin * h)
    bw = int(rel_box.width * w)
    bh = int(rel_box.height * h)
    x1 = clamp(x, 0, w - 1)
    y1 = clamp(y, 0, h - 1)
    x2 = clamp(x + bw, 0, w - 1)
    y2 = clamp(y + bh, 0, h - 1)
    return x1, y1, x2, y2

def iou(a, b):
    """a,b: (x1,y1,x2,y2)"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def timestamp_s(frame_idx, fps):
    return frame_idx / fps if fps > 0 else 0.0


# -----------------------------
# Tracker simples (por IoU)
# -----------------------------

class SimpleTracker:
    """
    Tracker simples: associa bbox -> track_id pelo maior IoU com tracks anteriores.
    Serve para manter IDs consistentes no vídeo anotado e no relatório.
    """
    def __init__(self, iou_threshold=0.3, max_missed=15):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks = {}  # track_id -> {"bbox":..., "missed": int}

    def update(self, bboxes):
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["missed"] += 1

        assignments = []
        used_tracks = set()

        for bbox in bboxes:
            best_tid = None
            best_iou = 0.0
            for tid, st in self.tracks.items():
                if tid in used_tracks:
                    continue
                score = iou(bbox, st["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_tid = tid

            if best_tid is not None and best_iou >= self.iou_threshold:
                self.tracks[best_tid]["bbox"] = bbox
                self.tracks[best_tid]["missed"] = 0
                used_tracks.add(best_tid)
                assignments.append((best_tid, bbox))
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": bbox, "missed": 0}
                used_tracks.add(tid)
                assignments.append((tid, bbox))

        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["missed"] > self.max_missed:
                del self.tracks[tid]

        return assignments


# -----------------------------
# Atividade e anomalia (movimento)
# -----------------------------

def motion_energy(prev_gray, gray):
    """Energia média de movimento via absdiff."""
    diff = cv2.absdiff(gray, prev_gray)
    return float(np.mean(diff))

def categorize_activity(energy, idle_thr=2.0, moving_thr=8.0):
    """
    Activity classification based on motion intensity:
    - idle: low motion intensity
    - moving: moderate motion intensity
    - abrupt_movement: high motion intensity (peaks)
    """
    if energy < idle_thr:
        return "idle"
    if energy < moving_thr:
        return "moving"
    return "abrupt_movement"

class RollingAnomaly:
    """
    Anomaly detector:
    flags anomaly when energy > mean + k*std using a rolling window.
    """
    def __init__(self, window=60, k=3.0):
        self.window = window
        self.k = k
        self.values = []

    def update(self, x):
        self.values.append(x)
        if len(self.values) > self.window:
            self.values.pop(0)

        if len(self.values) < max(10, self.window // 3):
            return False, None, None

        arr = np.array(self.values, dtype=np.float32)
        mu = float(arr.mean())
        sd = float(arr.std() + 1e-6)
        thr = mu + self.k * sd
        return (x > thr), mu, thr


# -----------------------------
# Emoção (DeepFace)
# -----------------------------

def safe_deepface_emotion(face_bgr):
    """
    Retorna (dominant_emotion, score 0..1) ou (None, None).
    enforce_detection=False evita crash quando o crop é ruim.
    """
    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        res = DeepFace.analyze(
            img_path=face_rgb,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv"
        )
        if isinstance(res, list) and len(res) > 0:
            res = res[0]
        emo = res.get("dominant_emotion")
        scores = res.get("emotion", {})
        score = float(scores.get(emo, 0.0)) if emo else 0.0
        if score > 1.0:
            score = score / 100.0
        return emo, score
    except Exception:
        return None, None


# -----------------------------
# Face recognition (face_recognition)
# -----------------------------

def face_encoding_from_crop(face_bgr):
    """
    Obtém encoding do rosto a partir do crop.
    Usa face_locations no próprio crop para reduzir erro.
    """
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model="hog")
    if not locs:
        return None
    encs = face_recognition.face_encodings(rgb, known_face_locations=locs)
    return encs[0] if encs else None

def match_or_register(encoding, known_encodings, known_ids, tol=0.50):
    """
    Se bater com alguém conhecido -> retorna face_id existente.
    Se não -> registra novo face_id.
    """
    if encoding is None:
        return None

    if len(known_encodings) == 0:
        new_id = 1
        known_encodings.append(encoding)
        known_ids.append(new_id)
        return new_id

    dists = face_recognition.face_distance(known_encodings, encoding)
    best_idx = int(np.argmin(dists))
    if float(dists[best_idx]) <= tol:
        return known_ids[best_idx]

    new_id = max(known_ids) + 1
    known_encodings.append(encoding)
    known_ids.append(new_id)
    return new_id


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Caminho do vídeo de entrada")
    parser.add_argument("--output-dir", default="./out", help="Pasta de saída")
    parser.add_argument("--frame-skip", type=int, default=1, help="Processa 1 a cada N frames")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = sem limite")
    parser.add_argument("--write-annotated-video", action="store_true", help="Gera annotated.mp4")
    parser.add_argument("--emotion-every-n", type=int, default=10, help="Emoção a cada N frames (por face)")
    parser.add_argument("--anomaly-k", type=float, default=3.0, help="Limiar k para anomalia (mean + k*std)")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    events_path = os.path.join(args.output_dir, "events.jsonl")
    report_path = os.path.join(args.output_dir, "report.md")
    annotated_path = os.path.join(args.output_dir, "annotated.mp4")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {args.input}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (nframes / fps) if fps > 0 else 0.0

    writer = None
    if args.write_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_path, fourcc, fps if fps > 0 else 25.0, (w, h))

    # Nova API do MediaPipe para detecção facial
    model_path = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )
    face_detector = vision.FaceDetector.create_from_options(options)

    tracker = SimpleTracker(iou_threshold=0.3, max_missed=15)
    anomaly = RollingAnomaly(window=60, k=args.anomaly_k)

    known_encodings = []
    known_face_ids = []

    last_emotion_by_face = {}  # face_id -> (emo, score)
    emotion_counters = defaultdict(Counter)
    activity_counter = Counter()
    anomaly_timestamps = []

    prev_gray = None
    processed_frames = 0

    with open(events_path, "w", encoding="utf-8") as f_events:
        frame_idx = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if args.frame_skip > 1 and (frame_idx % args.frame_skip != 0):
                if writer is not None:
                    writer.write(frame)
                continue

            if args.max_frames and processed_frames >= args.max_frames:
                break

            processed_frames += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            act_label = None
            energy = None
            is_anom = False
            mu = thr = None
            if prev_gray is not None:
                energy = motion_energy(prev_gray, gray)
                act_label = categorize_activity(energy)
                activity_counter[act_label] += 1

                is_anom, mu, thr = anomaly.update(energy)
                if is_anom:
                    anomaly_timestamps.append(timestamp_s(frame_idx, fps))

            prev_gray = gray

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            det = face_detector.detect(mp_image)

            bboxes = []
            if det.detections:
                for d in det.detections:
                    bbox = d.bounding_box
                    x1 = clamp(bbox.origin_x, 0, w - 1)
                    y1 = clamp(bbox.origin_y, 0, h - 1)
                    x2 = clamp(bbox.origin_x + bbox.width, 0, w - 1)
                    y2 = clamp(bbox.origin_y + bbox.height, 0, h - 1)
                    if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                        bboxes.append((x1, y1, x2, y2))

            tracks = tracker.update(bboxes)

            faces_out = []
            for tid, (x1, y1, x2, y2) in tracks:
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                enc = face_encoding_from_crop(face_crop)
                face_id = match_or_register(enc, known_encodings, known_face_ids, tol=0.50)

                emo, emo_score = None, None
                if face_id is not None:
                    if (processed_frames % args.emotion_every_n) == 0 or face_id not in last_emotion_by_face:
                        emo, emo_score = safe_deepface_emotion(face_crop)
                        if emo:
                            last_emotion_by_face[face_id] = (emo, emo_score)
                            emotion_counters[face_id][emo] += 1
                    else:
                        emo, emo_score = last_emotion_by_face.get(face_id, (None, None))

                faces_out.append({
                    "track_id": int(tid),
                    "face_id": int(face_id) if face_id is not None else None,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "emotion": emo,
                    "emotion_score": float(emo_score) if emo_score is not None else None
                })

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"T{tid}"
                if face_id is not None:
                    label += f" F{face_id}"
                if emo:
                    label += f" {emo}"
                cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            y0 = 20
            if act_label is not None:
                cv2.putText(frame, f"Activity: {act_label}", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                y0 += 22
            if energy is not None:
                cv2.putText(frame, f"Motion energy: {energy:.2f}", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                y0 += 22
            if is_anom:
                cv2.putText(frame, "ANOMALY!", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3, cv2.LINE_AA)

            event = {
                "frame_idx": int(frame_idx),
                "timestamp_s": float(timestamp_s(frame_idx, fps)),
                "faces": faces_out,
                "activity": {"label": act_label, "score": None},
                "motion_energy": energy,
                "is_anomaly": bool(is_anom),
                "anomaly_baseline_mean": mu,
                "anomaly_threshold": thr
            }
            f_events.write(json.dumps(event, ensure_ascii=False) + "\n")

            if writer is not None:
                writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    global_emotions = Counter()
    for fid, cnt in emotion_counters.items():
        global_emotions.update(cnt)

    dominant_by_face = {}
    for fid, cnt in emotion_counters.items():
        dominant_by_face[fid] = cnt.most_common(1)[0][0] if cnt else None

    total_anoms = len(anomaly_timestamps)

    lines = []
    lines.append("# Report – Tech Challenge (Fase 4)\n")
    lines.append("## Metadados do vídeo\n")
    lines.append(f"- Resolução: {w}x{h}\n")
    lines.append(f"- FPS: {fps:.2f}\n")
    lines.append(f"- Frames (estimado): {nframes}\n")
    lines.append(f"- Duração (s): {duration:.2f}\n")

    lines.append("\n## Processamento\n")
    lines.append(f"- Frames analisados: {processed_frames}\n")
    lines.append(f"- Frame-skip: {args.frame_skip}\n")

    lines.append("\n## Anomalias\n")
    lines.append(f"- Número de anomalias detectadas: **{total_anoms}**\n")
    if total_anoms:
        lines.append("- Timestamps (s) das anomalias (primeiras 30):\n")
        sample = anomaly_timestamps[:30]
        lines.append("  - " + ", ".join([f"{t:.2f}" for t in sample]) + "\n")

    lines.append("\n## Detecção de atividades\n")
    total_act = sum(activity_counter.values()) or 1
    for k, v in activity_counter.most_common():
        lines.append(f"- {k}: {v} ({100.0*v/total_act:.1f}%)\n")

    lines.append("\n## Emoções (global)\n")
    if global_emotions:
        total_emo = sum(global_emotions.values()) or 1
        for k, v in global_emotions.most_common():
            lines.append(f"- {k}: {v} ({100.0*v/total_emo:.1f}%)\n")
    else:
        lines.append("- Nenhuma emoção consolidada (talvez poucos rostos detectados ou DeepFace falhou nos crops)\n")

    lines.append("\n## Emoção dominante por pessoa (face_id)\n")
    if dominant_by_face:
        for fid in sorted(dominant_by_face.keys()):
            lines.append(f"- Face {fid}: {dominant_by_face[fid]}\n")
    else:
        lines.append("- Nenhuma face_id registrada.\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print("Concluído.")
    print(f"- events: {events_path}")
    print(f"- report: {report_path}")
    if args.write_annotated_video:
        print(f"- annotated video: {annotated_path}")


if __name__ == "__main__":
    main()