"""
Tech Challenge (IADT) – Fase 4
Aplicação completa de análise de vídeo com:
1) Detecção facial (OpenCV Haar Cascade)
2) Análise de expressões emocionais (DeepFace)
3) Detecção de atividades e anomalias por movimento
4) Geração de relatório final

Dependências:
pip install opencv-python numpy deepface tqdm
"""

import os
import json
import argparse
from collections import defaultdict, Counter

import cv2
import numpy as np
from deepface import DeepFace
from tqdm import tqdm


# -----------------------------
# Utilitários
# -----------------------------

def ensure_dir(path: str) -> None:
    """Cria diretório se não existir."""
    os.makedirs(path, exist_ok=True)


def clamp(v, lo, hi):
    """Limita valor entre min e max."""
    return max(lo, min(hi, v))


def timestamp_s(frame_idx, fps):
    """Converte frame para timestamp em segundos."""
    return frame_idx / fps if fps > 0 else 0.0


def iou(a, b):
    """Calcula IoU entre duas bounding boxes (x1,y1,x2,y2)."""
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


# -----------------------------
# Tracker simples (por IoU)
# -----------------------------

class SimpleTracker:
    """Tracker que associa bbox -> track_id pelo maior IoU."""
    def __init__(self, iou_threshold=0.3, max_missed=15):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks = {}

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
# Atividade e Anomalia
# -----------------------------

def motion_energy(prev_gray, gray):
    """Calcula energia média de movimento via absdiff."""
    diff = cv2.absdiff(gray, prev_gray)
    return float(np.mean(diff))


# Dicionário de tradução para português BR
EMOTION_TRANSLATION = {
    "angry": "raiva",
    "disgust": "nojo",
    "fear": "medo",
    "happy": "feliz",
    "sad": "tristeza",
    "surprise": "surpresa",
    "neutral": "neutro",
    "neutro": "neutro"
}

ACTIVITY_TRANSLATION = {
    "idle": "parado",
    "moving": "movimento",
    "abrupt_movement": "movimento_brusco"
}


def categorize_activity(energy, idle_thr=2.0, moving_thr=8.0):
    """Classifica atividade baseada na intensidade de movimento."""
    if energy < idle_thr:
        return "idle"
    if energy < moving_thr:
        return "moving"
    return "abrupt_movement"


class RollingAnomaly:
    """Detector de anomalias usando janela rolante."""
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
    """Retorna (dominant_emotion, score) ou ('neutro', 0.5) como fallback."""
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
        
        if not emo or score < 0.15:
            emo = "neutro"
            score = 0.5
        
        return emo, score
    except Exception:
        return "neutro", 0.5


# -----------------------------
# Processamento de Vídeo
# -----------------------------

def process_video(video_path, output_dir, frame_skip=1, emotion_every_n=10, 
                  anomaly_k=3.0, write_video=False):
    """Processa um único vídeo e gera relatório."""
    print(f"\n>>> Processando: {os.path.basename(video_path)}")
    
    ensure_dir(output_dir)
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    events_path = os.path.join(output_dir, f"{base_name}_events.jsonl")
    report_path = os.path.join(output_dir, f"{base_name}_report.txt")
    annotated_path = os.path.join(output_dir, f"{base_name}_annotated.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (nframes / fps) if fps > 0 else 0.0

    writer = None
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_path, fourcc, fps if fps > 0 else 25.0, (w, h))

    # Inicializar detector de rosto com OpenCV Haar Cascade
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Não foi possível carregar o classificador Haar Cascade")

    tracker = SimpleTracker(iou_threshold=0.3, max_missed=15)
    anomaly = RollingAnomaly(window=60, k=anomaly_k)

    last_emotion_by_track = {}
    emotion_counters = defaultdict(Counter)
    activity_counter = Counter()
    anomaly_timestamps = []
    dominant_by_track = {}
    total_faces_detected = 0
    prev_gray = None
    processed_frames = 0

    with open(events_path, "w", encoding="utf-8") as f_events:
        frame_idx = -1
        pbar = tqdm(total=nframes, desc="Frames", leave=False)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            pbar.update(1)

            if frame_skip > 1 and (frame_idx % frame_skip != 0):
                if writer is not None:
                    writer.write(frame)
                continue

            processed_frames += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            act_label = None
            energy = None
            is_anom = False
            
            if prev_gray is not None:
                energy = motion_energy(prev_gray, gray)
                act_label = categorize_activity(energy)
                activity_counter[act_label] += 1

                is_anom, mu, thr = anomaly.update(energy)
                if is_anom:
                    anomaly_timestamps.append(timestamp_s(frame_idx, fps))

            prev_gray = gray

            # Detectar rostos com Haar Cascade
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            bboxes = []
            for (x1, y1, fw, fh) in faces:
                x2 = x1 + fw
                y2 = y1 + fh
                if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                    bboxes.append((x1, y1, x2, y2))

            tracks = tracker.update(bboxes)
            total_faces_detected += len(tracks)

            faces_out = []
            for tid, (x1, y1, x2, y2) in tracks:
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                emo, emo_score = None, None
                needs_calc = (processed_frames % emotion_every_n == 0) or (tid not in last_emotion_by_track)
                if needs_calc:
                    emo, emo_score = safe_deepface_emotion(face_crop)
                    if emo:
                        last_emotion_by_track[tid] = (emo, emo_score)
                        emotion_counters[tid][emo] += 1
                        dominant_by_track[tid] = emotion_counters[tid].most_common(1)[0][0]
                else:
                    emo, emo_score = last_emotion_by_track.get(tid, (None, None))

                if emo is None:
                    emo = "neutro"
                    emo_score = 0.5

                # Traduzir emoção para português
                emo_translated = EMOTION_TRANSLATION.get(emo.lower(), emo)
                
                faces_out.append({
                    "track_id": int(tid),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "emotion": emo,
                    "emotion_translated": emo_translated,
                    "emotion_score": float(emo_score) if emo_score is not None else 0.5
                })

                # Quadrado vermelho para detecção facial
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"T{tid} | {emo_translated}"
                cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            y0 = 20
            if act_label is not None:
                # Traduzir atividade para português
                act_translated = ACTIVITY_TRANSLATION.get(act_label, act_label)
                act_display = act_translated.replace("_", " ").title()
                
                act_color = (255, 255, 255)
                if act_label == "abrupt_movement":
                    act_color = (0, 165, 255)
                elif act_label == "idle":
                    act_color = (128, 128, 128)
                cv2.putText(frame, f"Atividade: {act_display}", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, act_color, 2, cv2.LINE_AA)
                y0 += 25
            
            if energy is not None:
                cv2.putText(frame, f"Energia mov: {energy:.1f}", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                y0 += 25
            
            if is_anom:
                cv2.putText(frame, "ANOMALIA!", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3, cv2.LINE_AA)

            event = {
                "frame_idx": int(frame_idx),
                "timestamp_s": float(timestamp_s(frame_idx, fps)),
                "faces": faces_out,
                "activity": {"label": act_label, "energy": energy},
                "is_anomaly": bool(is_anom)
            }
            f_events.write(json.dumps(event, ensure_ascii=False) + "\n")

            # Mostrar vídeo em tempo real
            cv2.imshow("Análise de Vídeo - Detecção Facial", frame)
            
            # Pressione 'q' para sair antecipadamente
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nInterrompido pelo usuário!")
                break

            if writer is not None:
                writer.write(frame)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()
    
    # Fechar janela de visualização
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Espera um momento para fechar todas as janelas

    # Gera relatório completo em português BR
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("RELATÓRIO DE ANÁLISE DE VÍDEO")
    report_lines.append("=" * 60)
    report_lines.append(f"\nVídeo analisado: {os.path.basename(video_path)}")
    report_lines.append(f"Resolução: {w}x{h} pixels")
    report_lines.append(f"FPS (Quadros por segundo): {fps:.2f}")
    report_lines.append(f"Duração total: {duration:.2f} segundos")
    report_lines.append(f"Total de quadros do vídeo: {nframes}")
    report_lines.append(f"Quadros processados: {processed_frames}")
    report_lines.append("")
    
    report_lines.append("-" * 60)
    report_lines.append("DETECÇÃO FACIAL E ANÁLISE DE SENTIMENTOS")
    report_lines.append("-" * 60)
    report_lines.append(f"Total de rostos detectados: {total_faces_detected}")
    report_lines.append(f"Número de pessoas identificadas (track IDs): {len(dominant_by_track)}")
    report_lines.append("")
    
    if dominant_by_track:
        report_lines.append("DETALHAMENTO POR PESSOA:")
        report_lines.append("")
        for tid in sorted(dominant_by_track.keys()):
            emo = dominant_by_track.get(tid, "neutro")
            emo_translated = EMOTION_TRANSLATION.get(emo.lower(), emo)
            count = emotion_counters[tid].total()
            
            # Calcular porcentagens de cada emoção
            emo_counts = emotion_counters[tid]
            total_emos = sum(emo_counts.values())
            
            report_lines.append(f"  Pessoa {tid}:")
            report_lines.append(f"    Sentimento predominante: {emo_translated.title()}")
            report_lines.append(f"    Total de detecções: {count}")
            report_lines.append(f"    Porcentagem de cada sentimento:")
            
            # Ordenar emoções por quantidade
            for emotion, emotion_count in emo_counts.most_common():
                emotion_translated = EMOTION_TRANSLATION.get(emotion.lower(), emotion)
                pct = 100.0 * emotion_count / total_emos if total_emos > 0 else 0
                report_lines.append(f"      - {emotion_translated.title()}: {pct:.1f}%")
            
            # Encontrar timestamp da emoção predominante
            emo_frames = [k for k, v in emotion_counters[tid].items() if v == emo_counts.most_common(1)[0][1]]
            if emo_frames:
                first_emo_frame = list(emo_counts.elements()).count(emo_frames[0])
                timestamp = first_emo_frame * emotion_every_n * frame_skip / fps if fps > 0 else 0
                report_lines.append(f"    Momento do sentimento predominante: ~{timestamp:.1f}s")
            
            report_lines.append("")
    
    report_lines.append("")
    report_lines.append("-" * 60)
    report_lines.append("ANÁLISE DE MOVIMENTO E ATIVIDADES")
    report_lines.append("-" * 60)
    total_act = sum(activity_counter.values()) or 1
    report_lines.append(f"Classificação das atividades detectadas:")
    report_lines.append("")
    for act, count in activity_counter.most_common():
        pct = 100.0 * count / total_act
        act_translated = ACTIVITY_TRANSLATION.get(act, act).replace("_", " ").title()
        report_lines.append(f"  {act_translated}: {count} ocorrências ({pct:.1f}%)")
    
    report_lines.append("")
    report_lines.append("-" * 60)
    report_lines.append("DETECÇÃO DE ANOMALIAS DE MOVIMENTO")
    report_lines.append("-" * 60)
    total_anoms = len(anomaly_timestamps)
    report_lines.append(f"Total de anomalias detectadas: {total_anoms}")
    if anomaly_timestamps:
        report_lines.append(f"Timestamps das anomalias detectadas:")
        for i, ts in enumerate(anomaly_timestamps[:20], 1):
            report_lines.append(f"  {i:2d}. {ts:.2f} segundos")
        if len(anomaly_timestamps) > 20:
            report_lines.append(f"  ... e mais {len(anomaly_timestamps) - 20} anomalias")
    else:
        report_lines.append("  Nenhuma anomalia significativa detectada.")
    
    report_lines.append("")
    report_lines.append("-" * 60)
    report_lines.append("RESUMO EXECUTIVO")
    report_lines.append("-" * 60)
    
    # Resumo em português
    if dominant_by_track:
        emotions_summary = []
        for tid in sorted(dominant_by_track.keys()):
            emo = dominant_by_track.get(tid, "neutro")
            emo_translated = EMOTION_TRANSLATION.get(emo.lower(), emo)
            emotions_summary.append(f"Pessoa {tid} ({emo_translated.title()})")
        report_lines.append(f"Pessoas detectadas: {', '.join(emotions_summary)}")
    
    if activity_counter:
        main_activity = activity_counter.most_common(1)[0][0]
        main_activity_translated = ACTIVITY_TRANSLATION.get(main_activity, main_activity).replace("_", " ").title()
        main_activity_pct = 100.0 * activity_counter[main_activity] / total_act
        
        if main_activity == "idle":
            activity_summary = f"Vídeo predominantemente estático ({main_activity_translated}, {main_activity_pct:.1f}%)"
        elif main_activity == "moving":
            activity_summary = f"Vídeo com movimento moderado e regular ({main_activity_translated}, {main_activity_pct:.1f}%)"
        else:
            activity_summary = f"Vídeo com momentos de movimento intenso/brusco ({main_activity_translated}, {main_activity_pct:.1f}%)"
    else:
        activity_summary = "Não foi possível classificar as atividades do vídeo"
    
    report_lines.append(f"Comportamento geral: {activity_summary}")
    report_lines.append(f"Rostos detectados: {total_faces_detected} em {processed_frames} quadros processados")
    
    if total_anoms > 0:
        report_lines.append(f"Anomalias de movimento: {total_anoms} momento(s) de movimento atípico detectado(s)")
    else:
        report_lines.append("Anomalias de movimento: Nenhuma detectada")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("FIM DO RELATÓRIO")
    report_lines.append("=" * 60)
    
    report_text = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print("\n" + report_text)
    
    print(f"\n>>> Concluído: {os.path.basename(video_path)}")
    print(f"    Events: {events_path}")
    print(f"    Relatório: {report_path}")
    if write_video:
        print(f"    Vídeo anotado: {annotated_path}")
    
    return {
        "video": os.path.basename(video_path),
        "frames_analyzed": processed_frames,
        "faces_detected": total_faces_detected,
        "anomalies": total_anoms,
        "activity_summary": activity_counter.most_common()
    }


# -----------------------------
# Main - Processa todos os vídeos
# -----------------------------

def main():
    """Processa todos os vídeos da pasta data e gera relatório consolidado."""
    parser = argparse.ArgumentParser(description="Análise de vídeo - Tech Challenge Fase 4")
    parser.add_argument("--data-dir", default="./data", help="Pasta com vídeos de entrada")
    parser.add_argument("--output-dir", default="./output", help="Pasta de saída")
    parser.add_argument("--frame-skip", type=int, default=1, help="Processa 1 a cada N frames")
    parser.add_argument("--emotion-every-n", type=int, default=10, help="Emoção a cada N frames")
    parser.add_argument("--anomaly-k", type=float, default=3.0, help="Limiar k para anomalia")
    parser.add_argument("--write-video", action="store_true", help="Gera vídeos anotados")
    args = parser.parse_args()

    video_exts = {".mp4", ".webm", ".avi", ".mov", ".mkv"}

    if not os.path.exists(args.data_dir):
        print(f"ERRO: Pasta não encontrada: {args.data_dir}")
        return

    videos = [
        os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)
        if os.path.isfile(os.path.join(args.data_dir, f)) and 
           os.path.splitext(f)[1].lower() in video_exts
    ]

    if not videos:
        print(f"Nenhum vídeo encontrado em: {args.data_dir}")
        return

    print("=" * 60)
    print("ANÁLISE DE VÍDEO - TECH CHALLENGE FASE 4")
    print("=" * 60)
    print(f"\nVídeos encontrados: {len(videos)}")
    for v in videos:
        print(f"  - {os.path.basename(v)}")
    print()

    ensure_dir(args.output_dir)

    all_results = []
    for video_path in sorted(videos):
        try:
            result = process_video(
                video_path=video_path,
                output_dir=args.output_dir,
                frame_skip=args.frame_skip,
                emotion_every_n=args.emotion_every_n,
                anomaly_k=args.anomaly_k,
                write_video=args.write_video
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERRO ao processar {os.path.basename(video_path)}: {e}")

    print("\n" + "=" * 60)
    print("RESUMO CONSOLIDADO")
    print("=" * 60)

    total_frames = sum(r["frames_analyzed"] for r in all_results)
    total_faces = sum(r["faces_detected"] for r in all_results)
    total_anomalies = sum(r["anomalies"] for r in all_results)

    print(f"\nTotal de vídeos processados: {len(all_results)}")
    print(f"Frames analisados (total): {total_frames}")
    print(f"Rostos detectados (total): {total_faces}")
    print(f"Anomalias detectadas (total): {total_anomalies}")

    print("\n" + "-" * 40)
    print("Por vídeo:")
    for r in all_results:
        print(f"\n  {r['video']}:")
        print(f"    Frames: {r['frames_analyzed']}")
        print(f"    Rostos: {r['faces_detected']}")
        print(f"    Anomalias: {r['anomalies']}")

    print("\n" + "=" * 60)
    print("PROCESSAMENTO CONCLUÍDO")
    print(f"Resultados em: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
