import os
import cv2
import time
import threading
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# URLs des caméras (RTSP_URL_1 à RTSP_URL_5)
rtsp_urls = [os.getenv(f"RTSP_URL_{i}") for i in range(1, 6)]
rtsp_urls = [url for url in rtsp_urls if url]

# Modèle YOLO pour détection de visages
model = YOLO("yolov8n-face.pt")

# Dossier de sauvegarde des visages capturés
CAPTURE_DIR = "/app/images/captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Suivi du temps de dernière capture
last_capture_time = {}
MIN_INTERVAL = 5  # secondes

# Traitement YOLO pour une caméra
def process_camera(idx, url):
    print(f"[CAM {idx+1}] ✅ Démarrée")

    while True:
        time.sleep(1)

        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"[CAM {idx+1}] ❌ Échec de connexion")
            time.sleep(5)
            continue

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"[CAM {idx+1}] ⚠️ Frame non lue")
            continue

        results = model(frame, conf=0.3, verbose=False)
        boxes = results[0].boxes

        now = time.time()
        last_time = last_capture_time.get(idx, 0)

        if now - last_time >= MIN_INTERVAL:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(x1 - 20, 0)
                y1 = max(y1 - 20, 0)
                x2 = x2 + 20
                y2 = y2 + 20

                crop = frame[y1:y2, x1:x2]

                conf = float(box.conf[0]) if box.conf is not None else 0.0
                conf_str = f"{conf:.2f}"

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cam{idx+1}_face{i}_{ts}_conf{conf_str}.jpg"
                cv2.imwrite(os.path.join(CAPTURE_DIR, filename), crop)
                print(f"[CAM {idx+1}] 📸 Visage capturé : {filename}")

            last_capture_time[idx] = now
        else:
            print(f"[CAM {idx+1}] ⏳ Trop tôt pour une nouvelle capture")

# Démarrer un thread pour chaque caméra
for i, url in enumerate(rtsp_urls):
    threading.Thread(target=process_camera, args=(i, url), daemon=True).start()

# Garder le script vivant
while True:
    time.sleep(60)
