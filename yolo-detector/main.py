import os
import cv2
import time
import threading
from datetime import datetime
from flask import Flask, Response
from ultralytics import YOLO
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# URLs des caméras
rtsp_urls = [os.getenv(f"RTSP_URL_{i}") for i in range(1, 6)]
rtsp_urls = [url for url in rtsp_urls if url]

# Modèle YOLO
model = YOLO("yolov8n-face.pt")

# Répertoire des captures
CAPTURE_DIR = "/app/images/captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Flask App
app = Flask(__name__)
frame_buffers = [None for _ in rtsp_urls]
last_capture_time = {}
MIN_INTERVAL = 10  # secondes

# Traitement YOLO par caméra (avec reconnect RTSP)
def process_camera(idx, url):
    print(f"[CAM {idx+1}] ✅ Démarrée")

    while True:
        time.sleep(1)

        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"[CAM {idx+1}] ❌ Échec de connexion")
            time.sleep(5)
            continue

        # Lire une frame fraîche (évite les frames en retard)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"[CAM {idx+1}] ⚠️ Frame non lue")
            continue

        # Détection YOLO
        results = model(frame, conf=0.3, verbose=False)
        annotated = results[0].plot()
        boxes = results[0].boxes
        
        now = time.time()
        last_time = last_capture_time.get(idx, 0)
        print(now - last_time)
        if now - last_time >= MIN_INTERVAL:
            print("gogogo")
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

        # MJPEG buffer
        _, jpeg = cv2.imencode(".jpg", annotated)
        frame_buffers[idx] = jpeg.tobytes()

# Lancer un thread pour chaque caméra
for i, url in enumerate(rtsp_urls):
    threading.Thread(target=process_camera, args=(i, url), daemon=True).start()

# Routes Flask
@app.route("/")
def index():
    return "<h2>Flux caméras</h2>" + "".join(
        [f"<h3>Cam {i+1}</h3><img src='/video_feed_cam{i+1}'><br>" for i in range(len(rtsp_urls))]
    )

@app.route("/video_feed_cam<int:cam_id>")
def video_feed(cam_id):
    def generate():
        while True:
            frame = frame_buffers[cam_id - 1]
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            else:
                time.sleep(0.1)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Lancement Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
