import os
import pickle
import face_recognition
import mysql.connector
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import cv2


print("=== DEMARRAGE DU SCRIPT ===")

load_dotenv()

# Dossiers
CAPTURE_DIR = "/app/images/captures"
UNKNOWN_DIR = "/app/images/faces_unknown"

os.makedirs(UNKNOWN_DIR, exist_ok=True)

# Connexion MySQL
conn = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST", "mysql-db"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE")
)
cursor = conn.cursor()

# Charger tous les visages connus (encodages en BLOB -> vecteurs)
cursor.execute("SELECT id, name, encoding FROM faces")
known_faces = []
for face_id, name, encoding_blob in cursor.fetchall():
    encoding = pickle.loads(encoding_blob)
    known_faces.append((face_id, name, encoding))

print(f"✅ {len(known_faces)} visages connus chargés.")

# Parcours des images à traiter
for filename in os.listdir(CAPTURE_DIR):
    path = os.path.join(CAPTURE_DIR, filename)
    if not filename.lower().endswith(".jpg"):
        continue

    print(f"📸 Traitement de {filename}")
    image = face_recognition.load_image_file(path)
    locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, locations)

    if len(encodings) == 0:
        print(f"⚠️ Aucun visage détecté dans {filename}")
        continue

    # On ne traite que le 1er visage par image
    encoding = encodings[0]
    distances = [face_recognition.face_distance([k[2]], encoding)[0] for k in known_faces]

    if distances and min(distances) < 0.45:
        best_index = distances.index(min(distances))
        best_id, best_name, _ = known_faces[best_index]
        confidence = 1.0 - distances[best_index]
        print(f"✅ Visage reconnu : {best_name} (conf={confidence:.2f})")

        cursor.execute("""
            INSERT INTO detections (face_id, cam_id, image_path, match_confidence, is_unknown)
            VALUES (%s, %s, %s, %s, %s)
        """, (best_id, 1, path, distances[best_index], False))

    else:
        print("❓ Visage inconnu, enregistrement comme inconnu.")
        new_path = os.path.join(UNKNOWN_DIR, filename)
        os.rename(path, new_path)

        cursor.execute("""
            INSERT INTO detections (face_id, cam_id, image_path, match_confidence, is_unknown)
            VALUES (%s, %s, %s, %s, %s)
        """, (None, 1, new_path, None, True))

    conn.commit()

# Nettoyage
cursor.close()
conn.close()
print("🧼 Traitement terminé.")
