import os
import cv2
import face_recognition
import pickle

# Dossier des visages connus
KNOWN_DIR = "faces_known"
ENCODINGS_PATH = "known_faces.pkl"

known_encodings = []
known_names = []

for person_name in os.listdir(KNOWN_DIR):
    person_dir = os.path.join(KNOWN_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        filepath = os.path.join(person_dir, filename)
        image = cv2.imread(filepath)
        if image is None:
            print(f"[⚠️] Image illisible : {filepath}")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_image)
        encodings = face_recognition.face_encodings(rgb_image, boxes)

        if not encodings:
            print(f"[❌] Aucun visage détecté dans : {filename}")
            continue

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)
            print(f"[✅] {person_name} -> {filename}")

# Sauvegarder les encodages
data = {"encodings": known_encodings, "names": known_names}
with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump(data, f)

print(f"\n💾 Encodages sauvegardés dans : {ENCODINGS_PATH}")
