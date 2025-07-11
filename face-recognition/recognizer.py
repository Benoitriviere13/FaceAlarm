import os
import shutil
import time
import pickle
import face_recognition



# Répertoires et chemins
BASE_DIR = "/app"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_FILE = os.path.join(BASE_DIR, "known_faces.pkl")
CAPTURE_DIR = os.path.join(BASE_DIR, "images", "captures")
FACES_UNKNOWN_DIR = os.path.join(BASE_DIR, "images", "faces_unknown")

# Créer le dossier faces_unknown s'il n'existe pas
os.makedirs(FACES_UNKNOWN_DIR, exist_ok=True)

# Chargement des visages connus
if not os.path.exists(PKL_FILE):
    print("❌ Fichier 'encodings.pkl' introuvable.")
    exit(1)

with open(PKL_FILE, "rb") as f:
    data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]

print(f"✅ {len(known_names)} visages connus chargés.")

# Fonction de traitement
def traiter_images():
    images = [f for f in os.listdir(CAPTURE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        print("📂 Aucune image à traiter.")
        return

    for image_name in images:
        image_path = os.path.join(CAPTURE_DIR, image_name)
        print(f"\n📸 Traitement : {image_name}")

        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if not face_encodings:
            print("⚠️ Aucun visage détecté.")
            continue

        match_found = False
        for i, face_encoding in enumerate(face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            if True in matches:
                matched_index = matches.index(True)
                matched_name = known_names[matched_index]
                print(f"   ✅ Visage reconnu : {matched_name}. Suppression de l'image.")
                match_found = True
                break

        if match_found:
            os.remove(image_path)
        else:
            print(f"   ❓ Aucun visage reconnu. Déplacement dans 'faces_unknown'.")
            shutil.move(image_path, os.path.join(FACES_UNKNOWN_DIR, image_name))

# Boucle toutes les 2 minutes
print("⏳ Démarrage du script de reconnaissance toutes les 2 minutes...")
while True:
    traiter_images()
    time.sleep(120)