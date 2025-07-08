-- 1. Table des visages connus
CREATE TABLE IF NOT EXISTS faces (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    encoding BLOB NOT NULL,         -- Encodage du visage (vecteur 128D, serialisé)
    image_path VARCHAR(255),        -- Pour afficher l’image de référence
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. Table des détections
CREATE TABLE IF NOT EXISTS detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    face_id INT,                    -- Référence à faces.id, peut être NULL si inconnu
    cam_id INT NOT NULL,
    image_path VARCHAR(255) NOT NULL,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    match_confidence FLOAT,         -- Distance dB de face_recognition (plus c’est petit, mieux c’est)
    is_unknown BOOLEAN DEFAULT FALSE,

    FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE SET NULL
);
