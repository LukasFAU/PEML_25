import streamlit as st
import torch
import cv2
import numpy as np
import os
from PIL import Image

# Modell laden (stellen Sie sicher, dass best.pt im richtigen Ordner liegt)
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Das Modell muss in deinem Repository liegen
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
    return model

# Bildverarbeitung mit YOLOv5
def detect_objects(image, model):
    results = model(image)
    detections = []
    
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        class_name = model.names[int(cls)]
        detections.append((class_name, x1, y1, x2, y2, conf.item()))

        # Bounding Box zeichnen
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} ({conf:.2f})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, detections

# Streamlit UI
st.title("🔍 YOLOv5 Objekterkennung")

# Datei-Upload
uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    model = load_model()

    # Objekterkennung durchführen
    processed_image, detections = detect_objects(image, model)

    # Ergebnisse anzeigen
    st.image(processed_image, caption="Erkannte Objekte", use_column_width=True)
    
    # Ergebnisse als Tabelle ausgeben
    st.write("### 🔎 Ergebnisse")
    for det in detections:
        st.write(f"**{det[0]}** bei ({det[1]}, {det[2]}) - ({det[3]}, {det[4]}), Vertrauen: {det[5]:.2f}")
