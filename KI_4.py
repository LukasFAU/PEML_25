import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# Modell laden
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Stelle sicher, dass best.pt vorhanden ist
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

# Mittelpunkte berechnen
def calculate_midpoints(detections):
    class_positions = defaultdict(list)
    for det in detections:
        class_name, x1, y1, x2, y2, _ = det
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2
        class_positions[class_name].append((x_mid, y_mid))
    return class_positions

# Visualisierung der Klassendaten
def visualize_class_positions(class_positions):
    plt.figure(figsize=(10, 6))
    unique_colors = {class_name: (random.random(), random.random(), random.random()) for class_name in class_positions}

    for class_name, positions in class_positions.items():
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        color = unique_colors[class_name]
        plt.scatter(x_coords, y_coords, label=class_name, color=color)

    plt.title("Verlauf der Mittelpunkte der Klassen")
    plt.xlabel("X-Koordinate")
    plt.ylabel("Y-Koordinate")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Streamlit UI
st.title("🔍 YOLOv5 Objekterkennung für Bilder")

uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    model = load_model()
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    processed_image, detections = detect_objects(image, model)
    st.image(processed_image, caption="Erkannte Objekte", use_column_width=True)

    st.write("### 🔎 Ergebnisse")
    for det in detections:
        st.write(f"**{det[0]}** bei ({det[1]}, {det[2]}) - ({det[3]}, {det[4]}), Vertrauen: {det[5]:.2f}")

    # Mittelpunkte berechnen und visualisieren
    class_positions = calculate_midpoints(detections)
    visualize_class_positions(class_positions)
