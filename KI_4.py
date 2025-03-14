import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import csv

# Modell laden
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Stelle sicher, dass best.pt vorhanden ist
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
    return model

# Bildverarbeitung mit YOLOv5
def detect_objects(image, model, frame_number):
    results = model(image)
    detections = []
    
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        class_name = model.names[int(cls)]
        detections.append([f"frame_{frame_number}.png", class_name, x1, y1, x2, y2, conf.item()])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} ({conf:.2f})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image, detections

# Videoverarbeitung mit Speicherung der Bounding Boxen
def process_video(video_path, model, frame_step):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Fehler: Das Video konnte nicht geöffnet werden!")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = max(1, total_frames // frame_step)
    progress_bar = st.progress(0)
    st_frame = st.empty()
    all_detections = []
    
    frame_index = 0
    processed_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_index % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, detections = detect_objects(frame_rgb, model, frame_index)
            all_detections.extend(detections)
            st_frame.image(processed_frame, channels="RGB", caption=f"Frame {frame_index}")
            processed_frames += 1
            progress_bar.progress(min(processed_frames / frames_to_process, 1.0))
        
        frame_index += 1
    
    cap.release()
    
    return all_detections

# Mittelpunkte berechnen und speichern
def calculate_midpoints(detections):
    class_positions = defaultdict(list)
    output_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    
    with open(output_csv, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Image", "Class", "X_mid", "Y_mid"])
        
        for row in detections:
            image_file, class_name, x1, y1, x2, y2, confidence = row
            x_mid = (x1 + x2) / 2
            y_mid = (y1 + y2) / 2
            writer.writerow([image_file, class_name, x_mid, y_mid])
            class_positions[class_name].append((x_mid, y_mid))
    
    return class_positions, output_csv

# Visualisierung der Bounding Box Mittelpunkte
def visualize_class_positions(class_positions):
    plt.figure(figsize=(10, 6))
    unique_colors = {class_name: (random.random(), random.random(), random.random()) for class_name in class_positions}
    
    for class_name, positions in class_positions.items():
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        color = unique_colors[class_name]
        plt.plot(x_coords, y_coords, marker='o', label=f"{class_name}", linestyle='-', color=color)
    
    plt.title("Verlauf der Mittelpunkte der Klassen")
    plt.xlabel("X-Koordinate")
    plt.ylabel("Y-Koordinate")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Streamlit UI
st.title("🔍 YOLOv5 Objekterkennung für Bilder & Videos")
uploaded_file = st.file_uploader("Lade ein Bild oder Video hoch", type=["jpg", "png", "jpeg", "mp4"])
st.subheader("Geben Sie den Anteil an verwendeten Frames ein")
frame_step = st.number_input("Nur jeden n-ten Frame analysieren", min_value=1, value=1, step=1)

if uploaded_file is not None:
    model = load_model()
    detections = []
    
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
        processed_image, detections = detect_objects(image, model, 0)
        st.image(processed_image, caption="Erkannte Objekte", use_column_width=True)
    
    elif uploaded_file.type == "video/mp4":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_video_path = temp_file.name
        st.video(temp_video_path)
        detections = process_video(temp_video_path, model, frame_step)
        os.remove(temp_video_path)
    
    if detections:
        class_positions, output_csv = calculate_midpoints(detections)
        visualize_class_positions(class_positions)
        st.download_button("📥 CSV mit Mittelpunkten herunterladen", data=open(output_csv, "rb").read(), file_name="processed_results.csv")
