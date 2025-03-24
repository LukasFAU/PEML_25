import streamlit as st
import torch
import numpy as np
import tempfile
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import csv

# Modell laden
@st.cache_resource
def load_model():
    model_path = "best (1).pt"
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
    return model

# Videoverarbeitung mit Fehlerbehandlung
def process_video(video_path, model, frame_step, confidence_threshold):
    st.write("🚀 **Starte Videoverarbeitung...**")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Fehler: Das Video konnte nicht geöffnet werden!")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_process = max(1, total_frames // frame_step)
    
    st.write(f"🎥 **Gesamtzahl der Frames:** {total_frames}")
    st.write(f"⏱ **Framerate (FPS):** {fps}")
    st.write(f"⚡ **Geplante Verarbeitung:** {frames_to_process} Frames")
    
    progress_bar = st.progress(0)
    st_frame = st.empty()
    results_list = []
    frame_numbers = []
    class_names = []
    
    frame_index = 0
    processed_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, detections = detect_objects(frame_rgb, model, confidence_threshold)
            results_list.append(detections)
            
            for det in detections:
                frame_numbers.append(frame_index)
                class_names.append(det[0])

            st_frame.image(processed_frame, channels="RGB", caption=f"Frame {frame_index}/{total_frames}")
            processed_frames += 1
            progress = processed_frames / frames_to_process
            progress_bar.progress(min(progress, 1.0))
        
        frame_index += 1

    cap.release()
    plot_results(frame_numbers, class_names)
    return results_list

# Bildverarbeitung mit YOLOv5
def detect_objects(image, model, confidence_threshold):
    results = model(image)
    detections = []
    
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf.item() >= confidence_threshold:
            x1, y1, x2, y2 = map(int, xyxy)
            class_name = model.names[int(cls)]
            detections.append((class_name, x1, y1, x2, y2, conf.item()))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image, detections

# Ergebnisse visualisieren
def plot_results(frame_numbers, class_names):
    plt.figure(figsize=(10, 6))
    plt.scatter(frame_numbers, class_names, marker='o', color='blue', alpha=0.6)
    plt.xlabel("Frame Nummer")
    plt.ylabel("Klassennamen")
    plt.title("Erkannte Objekte über Frames hinweg")
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)

# Streamlit UI
st.title("🔍 YOLOv5 Objekterkennung für Bilder & Videos")
uploaded_file = st.file_uploader("Lade ein Bild oder Video hoch", type=["jpg", "png", "jpeg", "mp4"])
frame_step = st.number_input("Nur jeden n-ten Frame analysieren", min_value=1, value=1, step=1)
confidence_threshold = st.slider("Mindestkonfidenz für Erkennung", 0.0, 1.0, 0.5, 0.05)

def save_detections_to_csv(detections, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Class", "X1", "Y1", "X2", "Y2", "Confidence"])
        for detection in detections:
            writer.writerow(detection)

if uploaded_file is not None:
    model = load_model()
    
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
        processed_image, detections = detect_objects(image, model, confidence_threshold)
        st.image(processed_image, caption="Erkannte Objekte", use_column_width=True)
    
    elif uploaded_file.type == "video/mp4":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_video_path = temp_file.name
        
        st.video(temp_video_path)
        results = process_video(temp_video_path, model, frame_step, confidence_threshold)
        os.remove(temp_video_path)
