import streamlit as st
import subprocess
import numpy as np
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pandas as pd
import torch
from moviepy.editor import VideoFileClip

st.write("🔍 Überprüfe installierte Pakete...")

try:
    import PIL
    st.success("✅ Pillow ist installiert (statt OpenCV)!")
except ImportError:
    st.error("❌ Pillow ist nicht installiert!")

installed_packages = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
st.text(installed_packages.stdout)

@st.cache_resource
def load_model():
    model_path = "best.pt"  # Stelle sicher, dass best.pt vorhanden ist
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
    return model

# Videoverarbeitung mit MoviePy statt OpenCV
def process_video(video_path, model, frame_step):
    st.write("🚀 **Starte Videoverarbeitung...**")
    clip = VideoFileClip(video_path)
    total_frames = int(clip.fps * clip.duration)
    fps = clip.fps
    
    if total_frames == 0:
        st.error("❌ Fehler: Das Video enthält keine Frames oder kann nicht gelesen werden.")
        return []
    
    frames_to_process = max(1, total_frames // frame_step)
    st.write(f"🎥 **Gesamtzahl der Frames:** {total_frames}")
    st.write(f"⏱ **Framerate (FPS):** {fps}")
    st.write(f"⚡ **Geplante Verarbeitung:** {frames_to_process} Frames (jeder {frame_step}. Frame)")

    progress_bar = st.progress(0)
    st_frame = st.empty()
    results_list = []
    frame_numbers = []
    class_names = []

    for frame_index, frame in enumerate(clip.iter_frames(fps=frame_step)):
        image = Image.fromarray(frame)
        processed_frame, detections = detect_objects(image, model)
        results_list.append(detections)
        
        for det in detections:
            frame_numbers.append(frame_index)
            class_names.append(det[0])
        
        st_frame.image(processed_frame, caption=f"Frame {frame_index}/{total_frames}")
        progress_bar.progress(min(frame_index / frames_to_process, 1.0))

    plot_results(frame_numbers, class_names)
    return results_list, frame_numbers, class_names

# Bildverarbeitung mit YOLOv5 und Pillow
def detect_objects(image, model):
    results = model(np.array(image))
    detections = []
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        class_name = model.names[int(cls)]
        detections.append((class_name, x1, y1, x2, y2, conf.item()))
        
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        draw.text((x1, y1 - 10), f"{class_name} ({conf:.2f})", fill="green", font=font)
    
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
