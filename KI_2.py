import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
from PIL import Image

# Modell laden
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Stelle sicher, dass best.pt vorhanden ist
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
    return model

# Videoverarbeitung mit Fehlerbehandlung
def process_video(video_path, model, frame_step):
    st.write("🚀 **Starte Videoverarbeitung...**")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Fehler: Das Video konnte nicht geöffnet werden!")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

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

    frame_index = 0
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, detections = detect_objects(frame_rgb, model)
            results_list.append(detections)

            st_frame.image(processed_frame, channels="RGB", caption=f"Frame {frame_index}/{total_frames}")

            # Fortschritt berechnen und anzeigen
            processed_frames += 1
            progress = processed_frames / frames_to_process
            progress_bar.progress(min(progress, 1.0))

        frame_index += 1

    cap.release()
    return results_list

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
st.title("🔍 YOLOv5 Objekterkennung für Bilder & Videos")

# Datei-Upload (Bilder & Videos)
uploaded_file = st.file_uploader("Lade ein Bild oder Video hoch", type=["jpg", "png", "jpeg", "mp4"])

# Eingabefeld für Frame-Rate
st.subheader("Geben Sie den Anteil an verwendeten Frames ein")
frame_step = st.number_input("Nur jeden n-ten Frame analysieren", min_value=1, value=1, step=1)

if uploaded_file is not None:
    model = load_model()

    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

        processed_image, detections = detect_objects(image, model)
        st.image(processed_image, caption="Erkannte Objekte", use_column_width=True)

        st.write("### 🔎 Ergebnisse")
        for det in detections:
            st.write(f"**{det[0]}** bei ({det[1]}, {det[2]}) - ({det[3]}, {det[4]}), Vertrauen: {det[5]:.2f}")

    elif uploaded_file.type == "video/mp4":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_video_path = temp_file.name

        st.video(temp_video_path)
        results = process_video(temp_video_path, model, frame_step)

        st.write("### 🔎 Ergebnisse aus dem Video")
        for idx, frame_detections in enumerate(results):
            st.write(f"**Frame {idx * frame_step}**:")
            for det in frame_detections:
                st.write(f"**{det[0]}** bei ({det[1]}, {det[2]}) - ({det[3]}, {det[4]}), Vertrauen: {det[5]:.2f}")

        os.remove(temp_video_path)
