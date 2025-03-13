import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
from PIL import Image

import pandas as pd
# import csv
# import os
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import tempfile

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


####
st.title("📊 Bounding Box Datenverarbeitung & Visualisierung")

 # Funktion zum Extrahieren der Bildnummer (z. B. frame_0001.png → 1)
def extract_image_number(image_filename):
    return int(image_filename.split('_')[1].split('.')[0])

# CSV-Datei einlesen und sortieren
def sort_csv(input_csv, output_csv):
    with open(input_csv, mode='r') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        rows = list(reader)

        # Nach Bildnummer sortieren
        rows.sort(key=lambda x: extract_image_number(x[0]))

    with open(output_csv, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(rows)

# Mittelpunkte berechnen
def process_csv(input_csv):
    class_positions = defaultdict(list)
    output_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name  # Temporäre Datei für Ausgabe

    with open(input_csv, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = list(reader)
        rows.sort(key=lambda x: extract_image_number(x[0]))

        with open(output_csv, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["Image", "Class", "X_mid", "Y_mid"])  

            for row in rows:
                image_file, class_name, x1, y1, x2, y2, confidence = row
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                x_mid = (x1 + x2) / 2
                y_mid = (y1 + y2) / 2

                writer.writerow([image_file, class_name, x_mid, y_mid])
                class_positions[class_name].append((x_mid, y_mid))

    return class_positions, output_csv

# Referenzpunkte einlesen
def process_reference_csv(reference_csv):
    reference_positions = defaultdict(list)
    with open(reference_csv, mode='r') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            image_file, class_name, x_mid, y_mid = row
            x_mid, y_mid = map(float, [x_mid, y_mid])

            reference_positions[class_name].append((x_mid, y_mid))
    
    return reference_positions

# # Visualisierung der Klassendaten
# def visualize_class_positions(class_positions, reference_positions):
#     plt.figure(figsize=(10, 6))
#     unique_colors = {class_name: (random.random(), random.random(), random.random()) for class_name in class_positions}

#     for class_name, positions in class_positions.items():
#         x_coords = [pos[0] for pos in positions]
#         y_coords = [pos[1] for pos in positions]
#         color = unique_colors[class_name]

#         plt.plot(x_coords, y_coords, marker='o', label=f"{class_name} (Haupt)", linestyle='-', color=color)

#         if class_name in reference_positions:
#             ref_positions = reference_positions[class_name]
#             ref_x_coords = [pos[0] for pos in ref_positions]
#             ref_y_coords = [pos[1] for pos in ref_positions]
#             plt.plot(ref_x_coords, ref_y_coords, marker='o', label=f"{class_name} (Referenz)", linestyle='--', color=color)

#     plt.title("Verlauf der Mittelpunkte der Klassen")
#     plt.xlabel("X-Koordinate")
#     plt.ylabel("Y-Koordinate")
#     plt.legend()
#     plt.grid(True)
#     st.pyplot(plt)

def visualize_class_positions(class_positions, reference_positions=None):
    plt.figure(figsize=(10, 6))
    unique_colors = {class_name: (random.random(), random.random(), random.random()) for class_name in class_positions}

    for class_name, positions in class_positions.items():
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        color = unique_colors[class_name]

        # Visualisierung der Positionspunkte der Klasse
        plt.plot(x_coords, y_coords, marker='o', label=f"{class_name} (Haupt)", linestyle='-', color=color)

        # Falls Referenzpositionen vorhanden sind, diese ebenfalls visualisieren
        if reference_positions and class_name in reference_positions:
            ref_positions = reference_positions[class_name]
            ref_x_coords = [pos[0] for pos in ref_positions]
            ref_y_coords = [pos[1] for pos in ref_positions]
            plt.plot(ref_x_coords, ref_y_coords, marker='o', label=f"{class_name} (Referenz)", linestyle='--', color=color)

    plt.title("Verlauf der Mittelpunkte der Klassen")
    plt.xlabel("X-Koordinate")
    plt.ylabel("Y-Koordinate")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Datei-Uploads in Streamlit
uploaded_csv = st.file_uploader("🔼 Lade die Ergebnisse-CSV hoch", type=["csv"])
uploaded_ref_csv = st.file_uploader("🔼 Lade die Referenz-CSV hoch (optional)", type=["csv"])

if uploaded_csv is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(uploaded_csv.read())
        input_csv_path = temp_file.name

    # Sortieren der CSV
    sorted_csv_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    sort_csv(input_csv_path, sorted_csv_path)
    st.success("✅ Datei sortiert!")

    # Mittelpunkte berechnen
    class_positions, processed_csv_path = process_csv(sorted_csv_path)
    st.success("✅ Mittelpunkte berechnet!")

    # Falls Referenzdatei vorhanden ist, laden
    reference_positions = {}
    if uploaded_ref_csv is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as ref_file:
            ref_file.write(uploaded_ref_csv.read())
            reference_csv_path = ref_file.name

        reference_positions = process_reference_csv(reference_csv_path)
        st.success("✅ Referenzdaten geladen!")

# Daten visualisieren
visualize_class_positions(class_positions, reference_positions)

# Download-Link für bearbeitete CSV
st.download_button("📥 Sortierte CSV herunterladen", data=open(sorted_csv_path, "rb").read(), file_name="sorted_results.csv")
st.download_button("📥 CSV mit Mittelpunkten herunterladen", data=open(processed_csv_path, "rb").read(), file_name="processed_results.csv")

