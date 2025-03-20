
 import streamlit as st
import tempfile
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import torch
from moviepy.editor import VideoFileClip

# Modell laden
@st.cache_resource
def load_model():
    model_path = "best.pt"  
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
    return model

# Streamlit UI
st.title("🔍 YOLOv5 Objekterkennung für Bilder & Videos")

# Datei-Upload (Bilder & Videos)
uploaded_file = st.file_uploader("Lade ein Bild oder Video hoch", type=["jpg", "png", "jpeg", "mp4"])

if uploaded_file is None:
    st.warning("⚠️ Bitte lade eine Datei hoch!")

# Frame-Schrittgröße eingeben
frame_step = st.number_input("Nur jeden n-ten Frame analysieren", min_value=1, value=1, step=1)

# Bildverarbeitung
if uploaded_file is not None:
    model = load_model()

    if uploaded_file.type.startswith("image"):
        st.write("📷 **Bild hochgeladen**")
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

        # Bildverarbeitung (Platzhalter-Funktion, falls du eine hast)
        processed_image, detections = detect_objects(image, model)
        st.image(processed_image, caption="Erkannte Objekte", use_column_width=True)

    elif uploaded_file.type == "video/mp4":
        st.write("🎥 **Video hochgeladen**")

        # Temporäre Datei speichern
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_video_path = temp_file.name
        
        st.video(temp_video_path)

        # Video verarbeiten (Platzhalter-Funktion, falls du eine hast)
        results = process_video(temp_video_path, model, frame_step)

        # CSV für Ergebnisse
        df = pd.DataFrame(results)
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 CSV herunterladen", data=csv_data, file_name="detektionen.csv", mime='text/csv')

        os.remove(temp_video_path)
