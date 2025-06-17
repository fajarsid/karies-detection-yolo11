# components/labeling.py
import streamlit as st
from PIL import Image
import os
import config
from utils import load_model, process_image_detection, save_yolo_labels  # ✅ pastikan ada fungsi save_yolo_labels

def show_labeling_page():
    st.header("🏷️ Labeling Otomatis dengan YOLOv11")

    model = load_model("Detection", config)
    if model is None:
        st.error("Gagal load model.")
        return

    input_folder = st.text_input("Folder Gambar", "data/unlabeled")
    output_label_folder = st.text_input("Folder Output Label", "data/labels")

    confidence = st.slider("Confidence", 0.2, 1.0, 0.4)

    if st.button("Mulai Labeling"):
        images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        progress = st.progress(0)

        for i, img_file in enumerate(images):
            img_path = os.path.join(input_folder, img_file)
            image = Image.open(img_path)
            result = process_image_detection(model, image, confidence)

            if result["success"]:
                label_path = os.path.join(output_label_folder, img_file.replace('.jpg', '.txt'))
                save_yolo_labels(result["detection_data"], label_path)
            progress.progress((i+1)/len(images))

        st.success(f"Labeling selesai! Total gambar: {len(images)}")
    else:
        st.header("Labeling Dataset")
        st.info("Halaman labeling dataset belum diimplementasikan.")
    
    input_folder = st.text_input("Path Folder Gambar Belum Berlabel", value="data/unlabeled")
    label_folder = st.text_input("Path Output Label", value="data/labels")

    if not os.path.exists(input_folder):
        st.warning(f"Folder gambar tidak ditemukan: {input_folder}")
        return

    if st.button("Muat Daftar Gambar"):
        images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            st.info("Tidak ada gambar ditemukan di folder tersebut.")
            return
        st.write(f"Ditemukan {len(images)} gambar untuk labeling.")
