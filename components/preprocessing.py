import os
import shutil
from PIL import Image
import streamlit as st

def show_preprocessing_page():
    st.header("Preprocessing Dataset")
    input_folder = st.text_input("Path Dataset Mentah", value="data/raw")
    output_folder = st.text_input("Path Output Dataset Siap Latih", value="data/processed")
    if st.button("Jalankan Preprocessing"):
        # Cek dua kemungkinan struktur folder
        image_dir1 = os.path.join(input_folder, "images")
        label_dir1 = os.path.join(input_folder, "labels")
        image_dir2 = os.path.join(input_folder, "dataset/train/images")
        label_dir2 = os.path.join(input_folder, "dataset/train/labels")

        if os.path.exists(image_dir1) and os.path.exists(label_dir1):
            preprocess_dataset(image_dir1, label_dir1, output_folder)
            st.success("Preprocessing selesai.")
        elif os.path.exists(image_dir2) and os.path.exists(label_dir2):
            preprocess_dataset(image_dir2, label_dir2, output_folder)
            st.success("Preprocessing selesai.")
        else:
            st.error(
                f"Folder gambar tidak ditemukan: {image_dir1} atau {image_dir2} "
                f"dan/atau folder label tidak ditemukan: {label_dir1} atau {label_dir2}"
            )

def preprocess_dataset(image_dir, label_dir, output_folder, size=(640, 640)):
    st.write("📦 Memulai preprocessing dataset...")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    out_image_dir = os.path.join(output_folder, "images")
    out_label_dir = os.path.join(output_folder, "labels")
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    for fname in os.listdir(image_dir):
        if fname.endswith((".jpg", ".png", ".jpeg")):
            try:
                img_path = os.path.join(image_dir, fname)
                img = Image.open(img_path).resize(size)
                img.save(os.path.join(out_image_dir, fname))
                # Copy label file
                label_name = os.path.splitext(fname)[0] + ".txt"
                label_path = os.path.join(label_dir, label_name)
                if os.path.exists(label_path):
                    shutil.copy(label_path, os.path.join(out_label_dir, label_name))
            except Exception as e:
                st.error(f"Gagal preprocess {fname}: {e}")

st.success("✅ Preprocessing selesai.")

