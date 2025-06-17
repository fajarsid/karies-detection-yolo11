import os
import streamlit as st
from utils import start_training
from components.preprocessing import preprocess_dataset

def show_training_page():
    st.header("Latih Model Deteksi Karies Gigi (YOLO11)")
    st.markdown("""
    Jalankan pelatihan model YOLO11 secara langsung dari web ini.  
    Pastikan dataset dan file konfigurasi sudah tersedia di server.
    """)

    dataset_path = st.text_input("Path Dataset Mentah", value="data/raw")
    output_path = st.text_input("Path Output Dataset Siap Latih", value="data/processed")

    with st.form("train_form"):
        dataset_config = st.text_input("Path file konfigurasi dataset (.yaml)", value="karies.yaml")
        model_arch = st.selectbox("Pilih Arsitektur Model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"])
        epochs = st.number_input("Jumlah Epoch", min_value=1, max_value=300, value=100)
        batch = st.number_input("Batch Size", min_value=1, max_value=128, value=16)
        imgsz = st.number_input("Ukuran Gambar", min_value=320, max_value=1280, value=640)
        submit = st.form_submit_button("Mulai Training")

    if submit:
        start_training(dataset_config, model_arch, epochs, batch, imgsz)

    if st.button("Preprocessing Dataset"):
        # Cek dua kemungkinan struktur folder
        image_dir1 = os.path.join(dataset_path, "images")
        label_dir1 = os.path.join(dataset_path, "labels")
        image_dir2 = os.path.join(dataset_path, "dataset/train/images")
        label_dir2 = os.path.join(dataset_path, "dataset/train/labels")

        if os.path.exists(image_dir1) and os.path.exists(label_dir1):
            preprocess_dataset(image_dir1, label_dir1, output_path)
            st.success("✅ Preprocessing selesai.")
        elif os.path.exists(image_dir2) and os.path.exists(label_dir2):
            preprocess_dataset(image_dir2, label_dir2, output_path)
            st.success("✅ Preprocessing selesai.")
        else:
            st.error("❌ Folder gambar/label tidak ditemukan di struktur yang dikenali.")

    if st.button("Mulai Latih Model"):
        st.write("🚀 Melatih model dengan dataset hasil preprocessing...")
