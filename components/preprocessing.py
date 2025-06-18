import os
import shutil
from PIL import Image
import streamlit as st

def show_preprocessing_page():
    
    col1, col2 = st.columns(2)
    with col1:
        input_folder = st.text_input("Path Dataset Mentah", value="data/raw")
    with col2:
        output_folder = st.text_input("Path Output Dataset Siap Latih", value="data/processed")
    
    if st.button("🚀 Jalankan Preprocessing", type="primary"):
        # Validasi path
        if not os.path.exists(input_folder):
            st.error(f"Folder input tidak ditemukan: {input_folder}")
            return
            
        # Cek struktur folder
        image_dir1 = os.path.join(input_folder, "images")
        label_dir1 = os.path.join(input_folder, "labels")
        image_dir2 = os.path.join(input_folder, "dataset/train/images")
        label_dir2 = os.path.join(input_folder, "dataset/train/labels")

        if os.path.exists(image_dir1) and os.path.exists(label_dir1):
            preprocess_dataset(image_dir1, label_dir1, output_folder)
        elif os.path.exists(image_dir2) and os.path.exists(label_dir2):
            preprocess_dataset(image_dir2, label_dir2, output_folder)
        else:
            st.error(
                "Struktur folder tidak valid. Pastikan terdapat subfolder:\n"
                "- images/ dan labels/ ATAU\n"
                "- dataset/train/images/ dan dataset/train/labels/"
            )

def preprocess_dataset(image_dir, label_dir, output_folder, size=(640, 640)):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Setup output folder
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        out_image_dir = os.path.join(output_folder, "images")
        out_label_dir = os.path.join(output_folder, "labels")
        os.makedirs(out_image_dir, exist_ok=True)
        os.makedirs(out_label_dir, exist_ok=True)

        # Get image files
        image_files = [f for f in os.listdir(image_dir) 
                     if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        total_files = len(image_files)
        
        if total_files == 0:
            st.warning("Tidak ada file gambar yang ditemukan!")
            return

        success_count = 0
        for i, fname in enumerate(image_files):
            try:
                # Update progress
                progress = (i + 1) / total_files
                progress_bar.progress(progress)
                status_text.text(f"Memproses {i+1}/{total_files}: {fname}")

                # Process image
                img_path = os.path.join(image_dir, fname)
                img = Image.open(img_path)
                img = img.resize(size)
                img.save(os.path.join(out_image_dir, fname))

                # Process label
                label_name = os.path.splitext(fname)[0] + ".txt"
                label_path = os.path.join(label_dir, label_name)
                if os.path.exists(label_path):
                    shutil.copy(label_path, os.path.join(out_label_dir, label_name))
                
                success_count += 1
            except Exception as e:
                st.error(f"Gagal memproses {fname}: {str(e)}")

        # Final status
        progress_bar.empty()
        status_text.empty()
        st.success(f"""
        ✅ Preprocessing selesai!
        - Total file diproses: {success_count}/{total_files}
        - Output disimpan di: {output_folder}
        """)
        
    except Exception as e:
        st.error(f"Error saat preprocessing: {str(e)}")
        progress_bar.empty()
        status_text.empty()