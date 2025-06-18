# components/labeling.py
import streamlit as st
from PIL import Image
import os
import config
from utils import load_model, process_image_detection, save_yolo_labels
import pandas as pd

def show_labeling_page():
    
    # Tab system untuk berbagai mode labeling
    tab1, tab2 = st.tabs(["Labeling Otomatis", " Labeling Manual"])
    
    with tab1:
        st.subheader("Labeling Otomatis Menggunakan Model")
        model = load_model("Detection", config)
        
        if model is None:
            st.error("Gagal load model. Pastikan model YOLOv11 sudah tersedia.")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            input_folder = st.text_input("Folder Gambar", "data/unlabeled", key="auto_input")
        with col2:
            output_folder = st.text_input("Folder Output Label", "data/labels", key="auto_output")
        
        confidence = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
        batch_size = st.number_input("Jumlah Gambar per Batch", 1, 100, 10)
        
        if st.button("🚀 Jalankan Labeling Otomatis", type="primary"):
            if not os.path.exists(input_folder):
                st.error(f"Folder tidak ditemukan: {input_folder}")
                return
                
            images = [f for f in os.listdir(input_folder) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not images:
                st.warning("Tidak ada gambar ditemukan di folder tersebut")
                return
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            for i, img_file in enumerate(images[:batch_size]):
                try:
                    # Update progress
                    progress = (i + 1) / min(batch_size, len(images))
                    progress_bar.progress(progress)
                    status_text.text(f"Memproses {i+1}/{min(batch_size, len(images))}: {img_file}")
                    
                    # Process image
                    img_path = os.path.join(input_folder, img_file)
                    image = Image.open(img_path)
                    result = process_image_detection(model, image, confidence)
                    
                    if result["success"]:
                        label_path = os.path.join(output_folder, 
                                               os.path.splitext(img_file)[0] + ".txt")
                        save_yolo_labels(result["detection_data"], label_path)
                        results.append({
                            "File": img_file,
                            "Status": "✅ Berhasil",
                            "Objek Terdeteksi": len(result["detection_data"])
                        })
                    else:
                        results.append({
                            "File": img_file, 
                            "Status": "❌ Gagal",
                            "Objek Terdeteksi": 0
                        })
                except Exception as e:
                    results.append({
                        "File": img_file,
                        "Status": f"❌ Error: {str(e)}",
                        "Objek Terdeteksi": 0
                    })
            
            # Show results
            progress_bar.empty()
            status_text.empty()
            st.success(f"Labeling selesai! {len([r for r in results if '✅' in r['Status']])}/{len(results)} berhasil")
            
            # Results table
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)
            
            # Sample preview
            st.subheader("Pratinjau Hasil")
            cols = st.columns(3)
            for idx, img_file in enumerate(images[:3]):
                if idx < len(cols):
                    with cols[idx]:
                        st.image(Image.open(os.path.join(input_folder, img_file)), 
                                caption=img_file, width=200)
    
    with tab2:
        st.subheader("Labeling Manual/Review")
        manual_input = st.text_input("Folder Gambar", "data/unlabeled", key="manual_input")
        manual_label = st.text_input("Folder Label", "data/labels", key="manual_output")
        
        if st.button("Muat Gambar untuk Review"):
            if not os.path.exists(manual_input):
                st.error("Folder tidak ditemukan")
                return
                
            images = [f for f in os.listdir(manual_input) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not images:
                st.warning("Tidak ada gambar ditemukan")
                return
                
            st.session_state.images_to_label = images
            st.session_state.current_index = 0
            
        if "images_to_label" in st.session_state:
            img_file = st.session_state.images_to_label[st.session_state.current_index]
            img_path = os.path.join(manual_input, img_file)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(img_path, use_column_width=True)
                
                # Label viewer
                label_path = os.path.join(manual_label, os.path.splitext(img_file)[0] + ".txt")
                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        labels = f.read()
                    st.text_area("Label YOLO", labels, height=150)
                else:
                    st.warning("File label belum ada")
            
            with col2:
                # Navigation controls
                st.write(f"Gambar {st.session_state.current_index + 1}/{len(st.session_state.images_to_label)}")
                
                col_nav = st.columns(3)
                with col_nav[0]:
                    if st.button("⏮️ Prev") and st.session_state.current_index > 0:
                        st.session_state.current_index -= 1
                        st.rerun()
                with col_nav[1]:
                    if st.button("⏭️ Next") and st.session_state.current_index < len(st.session_state.images_to_label) - 1:
                        st.session_state.current_index += 1
                        st.rerun()
                with col_nav[2]:
                    if st.button("🔁 Refresh"):
                        st.rerun()
                
                # Label editor
                new_labels = st.text_area("Edit Label (YOLO format)", height=200)
                if st.button("💾 Simpan Perubahan"):
                    os.makedirs(manual_label, exist_ok=True)
                    with open(label_path, "w") as f:
                        f.write(new_labels)
                    st.success("Label berhasil disimpan!")