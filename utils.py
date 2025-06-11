# utils.py
import cv2
import streamlit as st
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import os
import yaml
import pandas as pd
import re 

from config import TRAIN_RUNS_DIR, get_latest_trained_model_path 

def load_model(model_type, config):
    """Load YOLO model based on type"""
    try:
        if model_type == 'Detection':
            model_path = config.DETECTION_MODEL
        elif model_type == 'Segmentation':
            model_path = config.SEGMENTATION_MODEL
        elif model_type == 'Pose Estimation':
            model_path = config.POSE_ESTIMATION_MODEL
        
        return YOLO(Path(model_path))
    except Exception as e:
        st.error(f"Unable to load model. Check path: {model_path}")
        st.error(e)
        return None

def display_detection_results(detection_data, classification_labels):
    """Menampilkan hasil deteksi dalam dua bagian yang terpisah: Ringkasan dan Tabel Detail."""

    # --- Bagian 1: Ringkasan Deteksi Gigi ---
    with st.expander("Kesimpulan Hasil Deteksi", expanded=True): 
        st.write("#### Ringkasan Deteksi Gigi") # Judul untuk bagian ringkasan

        if detection_data is None or len(detection_data) == 0:
            st.write("Tidak ada objek yang terdeteksi.")
            st.info("Kesimpulan: Model tidak mengidentifikasi adanya gigi atau karies dalam gambar.")
            return
        
        # Hitung jumlah deteksi untuk masing-masing kelas
        carious_detections_count = 0
        total_gigi_detections_count = 0 
        
        # Iterasi data deteksi untuk menghitung
        for box in detection_data:
            class_id = int(box.data[0][5]) # Ambil class_id dari tensor box.data
            
            # Asumsi: class_id 0 = Karies, class_id 1 = Gigi
            if class_id == 0: # Jika ini adalah class_id untuk 'Karies'
                carious_detections_count += 1
            elif class_id == 1: # Jika ini adalah class_id untuk 'Gigi'
                total_gigi_detections_count += 1
                
        # Tampilkan Kesimpulan
        if total_gigi_detections_count > 0:
            st.write(f"Terdapat **{carious_detections_count}** titik lokasi **{classification_labels[0]}** dari **{total_gigi_detections_count}** **{classification_labels[1]}** yang terdeteksi.")
            
            st.markdown(f"**Kesimpulan:** Ada **{carious_detections_count}** titik lokasi karies yang diidentifikasi oleh model pada gambar ini.")
            
            if carious_detections_count > 0:
                st.warning("Disarankan untuk konsultasi lebih lanjut dengan profesional gigi.")
            else:
                st.success("Berdasarkan deteksi model, tidak ada karies yang teridentifikasi pada gambar ini.")
        else:
            st.info("Model tidak mengidentifikasi adanya gigi dalam gambar ini untuk dianalisis.")

    # --- Bagian 2: Detail Hasil Deteksi ---
    # Bagian ini tetap di dalam expander, yang secara implisit berfungsi sebagai container
    with st.expander("Detail Hasil Deteksi", expanded=True): 
        st.write("#### Objek Terdeteksi") # Judul untuk tabel detail
        
        table_data = []
        for i, box in enumerate(detection_data):
            data = box.data[0].tolist()
            x1, y1, x2, y2, conf, class_id = data
            
            class_name = classification_labels[int(class_id)]
            
            row = {
                "Deteksi": f"#{i+1}",
                "Kelas": class_name,
                "Keyakinan": f"{conf:.2f} ({conf*100:.0f}%)",
                "Posisi (x1, y1, x2, y2)": f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
            }
            table_data.append(row)
        
        st.dataframe(pd.DataFrame(table_data), use_container_width=True) # Menampilkan data dalam DataFrame

def process_image_detection(model, image, confidence):
    """Process single image detection"""
    try:
        result = model.predict(image, conf=confidence)
        boxes = result[0].boxes
        result_plotted = result[0].plot()[:, :, ::-1]
        
        return {
            "success": True,
            "plotted_image": result_plotted,
            "detection_data": boxes
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def process_video_detection(model, video_path, confidence):
    """Process video detection with real-time display"""
    try:
        video_cap = cv2.VideoCapture(str(video_path))
        st_frame = st.empty()
        
        while video_cap.isOpened():
            success, image = video_cap.read()
            if success:
                image = cv2.resize(image, (720, int(720 * (9/16))))
                result = model.predict(image, conf=confidence)
                result_plotted = result[0].plot()
                
                st_frame.image(
                    result_plotted, 
                    caption="Detected Video",
                    channels="BGR",
                    use_container_width=True
                )
            else:
                video_cap.release()
                break
    except Exception as e:
        st.sidebar.error(f"Error Loading Video: {str(e)}")

def validate_training_config(dataset_config):
    """Validate training configuration"""
    try:
        with open(dataset_config, "r") as f:
            data_yaml = yaml.safe_load(f)
        train_path = data_yaml.get("train") # Gunakan .get() untuk keamanan
        val_path = data_yaml.get("val") # Gunakan .get()

        if not train_path or not os.path.exists(train_path):
            st.error(f"Folder train tidak ditemukan atau path tidak valid di {dataset_config}: {train_path}")
            return False
        if not val_path or not os.path.exists(val_path):
            st.error(f"Folder val tidak ditemukan atau path tidak valid di {dataset_config}: {val_path}")
            return False
        return True
    except FileNotFoundError:
        st.error(f"File konfigurasi dataset tidak ditemukan: {dataset_config}")
        return False
    except yaml.YAMLError as e:
        st.error(f"Error parsing file YAML: {e}")
        return False
    except Exception as e:
        st.error(f"Terjadi error saat memvalidasi konfigurasi dataset: {e}")
        return False

def start_training(dataset_config, model_arch, epochs, batch, imgsz):
    """Start model training"""
    st.info("Training dimulai...")
    
    if not validate_training_config(dataset_config):
        return
    
    try:
        model = YOLO(model_arch)
        run_name = 'karies_yolo11_web' 
        
        with st.spinner("Sedang melatih model..."):
            results = model.train(
                data=dataset_config,
                epochs=int(epochs),
                imgsz=int(imgsz),
                batch=int(batch),
                project=str(TRAIN_RUNS_DIR.parent),
                name=run_name
            )
        latest_model_path = get_latest_trained_model_path(base_name=run_name)
        if latest_model_path and latest_model_path.exists():
            st.success(f"Training selesai! Model terbaik disimpan di: `{latest_model_path}`")
            st.info("Model deteksi akan otomatis diperbarui untuk sesi berikutnya.")
        else:
            st.success("Training selesai! Namun path model terbaik tidak dapat ditemukan secara otomatis.")
            st.info(f"Cek folder `{TRAIN_RUNS_DIR}` untuk hasilnya.")
        
    except Exception as e:
        st.error(f"Terjadi error saat training: {e}")