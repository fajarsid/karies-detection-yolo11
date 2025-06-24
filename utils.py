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

        if not os.path.exists(model_path):
            st.error(f"File model tidak ditemukan: {model_path}")
            return None

        return YOLO(Path(model_path))
    except Exception as e:
        st.error(f"Unable to load model. Check path: {model_path}")
        st.error(e)
        return None

def display_detection_results(detection_data, classification_labels):
    """Menampilkan hasil deteksi dalam dua bagian: Ringkasan dan Tabel"""
    print(detection_data is None or len(detection_data) == 0)

    with st.expander("Kesimpulan Hasil Deteksi", expanded=True):
        st.write("#### Ringkasan Deteksi Gigi")
        if detection_data is None or len(detection_data) == 0:
            st.write("Tidak ada objek yang terdeteksi.")
            st.info("Kesimpulan: Model tidak mengidentifikasi adanya gigi atau karies dalam gambar.")
            return

        carious_detections_count = 0
        total_gigi_detections_count = 0

        for box in detection_data:
            class_id = int(box.data[0][5])
            if class_id == 0:
                carious_detections_count += 1
            elif class_id == 1:
                total_gigi_detections_count += 1

        st.write(f"Terdapat **{carious_detections_count}** titik lokasi **{classification_labels[0]}**"
                 f"{f' dari **{total_gigi_detections_count}** **{classification_labels[1]}** yang terdeteksi.' if total_gigi_detections_count > 0 else '.'}")
        st.markdown(f"**Kesimpulan:** Ada **{carious_detections_count}** titik lokasi karies yang diidentifikasi oleh model pada gambar ini.")
        if carious_detections_count > 0:
            st.warning("Disarankan untuk konsultasi lebih lanjut dengan profesional gigi.")
        else:
            st.success("Berdasarkan deteksi model, tidak ada karies yang teridentifikasi pada gambar ini.")

    with st.expander("Detail Hasil Deteksi", expanded=True):
        st.write("#### Objek Terdeteksi")
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
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

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

def validate_training_config(dataset_config):
    """Validasi file YAML YOLO dan pastikan path dataset valid"""
    try:
        with open(dataset_config, "r") as f:
            data_yaml = yaml.safe_load(f)

        base_path = data_yaml.get("path", "")
        train_rel = data_yaml.get("train", "")
        val_rel = data_yaml.get("val", "")

        config_dir = os.path.dirname(os.path.abspath(dataset_config))

        if not base_path:
            base_path_abs = config_dir
        elif not os.path.isabs(base_path):
            base_path_abs = os.path.abspath(os.path.join(config_dir, base_path))
        else:
            base_path_abs = base_path

        train_path = os.path.join(base_path_abs, train_rel) if not os.path.isabs(train_rel) else train_rel
        val_path = os.path.join(base_path_abs, val_rel) if not os.path.isabs(val_rel) else val_rel

        print(f"[VALIDATOR] base_path: {base_path_abs}")
        print(f"[VALIDATOR] train_path: {train_path}")
        print(f"[VALIDATOR] val_path: {val_path}")

        if not os.path.exists(train_path):
            st.error(f"Folder train tidak ditemukan: {train_path}")
            return False
        if not os.path.exists(val_path):
            st.error(f"Folder val tidak ditemukan: {val_path}")
            return False
        return True

    except FileNotFoundError:
        st.error(f"File konfigurasi dataset tidak ditemukan: {dataset_config}")
        return False
    except yaml.YAMLError as e:
        st.error(f"Error membaca YAML: {e}")
        return False
    except Exception as e:
        st.error(f"Terjadi error saat memvalidasi konfigurasi dataset: {e}")
        return False

def start_training(dataset_config, model_arch, epochs, batch, imgsz):
    """Jalankan training YOLO"""
    st.info("Training dimulai...")

    if not validate_training_config(dataset_config):
        return False

    try:
        model = YOLO(model_arch)
        run_name = 'karies_yolo11_web'

        with st.spinner("Sedang melatih model..."):
            results = model.train(
                data=dataset_config,
                epochs=int(epochs),
                imgsz=int(imgsz),
                batch=int(batch),
                project='runs/train',
                name=run_name
            )

        latest_model_path = get_latest_trained_model_path(base_name=run_name)
        if latest_model_path and latest_model_path.exists():
            st.success(f"Training selesai! Model terbaik disimpan di: `{latest_model_path}`")
            st.info("Model deteksi akan otomatis diperbarui untuk sesi berikutnya.")
            return True
        else:
            st.warning("Training selesai! Namun model terbaik tidak ditemukan otomatis.")
            st.info(f"Cek folder `{TRAIN_RUNS_DIR}` untuk melihat hasil.")
            return True

    except Exception as e:
        st.error(f"Terjadi error saat training: {e}")
        return False

def save_yolo_labels(detection_data, label_path):
    """Simpan bounding box ke file label format YOLO"""
    with open(label_path, "w") as f:
        for item in detection_data:
            class_id = item["class_id"]
            x_center = item["x_center"]
            y_center = item["y_center"]
            width = item["width"]
            height = item["height"]
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
