import streamlit as st
from PIL import Image
import config
from utils import (
    load_model, display_detection_results, process_image_detection
)

def render_page_config():
    st.subheader("Konfigurasi Model")
    model_type = "Detection"
    confidence_value = float(st.slider("Tingkat Kepercayaan", 25, 100, 40)) / 100
    source_radio = config.SOURCES_LIST[0]
    return {
        "model_type": model_type,
        "confidence": confidence_value,
        "source_type": source_radio
    }

def show_detection_page():
    st.header("Deteksi Karies Gigi dengan YOLO11")

    # Konfigurasi model
    config_data = render_page_config()

    # Load model
    model = load_model(config_data["model_type"], config)
    if model is None:
        return

    # Jalankan deteksi gambar
    if config_data["source_type"] == "Image":
        show_image_detection(model, config_data["confidence"])

def show_image_detection(model, confidence):
    st.subheader("Unggah Gambar Gigi")

    source_image = st.file_uploader(
        "Unggah file (jpg/png/jpeg/bmp/webp)",
        type=("jpg", "png", "jpeg", "bmp", "webp")
    )

    detection_results_data = None
    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_image is None:
                st.image(str(config.DEFAULT_IMAGE), caption="Gambar Default", use_container_width=True)
            else:
                st.image(source_image, caption="Gambar Diunggah", use_container_width=True)
        except Exception as e:
            st.error("Gagal memuat gambar.")
            st.error(e)

    with col2:
        try:
            if source_image is None:
                st.image(str(config.DEFAULT_DETECT_IMAGE), caption="Deteksi Default", use_container_width=True)
            else:
                if st.button("Deteksi Karies Gigi"):
                    uploaded_image = Image.open(source_image)
                    result = process_image_detection(model, uploaded_image, confidence)
                    if result["success"]:
                        st.image(result["plotted_image"], caption="Hasil Deteksi", use_container_width=True)
                        detection_results_data = result["detection_data"]
                    else:
                        st.error(f"Deteksi gagal: {result['error']}")
        except Exception as e:
            st.error("Terjadi kesalahan saat proses deteksi.")
            st.error(e)

    if detection_results_data is not None:
        st.markdown("---")
        display_detection_results(detection_results_data, config.CLASSIFICATION)
