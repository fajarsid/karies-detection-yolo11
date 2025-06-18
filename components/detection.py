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
    
    # Validasi model terlebih dahulu
    if model is None:
        st.error("Model tidak berhasil dimuat. Silakan coba lagi atau hubungi administrator.")
        return

    source_image = st.file_uploader(
        "Unggah file (jpg/png/jpeg/bmp/webp)",
        type=("jpg", "png", "jpeg", "bmp", "webp"),
        accept_multiple_files=False
    )

    col1, col2 = st.columns(2)
    detection_results_data = None

    with col1:
        try:
            default_img = Image.open(config.DEFAULT_IMAGE) if isinstance(config.DEFAULT_IMAGE, str) else config.DEFAULT_IMAGE
            if source_image is None:
                st.image(default_img, caption="Gambar Default", use_column_width=True)
            else:
                uploaded_img = Image.open(source_image)
                st.image(uploaded_img, caption="Gambar Diunggah", use_column_width=True)
        except Exception as e:
            st.error(f"Error memuat gambar: {str(e)}")
            st.stop()

    with col2:
        if source_image:
            if st.button("🔍 Deteksi Karies Gigi", type="primary"):
                with st.spinner("Memproses deteksi..."):
                    try:
                        result = process_image_detection(model, uploaded_img, confidence)
                        if result["success"]:
                            st.image(result["plotted_image"], caption="Hasil Deteksi", use_column_width=True)
                            detection_results_data = result["detection_data"]
                            st.success("Deteksi berhasil!")
                        else:
                            st.error(f"Deteksi gagal: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat proses deteksi: {str(e)}")
                        st.stop()
        else:
            default_detect = Image.open(config.DEFAULT_DETECT_IMAGE) if isinstance(config.DEFAULT_DETECT_IMAGE, str) else config.DEFAULT_DETECT_IMAGE
            st.image(default_detect, caption="Deteksi Default", use_column_width=True)

    if detection_results_data:
        st.markdown("---")
        with st.expander("📊 Detail Hasil Deteksi", expanded=True):
            display_detection_results(detection_results_data, config.CLASSIFICATION)