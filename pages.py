# pages.py
import streamlit as st
from PIL import Image
import config
from utils import (
    load_model, display_detection_results, process_image_detection,
    process_video_detection, start_training
)

def render_sidebar():
    """Render sidebar and return configuration"""
    st.sidebar.header("Model Configurations")
    
    model_type = st.sidebar.radio("Task", ["Detection", "Segmentation", "Pose Estimation"])
    confidence_value = float(st.sidebar.slider("Select Model Confidence Value", 25, 100, 40)) / 100
    
    st.sidebar.header("Image/Video Config")
    source_radio = st.sidebar.radio("Select Source", config.SOURCES_LIST)
    
    return {
        "model_type": model_type,
        "confidence": confidence_value,
        "source_type": source_radio
    }

def show_detection_page():
    """Main detection page"""
    st.header("Deteksi Karies Gigi dengan YOLO11")
    
    # Get sidebar config
    sidebar_config = render_sidebar()
    
    # Load model
    model = load_model(sidebar_config["model_type"], config)
    if model is None:
        return
    
    if sidebar_config["source_type"] == "Image":
        show_image_detection(model, sidebar_config["confidence"])
    elif sidebar_config["source_type"] == "Video":
        show_video_detection(model, sidebar_config["confidence"])

def show_image_detection(model, confidence):
    """Handle image detection"""
    # Image upload
    source_image = st.sidebar.file_uploader(
        "Choose an Image....", 
        type=("jpg", "png", "jpeg", "bmp", "webp")
    )
    
    col1, col2 = st.columns(2)
    
    # Display input image
    with col1:
        try:
            if source_image is None:
                st.image(str(config.DEFAULT_IMAGE), caption="Default Image", use_container_width=True)
            else:
                st.image(source_image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error("Error occurred while opening the image")
            st.error(e)
    
    # Process and display detection
    detection_results_data = None
    
    with col2:
        try:
            if source_image is None:
                st.image(str(config.DEFAULT_DETECT_IMAGE), caption="Detected Image", use_container_width=True)
            else:
                if st.sidebar.button("Detect Objects"):
                    uploaded_image = Image.open(source_image)
                    result = process_image_detection(model, uploaded_image, confidence)
                    
                    if result["success"]:
                        st.image(result["plotted_image"], caption="Detected Image", use_container_width=True)
                        detection_results_data = result["detection_data"]
                    else:
                        st.error(f"Detection failed: {result['error']}")
        except Exception as e:
            st.error("Error occurred while processing detection")
            st.error(e)
    
    # Show detection results
    if detection_results_data is not None:
        st.markdown("---")
        display_detection_results(detection_results_data, config.CLASSIFICATION)
        # with st.expander("Detail Hasil Deteksi", expanded=True):

def show_video_detection(model, confidence):
    """Handle video detection"""
    source_video = st.sidebar.selectbox("Choose a Video...", config.VIDEOS_DICT.keys())
    
    # Display video
    with open(config.VIDEOS_DICT.get(source_video), 'rb') as video_file:
        video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)
    
    # Process video
    if st.sidebar.button("Detect Video Objects"):
        process_video_detection(model, config.VIDEOS_DICT.get(source_video), confidence)

def show_training_page():
    """Training page"""
    st.header("Latih Model Deteksi Karies Gigi (YOLO11)")
    st.markdown("""
    Jalankan pelatihan model YOLO11 secara langsung dari web ini.  
    Pastikan dataset dan file konfigurasi sudah tersedia di server.
    """)
    
    with st.form("train_form"):
        dataset_config = st.text_input("Path file konfigurasi dataset (.yaml)", value="karies.yaml")
        model_arch = st.selectbox("Pilih Arsitektur Model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"])
        epochs = st.number_input("Jumlah Epoch", min_value=1, max_value=300, value=100)
        batch = st.number_input("Batch Size", min_value=1, max_value=128, value=16)
        imgsz = st.number_input("Ukuran Gambar", min_value=320, max_value=1280, value=640)
        submit = st.form_submit_button("Mulai Training")
    
    if submit:
        start_training(dataset_config, model_arch, epochs, batch, imgsz)

def show_about_page():
    """About page"""
    st.header("Tentang Aplikasi Deteksi Karies Gigi")
    st.markdown("""
    Aplikasi ini menggunakan model YOLO11 untuk mendeteksi karies pada gigi dari gambar atau video.
    
    **Fitur:**
    - Deteksi karies gigi pada gambar dan video
    - Model dapat dikonfigurasi sesuai kebutuhan
    - Pelatihan model secara langsung dari aplikasi web

    **Pengembangan:**  
    - Dataset dan model dapat diganti sesuai kebutuhan penelitian Anda.
    - Kode dirancang sederhana untuk kemudahan maintenance.
    """)