# Aplikasi Deteksi Karies Gigi menggunakan YOLO11
# Fitur: Deteksi karies pada gambar/video, pelatihan model, dan pengaturan model via Streamlit

#Import All the Required Libraries
import cv2
import streamlit as st
from pathlib import Path
import sys
from ultralytics import YOLO
from PIL import Image

#Get the absolute path of the current file
FILE = Path(__file__).resolve()

#Get the parent directory of the current file
ROOT = FILE.parent

#Add the root path to the sys.path list
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

#Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

#Sources
IMAGE = 'Image'
VIDEO = 'Video'

SOURCES_LIST = [IMAGE, VIDEO]

#Image Config
IMAGES_DIR = ROOT/'images'
DEFAULT_IMAGE = IMAGES_DIR/'image1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR/'detectedimage1.jpg'

#Videos Config
VIDEO_DIR = ROOT/'videos'
VIDEOS_DICT = {
    'video 1': VIDEO_DIR/'video1.mp4',
    'video 2': VIDEO_DIR/'video2.mp4'
}

#Model Configurations
MODEL_DIR = ROOT/'weights'
DETECTION_MODEL = MODEL_DIR/'yolo11n.pt'

#In case of your custom model
#DETECTION_MODEL = MODEL_DIR/'custom_model_weight.pt'

SEGMENTATION_MODEL  = MODEL_DIR/'yolo11n-seg.pt'

POSE_ESTIMATION_MODEL = MODEL_DIR/'yolo11n-pose.pt'

#Page Layout
st.set_page_config(
    page_title = "YOLO11",
    page_icon = "🤖"
)

# Sidebar Menu
menu = st.sidebar.selectbox(
    "Menu", 
    ["Deteksi Karies Gigi", "Latih Model", "Tentang"]
)

if menu == "Deteksi Karies Gigi":
    st.header("Deteksi Karies Gigi dengan YOLO11")
    st.sidebar.header("Model Configurations")

    #Choose Model: Detection, Segmentation or Pose Estimation
    model_type = st.sidebar.radio("Task", ["Detection", "Segmentation", "Pose Estimation"])

    #Select Confidence Value
    confidence_value = float(st.sidebar.slider("Select Model Confidence Value", 25, 100, 40))/100

    #Selecting Detection, Segmentation, Pose Estimation Model
    if model_type == 'Detection':
        model_path = Path(DETECTION_MODEL)
    elif model_type == 'Segmentation':
        model_path = Path(SEGMENTATION_MODEL)
    elif model_type ==  'Pose Estimation':
        model_path = Path(POSE_ESTIMATION_MODEL)

    #Load the YOLO Model
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Unable to load model. Check the sepcified path: {model_path}")
        st.error(e)

    #Image / Video Configuration
    st.sidebar.header("Image/Video Config")
    source_radio = st.sidebar.radio(
        "Select Source", SOURCES_LIST
    )

    source_image = None
    if source_radio == IMAGE:
        source_image = st.sidebar.file_uploader(
            "Choose an Image....", type = ("jpg", "png", "jpeg", "bmp", "webp")
        )
        col1, col2 = st.columns(2)
        with col1:
            try:
                if source_image is None:
                    default_image_path = str(DEFAULT_IMAGE)
                    default_image = Image.open(default_image_path)
                    st.image(default_image_path, caption = "Default Image", use_column_width=True)
                else:
                    uploaded_image  =Image.open(source_image)
                    st.image(source_image, caption = "Uploaded Image", use_column_width = True)
            except Exception as e:
                st.error("Error Occured While Opening the Image")
                st.error(e)
        with col2:
            try:
                if source_image is None:
                    default_detected_image_path = str(DEFAULT_DETECT_IMAGE)
                    default_detected_image = Image.open(default_detected_image_path)
                    st.image(default_detected_image_path, caption = "Detected Image", use_column_width = True)
                else:
                    if st.sidebar.button("Detect Objects"):
                        result = model.predict(uploaded_image, conf = confidence_value)
                        boxes = result[0].boxes
                        result_plotted = result[0].plot()[:,:,::-1]
                        st.image(result_plotted, caption = "Detected Image", use_column_width = True)

                        try:
                            with st.expander("Detection Results"):
                                for box in boxes:
                                    st.write(box.data)
                        except Exception as e:
                            st.error(e)
            except Exception as e:
                st.error("Error Occured While Opening the Image")
                st.error(e)

    elif source_radio == VIDEO:
        source_video = st.sidebar.selectbox(
            "Choose a Video...", VIDEOS_DICT.keys()
        )
        with open(VIDEOS_DICT.get(source_video), 'rb') as video_file:
            video_bytes = video_file.read()
            if video_bytes:
                st.video(video_bytes)
            if st.sidebar.button("Detect Video Objects"):
                try:
                    video_cap = cv2.VideoCapture(
                        str(VIDEOS_DICT.get(source_video))
                    )
                    st_frame = st.empty()
                    while (video_cap.isOpened()):
                        success, image = video_cap.read()
                        if success:
                            image = cv2.resize(image, (720, int(720 * (9/16))))
                            #Predict the objects in the image using YOLO11
                            result = model.predict(image, conf = confidence_value)
                            #Plot the detected objects on the video frame
                            result_plotted = result[0].plot()
                            st_frame.image(result_plotted, caption = "Detected Video",
                                           channels = "BGR",
                                           use_column_width=True)
                        else:
                            video_cap.release()
                            break
                except Exception as e:
                    st.sidebar.error("Error Loading Video"+str(e))

elif menu == "Latih Model":
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
        import os
        import yaml
        st.info("Training dimulai...")
        # Cek file yaml dan folder dataset
        try:
            with open(dataset_config, "r") as f:
                data_yaml = yaml.safe_load(f)
            train_path = data_yaml["train"]
            val_path = data_yaml["val"]
            if not os.path.exists(train_path):
                st.error(f"Folder train tidak ditemukan: {train_path}")
            elif not os.path.exists(val_path):
                st.error(f"Folder val tidak ditemukan: {val_path}")
            else:
                from ultralytics import YOLO
                model = YOLO(model_arch)
                with st.spinner("Sedang melatih model..."):
                    results = model.train(
                        data=dataset_config,
                        epochs=int(epochs),
                        imgsz=int(imgsz),
                        batch=int(batch),
                        project='runs/train',
                        name='karies_yolo11_web'
                    )
                st.success("Training selesai! Cek folder runs/train/karies_yolo11_web untuk hasilnya.")
        except Exception as e:
            st.error(f"Terjadi error saat training: {e}")

elif menu == "Tentang":
    st.header("Tentang Aplikasi Deteksi Karies Gigi")
    st.markdown("""
    Aplikasi ini menggunakan model YOLO11 untuk mendeteksi karies pada gigi dari gambar atau video.
    
    **Fitur:**
    - Deteksi karies gigi pada gambar dan video
    - Model dapat dikonfigurasi sesuai kebutuhan

    **Pengembangan:**  
    - Dataset dan model dapat diganti sesuai kebutuhan penelitian Anda.
    """)