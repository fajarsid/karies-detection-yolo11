import os
import streamlit as st
from utils import start_training
from components.preprocessing import show_preprocessing_page

def show_training_page():
    # Tab system
    tab1, tab2 = st.tabs(["⚙️ Training Model", "🛠️ Preprocessing Data"])
    
    with tab1:
        st.markdown("""
        <div style='background-color:#f0f2f6; padding:15px; border-radius:10px; margin-bottom:20px;'>
        <b>Persyaratan:</b>
        <ol>
            <li>Dataset sudah melalui preprocessing</li>
            <li>File konfigurasi dataset (.yaml) valid</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("training_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                dataset_config = st.text_input(
                    "Path config.yaml",
                    "config/karies.yaml",
                    help="File konfigurasi dataset YOLOv11"
                )
                model_arch = st.selectbox(
                    "Arsitektur Model",
                    ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"],
                    index=1
                )
                
            with col2:
                epochs = st.slider("Epochs", 1, 300, 50)
                batch_size = st.selectbox(
                    "Batch Size",
                    [8, 16, 32, 64],
                    index=1
                )
                img_size = st.selectbox(
                    "Image Size",
                    [320, 416, 512, 640],
                    index=3
                )
            
            if st.form_submit_button("🚀 Mulai Training", type="primary"):
                if not os.path.exists(dataset_config):
                    st.error("File config.yaml tidak ditemukan")
                else:
                    try:
                        with st.spinner("Training sedang berjalan..."):
                            # Callback untuk update progress
                            def on_epoch_end(epoch, total_epochs, metrics):
                                progress = (epoch + 1) / total_epochs
                                st.session_state.training_progress = progress
                                st.session_state.training_metrics = metrics
                            
                            start_training(
                                dataset_config,
                                model_arch,
                                epochs,
                                batch_size,
                                img_size,
                                callback=on_epoch_end
                            )
                            st.success("🎉 Training selesai!")
                    except Exception as e:
                        st.error(f"Gagal training: {str(e)}")
        
        # Tampilkan progress training
        if "training_progress" in st.session_state:
            st.progress(st.session_state.training_progress)
            if "training_metrics" in st.session_state:
                metrics = st.session_state.training_metrics
                st.metric("Loss", f"{metrics['loss']:.4f}")
                st.metric("Accuracy", f"{metrics['accuracy']:.2f}%")
    
    with tab2:
        # Gunakan komponen preprocessing yang terpisah
        show_preprocessing_page()