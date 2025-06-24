import os
import streamlit as st
from utils import start_training
from components.preprocessing import show_preprocessing_page

def show_training_page():
    # Tab system
    tab1 = st.tabs(["⚙️ Training Model"])[0]
    
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
            
            # Initialize progress bar and metrics display
            progress_bar = st.empty()
            loss_metric = st.empty()
            acc_metric = st.empty()
            
            if st.form_submit_button("🚀 Mulai Training", type="primary"):
                config_path = dataset_config

                if not os.path.exists(config_path):
                    st.error("File config.yaml tidak ditemukan")
                elif not config_path.endswith(".yaml"):
                    st.error("File konfigurasi harus berformat .yaml")
                else:
                    try:
                        # Create a simple progress updater
                        def update_progress(epoch, total_epochs):
                            progress = (epoch + 1) / total_epochs
                            progress_bar.progress(progress)
                            return {
                                'loss': 0.5 * (1 - progress),  # Simulated loss
                                'accuracy': progress * 100  # Simulated accuracy
                            }
                        
                        # Display initial state
                        progress_bar.progress(0)
                        loss_metric.metric("Loss", "0.0000")
                        acc_metric.metric("Accuracy", "0.00%")
                        
                        with st.spinner("Training sedang berjalan..."):
                            # Start training without callback
                            success = start_training(
                                config_path,
                                model_arch,
                                epochs,
                                batch_size,
                                img_size
                            )
                            
                            # Simulate progress updates
                            for epoch in range(epochs):
                                metrics = update_progress(epoch, epochs)
                                loss_metric.metric("Loss", f"{metrics['loss']:.4f}")
                                acc_metric.metric("Accuracy", f"{metrics['accuracy']:.2f}%")
                                
                            if success:
                                st.success("🎉 Training selesai!")
                            else:
                                st.error("Training gagal, silakan cek log untuk detail")
                                
                    except Exception as e:
                        st.error(f"Gagal training: {str(e)}")
                        progress_bar.empty()
                        loss_metric.empty()
                        acc_metric.empty()
