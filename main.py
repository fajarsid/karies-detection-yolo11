# main.py
import sys
import streamlit as st
import config
from pages import show_detection_page, show_training_page, show_about_page

def setup_app():
    """Setup application configuration"""
    # Add root path to sys.path
    if config.ROOT not in sys.path:
        sys.path.append(str(config.ROOT))
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="YOLO11 | Deteksi Karies",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Custom CSS for better appearance
    st.markdown("""
        <style>
        .main {background-color: #630039;}
        .stSidebar {background-color: #630039;}
        .css-1d391kg {background: #630039;}
        .css-18e3th9 {padding-top: 2rem;}
        .header-title {font-size:2.5rem; font-weight:700; color:#2b4162;}
        .desc {font-size:1.1rem; color:#4b5563;}
        .st-emotion-cache-13k62yr {border-top: 1px solid #fff !important;}
        </style>
    """, unsafe_allow_html=True)

def main():
    """Main application entry point"""
    setup_app()
    
    # Custom sidebar with logo and info
    st.sidebar.image("images/ti.png")
    st.sidebar.markdown("<h2 style='color:#fff;'>YOLO11 Karies</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    menu = st.sidebar.selectbox(
        "Menu", 
        ["Deteksi Karies Gigi", "Latih Model", "Tentang"]
    )

    # Main header and description
    st.markdown("<div class='header-title'>🤖 Deteksi Karies Gigi Otomatis</div>", unsafe_allow_html=True)
    st.markdown("<div class='desc'>Selamat datang di aplikasi deteksi karies gigi menggunakan YOLOv8. Silakan pilih menu di samping untuk memulai deteksi, melatih model, atau mengetahui informasi aplikasi.</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Route to appropriate page
    if menu == "Deteksi Karies Gigi":
        show_detection_page()
    elif menu == "Latih Model":
        show_training_page()
    elif menu == "Tentang":
        show_about_page()

if __name__ == "__main__":
    main()

    st.sidebar.markdown("---")
    st.sidebar.info("Developed by Intan 2025 | Teknik Informatika Universitas Nusaputra")