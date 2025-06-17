from components.login import show_login_page
import streamlit as st
st.set_page_config(
    page_title="YOLO11 | Deteksi Karies",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


import sys
import config
from components.detection import show_detection_page
from components.training import show_training_page
from components.about import show_about_page
from components.preprocessing import show_preprocessing_page 
from components.labeling import show_labeling_page 


def setup_app():

    
    """Setup application configuration"""
    if config.ROOT not in sys.path:
        sys.path.append(str(config.ROOT))
    
    # Optional: Tambahkan gaya khusus
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
    setup_app()
    
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # Jika belum login, tampilkan login page dan hentikan eksekusi lebih lanjut
    if not st.session_state["logged_in"]:
        show_login_page()
        return

    # Jika sudah login, tampilkan sidebar menu dan tombol logout
    with st.sidebar:
        st.markdown("<h2 style='color:#fff; text-align:center;'>YOLO11 Karies</h2>", unsafe_allow_html=True)
        menu = st.radio("Menu", [
            "Tentang", 
            "Preprocessing",
            "Labeling", 
            "Latih Model", 
            "Deteksi Karies Gigi"
        ], key="main_menu")
        st.markdown("---")
        st.caption("© 2025 Intan | Teknik Informatika Universitas Nusaputra")
        # Pindahkan tombol Logout ke bagian paling bawah sidebar agar selalu tampil
        logout_clicked = st.button("Logout", key="logout_btn")

    # Logout harus dicek di luar with agar selalu dieksekusi
    if st.session_state.get("logged_in") and logout_clicked:
        st.session_state["logged_in"] = False
        st.rerun()

    # Routing halaman (jika sudah login)
    if menu == "Deteksi Karies Gigi":
        show_detection_page()
    elif menu == "Latih Model":
        show_training_page()
    elif menu == "Tentang":
        show_about_page()
    elif menu == "Preprocessing":
        show_preprocessing_page()
    elif menu == "Labeling":
        show_labeling_page()

if __name__ == "__main__":
    main()