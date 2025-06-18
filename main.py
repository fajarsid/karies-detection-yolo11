import streamlit as st

# ========== Page Configuration ==========
# HARUS menjadi perintah Streamlit pertama
st.set_page_config(
    page_title="YOLO11 | Deteksi Karies",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setelah itu baru import komponen lain
import sys
import config
from components.login import show_login_page
from components.detection import show_detection_page
from components.training import show_training_page
from components.dashboard import show_about_page
from components.preprocessing import show_preprocessing_page 
from components.labeling import show_labeling_page 

# ========== Custom CSS Styling ==========
def setup_app():
    """Setup application configuration and styling"""
    if config.ROOT not in sys.path:
        sys.path.append(str(config.ROOT))
    
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');
        
        /* Base styles */
        * {
            font-family: 'Poppins', sans-serif !important;
        }
        
        /* Main content area */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background-color: #f8f9fa;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #630039 0%, #3a0123 100%) !important;
            color: white !important;
            padding: 1rem !important;
        }
        
        /* Sidebar buttons */
        .stButton button {
            text-align: left !important;
            justify-content: flex-start !important;
            padding-left: 20px !important;
            width: 100%;
            border-radius: 8px;
            border: 1px solid transparent;
            margin: 2px 0 !important;
            padding: 8px 16px !important;
            background-color: rgba(255,255,255,0.1);
            color: white;
            transition: all 0.3s;
        }
        
        .stButton button:hover {
            background-color: rgba(255,255,255,0.2);
            border-color: white;
        }
        
        /* Active menu button */
        .stButton button:focus:not(:active) {
            background-color: rgba(255,255,255,0.3);
            box-shadow: 0 0 0 0.2rem rgba(255,255,255,0.25);
        }
        
        /* Sidebar header */
        .sidebar-header {
            text-align: center;
            margin-bottom: 1.5rem;
            color: white;
        }
        
        /* Footer */
        .sidebar-footer {
            text-align: center;
            margin-top: 1rem;
            color: rgba(255,255,255,0.7);
            font-size: 0.8rem;
        }
        
        /* Card styling */
        .st-emotion-cache-1y4p8pa {
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    setup_app()
    
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # Login check
    if not st.session_state["logged_in"]:
        show_login_page()
        return

    # ========== Sidebar Navigation ==========
    with st.sidebar:
        # st.image("assets/logo.png", width=150)  # Make sure you have this image
        st.markdown("<h2 class='sidebar-header'>YOLO11 Karies</h2>", unsafe_allow_html=True)
        
        menu_options = [
        "Dashboard",
        "Preprocessing",
        "Labeling", 
        "Latih Model",
        "Deteksi Karies Gigi"
        ]
        
        # Create menu buttons
        for option in menu_options:
            if st.button(option, key=f"menu_{option}"):
                st.session_state.menu_selected = option
        
        # Set default menu if not selected
        if "menu_selected" not in st.session_state:
            st.session_state.menu_selected = next(iter(menu_options))
        
        # Spacer and logout button
        st.markdown("<div style='flex:1; min-height: 5vh;'></div>", unsafe_allow_html=True)
        
        if st.button("Logout", type="primary"):
            st.session_state["logged_in"] = False
            st.rerun()
            
        st.markdown("<p class='sidebar-footer'>© 2025 Intan | Teknik Informatika Universitas Nusaputra</p>", 
                   unsafe_allow_html=True)

    # ========== Page Routing ==========
    with st.container():
        st.markdown(f"## {st.session_state.menu_selected}")
        st.markdown("---")
        
        if st.session_state.menu_selected == "Deteksi Karies Gigi":
            show_detection_page()
        elif st.session_state.menu_selected == "Latih Model":
            show_training_page()
        elif st.session_state.menu_selected == "Dashboard":
            show_about_page()
        elif st.session_state.menu_selected == "Preprocessing":
            show_preprocessing_page()
        elif st.session_state.menu_selected == "Labeling":
            show_labeling_page()

if __name__ == "__main__":
    main()