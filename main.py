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
        page_icon="🤖"
    )

def main():
    """Main application entry point"""
    setup_app()
    
    # Sidebar menu
    menu = st.sidebar.selectbox(
        "Menu", 
        ["Deteksi Karies Gigi", "Latih Model", "Tentang"]
    )
    
    # Route to appropriate page
    if menu == "Deteksi Karies Gigi":
        show_detection_page()
    elif menu == "Latih Model":
        show_training_page()
    elif menu == "Tentang":
        show_about_page()

if __name__ == "__main__":
    main()