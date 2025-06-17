import streamlit as st

def show_about_page():
    st.markdown("<div class='header-title'>🤖 Deteksi Karies Gigi Otomatis</div>", unsafe_allow_html=True)
    st.markdown("<div class='desc'>Selamat datang di aplikasi deteksi karies gigi menggunakan YOLOv11. Silakan pilih menu di samping untuk memulai deteksi, melatih model, atau mengetahui informasi aplikasi.</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.header("ℹ️ Tentang Aplikasi Ini")
    st.markdown("""
    Aplikasi ini dibuat untuk mendeteksi karies gigi menggunakan algoritma YOLOv11.
    
    **Fitur utama:**
    - Deteksi otomatis dari gambar rontgen
    - Pelatihan model menggunakan dataset sendiri
    - Antarmuka berbasis web untuk dokter gigi

    """)
