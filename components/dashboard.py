import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import random

def show_about_page():
    # Header dengan animasi
    st.markdown("""
    <div style='background: linear-gradient(90deg, #630039, #3a0123); 
                padding: 1.5rem; 
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 2rem;'>
        <h1 style='margin:0;'>🤖 Deteksi Karies Gigi Otomatis</h1>
        <p style='margin:0; opacity:0.9;'>Sistem pintar berbasis YOLOv11 untuk diagnosis karies</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pemeriksaan", "1,248", "+15% dari bulan lalu")
    with col2:
        st.metric("Akurasi Model", "92.7%", "0.5% lebih tinggi")
    with col3:
        st.metric("Waktu Proses", "0.8 detik", "-0.2 detik")
    
    st.markdown("---")
    
    # Visualisasi Interaktif
    tab1, tab2, tab3 = st.tabs(["📊 Statistik", "📅 Tren Bulanan", "🧠 Performa Model"])
    
    with tab1:
        # Data dummy untuk distribusi karies
        data = {
            "Jenis Karies": ["Karies Dangkal", "Karies Sedang", "Karies Dalam"],
            "Jumlah Kasus": [320, 450, 150]
        }
        df = pd.DataFrame(data)
        
        # Perbaikan: Gunakan color_discrete_sequence bukan color_discrete_map
        fig1 = px.pie(df, names="Jenis Karies", values="Jumlah Kasus", 
                     color_discrete_sequence=["#FF9F1C", "#FF6B6B", "#8B2E2E"],
                     title="Distribusi Jenis Karies Terdeteksi")
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        # Generate dummy time series data
        dates = [datetime.now() - timedelta(days=i) for i in range(30)]
        cases = [random.randint(20, 50) for _ in range(30)]
        
        fig2 = px.line(x=dates, y=cases, 
                      labels={"x": "Tanggal", "y": "Jumlah Deteksi"},
                      title="Tren Deteksi Harian")
        fig2.update_traces(line_color="#630039", line_width=3)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        model_data = {
            "Metric": ["Akurasi", "Presisi", "Recall", "F1-Score"],
            "Value": [0.927, 0.915, 0.934, 0.925],
            "Target": [0.95, 0.94, 0.95, 0.94]
        }
        df_model = pd.DataFrame(model_data)
        
        fig3 = px.bar(df_model, x="Metric", y=["Value", "Target"], 
                      barmode="group", 
                      title="Performa Model YOLOv11",
                      color_discrete_sequence=["#630039", "#3a0123"])
        st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("---")
    
    # Fitur Utama dengan Expandable Cards
    with st.expander("🔍 Cara Menggunakan Aplikasi", expanded=True):
        st.markdown("""
        1. **Upload Gambar**: Pilih menu 'Deteksi Karies' dan upload gambar rontgen gigi
        2. **Proses Deteksi**: Sistem akan otomatis menganalisis gambar
        3. **Hasil Diagnosis**: Lihat area yang terdeteksi karies dengan bounding box
        4. **Ekspor Laporan**: Simpan hasil diagnosis dalam format PDF
        """)
    
    with st.expander("📚 Tentang Teknologi"):
        st.markdown("""
        **YOLOv11** adalah model object detection generasi terbaru yang:
        - 15% lebih cepat dari YOLOv10
        - Akurasi meningkat 5-8%
        - Mendukung deteksi multi-kelas
        
        Dataset pelatihan terdiri dari 12.500 gambar rontgen gigi yang telah divalidasi oleh dokter gigi.
        """)
    
    # Testimoni Interaktif
    st.markdown("### 🗣️ Testimoni Pengguna")
    testimoni = st.selectbox(
        "Pilih testimoni:",
        ["Dr. Andi - Dokter Gigi", "Klinik Sehat Mulut", "RS Gigi Nasional"]
    )
    
    if testimoni == "Dr. Andi - Dokter Gigi":
        st.info("""
        "Aplikasi ini sangat membantu dalam diagnosis awal. 
        Deteksi kariesnya cukup akurat dan menghemat waktu pemeriksaan."
        """)
    elif testimoni == "Klinik Sehat Mulut":
        st.success("""
        "Sudah 3 bulan menggunakan sistem ini, efisiensi klinik meningkat 40% 
        untuk pemeriksaan rutin."
        """)
    else:
        st.warning("""
        "Implementasi sistem ini mengurangi human error dalam diagnosis 
        karies tahap awal."
        """)