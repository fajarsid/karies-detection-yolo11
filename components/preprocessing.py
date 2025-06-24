import os
import shutil
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from pathlib import Path

def show_preprocessing_page():
    
    # Membuat kolom tampilan
    kolom_config, kolom_visual = st.columns([1, 2])
    
    with kolom_config:
        st.subheader("Pengaturan")
        folder_input = st.text_input("Folder Input", "dataset_raw")
        folder_output = st.text_input("Folder Output", "datasets/karies")
        ukuran_gambar = st.selectbox("Ukuran Gambar", [320, 416, 512, 640], index=3)
        
        st.subheader("Peningkatan Kualitas Gambar")
        perbaikan_ketajaman = st.checkbox("Perbaikan Ketajaman", True)
        penghilangan_noise = st.checkbox("Penghilangan Noise", True)
        peningkatan_kontras = st.checkbox("Peningkatan Kontras", False)
        
        if st.button("🚀 Mulai Preprocessing", type="primary"):
            proses_dataset(folder_input, folder_output, ukuran_gambar, kolom_visual,
                         perbaikan_ketajaman, penghilangan_noise, peningkatan_kontras)
    
    with kolom_visual:
        st.subheader("Visualisasi Proses")
        if not st.session_state.get('sedang_proses'):
            st.info("Atur konfigurasi dan jalankan preprocessing untuk melihat visualisasi")

def perbaikan_gambar(img, ketajaman=True, noise=True, kontras=False):
    """Menerapkan teknik peningkatan kualitas gambar"""
    # Konversi dari PIL Image ke format OpenCV (BGR)
    img_cv = np.array(img)[:, :, ::-1].copy()  # RGB ke BGR
    
    if noise:
        img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)
    
    if ketajaman:
        kernel = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]])
        img_cv = cv2.filter2D(img_cv, -1, kernel)
    
    if kontras:
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        img_cv = cv2.merge((l,a,b))
    
    # Konversi kembali ke PIL Image
    return Image.fromarray(img_cv[:, :, ::-1])  # BGR ke RGB

def proses_dataset(folder_input, folder_output, ukuran_gambar, kolom_visual,
                  ketajaman, noise, kontras):
    path_input = Path(folder_input)
    path_output = Path(folder_output)
    
    # Inisialisasi status proses
    st.session_state.sedang_proses = True
    
    try:
        # Setup progress tracking
        progress_bar = kolom_visual.progress(0)
        status_text = kolom_visual.empty()
        tempat_gambar = kolom_visual.empty()
        tempat_metrics = kolom_visual.empty()
        
        total_file = 0
        file_terproses = 0
        
        # Hitung total file dulu
        for bagian in ['train', 'valid']:
            dir_gambar = path_input / bagian / 'images'
            total_file += len(list(dir_gambar.glob("*.[jJpP]*[gG]")))
        
        # Proses file
        for bagian in ['train', 'valid']:
            dir_gambar = path_input / bagian / 'images'
            dir_label = path_input / bagian / 'labels'
            
            # Buat folder output
            (path_output / bagian / 'images').mkdir(parents=True, exist_ok=True)
            (path_output / bagian / 'labels').mkdir(parents=True, exist_ok=True)
            
            for file_gambar in dir_gambar.glob("*.[jJpP]*[gG]"):
                try:
                    file_terproses += 1
                    progress = file_terproses / total_file
                    
                    # Update UI
                    progress_bar.progress(progress)
                    status_text.text(f"🔧 Memproses {bagian}: {file_gambar.name}")
                    
                    # Proses dan tampilkan gambar
                    with Image.open(file_gambar) as img:
                        # Tampilkan sebelum/sesudah
                        kolom = tempat_gambar.columns(2)
                        kolom[0].image(img, caption="Asli", use_column_width=True)
                        
                        # Lakukan perbaikan gambar
                        img_diperbaiki = perbaikan_gambar(img, ketajaman, noise, kontras)
                        img_resize = img_diperbaiki.resize((ukuran_gambar, ukuran_gambar))
                        
                        kolom[1].image(img_resize, 
                                     caption=f"Hasil Perbaikan {ukuran_gambar}px", 
                                     use_column_width=True)
                        
                        # Simpan gambar yang sudah diproses
                        img_resize.save(path_output / bagian / 'images' / file_gambar.name)
                    
                    # Salin label yang sesuai
                    file_label = dir_label / f"{file_gambar.stem}.txt"
                    if file_label.exists():
                        shutil.copy(file_label, path_output / bagian / 'labels' / file_label.name)
                    
                    # Update metrics
                    tempat_metrics.markdown(f"""
                    **Status Proses**
                    - File diproses: `{file_terproses}/{total_file}`
                    - Bagian: `{bagian}`
                    - Ukuran Gambar: `{ukuran_gambar}x{ukuran_gambar}`
                    """)
                    
                except Exception as e:
                    kolom_visual.error(f"⚠️ Gagal memproses {file_gambar.name}: {str(e)}")
        
        kolom_visual.success("✔️ Preprocessing selesai dengan sukses!")
        
    except Exception as e:
        kolom_visual.error(f"❌ Error fatal: {str(e)}")
    finally:
        st.session_state.sedang_proses = False
        progress_bar.empty()
        status_text.empty()