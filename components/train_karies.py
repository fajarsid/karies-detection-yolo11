from ultralytics import YOLO

# Path ke file konfigurasi dataset
DATASET_CONFIG = 'config/karies.yaml'  # pastikan file ini sudah ada dan benar

# Pilih model YOLO11 yang akan dilatih (bisa yolo11n.pt, yolo11s.pt, dst)
MODEL = 'yolo11n.pt'

# Mulai training
if __name__ == "__main__":
    model = YOLO(MODEL)
    model.train(
        data=DATASET_CONFIG,
        epochs=100,           # jumlah epoch, bisa disesuaikan
        imgsz=640,            # ukuran gambar
        batch=16,             # batch size, sesuaikan dengan GPU
        project='runs/train',
        name='karies_yolo11'
    )
