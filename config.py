# config.py
from pathlib import Path
import re

# Setup paths
FILE = Path(__file__).resolve()
ROOT = FILE.parent
# ROOT = ROOT.relative_to(Path.cwd())
ROOT = Path(__file__).resolve().parent

# Directories
IMAGES_DIR = ROOT / 'images'
VIDEO_DIR = ROOT / 'videos'
MODEL_DIR = ROOT / 'weights'

# Default files
DEFAULT_IMAGE = IMAGES_DIR / 'image1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'detectedimage1.jpg'

# Base path for trained models
TRAIN_RUNS_DIR = ROOT / 'runs' / 'train'

# --- Fungsi untuk mendapatkan path model terbaik terbaru secara dinamis ---
def get_latest_trained_model_path(base_name='karies_yolo11_web'):
    """
    Mencari folder training terbaru (berdasarkan nama dan angka tertinggi)
    dan mengembalikan path ke best.pt di dalamnya.
    """
    latest_run_path = None
    latest_run_number = -1 # Untuk mencari angka tertinggi

    if not TRAIN_RUNS_DIR.exists():
        return None # Folder runs/train belum ada

    # Pola regex untuk mencari folder 'karies_yolo11_web' diikuti dengan angka opsional
    # Misalnya: karies_yolo11_web, karies_yolo11_web1, karies_yolo11_web2
    pattern = re.compile(rf"^{re.escape(base_name)}(\d*)$")

    for folder in TRAIN_RUNS_DIR.iterdir():
        if folder.is_dir():
            match = pattern.match(folder.name)
            if match:
                # Ambil nomor di akhir nama folder, atau 0 jika tidak ada nomor (e.g., 'karies_yolo11_web')
                current_run_number = int(match.group(1)) if match.group(1) else 0
                
                if current_run_number > latest_run_number:
                    latest_run_number = current_run_number
                    latest_run_path = folder / 'weights' / 'best.pt'
                    
                    # Cek apakah best.pt benar-benar ada di folder ini
                    if not latest_run_path.exists():
                        latest_run_path = None # Reset jika best.pt tidak ada
    
    return latest_run_path

# Model paths
DETECTION_MODEL = get_latest_trained_model_path() or (MODEL_DIR / 'default_yolo_detection.pt') 
SEGMENTATION_MODEL = MODEL_DIR / 'yolo11n-seg.pt'
POSE_ESTIMATION_MODEL = MODEL_DIR / 'yolo11n-pose.pt'

# Video files
VIDEOS_DICT = {
    'video 1': VIDEO_DIR / 'video1.mp4',
    'video 2': VIDEO_DIR / 'video2.mp4'
}

# Constants
CLASSIFICATION = ["karies", "gigi"]
SOURCES_LIST = ['Image', 'Video']
