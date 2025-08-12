#!/bin/bash
# GCP VM 인스턴스 내부에서 실행할 환경 설정 스크립트
# Training Environment Setup Script (Run inside VM)

echo "========================================"
echo "🔧 Sports AI 학습 환경 설치"
echo "========================================"

# 1. 시스템 업데이트
echo "1️⃣ 시스템 업데이트..."
sudo apt-get update && sudo apt-get upgrade -y

# 2. CUDA 확인 (Deep Learning VM은 이미 설치됨)
echo "2️⃣ CUDA 상태 확인..."
nvidia-smi
nvcc --version

# 3. Python 환경 설정
echo "3️⃣ Python 환경 설정..."
conda create -n sports_ai python=3.9 -y
conda activate sports_ai

# 4. PyTorch 설치 (CUDA 11.8)
echo "4️⃣ PyTorch 및 딥러닝 라이브러리 설치..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 5. 필수 라이브러리 설치
echo "5️⃣ 필수 라이브러리 설치..."
pip install -r requirements_gcp.txt

# requirements_gcp.txt 생성
cat > requirements_gcp.txt << EOF
# Deep Learning
tensorflow==2.13.0
transformers==4.35.0
accelerate==0.24.0
timm==0.9.10

# Computer Vision
opencv-python==4.8.1.78
mediapipe==0.10.7
albumentations==1.3.1
scikit-image==0.22.0

# Data Processing
numpy==1.24.3
pandas==2.1.3
scipy==1.11.4
scikit-learn==1.3.2

# Monitoring & Logging
tensorboard==2.13.0
wandb==0.16.0
tqdm==4.66.1
matplotlib==3.8.1
seaborn==0.13.0

# Video Processing
yt-dlp==2023.11.16
moviepy==1.0.3
imageio==2.33.0
imageio-ffmpeg==0.4.9

# Cloud Storage
google-cloud-storage==2.10.0
google-cloud-aiplatform==1.36.4

# Optimization
tensorrt==8.6.1
onnx==1.15.0
onnxruntime-gpu==1.16.3

# Utils
pyyaml==6.0.1
click==8.1.7
python-dotenv==1.0.0
EOF

pip install -r requirements_gcp.txt

# 6. Jupyter 설정
echo "6️⃣ Jupyter Notebook 설정..."
pip install jupyter jupyterlab

# Jupyter 설정 파일 생성
jupyter notebook --generate-config
cat >> ~/.jupyter/jupyter_notebook_config.py << EOF
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.allow_root = True
c.NotebookApp.token = ''
c.NotebookApp.password = ''
EOF

# 7. 프로젝트 코드 클론
echo "7️⃣ 프로젝트 코드 다운로드..."
cd ~
git clone https://github.com/yourusername/HealthcareAI.git
cd HealthcareAI

# 8. GCS 데이터 다운로드
echo "8️⃣ 학습 데이터 다운로드..."
mkdir -p data
gsutil -m cp -r gs://sports-ai-data-bucket/collected_data/* ./data/

# 9. 모니터링 스크립트 설정
echo "9️⃣ 모니터링 스크립트 설정..."
cat > monitor_training.py << 'EOF'
import psutil
import GPUtil
import time
from datetime import datetime

def monitor_system():
    """시스템 리소스 모니터링"""
    while True:
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 메모리 사용률
        memory = psutil.virtual_memory()
        
        # GPU 사용률
        gpus = GPUtil.getGPUs()
        
        # 디스크 사용률
        disk = psutil.disk_usage('/')
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 시스템 모니터링")
        print(f"CPU: {cpu_percent}%")
        print(f"RAM: {memory.percent}% ({memory.used/1e9:.1f}/{memory.total/1e9:.1f} GB)")
        
        for gpu in gpus:
            print(f"GPU: {gpu.load*100:.1f}% | VRAM: {gpu.memoryUsed}/{gpu.memoryTotal} MB | Temp: {gpu.temperature}°C")
        
        print(f"Disk: {disk.percent}% ({disk.used/1e9:.1f}/{disk.total/1e9:.1f} GB)")
        
        time.sleep(30)

if __name__ == "__main__":
    monitor_system()
EOF

# 10. 자동 재시작 스크립트 (Preemptible 대비)
echo "🔟 자동 재시작 스크립트 설정..."
cat > auto_restart.sh << 'EOF'
#!/bin/bash
# Preemptible 인스턴스 자동 재시작

CHECKPOINT_DIR="gs://sports-ai-data-bucket-checkpoints/latest/"

# 체크포인트 다운로드
gsutil -m cp -r $CHECKPOINT_DIR ./checkpoints/

# 학습 재개
python train_sports_ai.py --resume --checkpoint ./checkpoints/latest.pth

# 학습 완료 후 체크포인트 업로드
gsutil -m cp -r ./checkpoints/* $CHECKPOINT_DIR
EOF

chmod +x auto_restart.sh

echo "========================================"
echo "✅ 환경 설정 완료!"
echo "========================================"
echo ""
echo "🚀 학습 시작 방법:"
echo "1. Jupyter 실행: jupyter lab --ip=0.0.0.0 --no-browser"
echo "2. Python 스크립트: python train_sports_ai.py"
echo "3. 모니터링: python monitor_training.py"
echo ""
echo "💡 팁:"
echo "- tmux 사용으로 세션 유지: tmux new -s training"
echo "- TensorBoard 실행: tensorboard --logdir=logs --host=0.0.0.0"
echo "========================================"