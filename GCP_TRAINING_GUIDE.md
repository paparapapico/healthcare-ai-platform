# GCP Sports AI Training Guide
Google Cloud Platform에서 Sports AI 모델 학습 완전 가이드

## 현재 상태
- 데이터 수집 완료: 1,176,478개 샘플 식별
- 학습 스크립트 준비 완료
- GCP $300 무료 크레딧 활용 예정

## Step 1: GCP 계정 생성 및 크레딧 활성화

### 1.1 GCP 계정 생성
```bash
# 브라우저에서 접속
https://cloud.google.com/free

# 필요 정보:
- Google 계정
- 신용카드 (확인용, 자동 결제 안됨)
- 전화번호 인증
```

### 1.2 $300 무료 크레딧 확인
- 계정 생성 시 자동으로 $300 크레딧 부여
- 유효기간: 90일
- 사용 가능 서비스: 모든 GCP 서비스

## Step 2: Google Cloud SDK 설치

### Windows에서 설치
```powershell
# PowerShell 관리자 권한으로 실행

# 1. SDK 다운로드 및 설치
# https://cloud.google.com/sdk/docs/install 에서 다운로드

# 2. 설치 후 초기화
gcloud init

# 3. 계정 로그인
gcloud auth login

# 4. 프로젝트 생성
gcloud projects create sports-ai-training --name="Sports AI Training"

# 5. 프로젝트 설정
gcloud config set project sports-ai-training
```

## Step 3: GCP 프로젝트 설정

### 3.1 결제 계정 연결
```bash
# 브라우저에서 수동 연결
https://console.cloud.google.com/billing

# 프로젝트에 결제 계정 연결 확인
gcloud beta billing accounts list
```

### 3.2 필요한 API 활성화
```bash
# Compute Engine API
gcloud services enable compute.googleapis.com

# Cloud Storage API  
gcloud services enable storage-api.googleapis.com

# AI Platform API
gcloud services enable ml.googleapis.com

# 활성화 확인
gcloud services list --enabled
```

## Step 4: GPU 인스턴스 생성

### 4.1 GPU 할당량 확인
```bash
# GPU 할당량 확인
gcloud compute project-info describe --project=sports-ai-training

# T4 GPU 할당량 요청 (필요시)
# Console에서: IAM & Admin > Quotas > Filter by "GPU"
```

### 4.2 Preemptible T4 인스턴스 생성
```bash
# 변수 설정
$ZONE = "us-central1-a"
$INSTANCE_NAME = "sports-ai-gpu-instance"

# 인스턴스 생성 (Preemptible T4, $0.10/hour)
gcloud compute instances create $INSTANCE_NAME `
    --machine-type=n1-standard-8 `
    --accelerator=type=nvidia-tesla-t4,count=1 `
    --maintenance-policy=TERMINATE `
    --preemptible `
    --boot-disk-size=200GB `
    --boot-disk-type=pd-ssd `
    --image-family=deep-learning-vm `
    --image-project=deeplearning-platform-release `
    --metadata="install-nvidia-driver=True" `
    --scopes=https://www.googleapis.com/auth/cloud-platform `
    --zone=$ZONE
```

### 4.3 방화벽 규칙 설정
```bash
# Jupyter Notebook 포트 열기
gcloud compute firewall-rules create allow-jupyter `
    --allow=tcp:8888 `
    --source-ranges=0.0.0.0/0

# TensorBoard 포트 열기
gcloud compute firewall-rules create allow-tensorboard `
    --allow=tcp:6006 `
    --source-ranges=0.0.0.0/0
```

## Step 5: Storage 버킷 생성

```bash
# 버킷 생성
$BUCKET_NAME = "sports-ai-data-bucket"
$REGION = "us-central1"

# 데이터 버킷
gsutil mb -p sports-ai-training -l $REGION gs://$BUCKET_NAME/

# 체크포인트 버킷
gsutil mb -p sports-ai-training -l $REGION gs://$BUCKET_NAME-checkpoints/

# 모델 버킷
gsutil mb -p sports-ai-training -l $REGION gs://$BUCKET_NAME-models/
```

## Step 6: 인스턴스 접속 및 환경 설정

### 6.1 SSH 접속
```bash
# SSH로 인스턴스 접속
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE

# 또는 브라우저 SSH
# Console > Compute Engine > VM instances > SSH 버튼 클릭
```

### 6.2 학습 환경 설치 (인스턴스 내부에서)
```bash
# GPU 확인
nvidia-smi

# Conda 환경 생성
conda create -n sports_ai python=3.9 -y
conda activate sports_ai

# PyTorch 설치
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 필수 라이브러리 설치
pip install tensorflow==2.13.0 transformers==4.35.0 opencv-python mediapipe
pip install wandb tqdm google-cloud-storage numpy pandas scikit-learn
```

## Step 7: 프로젝트 코드 업로드

### 7.1 로컬에서 GCS로 업로드
```bash
# 로컬 PowerShell에서 실행
cd C:\Users\Leehanjun\HealthcareAI

# ai_models 폴더 업로드
gsutil -m cp -r ai_models gs://sports-ai-data-bucket/

# gcp 학습 스크립트 업로드
gsutil -m cp -r gcp gs://sports-ai-data-bucket/

# 수집된 데이터 업로드
gsutil -m cp -r ai_models/data_collection/collected_sports_data gs://sports-ai-data-bucket/
```

### 7.2 인스턴스에서 다운로드
```bash
# SSH 세션에서
cd ~
mkdir HealthcareAI
cd HealthcareAI

# 코드 다운로드
gsutil -m cp -r gs://sports-ai-data-bucket/ai_models ./
gsutil -m cp -r gs://sports-ai-data-bucket/gcp ./
gsutil -m cp -r gs://sports-ai-data-bucket/collected_sports_data ./data/
```

## Step 8: 학습 시작

### 8.1 tmux 세션 시작 (연결 끊김 방지)
```bash
# tmux 설치 및 시작
sudo apt-get install tmux -y
tmux new -s training

# tmux 명령어
# Ctrl+B, D : 세션 분리
# tmux attach -t training : 세션 재연결
```

### 8.2 학습 실행
```bash
cd ~/HealthcareAI/gcp
conda activate sports_ai

# Phase 1: 실험 단계 (T4, 2-3시간, $0.30)
python train_sports_ai.py --phase 1 --no_wandb

# Phase 2: 본격 학습 (V100 필요시)
# 인스턴스를 V100으로 업그레이드 후 실행
# python train_sports_ai.py --phase 2

# 전체 파이프라인 실행
# python train_sports_ai.py --phase all
```

### 8.3 모니터링
```bash
# 새 터미널 창에서
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE

# GPU 모니터링
watch -n 1 nvidia-smi

# 학습 로그 확인
tail -f ~/HealthcareAI/training.log

# TensorBoard 실행
tensorboard --logdir=~/HealthcareAI/logs --host=0.0.0.0 --port=6006
```

## Step 9: 비용 관리

### 9.1 예상 비용
```
Phase 1 (T4 Preemptible): 
- 시간: 10시간
- 비용: $1.00

Phase 2 (V100 Regular):
- 시간: 72시간  
- 비용: $178.56

Phase 3 (T4 Preemptible):
- 시간: 15시간
- 비용: $1.50

Storage & Network:
- 비용: ~$20

총 예상 비용: $201.06 (크레딧 내 가능)
```

### 9.2 비용 모니터링
```bash
# 비용 확인
gcloud billing accounts list
gcloud alpha billing budgets list

# 일일 비용 리포트
gcloud billing accounts get-iam-policy [BILLING_ACCOUNT_ID]
```

### 9.3 인스턴스 중지 (사용 안할 때)
```bash
# 인스턴스 중지 (비용 절약)
gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE

# 인스턴스 시작
gcloud compute instances start $INSTANCE_NAME --zone=$ZONE

# 인스턴스 삭제 (완전히 끝난 후)
gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE
```

## Step 10: 학습 완료 후

### 10.1 모델 다운로드
```bash
# 로컬로 모델 다운로드
gsutil -m cp -r gs://sports-ai-data-bucket-models/* ./trained_models/
```

### 10.2 모델 배포 준비
```bash
# 모델을 backend로 복사
copy trained_models\*.pth backend\app\models\
```

### 10.3 정리
```bash
# 버킷 삭제 (선택사항)
gsutil rm -r gs://sports-ai-data-bucket
gsutil rm -r gs://sports-ai-data-bucket-checkpoints
gsutil rm -r gs://sports-ai-data-bucket-models

# 프로젝트 삭제 (모든 리소스 삭제)
gcloud projects delete sports-ai-training
```

## 문제 해결

### GPU 할당 오류
```bash
# 다른 존 시도
zones = ["us-central1-a", "us-central1-b", "us-west1-b", "europe-west4-a"]
```

### Preemptible 인스턴스 중단
```bash
# 자동 재시작 스크립트 설정
gcloud compute instances create ... --metadata="startup-script=..."
```

### 메모리 부족
```bash
# 배치 크기 줄이기
# gradient accumulation 늘리기
# mixed precision 사용
```

## 다음 단계

1. GCP 계정 생성 완료 후 알려주세요
2. SDK 설치 후 프로젝트 ID 확인
3. 인스턴스 생성 시 발생하는 오류 공유
4. 학습 진행 상황 모니터링

준비되시면 Step 1부터 시작하세요!