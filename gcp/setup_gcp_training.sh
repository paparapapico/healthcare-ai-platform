#!/bin/bash
# GCP 프로젝트 설정 및 학습 환경 구축 스크립트
# Google Cloud Platform Training Setup Script

echo "========================================"
echo "🚀 GCP Sports AI 학습 환경 설정"
echo "========================================"

# 프로젝트 설정
PROJECT_ID="sports-ai-training"
REGION="us-central1"
ZONE="us-central1-a"
INSTANCE_NAME="sports-ai-gpu-instance"
BUCKET_NAME="sports-ai-data-bucket"

# 1. 프로젝트 생성 및 설정
echo "1️⃣ 프로젝트 설정 중..."
gcloud projects create $PROJECT_ID --name="Sports AI Training"
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION
gcloud config set compute/zone $ZONE

# 2. 결제 계정 연결 (수동으로 해야 함)
echo "⚠️  브라우저에서 결제 계정을 연결하세요:"
echo "https://console.cloud.google.com/billing"
read -p "결제 계정 연결 완료 후 Enter 키를 누르세요..."

# 3. 필요한 API 활성화
echo "2️⃣ 필요한 API 활성화 중..."
gcloud services enable compute.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable ml.googleapis.com
gcloud services enable notebooks.googleapis.com

# 4. GPU 할당량 요청 (필요시)
echo "3️⃣ GPU 할당량 확인..."
gcloud compute project-info describe --project=$PROJECT_ID

# 5. Preemptible GPU 인스턴스 생성
echo "4️⃣ GPU 인스턴스 생성 중... (Tesla T4, Preemptible)"
gcloud compute instances create $INSTANCE_NAME \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE \
    --preemptible \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --image-family=deep-learning-vm \
    --image-project=deeplearning-platform-release \
    --metadata="install-nvidia-driver=True" \
    --scopes=https://www.googleapis.com/auth/cloud-platform

# 6. 방화벽 규칙 설정 (Jupyter, TensorBoard)
echo "5️⃣ 방화벽 규칙 설정 중..."
gcloud compute firewall-rules create allow-jupyter \
    --allow=tcp:8888 \
    --source-ranges=0.0.0.0/0 \
    --description="Allow Jupyter Notebook"

gcloud compute firewall-rules create allow-tensorboard \
    --allow=tcp:6006 \
    --source-ranges=0.0.0.0/0 \
    --description="Allow TensorBoard"

# 7. Cloud Storage 버킷 생성
echo "6️⃣ Cloud Storage 버킷 생성 중..."
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME/
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME-checkpoints/
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME-models/

# 8. 인스턴스 IP 가져오기
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo "========================================"
echo "✅ GCP 설정 완료!"
echo "========================================"
echo "🖥️  인스턴스: $INSTANCE_NAME"
echo "🌐 외부 IP: $EXTERNAL_IP"
echo "💰 예상 비용: $0.10/시간 (Preemptible T4)"
echo "📦 버킷: gs://$BUCKET_NAME/"
echo ""
echo "다음 단계:"
echo "1. SSH 접속: gcloud compute ssh $INSTANCE_NAME"
echo "2. Jupyter 실행: http://$EXTERNAL_IP:8888"
echo "3. TensorBoard: http://$EXTERNAL_IP:6006"
echo "========================================"