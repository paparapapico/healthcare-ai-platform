# GCP Quick Start Script for Windows PowerShell
# Sports AI Training 빠른 시작 스크립트

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🚀 GCP Sports AI Training Quick Setup" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan

# 설정 변수
$PROJECT_ID = "sports-ai-training"
$ZONE = "us-central1-a"
$INSTANCE_NAME = "sports-ai-gpu-instance"
$BUCKET_NAME = "sports-ai-data-bucket"
$REGION = "us-central1"

Write-Host "`n📋 설정 확인:" -ForegroundColor Yellow
Write-Host "Project ID: $PROJECT_ID"
Write-Host "Zone: $ZONE"
Write-Host "Instance: $INSTANCE_NAME"
Write-Host "Bucket: $BUCKET_NAME"

# Step 1: 프로젝트 확인
Write-Host "`n1️⃣ 프로젝트 설정 중..." -ForegroundColor Green
gcloud config set project $PROJECT_ID 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "프로젝트가 없습니다. 생성 중..." -ForegroundColor Yellow
    gcloud projects create $PROJECT_ID --name="Sports AI Training"
    gcloud config set project $PROJECT_ID
}

Write-Host "현재 프로젝트: " -NoNewline
gcloud config get-value project

# Step 2: API 활성화
Write-Host "`n2️⃣ 필요한 API 활성화 중..." -ForegroundColor Green
$apis = @(
    "compute.googleapis.com",
    "storage-api.googleapis.com",
    "ml.googleapis.com"
)

foreach ($api in $apis) {
    Write-Host "활성화: $api"
    gcloud services enable $api --quiet
}

# Step 3: GPU 인스턴스 생성
Write-Host "`n3️⃣ GPU 인스턴스 확인 중..." -ForegroundColor Green
$existing = gcloud compute instances list --filter="name=$INSTANCE_NAME" --format="value(name)" 2>$null

if ($existing) {
    Write-Host "인스턴스가 이미 존재합니다: $INSTANCE_NAME" -ForegroundColor Yellow
    $status = gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(status)"
    Write-Host "상태: $status"
    
    if ($status -eq "TERMINATED") {
        Write-Host "인스턴스 시작 중..."
        gcloud compute instances start $INSTANCE_NAME --zone=$ZONE
    }
} else {
    Write-Host "새 GPU 인스턴스 생성 중 (Tesla T4)..." -ForegroundColor Yellow
    Write-Host "예상 시간: 2-3분"
    
    gcloud compute instances create $INSTANCE_NAME `
        --machine-type=n1-standard-8 `
        --accelerator="type=nvidia-tesla-t4,count=1" `
        --maintenance-policy=TERMINATE `
        --preemptible `
        --boot-disk-size=200GB `
        --boot-disk-type=pd-ssd `
        --image-family=deep-learning-vm `
        --image-project=deeplearning-platform-release `
        --metadata="install-nvidia-driver=True" `
        --scopes="https://www.googleapis.com/auth/cloud-platform" `
        --zone=$ZONE `
        --quiet
}

# Step 4: 방화벽 규칙
Write-Host "`n4️⃣ 방화벽 규칙 설정 중..." -ForegroundColor Green
$rules = @{
    "allow-jupyter" = "tcp:8888"
    "allow-tensorboard" = "tcp:6006"
}

foreach ($rule in $rules.GetEnumerator()) {
    $exists = gcloud compute firewall-rules list --filter="name=$($rule.Key)" --format="value(name)" 2>$null
    if (-not $exists) {
        Write-Host "생성: $($rule.Key)"
        gcloud compute firewall-rules create $rule.Key `
            --allow=$rule.Value `
            --source-ranges="0.0.0.0/0" `
            --quiet
    } else {
        Write-Host "이미 존재: $($rule.Key)"
    }
}

# Step 5: Storage 버킷
Write-Host "`n5️⃣ Storage 버킷 확인 중..." -ForegroundColor Green
$buckets = @(
    $BUCKET_NAME,
    "$BUCKET_NAME-checkpoints",
    "$BUCKET_NAME-models"
)

foreach ($bucket in $buckets) {
    $exists = gsutil ls -b "gs://$bucket" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "생성: gs://$bucket"
        gsutil mb -p $PROJECT_ID -l $REGION "gs://$bucket/" 2>$null
    } else {
        Write-Host "이미 존재: gs://$bucket"
    }
}

# Step 6: 데이터 업로드
Write-Host "`n6️⃣ 학습 코드 업로드 중..." -ForegroundColor Green
$upload_items = @{
    "gcp" = "gs://$BUCKET_NAME/gcp/"
    "ai_models/models" = "gs://$BUCKET_NAME/models/"
    "ai_models/data_collection/collected_sports_data" = "gs://$BUCKET_NAME/data/"
}

foreach ($item in $upload_items.GetEnumerator()) {
    if (Test-Path $item.Key) {
        Write-Host "업로드: $($item.Key) -> $($item.Value)"
        gsutil -m cp -r $item.Key $item.Value 2>$null
    } else {
        Write-Host "경로 없음: $($item.Key)" -ForegroundColor Yellow
    }
}

# Step 7: 인스턴스 정보
Write-Host "`n7️⃣ 인스턴스 정보 가져오는 중..." -ForegroundColor Green
$EXTERNAL_IP = gcloud compute instances describe $INSTANCE_NAME `
    --zone=$ZONE `
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)"

# Step 8: 초기 설정 스크립트 생성
Write-Host "`n8️⃣ 초기 설정 스크립트 생성 중..." -ForegroundColor Green
$setup_script = @'
#!/bin/bash
echo 'Sports AI Training Setup'

# GPU 확인
nvidia-smi

# Python 환경 설정
conda create -n sports_ai python=3.9 -y
source activate sports_ai

# 필수 패키지 설치
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow transformers opencv-python mediapipe
pip install wandb tqdm google-cloud-storage numpy pandas

# 프로젝트 디렉토리 생성
mkdir -p ~/HealthcareAI
cd ~/HealthcareAI

# GCS에서 코드 다운로드
gsutil -m cp -r gs://sports-ai-data-bucket/gcp ./
gsutil -m cp -r gs://sports-ai-data-bucket/models ./ai_models/
gsutil -m cp -r gs://sports-ai-data-bucket/data ./data/

echo 'Setup Complete!'
echo 'Start training: python gcp/train_sports_ai.py --phase 1'
'@

$setup_script | Out-File -FilePath "initial_setup.sh" -Encoding UTF8

# Step 9: 완료 메시지
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "✅ GCP 환경 준비 완료!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n📋 인스턴스 정보:" -ForegroundColor Yellow
Write-Host "이름: $INSTANCE_NAME"
Write-Host "외부 IP: $EXTERNAL_IP"
Write-Host "GPU: Tesla T4 (Preemptible)"
Write-Host "비용: `$0.10/hour"

Write-Host "`n🔗 접속 방법:" -ForegroundColor Yellow
Write-Host "1. SSH: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
Write-Host "2. Jupyter: http://${EXTERNAL_IP}:8888"
Write-Host "3. TensorBoard: http://${EXTERNAL_IP}:6006"

Write-Host "`n📝 다음 단계:" -ForegroundColor Yellow
Write-Host "1. SSH로 인스턴스 접속"
Write-Host "2. initial_setup.sh 실행"
Write-Host "3. 학습 시작"

Write-Host "`n💡 유용한 명령어:" -ForegroundColor Yellow
Write-Host "인스턴스 중지: gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
Write-Host "인스턴스 시작: gcloud compute instances start $INSTANCE_NAME --zone=$ZONE"
Write-Host "비용 확인: gcloud billing accounts list"

Write-Host "`n⚠️  주의사항:" -ForegroundColor Red
Write-Host "- Preemptible 인스턴스는 최대 24시간 실행"
Write-Host "- 사용하지 않을 때는 인스턴스 중지"
Write-Host "- 체크포인트 자주 저장"

# SSH 접속 제안
$response = Read-Host "`n지금 SSH로 접속하시겠습니까? (y/n)"
if ($response -eq 'y') {
    Write-Host "SSH 접속 중..." -ForegroundColor Green
    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE
}