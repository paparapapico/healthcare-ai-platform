Write-Host "💳 Healthcare AI 결제 시스템 설정..." -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan

# 1. 결제 시스템 스크립트 디렉토리 생성
Write-Host "1️⃣ 스크립트 디렉토리 설정..." -ForegroundColor Yellow

if (-not (Test-Path "scripts")) {
    New-Item -ItemType Directory -Name "scripts"
    Write-Host "✅ scripts 디렉토리 생성" -ForegroundColor Green
}

# 2. Stripe 패키지 설치
Write-Host "`n2️⃣ Stripe 패키지 설치..." -ForegroundColor Yellow

# Backend Stripe 설치
Write-Host "📦 Backend Stripe 설치..." -ForegroundColor Cyan
Set-Location "backend"
pip install stripe python-dotenv

# Frontend Stripe 설치
Write-Host "📦 Frontend Stripe 설치..." -ForegroundColor Cyan
Set-Location "../frontend"
npm install @stripe/stripe-js @stripe/react-stripe-js

Set-Location ".."

# 3. 환경 변수 설정
Write-Host "`n3️⃣ 환경 변수 설정..." -ForegroundColor Yellow

# Backend .env 파일 업데이트
Write-Host "🔧 Backend 환경 변수 설정..." -ForegroundColor Cyan
$backendEnvPath = "backend\.env"

if (Test-Path $backendEnvPath) {
    $backendEnv = Get-Content $backendEnvPath
} else {
    $backendEnv = @()
}

# Stripe 설정 추가 (기존 내용 유지)
$stripeSettings = @"

# Stripe 결제 설정
STRIPE_SECRET_KEY=sk_test_51234567890abcdef_your_test_key_here
STRIPE_PUBLISHABLE_KEY=pk_test_51234567890abcdef_your_test_key_here  
STRIPE_WEBHOOK_SECRET=whsec_1234567890abcdef_your_webhook_secret

# 결제 설정
PAYMENT_CURRENCY=USD
PAYMENT_SUCCESS_URL=http://localhost:3000/payment/success
PAYMENT_CANCEL_URL=http://localhost:3000/payment/cancel
"@

($backendEnv + $stripeSettings) | Out-File -FilePath $backendEnvPath -Encoding UTF8
Write-Host "✅ Backend .env 업데이트 완료" -ForegroundColor Green

# Frontend .env 파일 업데이트
Write-Host "🔧 Frontend 환경 변수 설정..." -ForegroundColor Cyan
$frontendEnvPath = "frontend\.env"

if (Test-Path $frontendEnvPath) {
    $frontendEnv = Get-Content $frontendEnvPath
} else {
    $frontendEnv = @()
}

# Stripe 설정 추가
$frontendStripeSettings = @"

# Stripe 결제 설정 (Frontend)
VITE_STRIPE_PUBLISHABLE_KEY=pk_test_51234567890abcdef_your_test_key_here
VITE_PAYMENT_SUCCESS_URL=http://localhost:3000/payment/success
VITE_PAYMENT_CANCEL_URL=http://localhost:3000/payment/cancel
"@

($frontendEnv + $frontendStripeSettings) | Out-File -FilePath $frontendEnvPath -Encoding UTF8
Write-Host "✅ Frontend .env 업데이트 완료" -ForegroundColor Green

# 4. 결제 시스템 시작 스크립트 생성
Write-Host "`n4️⃣ 실행 스크립트 생성..." -ForegroundColor Yellow