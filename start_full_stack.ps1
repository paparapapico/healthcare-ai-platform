# start_full_stack.ps1 - Windows PowerShell 스크립트
Write-Host "Healthcare AI 전체 시스템 시작..." -ForegroundColor Green

# 1. Docker 서비스 시작
Write-Host "Docker 서비스 시작..." -ForegroundColor Yellow
try {
    docker-compose up -d
    Write-Host "Docker 서비스 준비 대기 (10초)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
} catch {
    Write-Host "Docker 실행 실패. Docker Desktop이 설치되고 실행 중인지 확인하세요." -ForegroundColor Red
}

# 2. 백엔드 시작 (새 PowerShell 창에서)
Write-Host "백엔드 시작..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\Leehanjun\HealthcareAI\backend'; .\venv\Scripts\Activate.ps1; uvicorn app.main:app --reload --port 8000"

# 3. 잠시 대기
Write-Host "백엔드 준비 대기 (5초)..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# 4. 프론트엔드 시작 (새 PowerShell 창에서)
Write-Host "프론트엔드 시작..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\Leehanjun\HealthcareAI\frontend'; npm run dev"

Write-Host ""
Write-Host "전체 시스템 실행 완료!" -ForegroundColor Green
Write-Host ""
Write-Host "서비스 접속 정보:" -ForegroundColor Cyan
Write-Host "  - 프론트엔드 웹사이트: http://localhost:3000" -ForegroundColor White
Write-Host "  - 백엔드 API 문서: http://localhost:8000/docs" -ForegroundColor White
Write-Host "  - Grafana 모니터링: http://localhost:3000" -ForegroundColor White
Write-Host "  - PostgreSQL: localhost:5432" -ForegroundColor White
Write-Host "  - Redis: localhost:6379" -ForegroundColor White
Write-Host ""
Write-Host "로그인 정보:" -ForegroundColor Cyan
Write-Host "  - Email: admin@healthcare.ai" -ForegroundColor White
Write-Host "  - Password: admin123" -ForegroundColor White
Write-Host ""
Write-Host "서비스를 종료하려면 각 창에서 Ctrl+C를 누르세요" -ForegroundColor Yellow