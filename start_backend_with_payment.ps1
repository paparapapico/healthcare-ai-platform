# ?뚯씪: HealthcareAI\start_backend_with_payment.ps1
# 寃곗젣 湲곕뒫???ы븿??諛깆뿏???쒖옉

Write-Host "?뮩 Healthcare AI Backend (寃곗젣 ?ы븿) ?쒖옉..." -ForegroundColor Green

Set-Location "backend"

# 媛?곹솚寃??쒖꽦???쒕룄
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
    Write-Host "??Python 媛?곹솚寃??쒖꽦?? -ForegroundColor Green
} elseif (Test-Path "env\Scripts\Activate.ps1") {
    & "env\Scripts\Activate.ps1"
    Write-Host "??Python 媛?곹솚寃??쒖꽦?? -ForegroundColor Green
}

# ?섍꼍 蹂??濡쒕뱶 ?뺤씤
if (Test-Path ".env") {
    Write-Host "???섍꼍 蹂???뚯씪 ?뺤씤?? -ForegroundColor Green
} else {
    Write-Host "?좑툘 .env ?뚯씪???놁뒿?덈떎!" -ForegroundColor Yellow
}

# Stripe ?⑦궎吏 ?뺤씤
Write-Host "?벀 Stripe ?⑦궎吏 ?뺤씤..." -ForegroundColor Cyan
pip show stripe

Write-Host "`n?? 諛깆뿏???쒕쾭 ?쒖옉 (?ы듃 8000)..." -ForegroundColor Cyan
Write-Host "?뮩 寃곗젣 API ?붾뱶?ъ씤?? http://localhost:8000/api/payments" -ForegroundColor Yellow
Write-Host "?뱥 API 臾몄꽌: http://localhost:8000/docs" -ForegroundColor Yellow

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
