# ?뚯씪: HealthcareAI\start_frontend_with_payment.ps1
# 寃곗젣 湲곕뒫???ы븿???꾨줎?몄뿏???쒖옉

Write-Host "?뮩 Healthcare AI Frontend (寃곗젣 ?ы븿) ?쒖옉..." -ForegroundColor Green

Set-Location "frontend"

# ?섍꼍 蹂???뺤씤
if (Test-Path ".env") {
    Write-Host "???섍꼍 蹂???뚯씪 ?뺤씤?? -ForegroundColor Green
    Write-Host "?뱞 ?섍꼍 蹂???댁슜:" -ForegroundColor Cyan
    Get-Content ".env" | Where-Object { $_ -like "*STRIPE*" }
} else {
    Write-Host "?좑툘 .env ?뚯씪???놁뒿?덈떎!" -ForegroundColor Yellow
}

# Stripe ?⑦궎吏 ?뺤씤
Write-Host "`n?벀 Stripe ?⑦궎吏 ?뺤씤..." -ForegroundColor Cyan
npm list @stripe/stripe-js @stripe/react-stripe-js 2>$null

Write-Host "`n?? ?꾨줎?몄뿏???쒕쾭 ?쒖옉 (?ы듃 3000)..." -ForegroundColor Cyan
Write-Host "?뮩 寃곗젣 ?섏씠吏: http://localhost:3000/payment" -ForegroundColor Yellow
Write-Host "?룧 ??쒕낫?? http://localhost:3000/admin" -ForegroundColor Yellow

npm run dev
