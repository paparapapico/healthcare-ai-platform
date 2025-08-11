# Healthcare AI Payment System Startup Script
# Windows PowerShell

Write-Host "Healthcare AI Payment System Starting..." -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan

# 1. Start Backend (Background)
Write-Host "1. Starting Backend..." -ForegroundColor Yellow
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    & ".\start_backend_with_payment.ps1"
}

# Wait 5 seconds
Start-Sleep -Seconds 5

# 2. Start Frontend (Background)
Write-Host "2. Starting Frontend..." -ForegroundColor Yellow
$frontendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    & ".\start_frontend_with_payment.ps1"
}

Write-Host ""
Write-Host "Payment System Started Successfully!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Access Information:" -ForegroundColor Yellow
Write-Host "  - Dashboard: http://localhost:3000/admin" -ForegroundColor Green
Write-Host "  - Payment Page: http://localhost:3000/payment" -ForegroundColor Green
Write-Host "  - Backend API: http://localhost:8000" -ForegroundColor Green
Write-Host "  - Payment API: http://localhost:8000/api/payments" -ForegroundColor Green
Write-Host ""
Write-Host "Login Credentials:" -ForegroundColor Yellow
Write-Host "  Email: admin@healthcare.ai" -ForegroundColor White
Write-Host "  Password: admin123" -ForegroundColor White
Write-Host ""
Write-Host "Test Card (Stripe):" -ForegroundColor Yellow
Write-Host "  Card Number: 4242 4242 4242 4242" -ForegroundColor White
Write-Host "  Expiry: 12/34" -ForegroundColor White
Write-Host "  CVC: 123" -ForegroundColor White
Write-Host ""
Write-Host "Job Management:" -ForegroundColor Yellow
Write-Host "  Get-Job                    # Check running jobs" -ForegroundColor Gray
Write-Host "  Stop-Job -Name Backend     # Stop backend" -ForegroundColor Gray
Write-Host "  Stop-Job -Name Frontend    # Stop frontend" -ForegroundColor Gray

# Status monitoring
try {
    while ($true) {
        Start-Sleep -Seconds 10
        $jobs = Get-Job
        if ($jobs.Count -ge 2) {
            Write-Host "Status: Backend($($jobs[0].State)), Frontend($($jobs[1].State))" -ForegroundColor Cyan
        }
    }
} finally {
    Write-Host ""
    Write-Host "Cleaning up..." -ForegroundColor Yellow
    Get-Job | Stop-Job
    Get-Job | Remove-Job
    Write-Host "Cleanup complete" -ForegroundColor Green
}
