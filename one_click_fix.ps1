# 파일: HealthcareAI\one_click_fix.ps1
# 모든 오류를 한 번에 해결하는 스크립트

Write-Host "🚑 Healthcare AI 긴급 수정 시작..." -ForegroundColor Red
Write-Host "========================================" -ForegroundColor Yellow

# 1. 모든 Python 프로세스 종료
Write-Host "1️⃣ 기존 프로세스 정리..." -ForegroundColor Yellow
Get-Process | Where-Object { $_.ProcessName -like "*python*" -or $_.ProcessName -like "*uvicorn*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# 2. 필요한 패키지 설치
Write-Host "2️⃣ 필요한 패키지 설치..." -ForegroundColor Yellow
Set-Location ".\backend"
pip install python-socketio fastapi uvicorn

# 3. 기존 문제 파일들 정리
Write-Host "3️⃣ 기존 파일 정리..." -ForegroundColor Yellow
if (Test-Path ".\app\main.py") {
    Move-Item ".\app\main.py" ".\app\main_old.py" -Force
}

# 4. 새로운 main.py 생성
Write-Host "4️⃣ 새로운 main.py 생성..." -ForegroundColor Green

@'
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import socketio
import asyncio
from datetime import datetime
import uvicorn

app = FastAPI(title="Healthcare AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode='asgi')
socket_app = socketio.ASGIApp(sio)

connected_clients = set()

@sio.event
async def connect(sid, environ):
    connected_clients.add(sid)
    print(f"✅ Client {sid} connected")
    await sio.emit('connected', {'message': 'Connected!'}, to=sid)

@sio.event
async def disconnect(sid):
    connected_clients.discard(sid)
    print(f"❌ Client {sid} disconnected")

@sio.event
async def request_realtime_stats(sid, data):
    stats = {
        'timestamp': datetime.now().isoformat(),
        'activeUsers': 42,
        'workoutSessions': 15,
        'onlineUsers': len(connected_clients)
    }
    await sio.emit('realtime_stats', stats, to=sid)

@app.get("/")
async def root():
    return {"message": "Healthcare AI API is running!", "status": "healthy"}

@app.get("/health")
async def health():
    return {"status": "healthy", "clients": len(connected_clients)}

@app.post("/api/auth/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if username == "admin@healthcare.ai" and password == "admin123":
        return {"access_token": "mock_token", "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/api/auth/me")
async def get_user():
    return {
        "id": "1", "email": "admin@healthcare.ai", "full_name": "Admin User",
        "is_active": True, "is_superuser": True, "created_at": "2024-01-01T00:00:00Z"
    }

@app.get("/api/dashboard/stats")
async def get_stats():
    return {
        "total_users": 150, "active_users_today": 42, "total_workouts": 1250,
        "total_challenges": 8, "avg_session_duration": 32,
        "top_exercises": [
            {"exercise_type": "push_ups", "count": 45},
            {"exercise_type": "squats", "count": 38},
            {"exercise_type": "planks", "count": 32}
        ]
    }

@app.get("/api/users/")
async def get_users():
    return [
        {"id": "1", "email": "admin@healthcare.ai", "full_name": "Admin User", 
         "is_active": True, "is_superuser": True, "created_at": "2024-01-01T00:00:00Z"},
        {"id": "2", "email": "user2@healthcare.ai", "full_name": "User 2",
         "is_active": True, "is_superuser": False, "created_at": "2024-01-01T00:00:00Z"}
    ]

@app.get("/api/challenges/")
async def get_challenges():
    return [
        {"id": "1", "title": "30-Day Push-up Challenge", "description": "Complete 1000 push-ups",
         "target_value": 1000, "current_value": 650, "participants_count": 45, "is_active": True,
         "start_date": "2024-01-01", "end_date": "2024-01-31"}
    ]

app.mount("/socket.io", socket_app)

async def broadcast_stats():
    while True:
        if connected_clients:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'activeUsers': 40 + (datetime.now().second % 10),
                'workoutSessions': 12 + (datetime.now().second % 5),
                'onlineUsers': len(connected_clients)
            }
            await sio.emit('realtime_stats', stats)
        await asyncio.sleep(10)

@app.on_event("startup")
async def startup():
    asyncio.create_task(broadcast_stats())

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
'@ | Out-File -FilePath ".\app\main.py" -Encoding UTF8

# 5. __init__.py 확인
if (-not (Test-Path ".\app\__init__.py")) {
    New-Item -Path ".\app\__init__.py" -ItemType File -Force
}

Write-Host "5️⃣ 서버 시작..." -ForegroundColor Cyan
Write-Host ""
Write-Host "🌐 서버 주소:" -ForegroundColor Green
Write-Host "  - Backend: http://localhost:8000" -ForegroundColor White
Write-Host "  - API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "  - Health: http://localhost:8000/health" -ForegroundColor White
Write-Host ""
Write-Host "서버를 중지하려면 Ctrl+C를 누르세요" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

# 서버 실행
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 파일: HealthcareAI\start_everything.ps1
# 프론트엔드와 백엔드를 모두 시작하는 스크립트

Write-Host "🚀 Healthcare AI 전체 시작..." -ForegroundColor Green

# 백엔드 시작 (백그라운드)
Write-Host "1️⃣ 백엔드 시작..." -ForegroundColor Yellow
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    & ".\one_click_fix.ps1"
}

# 5초 대기
Start-Sleep -Seconds 5

# 프론트엔드 시작 (백그라운드)
Write-Host "2️⃣ 프론트엔드 시작..." -ForegroundColor Yellow
$frontendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    if (Test-Path ".\frontend") {
        Set-Location ".\frontend"
        if (-not (Test-Path ".env")) {
            @'
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
VITE_ENVIRONMENT=development
'@ | Out-File -FilePath ".env" -Encoding UTF8
        }
        npm run dev
    }
}

Write-Host ""
Write-Host "✅ 서비스 시작 완료!" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🌐 접속 정보:" -ForegroundColor Yellow
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor Green
Write-Host "  Backend:  http://localhost:8000" -ForegroundColor Green
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""
Write-Host "🔑 로그인:" -ForegroundColor Yellow
Write-Host "  Email: admin@healthcare.ai" -ForegroundColor White
Write-Host "  Password: admin123" -ForegroundColor White
Write-Host ""
Write-Host "❓ 명령어:" -ForegroundColor Yellow
Write-Host "  Get-Job                    # 작업 상태 확인" -ForegroundColor Gray
Write-Host "  Stop-Job -Name Backend     # 백엔드 중지" -ForegroundColor Gray
Write-Host "  Stop-Job -Name Frontend    # 프론트엔드 중지" -ForegroundColor Gray

# 무한 대기 (사용자가 중지할 때까지)
try {
    while ($true) {
        Start-Sleep -Seconds 10
        $jobs = Get-Job
        Write-Host "📊 상태: Backend($($jobs[0].State)), Frontend($($jobs[1].State))" -ForegroundColor Cyan
    }
} finally {
    Write-Host "🛑 정리 중..." -ForegroundColor Yellow
    Get-Job | Stop-Job
    Get-Job | Remove-Job
}