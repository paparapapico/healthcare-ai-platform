# 파일: HealthcareAI\replace_main.ps1
# main.py 완전 교체 스크립트

Write-Host "🔄 main.py 파일 완전 교체..." -ForegroundColor Yellow

# backend 디렉토리로 이동
if (-not (Test-Path ".\backend\app")) {
    Write-Host "❌ backend\app 디렉토리를 찾을 수 없습니다." -ForegroundColor Red
    exit 1
}

# 기존 main.py 완전 삭제
Write-Host "🗑️ 기존 main.py 삭제..." -ForegroundColor Yellow
Remove-Item ".\backend\app\main.py" -Force -ErrorAction SilentlyContinue

# 새로운 main.py 생성 (완전히 독립적인 버전)
Write-Host "📄 새로운 main.py 생성..." -ForegroundColor Green

@'
"""
Healthcare AI Backend - Simplified Version
WebSocket 지원을 포함한 간단한 FastAPI 서버
"""

from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import socketio
import asyncio
from datetime import datetime
import uvicorn
from typing import List, Optional

# FastAPI 앱 생성
app = FastAPI(
    title="Healthcare AI API",
    description="AI-powered healthcare platform",
    version="1.0.0"
)

# CORS 설정 (개발용 - 모든 오리진 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socket.IO 서버 설정
sio = socketio.AsyncServer(
    cors_allowed_origins="*",
    async_mode='asgi'
)

# Socket.IO 앱 생성
socket_app = socketio.ASGIApp(sio)

# 데이터 모델
class User(BaseModel):
    id: str
    email: str
    full_name: str
    is_active: bool
    is_superuser: bool
    created_at: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str

# 전역 변수 (실시간 데이터용)
connected_clients = set()

# Socket.IO 이벤트
@sio.event
async def connect(sid, environ):
    connected_clients.add(sid)
    print(f"✅ Client {sid} connected. Total: {len(connected_clients)}")
    await sio.emit('connected', {'message': 'Welcome to Healthcare AI!'}, to=sid)

@sio.event
async def disconnect(sid):
    connected_clients.discard(sid)
    print(f"❌ Client {sid} disconnected. Total: {len(connected_clients)}")

@sio.event
async def request_realtime_stats(sid, data):
    """실시간 통계 요청"""
    stats = {
        'timestamp': datetime.now().isoformat(),
        'activeUsers': 42 + (datetime.now().second % 10),
        'workoutSessions': 15 + (datetime.now().second % 5),
        'onlineUsers': len(connected_clients)
    }
    await sio.emit('realtime_stats', stats, to=sid)

# 백그라운드 태스크: 실시간 데이터 브로드캐스트
async def broadcast_realtime_data():
    while True:
        if connected_clients:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'activeUsers': 40 + (datetime.now().minute % 15),
                'workoutSessions': 12 + (datetime.now().second % 8),
                'onlineUsers': len(connected_clients)
            }
            await sio.emit('realtime_stats', stats)
        await asyncio.sleep(10)

# REST API 엔드포인트
@app.get("/")
async def root():
    return {
        "message": "Healthcare AI API is running!",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "connected_clients": len(connected_clients),
        "timestamp": datetime.now().isoformat()
    }

# 인증 API
@app.post("/api/auth/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """로그인 API"""
    if username == "admin@healthcare.ai" and password == "admin123":
        return {
            "access_token": "mock_token_12345",
            "token_type": "bearer"
        }
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/api/auth/me")
async def get_current_user():
    """현재 사용자 정보"""
    return {
        "id": "1",
        "email": "admin@healthcare.ai",
        "full_name": "Admin User",
        "is_active": True,
        "is_superuser": True,
        "created_at": "2024-01-01T00:00:00Z"
    }

# 대시보드 API
@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """대시보드 통계"""
    return {
        "total_users": 150,
        "active_users_today": 42,
        "total_workouts": 1250,
        "total_challenges": 8,
        "avg_session_duration": 32,
        "top_exercises": [
            {"exercise_type": "push_ups", "count": 45},
            {"exercise_type": "squats", "count": 38},
            {"exercise_type": "planks", "count": 32},
            {"exercise_type": "burpees", "count": 28},
            {"exercise_type": "jumping_jacks", "count": 25}
        ]
    }

# 사용자 API
@app.get("/api/users/")
async def get_users(skip: int = 0, limit: int = 100):
    """사용자 목록"""
    users = []
    for i in range(1, 11):  # 10명의 모의 사용자
        users.append({
            "id": str(i),
            "email": f"user{i}@healthcare.ai",
            "full_name": f"User {i}",
            "is_active": i % 5 != 0,  # 5번째마다 비활성
            "is_superuser": i == 1,   # 첫 번째만 관리자
            "created_at": "2024-01-01T00:00:00Z"
        })
    return users[skip:skip+limit]

@app.get("/api/users/{user_id}")
async def get_user(user_id: str):
    """특정 사용자 정보"""
    return {
        "id": user_id,
        "email": f"user{user_id}@healthcare.ai",
        "full_name": f"User {user_id}",
        "is_active": True,
        "is_superuser": user_id == "1",
        "created_at": "2024-01-01T00:00:00Z"
    }

# 운동 API
@app.get("/api/workouts/")
async def get_workouts():
    """운동 목록"""
    return [
        {
            "id": "1",
            "user_id": "1",
            "exercise_type": "push_ups",
            "duration": 300,
            "calories_burned": 50,
            "pose_accuracy": 85,
            "created_at": "2024-01-01T10:00:00Z"
        },
        {
            "id": "2", 
            "user_id": "2",
            "exercise_type": "squats",
            "duration": 600,
            "calories_burned": 80,
            "pose_accuracy": 92,
            "created_at": "2024-01-01T11:00:00Z"
        }
    ]

# 챌린지 API
@app.get("/api/challenges/")
async def get_challenges():
    """챌린지 목록"""
    return [
        {
            "id": "1",
            "title": "30-Day Push-up Challenge",
            "description": "Complete 1000 push-ups in 30 days",
            "target_value": 1000,
            "current_value": 650,
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "participants_count": 45,
            "is_active": True
        },
        {
            "id": "2",
            "title": "Weekly Plank Master",
            "description": "Hold plank for 10 minutes total each week",
            "target_value": 600,
            "current_value": 480,
            "start_date": "2024-01-01", 
            "end_date": "2024-01-07",
            "participants_count": 32,
            "is_active": True
        }
    ]

# Socket.IO를 FastAPI에 마운트
app.mount("/socket.io", socket_app)

# 앱 시작 시 백그라운드 태스크 시작
@app.on_event("startup")
async def startup_event():
    """앱 시작 시 실행"""
    asyncio.create_task(broadcast_realtime_data())
    print("🚀 Healthcare AI Backend started!")
    print("📡 WebSocket server ready")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
'@ | Out-File -FilePath ".\backend\app\main.py" -Encoding UTF8

Write-Host "✅ 새로운 main.py 생성 완료!" -ForegroundColor Green

# __init__.py 파일 생성 (필요한 경우)
if (-not (Test-Path ".\backend\app\__init__.py")) {
    Write-Host "📄 __init__.py 생성..." -ForegroundColor Yellow
    New-Item -Path ".\backend\app\__init__.py" -ItemType File -Force
}

Write-Host "🎯 설정 완료! 이제 서버를 시작하세요:" -ForegroundColor Cyan
Write-Host "  cd backend" -ForegroundColor White
Write-Host "  python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000" -ForegroundColor White