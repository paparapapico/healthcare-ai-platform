from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import socketio
import asyncio
from datetime import datetime
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Healthcare AI API")

# 강화된 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socket.IO
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode='asgi')
socket_app = socketio.ASGIApp(sio)

connected_clients = set()

@sio.event
async def connect(sid, environ):
    connected_clients.add(sid)
    await sio.emit('connected', {'message': 'Connected!'}, to=sid)

@sio.event
async def disconnect(sid):
    connected_clients.discard(sid)

@app.get("/")
async def root():
    return {"message": "Healthcare AI API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# 로그인 API - 디버깅 강화
@app.post("/api/auth/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    # 요청 로깅
    logger.info(f"=== LOGIN ATTEMPT ===")
    logger.info(f"Username: {username}")
    logger.info(f"Password length: {len(password)}")
    logger.info(f"Request headers: {dict(request.headers)}")
    
    # 계정 확인
    if username == "admin@healthcare.ai" and password == "admin123":
        logger.info(" LOGIN SUCCESS")
        return {
            "access_token": "healthcare_admin_token_12345",
            "token_type": "bearer"
        }
    else:
        logger.error(f" LOGIN FAILED - Username: '{username}', Password: '{password}'")
        logger.error("Expected: admin@healthcare.ai / admin123")
        raise HTTPException(
            status_code=401, 
            detail={
                "error": "Invalid credentials",
                "received_username": username,
                "expected": "admin@healthcare.ai"
            }
        )

# OPTIONS 요청 처리
@app.options("/api/auth/login")
async def login_options():
    return JSONResponse(content={})

@app.get("/api/auth/me")
async def get_current_user():
    return {
        "id": "1",
        "email": "admin@healthcare.ai",
        "full_name": "Admin User",
        "is_active": True,
        "is_superuser": True,
        "created_at": "2024-01-01T00:00:00Z"
    }

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    return {
        "total_users": 150,
        "active_users_today": 42,
        "total_workouts": 1250,
        "total_challenges": 8,
        "avg_session_duration": 32,
        "top_exercises": [
            {"exercise_type": "push_ups", "count": 45},
            {"exercise_type": "squats", "count": 38},
            {"exercise_type": "planks", "count": 32}
        ]
    }

@app.get("/api/users/")
async def get_users():
    return [
        {
            "id": "1",
            "email": "admin@healthcare.ai",
            "full_name": "Admin User",
            "is_active": True,
            "is_superuser": True,
            "created_at": "2024-01-01T00:00:00Z"
        }
    ]

@app.get("/api/challenges/")
async def get_challenges():
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
        }
    ]

app.mount("/socket.io", socket_app)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
