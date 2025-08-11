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
    print(f"??Client {sid} connected")
    await sio.emit('connected', {'message': 'Connected!'}, to=sid)

@sio.event
async def disconnect(sid):
    connected_clients.discard(sid)
    print(f"??Client {sid} disconnected")

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
