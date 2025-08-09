# app/main.py 상단에 추가
import os
from app.core.monitoring import metrics_endpoint, MetricsCollector, track_request_metrics
from app.core.error_tracking import initialize_sentry
from fastapi import Request
import time

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app.api.v1 import auth, health, workouts, social, notifications
from typing import Optional, List
import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime, date
import base64
from app.db.database import get_db

from app.api.v1.auth import router as auth_router
from app.api.v1.health import router as health_router
from app.api.v1.workouts import router as workout_router
from app.api.v1.social import router as social_router

app = FastAPI(
    title="Healthcare AI Platform",
    description="AI 기반 헬스케어 자세 분석 플랫폼",
    version="1.0.0"
)

import logging  # 이 라인을 추가!

# Logger 설정 추가
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# app/main.py에 추가
from app.api.v1.social import router as social_router
app.include_router(social_router)
# app/main.py에 추가
from app.api.v1.health import router as health_router
app.include_router(health_router)
# app/main.py 상단에 추가
from app.api.v1.notifications import router as notifications_router
from app.services.notifications.scheduler import init_scheduler
from app.core.cache import cache_manager
from app.db.optimizations import initialize_optimizations

# 라우터 추가
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(health.router, prefix="/api/v1/health", tags=["health"])
app.include_router(workouts.router, prefix="/api/v1/workouts", tags=["workouts"])
app.include_router(social.router, prefix="/api/v1/social", tags=["social"])
app.include_router(notifications.router, prefix="/api/v1/notifications", tags=["notifications"])

# 앱 시작 이벤트에 추가
@app.on_event("startup")
async def startup_event():
    """앱 시작 시 실행"""
    # 데이터베이스 최적화
    initialize_optimizations()
    
    # 알림 스케줄러 시작
    init_scheduler()
    
    # Redis 연결 확인
    if cache_manager.redis_client:
        logger.info("Redis cache connected")
    else:
        logger.warning("Redis cache not available")

@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료 시 실행"""
    # 스케줄러 종료
    from app.services.notifications.scheduler import scheduler
    scheduler.shutdown()

# Sentry 초기화
initialize_sentry(environment=os.getenv("ENVIRONMENT", "development"))

# Prometheus 메트릭 엔드포인트 추가
@app.get("/metrics")
async def get_metrics():
    return await metrics_endpoint()

# 미들웨어 추가 - 요청 추적
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # 메트릭 수집
    MetricsCollector.collect_system_metrics()
    
    return response

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경에서만 사용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Pydantic 모델들
class HealthProfile(BaseModel):
    user_id: str
    age: int = Field(..., ge=1, le=120)
    height: float = Field(..., ge=50, le=250)  # cm
    weight: float = Field(..., ge=20, le=300)  # kg
    activity_level: str = Field(..., pattern="^(low|moderate|high)$")
    health_conditions: Optional[List[str]] = []

class PostureAnalysisRequest(BaseModel):
    image_data: str  # base64 encoded image
    user_id: str

class PostureResult(BaseModel):
    overall_score: float
    details: dict
    recommendations: List[str]
    timestamp: datetime

class ExerciseRecommendation(BaseModel):
    exercise_name: str
    duration: int  # minutes
    difficulty: str
    target_areas: List[str]
    instructions: List[str]

# 사용자 관리
class UserCreate(BaseModel):
    email: str
    password: str
    name: str
    birth_date: str  # "1990-01-01" 형태
    gender: str
    height: Optional[float] = None
    weight: Optional[float] = None
    subscription_tier: Optional[str] = "FREE"

class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    birth_date: Optional[str] = None
    gender: Optional[str] = None
    subscription_tier: str = "FREE"
    created_at: Optional[datetime] = None
    is_active: bool = True

# 임시 데이터 저장소
users_db = {}
health_profiles_db = {}
analysis_results_db = {}

# 헬스체크 엔드포인트
@app.get("/")
async def root():
    return {
        "message": "Healthcare AI Platform API", 
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

# 사용자 관리 API
@app.post("/api/v1/auth/register", response_model=UserResponse)
async def register_user(user_data: UserCreate):
    """회원가입 API"""
    try:
        # 기존 users_db에 저장 (기존 로직과 동일)
        user_id = len(users_db) + 1
        
        user_record = {
            "id": user_id,
            "email": user_data.email,
            "name": user_data.name,
            "birth_date": user_data.birth_date,
            "gender": user_data.gender,
            "height": user_data.height,
            "weight": user_data.weight,
            "subscription_tier": user_data.subscription_tier or "FREE",
            "created_at": datetime.now(),
            "is_active": True
        }
        
        users_db[f"user_{user_id}"] = user_record
        
        # 응답 데이터 생성 (모든 필드 포함)
        response_data = {
            "id": user_record["id"],
            "email": user_record["email"], 
            "name": user_record["name"],
            "birth_date": user_record["birth_date"],
            "gender": user_record["gender"],
            "subscription_tier": user_record["subscription_tier"],
            "created_at": user_record["created_at"],
            "is_active": user_record["is_active"]
        }
        
        return UserResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.get("/api/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(**users_db[user_id])

# 헬스 프로필 API
@app.post("/api/health-profile")
async def create_health_profile(profile: HealthProfile):
    if profile.user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    health_profiles_db[profile.user_id] = profile.dict()
    return {"message": "Health profile created successfully", "user_id": profile.user_id}

@app.get("/api/health-profile/{user_id}")
async def get_health_profile(user_id: str):
    if user_id not in health_profiles_db:
        raise HTTPException(status_code=404, detail="Health profile not found")
    return health_profiles_db[user_id]

# 자세 분석 함수
def analyze_pose(image_data: str):
    try:
        # base64 디코딩
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # BGR to RGB 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # MediaPipe로 자세 추정
        results = pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return {
                "overall_score": 0,
                "details": {"error": "No pose detected"},
                "recommendations": ["카메라에 전신이 보이도록 조정해주세요"]
            }
        
        # 간단한 자세 분석 로직
        landmarks = results.pose_landmarks.landmark
        
        # 어깨 기울기 분석
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
        
        # 목 자세 분석 (귀와 어깨 위치)
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        neck_alignment = abs(left_ear.x - left_shoulder.x)
        
        # 척추 정렬 분석
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        spine_alignment = abs(nose.x - left_hip.x)
        
        # 점수 계산 (0-100)
        shoulder_score = max(0, 100 - shoulder_slope * 1000)
        neck_score = max(0, 100 - neck_alignment * 500)
        spine_score = max(0, 100 - spine_alignment * 300)
        
        overall_score = (shoulder_score + neck_score + spine_score) / 3
        
        # 추천사항 생성
        recommendations = []
        if shoulder_score < 80:
            recommendations.append("어깨 균형을 위한 스트레칭을 권장합니다")
        if neck_score < 70:
            recommendations.append("목 자세 교정을 위한 운동이 필요합니다")
        if spine_score < 75:
            recommendations.append("척추 정렬을 위한 코어 운동을 추천합니다")
        
        if not recommendations:
            recommendations.append("훌륭한 자세입니다! 현재 상태를 유지하세요")
        
        return {
            "overall_score": round(overall_score, 1),
            "details": {
                "shoulder_balance": round(shoulder_score, 1),
                "neck_posture": round(neck_score, 1),
                "spine_alignment": round(spine_score, 1)
            },
            "recommendations": recommendations
        }
        
    except Exception as e:
        return {
            "overall_score": 0,
            "details": {"error": str(e)},
            "recommendations": ["이미지 분석 중 오류가 발생했습니다"]
        }

# 자세 분석 API
@app.post("/api/analyze-posture", response_model=PostureResult)
async def analyze_posture(request: PostureAnalysisRequest):
    if request.user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    analysis = analyze_pose(request.image_data)
    
    result = PostureResult(
        overall_score=analysis["overall_score"],
        details=analysis["details"],
        recommendations=analysis["recommendations"],
        timestamp=datetime.now()
    )
    
    # 결과 저장
    if request.user_id not in analysis_results_db:
        analysis_results_db[request.user_id] = []
    analysis_results_db[request.user_id].append(result.dict())
    
    return result

# 분석 기록 조회
@app.get("/api/analysis-history/{user_id}")
async def get_analysis_history(user_id: str, limit: int = 10):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    history = analysis_results_db.get(user_id, [])
    return {"user_id": user_id, "history": history[-limit:]}

# 운동 추천 API
@app.get("/api/exercise-recommendations/{user_id}")
async def get_exercise_recommendations(user_id: str):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # 최근 분석 결과 기반 운동 추천
    recent_analyses = analysis_results_db.get(user_id, [])
    
    recommendations = [
        ExerciseRecommendation(
            exercise_name="목 스트레칭",
            duration=5,
            difficulty="easy",
            target_areas=["neck", "shoulders"],
            instructions=[
                "천천히 목을 좌우로 돌리기",
                "목을 앞뒤로 숙이기",
                "어깨를 위아래로 움직이기"
            ]
        ),
        ExerciseRecommendation(
            exercise_name="코어 강화 운동",
            duration=15,
            difficulty="medium",
            target_areas=["core", "spine"],
            instructions=[
                "플랭크 자세 30초 유지",
                "사이드 플랭크 각 방향 20초",
                "데드버그 운동 10회씩"
            ]
        )
    ]
    
    return {"user_id": user_id, "recommendations": recommendations}

# 이미지 업로드 엔드포인트
@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    base64_encoded = base64.b64encode(contents).decode('utf-8')
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "base64_data": f"data:{file.content_type};base64,{base64_encoded}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# app/main.py 끝에 추가
from app.websocket_server import websocket_endpoint, get_websocket_metrics

# WebSocket 라우트는 자동으로 추가됨