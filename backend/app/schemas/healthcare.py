# app/schemas/healthcare.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import re

# 건강 프로필 스키마
class HealthProfileBase(BaseModel):
    height: Optional[float] = Field(None, ge=50, le=250, description="키 (cm)")
    weight: Optional[float] = Field(None, ge=20, le=300, description="몸무게 (kg)")
    activity_level: str = Field("moderate", pattern="^(low|moderate|high)$")
    health_conditions: List[str] = Field(default=[], description="건강 상태")
    medications: List[str] = Field(default=[], description="복용 약물")
    allergies: List[str] = Field(default=[], description="알레르기")
    fitness_goals: Optional[str] = Field(None, description="운동 목표")

class HealthProfileCreate(HealthProfileBase):
    user_id: int

class HealthProfileUpdate(HealthProfileBase):
    pass

class HealthProfileResponse(HealthProfileBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

# 자세 분석 스키마
class PoseAnalysisBase(BaseModel):
    overall_score: float = Field(..., ge=0, le=100, description="전체 점수")
    neck_score: Optional[float] = Field(None, ge=0, le=100)
    shoulder_score: Optional[float] = Field(None, ge=0, le=100)
    spine_score: Optional[float] = Field(None, ge=0, le=100)
    hip_score: Optional[float] = Field(None, ge=0, le=100)
    pose_landmarks: Optional[dict] = None
    analysis_details: Optional[dict] = None
    recommendations: List[str] = Field(..., description="개선 추천사항")
    analysis_duration: Optional[float] = None
    device_info: Optional[str] = None

class PoseAnalysisCreate(PoseAnalysisBase):
    user_id: int
    image_data: str = Field(..., description="Base64 인코딩된 이미지 데이터")

class PoseAnalysisResponse(PoseAnalysisBase):
    id: int
    user_id: int
    image_path: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

# 운동 스키마
class ExerciseBase(BaseModel):
    name: str = Field(..., max_length=200)
    category: str = Field(..., max_length=50)
    difficulty: str = Field("beginner", pattern="^(beginner|intermediate|advanced)$")
    duration_minutes: int = Field(5, ge=1, le=120)
    instructions: List[str] = Field(..., description="운동 방법 단계별 설명")
    target_areas: List[str] = Field(..., description="타겟 부위")
    benefits: Optional[str] = None
    precautions: Optional[str] = None
    video_url: Optional[str] = None
    image_url: Optional[str] = None

class ExerciseCreate(ExerciseBase):
    pass

class ExerciseResponse(ExerciseBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

# 운동 추천 스키마
class ExerciseRecommendationBase(BaseModel):
    priority: int = Field(1, ge=1, le=10)
    reason: Optional[str] = None

class ExerciseRecommendationCreate(ExerciseRecommendationBase):
    user_id: int
    pose_analysis_id: int
    exercise_id: int

class ExerciseRecommendationResponse(ExerciseRecommendationBase):
    id: int
    user_id: int
    pose_analysis_id: int
    exercise_id: int
    exercise: ExerciseResponse
    created_at: datetime

    class Config:
        from_attributes = True

# 사용자 진행 상황 스키마
class UserProgressBase(BaseModel):
    average_score: float = Field(..., ge=0, le=100)
    total_analyses: int = Field(0, ge=0)
    improvement_rate: Optional[float] = None
    avg_neck_score: Optional[float] = Field(None, ge=0, le=100)
    avg_shoulder_score: Optional[float] = Field(None, ge=0, le=100)
    avg_spine_score: Optional[float] = Field(None, ge=0, le=100)
    avg_hip_score: Optional[float] = Field(None, ge=0, le=100)

class UserProgressCreate(UserProgressBase):
    user_id: int

class UserProgressResponse(UserProgressBase):
    id: int
    user_id: int
    date: datetime

    class Config:
        from_attributes = True

# 자세 분석 요청 스키마
class PoseAnalysisRequest(BaseModel):
    user_id: int
    image_data: str = Field(..., description="Base64 인코딩된 이미지 데이터")

# 자세 분석 결과 스키마  
class PoseAnalysisResult(BaseModel):
    overall_score: float
    details: dict
    recommendations: List[str]
    timestamp: datetime

    class Config:
        from_attributes = True