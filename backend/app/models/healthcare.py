# app/models/healthcare.py
from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship  # 이렇게 수정
from sqlalchemy.sql import func
from backend.app.database import Base

class HealthProfile(Base):
    """건강 프로필 모델"""
    __tablename__ = "health_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    height = Column(Float, nullable=True)  # cm
    weight = Column(Float, nullable=True)  # kg
    activity_level = Column(String(20), default="moderate")  # low, moderate, high
    health_conditions = Column(JSON, default=list)  # 건강 상태 리스트
    medications = Column(JSON, default=list)  # 복용 약물
    allergies = Column(JSON, default=list)  # 알레르기
    fitness_goals = Column(Text, nullable=True)  # 운동 목표
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # 관계 설정 (기존 User 모델과 연결)
    # user = relationship("User", back_populates="health_profile")

class PoseAnalysis(Base):
    """자세 분석 결과 모델"""
    __tablename__ = "pose_analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    image_path = Column(String(500), nullable=True)  # 분석한 이미지 경로
    overall_score = Column(Float, nullable=False)  # 전체 점수 (0-100)
    
    # 상세 분석 점수
    neck_score = Column(Float, nullable=True)  # 목 자세 점수
    shoulder_score = Column(Float, nullable=True)  # 어깨 균형 점수
    spine_score = Column(Float, nullable=True)  # 척추 정렬 점수
    hip_score = Column(Float, nullable=True)  # 골반 균형 점수
    
    # MediaPipe 데이터
    pose_landmarks = Column(JSON, nullable=True)  # 포즈 랜드마크 좌표
    
    # 분석 결과 및 추천사항
    analysis_details = Column(JSON, nullable=True)  # 상세 분석 결과
    recommendations = Column(JSON, nullable=False)  # 개선 추천사항
    
    # 분석 메타데이터
    analysis_duration = Column(Float, nullable=True)  # 분석 소요 시간 (초)
    device_info = Column(String(200), nullable=True)  # 분석한 기기 정보
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 관계 설정
    # user = relationship("User", back_populates="pose_analyses")

class Exercise(Base):
    """운동 정보 모델"""
    __tablename__ = "exercises"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    category = Column(String(50), nullable=False)  # neck, shoulder, spine, core, etc.
    difficulty = Column(String(20), default="beginner")  # beginner, intermediate, advanced
    duration_minutes = Column(Integer, default=5)
    instructions = Column(JSON, nullable=False)  # 운동 방법 단계별 설명
    target_areas = Column(JSON, nullable=False)  # 타겟 부위
    benefits = Column(Text, nullable=True)  # 운동 효과
    precautions = Column(Text, nullable=True)  # 주의사항
    video_url = Column(String(500), nullable=True)  # 참고 비디오 URL
    image_url = Column(String(500), nullable=True)  # 참고 이미지 URL
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ExerciseRecommendation(Base):
    """운동 추천 모델"""
    __tablename__ = "exercise_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    pose_analysis_id = Column(Integer, ForeignKey("pose_analyses.id"), nullable=False)
    exercise_id = Column(Integer, ForeignKey("exercises.id"), nullable=False)
    priority = Column(Integer, default=1)  # 추천 우선순위 (1이 가장 높음)
    reason = Column(Text, nullable=True)  # 추천 이유
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 관계 설정
    # user = relationship("User")
    pose_analysis = relationship("PoseAnalysis")
    exercise = relationship("Exercise")

class UserProgress(Base):
    """사용자 진행 상황 모델"""
    __tablename__ = "user_progress"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    date = Column(DateTime(timezone=True), server_default=func.now())
    average_score = Column(Float, nullable=False)  # 하루 평균 점수
    total_analyses = Column(Integer, default=0)  # 하루 분석 횟수
    improvement_rate = Column(Float, nullable=True)  # 전일 대비 개선율
    
    # 부위별 평균 점수
    avg_neck_score = Column(Float, nullable=True)
    avg_shoulder_score = Column(Float, nullable=True)
    avg_spine_score = Column(Float, nullable=True)
    avg_hip_score = Column(Float, nullable=True)

    # 관계 설정
    # user = relationship("User")