"""
Database Models for Healthcare Platform
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON, Enum, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()

# ========================
# Enums
# ========================

class GenderEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"

class ExerciseTypeEnum(str, enum.Enum):
    SQUAT = "squat"
    PUSHUP = "pushup"
    PLANK = "plank"
    LUNGE = "lunge"
    JUMPING_JACK = "jumping_jack"
    SHOULDER_PRESS = "shoulder_press"

class SubscriptionTierEnum(str, enum.Enum):
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    PRO = "pro"

# ========================
# User Models
# ========================

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    
    # Profile
    name = Column(String, nullable=False)
    birth_date = Column(Date)
    gender = Column(String(10))  # Enum 대신 일반 문자열 사용
    height = Column(Float)  # cm
    weight = Column(Float)  # kg
    profile_image = Column(String)
    
    # Health Score
    health_score = Column(Float, default=75.0)
    
    # Subscription
    subscription_tier = Column(Enum(SubscriptionTierEnum), default=SubscriptionTierEnum.FREE)
    subscription_expires = Column(DateTime)
    
    # Metadata
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    workouts = relationship("Workout", back_populates="user", cascade="all, delete-orphan")
    health_metrics = relationship("HealthMetric", back_populates="user", cascade="all, delete-orphan")
    goals = relationship("Goal", back_populates="user", cascade="all, delete-orphan")
    achievements = relationship("UserAchievement", back_populates="user", cascade="all, delete-orphan")
     # ... 다른 필드들
    created_at = Column(DateTime, default=datetime.utcnow)  # 이 필드가 있어야 함
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
# ========================
# Workout Models
# ========================

class Workout(Base):
    __tablename__ = "workouts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Workout Details
    exercise_type = Column(Enum(ExerciseTypeEnum), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    duration = Column(Integer)  # seconds
    
    # Performance
    total_reps = Column(Integer, default=0)
    calories_burned = Column(Float, default=0)
    avg_form_score = Column(Float)
    max_form_score = Column(Float)
    min_form_score = Column(Float)
    
    # Analysis Data
    angles_data = Column(JSON)  # Store angle measurements over time
    stage_transitions = Column(JSON)  # Track stage changes
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="workouts")
    exercise_sets = relationship("ExerciseSet", back_populates="workout", cascade="all, delete-orphan")

class ExerciseSet(Base):
    __tablename__ = "exercise_sets"
    
    id = Column(Integer, primary_key=True, index=True)
    workout_id = Column(Integer, ForeignKey("workouts.id"), nullable=False)
    
    set_number = Column(Integer, nullable=False)
    reps = Column(Integer, nullable=False)
    duration = Column(Float)  # seconds
    form_score = Column(Float)
    rest_time = Column(Float)  # seconds before next set
    
    # Relationships
    workout = relationship("Workout", back_populates="exercise_sets")

# ========================
# Health Metrics
# ========================

class HealthMetric(Base):
    __tablename__ = "health_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Basic Metrics
    weight = Column(Float)  # kg
    body_fat_percentage = Column(Float)
    muscle_mass = Column(Float)  # kg
    bmi = Column(Float)
    
    # Vital Signs
    heart_rate_resting = Column(Integer)  # bpm
    heart_rate_max = Column(Integer)  # bpm
    blood_pressure_systolic = Column(Integer)  # mmHg
    blood_pressure_diastolic = Column(Integer)  # mmHg
    blood_oxygen = Column(Float)  # SpO2 percentage
    
    # Activity
    steps = Column(Integer)
    active_minutes = Column(Integer)
    calories_burned = Column(Float)
    distance_walked = Column(Float)  # km
    floors_climbed = Column(Integer)
    
    # Sleep
    sleep_hours = Column(Float)
    sleep_quality_score = Column(Float)
    deep_sleep_hours = Column(Float)
    rem_sleep_hours = Column(Float)
    
    # Wellness
    stress_level = Column(Integer)  # 1-10
    energy_level = Column(Integer)  # 1-10
    mood_score = Column(Integer)  # 1-10
    water_intake = Column(Float)  # liters
    
    # Timestamp
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="health_metrics")

# ========================
# Goals & Achievements
# ========================

class Goal(Base):
    __tablename__ = "goals"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    title = Column(String, nullable=False)
    description = Column(String)
    category = Column(String)  # weight_loss, muscle_gain, endurance, etc.
    
    # Target
    target_value = Column(Float)
    target_unit = Column(String)  # kg, reps, minutes, etc.
    current_value = Column(Float, default=0)
    
    # Timeline
    start_date = Column(Date, nullable=False)
    target_date = Column(Date, nullable=False)
    completed_date = Column(Date)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_completed = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="goals")

class Achievement(Base):
    __tablename__ = "achievements"
    
    id = Column(Integer, primary_key=True, index=True)
    
    name = Column(String, nullable=False, unique=True)
    description = Column(String)
    category = Column(String)  # workout, health, streak, milestone
    icon = Column(String)
    points = Column(Integer, default=10)
    
    # Requirements
    requirement_type = Column(String)  # count, streak, total
    requirement_value = Column(Integer)
    
    # Relationships
    user_achievements = relationship("UserAchievement", back_populates="achievement")

class UserAchievement(Base):
    __tablename__ = "user_achievements"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    achievement_id = Column(Integer, ForeignKey("achievements.id"), nullable=False)
    
    earned_at = Column(DateTime, default=datetime.utcnow)
    progress = Column(Float, default=0)  # 0-100 percentage
    
    # Relationships
    user = relationship("User", back_populates="achievements")
    achievement = relationship("Achievement", back_populates="user_achievements")

# ========================
# Social Features
# ========================

class Friendship(Base):
    __tablename__ = "friendships"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    friend_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    status = Column(String, default="pending")  # pending, accepted, blocked
    created_at = Column(DateTime, default=datetime.utcnow)
    accepted_at = Column(DateTime)

class Challenge(Base):
    __tablename__ = "challenges"
    
    id = Column(Integer, primary_key=True, index=True)
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    title = Column(String, nullable=False)
    description = Column(String)
    challenge_type = Column(String)  # individual, team
    exercise_type = Column(Enum(ExerciseTypeEnum))
    
    # Target
    target_reps = Column(Integer)
    target_duration = Column(Integer)  # minutes
    target_calories = Column(Float)
    
    # Timeline
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    participants = relationship("ChallengeParticipant", back_populates="challenge")

class ChallengeParticipant(Base):
    __tablename__ = "challenge_participants"
    
    id = Column(Integer, primary_key=True, index=True)
    challenge_id = Column(Integer, ForeignKey("challenges.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Progress
    current_reps = Column(Integer, default=0)
    current_duration = Column(Integer, default=0)
    current_calories = Column(Float, default=0)
    
    # Status
    joined_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    rank = Column(Integer)
    
    # Relationships
    challenge = relationship("Challenge", back_populates="participants")

# ========================
# Content & Programs
# ========================

class WorkoutProgram(Base):
    __tablename__ = "workout_programs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    name = Column(String, nullable=False)
    description = Column(String)
    difficulty = Column(String)  # beginner, intermediate, advanced
    duration_weeks = Column(Integer)
    
    # Content
    program_data = Column(JSON)  # Detailed program structure
    thumbnail_url = Column(String)
    
    # Metadata
    created_by = Column(String)  # admin, trainer name
    created_at = Column(DateTime, default=datetime.utcnow)
    is_premium = Column(Boolean, default=False)
    
    # Relationships
    enrollments = relationship("ProgramEnrollment", back_populates="program")

class ProgramEnrollment(Base):
    __tablename__ = "program_enrollments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    program_id = Column(Integer, ForeignKey("workout_programs.id"), nullable=False)
    
    # Progress
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    current_week = Column(Integer, default=1)
    current_day = Column(Integer, default=1)
    completion_percentage = Column(Float, default=0)
    
    # Relationships
    program = relationship("WorkoutProgram", back_populates="enrollments")