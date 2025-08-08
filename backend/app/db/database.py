"""
Database Configuration and Session Management
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/healthcare_db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    echo=False  # Set to True for SQL query logging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Dependency for FastAPI
def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency for FastAPI
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ========================
# Database Initialization
# ========================

def init_db():
    """
    Initialize database with tables and seed data
    """
    from app.models.models import Base, Achievement
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Seed initial achievements
    db = SessionLocal()
    try:
        # Check if achievements already exist
        existing = db.query(Achievement).first()
        if not existing:
            seed_achievements(db)
            print("✅ Database initialized with seed data")
        else:
            print("ℹ️ Database already initialized")
    finally:
        db.close()

def seed_achievements(db: Session):
    """
    Seed initial achievements
    """
    from app.models.models import Achievement
    
    achievements = [
        # Workout Achievements
        {
            "name": "첫 운동",
            "description": "첫 운동을 완료했습니다!",
            "category": "workout",
            "icon": "🎯",
            "points": 10,
            "requirement_type": "count",
            "requirement_value": 1
        },
        {
            "name": "운동 매니아",
            "description": "100회 운동 달성",
            "category": "workout",
            "icon": "💪",
            "points": 100,
            "requirement_type": "count",
            "requirement_value": 100
        },
        {
            "name": "스쿼트 마스터",
            "description": "스쿼트 1000개 달성",
            "category": "workout",
            "icon": "🏋️",
            "points": 50,
            "requirement_type": "total",
            "requirement_value": 1000
        },
        {
            "name": "완벽한 자세",
            "description": "자세 점수 95점 이상 달성",
            "category": "workout",
            "icon": "⭐",
            "points": 30,
            "requirement_type": "score",
            "requirement_value": 95
        },
        
        # Streak Achievements
        {
            "name": "3일 연속",
            "description": "3일 연속 운동 완료",
            "category": "streak",
            "icon": "🔥",
            "points": 20,
            "requirement_type": "streak",
            "requirement_value": 3
        },
        {
            "name": "일주일 전사",
            "description": "7일 연속 운동 완료",
            "category": "streak",
            "icon": "🔥🔥",
            "points": 50,
            "requirement_type": "streak",
            "requirement_value": 7
        },
        {
            "name": "한달 챔피언",
            "description": "30일 연속 운동 완료",
            "category": "streak",
            "icon": "🔥🔥🔥",
            "points": 200,
            "requirement_type": "streak",
            "requirement_value": 30
        },
        
        # Health Achievements
        {
            "name": "건강 점수 80",
            "description": "건강 점수 80점 달성",
            "category": "health",
            "icon": "❤️",
            "points": 30,
            "requirement_type": "health_score",
            "requirement_value": 80
        },
        {
            "name": "건강 점수 90",
            "description": "건강 점수 90점 달성",
            "category": "health",
            "icon": "💖",
            "points": 50,
            "requirement_type": "health_score",
            "requirement_value": 90
        },
        {
            "name": "칼로리 버너",
            "description": "하루 500kcal 소모",
            "category": "health",
            "icon": "🔥",
            "points": 25,
            "requirement_type": "daily_calories",
            "requirement_value": 500
        },
        
        # Milestone Achievements
        {
            "name": "첫 친구",
            "description": "첫 친구를 추가했습니다",
            "category": "social",
            "icon": "👥",
            "points": 15,
            "requirement_type": "friends",
            "requirement_value": 1
        },
        {
            "name": "챌린지 우승",
            "description": "챌린지에서 1등 달성",
            "category": "social",
            "icon": "🏆",
            "points": 100,
            "requirement_type": "challenge_win",
            "requirement_value": 1
        }
    ]
    
    for achievement_data in achievements:
        achievement = Achievement(**achievement_data)
        db.add(achievement)
    
    db.commit()
    print(f"✅ Seeded {len(achievements)} achievements")

# ========================
# Database Utilities
# ========================

def reset_database():
    """
    Drop all tables and recreate (DANGER: Deletes all data!)
    """
    from app.models.models import Base
    
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    init_db()
    print("⚠️ Database reset complete")

def get_db_stats():
    """
    Get database statistics
    """
    from app.models.models import User, Workout, HealthMetric
    
    db = SessionLocal()
    try:
        stats = {
            "total_users": db.query(User).count(),
            "total_workouts": db.query(Workout).count(),
            "total_health_metrics": db.query(HealthMetric).count(),
        }
        return stats
    finally:
        db.close()

# ========================
# Alembic Configuration
# ========================

"""
Alembic 마이그레이션 설정:

1. 초기화:
   alembic init alembic

2. alembic.ini 수정:
   sqlalchemy.url = postgresql://localhost/healthcare_db

3. alembic/env.py 수정:
   from app.models.models import Base
   target_metadata = Base.metadata

4. 첫 마이그레이션 생성:
   alembic revision --autogenerate -m "Initial migration"

5. 마이그레이션 실행:
   alembic upgrade head
"""

if __name__ == "__main__":
    # Initialize database when run directly
    init_db()