#!/usr/bin/env python3
"""
SQLite 데이터베이스 테이블 생성 스크립트
"""

import sys
import os
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

def create_sqlite_tables():
    """SQLite 데이터베이스와 테이블 생성"""
    try:
        # 1. database.py를 SQLite로 변경
        print("1. 데이터베이스 설정을 SQLite로 변경 중...")
        
        # 2. SQLite engine으로 테이블 생성
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import declarative_base
        
        # SQLite 데이터베이스 생성
        engine = create_engine("sqlite:///./healthcare_ai.db", echo=True)
        Base = declarative_base()
        
        # 3. 기본 테이블들 수동 생성
        with engine.connect() as conn:
            # Users 테이블
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    hashed_password VARCHAR(255) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    birth_date DATE,
                    gender VARCHAR(10),
                    height INTEGER,
                    weight FLOAT,
                    profile_image VARCHAR(500),
                    health_score FLOAT DEFAULT 0.0,
                    subscription_tier VARCHAR(20) DEFAULT 'basic',
                    subscription_expires DATETIME,
                    is_active BOOLEAN DEFAULT 1,
                    is_verified BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login DATETIME
                )
            """))
            
            # Health Metrics 테이블
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS health_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    weight FLOAT,
                    body_fat_percentage FLOAT,
                    muscle_mass FLOAT,
                    bmi FLOAT,
                    heart_rate_resting INTEGER,
                    heart_rate_max INTEGER,
                    blood_pressure_systolic INTEGER,
                    blood_pressure_diastolic INTEGER,
                    blood_oxygen FLOAT,
                    steps INTEGER,
                    active_minutes INTEGER,
                    calories_burned FLOAT,
                    distance_walked FLOAT,
                    floors_climbed INTEGER,
                    sleep_hours FLOAT,
                    sleep_quality_score FLOAT,
                    deep_sleep_hours FLOAT,
                    rem_sleep_hours FLOAT,
                    stress_level INTEGER,
                    energy_level INTEGER,
                    mood_score INTEGER,
                    water_intake INTEGER,
                    recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """))
            
            # Friendships 테이블
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS friendships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    friend_id INTEGER NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    accepted_at DATETIME,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (friend_id) REFERENCES users (id)
                )
            """))
            
            # Workouts 테이블
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS workouts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    exercise_type VARCHAR(100) NOT NULL,
                    duration INTEGER,
                    reps INTEGER,
                    sets INTEGER,
                    calories_burned FLOAT,
                    form_score FLOAT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """))
            
            # Challenges 테이블
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS challenges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    creator_id INTEGER NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    description TEXT,
                    exercise_type VARCHAR(100),
                    target_reps INTEGER,
                    target_duration INTEGER,
                    duration_days INTEGER,
                    is_public BOOLEAN DEFAULT 1,
                    start_date DATETIME,
                    end_date DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (creator_id) REFERENCES users (id)
                )
            """))
            
            conn.commit()
        
        print("✅ SQLite 데이터베이스와 테이블이 성공적으로 생성되었습니다!")
        print("📁 데이터베이스 파일: healthcare_ai.db")
        
        # 4. 테스트용 더미 데이터 생성
        print("\n2. 테스트용 더미 데이터 생성 중...")
        with engine.connect() as conn:
            # 테스트 사용자 생성
            conn.execute(text("""
                INSERT OR IGNORE INTO users (id, email, hashed_password, name, birth_date, gender, height, weight, is_active, is_verified)
                VALUES (1, 'test@example.com', '$2b$12$dummy_hash_for_testing', 'Test User', '1990-01-01', 'male', 175, 70.0, 1, 1)
            """))
            conn.commit()
        
        print("✅ 테스트용 더미 데이터가 생성되었습니다!")
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

if __name__ == "__main__":
    success = create_sqlite_tables()
    if success:
        print("\n🎉 데이터베이스 설정이 완료되었습니다!")
        print("이제 'python tests/test_api.py' 또는 'pytest tests/test_api.py -v'를 실행할 수 있습니다.")
    else:
        print("\n❌ 데이터베이스 설정에 실패했습니다.")