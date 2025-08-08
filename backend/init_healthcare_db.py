# simple_init.py - 간단한 Healthcare AI 데이터베이스 초기화
import sys
import os
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 데이터베이스 연결
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://healthcare_user:healthcare123@localhost:5432/healthcare_db")
engine = create_engine(DATABASE_URL)
Base = declarative_base()

def create_tables_directly():
    """SQL로 직접 테이블 생성"""
    print("📊 테이블 직접 생성 중...")
    
    try:
        with engine.connect() as conn:
            # 트랜잭션 시작
            trans = conn.begin()
            
            try:
                # 기존 테이블이 있다면 스키마 확인만
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """))
                existing_tables = [row[0] for row in result.fetchall()]
                
                # Healthcare 테이블들 생성 (존재하지 않는 경우만)
                healthcare_tables = [
                    """
                    CREATE TABLE IF NOT EXISTS health_profiles (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        height REAL,
                        weight REAL,
                        activity_level VARCHAR(20) DEFAULT 'moderate',
                        health_conditions JSON DEFAULT '[]',
                        medications JSON DEFAULT '[]',
                        allergies JSON DEFAULT '[]',
                        fitness_goals TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS exercises (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(200) NOT NULL,
                        category VARCHAR(50) NOT NULL,
                        difficulty VARCHAR(20) DEFAULT 'beginner',
                        duration_minutes INTEGER DEFAULT 5,
                        instructions JSON NOT NULL,
                        target_areas JSON NOT NULL,
                        benefits TEXT,
                        precautions TEXT,
                        video_url VARCHAR(500),
                        image_url VARCHAR(500),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS pose_analyses (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        image_path VARCHAR(500),
                        overall_score REAL NOT NULL,
                        neck_score REAL,
                        shoulder_score REAL,
                        spine_score REAL,
                        hip_score REAL,
                        pose_landmarks JSON,
                        analysis_details JSON,
                        recommendations JSON NOT NULL,
                        analysis_duration REAL,
                        device_info VARCHAR(200),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS exercise_recommendations (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        pose_analysis_id INTEGER,
                        exercise_id INTEGER,
                        priority INTEGER DEFAULT 1,
                        reason TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        FOREIGN KEY (pose_analysis_id) REFERENCES pose_analyses(id),
                        FOREIGN KEY (exercise_id) REFERENCES exercises(id)
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS user_progress (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        average_score REAL NOT NULL,
                        total_analyses INTEGER DEFAULT 0,
                        improvement_rate REAL,
                        avg_neck_score REAL,
                        avg_shoulder_score REAL,
                        avg_spine_score REAL,
                        avg_hip_score REAL
                    )
                    """
                ]
                
                for table_sql in healthcare_tables:
                    conn.execute(text(table_sql))
                    
                trans.commit()
                
                # 생성된 테이블 확인
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """))
                tables = [row[0] for row in result.fetchall()]
                print(f"✅ 테이블 생성 완료 ({len(tables)}개): {', '.join(tables)}")
                
                return True
                
            except Exception as e:
                trans.rollback()
                print(f"❌ 테이블 생성 실패: {e}")
                return False
                
    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        return False

def insert_sample_data():
    """샘플 운동 데이터 삽입"""
    print("💪 샘플 운동 데이터 삽입 중...")
    
    try:
        with engine.connect() as conn:
            # 기존 데이터 확인
            result = conn.execute(text("SELECT COUNT(*) FROM exercises"))
            count = result.fetchone()[0]
            
            if count > 0:
                print(f"ℹ️  기존 운동 데이터 {count}개 발견, 새 데이터는 추가하지 않습니다.")
                return True
            
            # 샘플 운동 데이터
            sample_exercises = [
                {
                    "name": "목 스트레칭",
                    "category": "neck",
                    "instructions": '["편안한 자세로 앉기", "천천히 목을 좌우로 기울이기", "15초씩 유지하기"]',
                    "target_areas": '["목", "어깨"]',
                    "benefits": "목 근육 긴장 완화, 거북목 개선"
                },
                {
                    "name": "어깨 돌리기", 
                    "category": "shoulder",
                    "instructions": '["팔을 자연스럽게 내리고 서기", "어깨를 천천히 앞뒤로 돌리기"]',
                    "target_areas": '["어깨", "상부 등"]',
                    "benefits": "어깨 근육 이완, 혈액순환 개선"
                },
                {
                    "name": "플랭크",
                    "category": "core",
                    "difficulty": "intermediate",
                    "duration_minutes": 3,
                    "instructions": '["팔꿈치로 몸 지탱하기", "30초간 유지하기", "3세트 반복"]',
                    "target_areas": '["코어", "척추"]',
                    "benefits": "코어 근력 강화, 자세 개선"
                }
            ]
            
            # 데이터 삽입
            for exercise in sample_exercises:
                insert_sql = text("""
                    INSERT INTO exercises (name, category, difficulty, duration_minutes, instructions, target_areas, benefits)
                    VALUES (:name, :category, :difficulty, :duration_minutes, :instructions, :target_areas, :benefits)
                """)
                
                conn.execute(insert_sql, {
                    'name': exercise['name'],
                    'category': exercise['category'],
                    'difficulty': exercise.get('difficulty', 'beginner'),
                    'duration_minutes': exercise.get('duration_minutes', 5),
                    'instructions': exercise['instructions'],
                    'target_areas': exercise['target_areas'],
                    'benefits': exercise['benefits']
                })
            
            conn.commit()
            print(f"✅ 샘플 운동 데이터 {len(sample_exercises)}개 삽입 완료!")
            return True
            
    except Exception as e:
        print(f"❌ 샘플 데이터 삽입 실패: {e}")
        return False

def main():
    print("🏥 Healthcare AI 데이터베이스 간단 초기화 시작...\n")
    
    # 1. 연결 테스트
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ 데이터베이스 연결 성공: {version[:50]}...\n")
    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        return False
    
    # 2. 테이블 생성
    if not create_tables_directly():
        return False
    
    # 3. 샘플 데이터 삽입
    if not insert_sample_data():
        return False
    
    print("\n🎉 Healthcare AI 데이터베이스 초기화 완료!")
    print("🚀 이제 FastAPI 서버를 시작할 수 있습니다:")
    print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)