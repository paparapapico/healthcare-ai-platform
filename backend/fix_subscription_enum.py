#!/usr/bin/env python3
"""
SQLite 데이터베이스의 enum 문제 해결 스크립트
"""

from sqlalchemy import create_engine, text

def fix_subscription_enum():
    """구독 tier 문제 해결"""
    try:
        engine = create_engine("sqlite:///./healthcare_ai.db", echo=True)
        
        with engine.connect() as conn:
            # 기존 users 테이블 수정 - subscription_tier를 VARCHAR로 변경
            print("구독 tier 컬럼 수정 중...")
            
            # 새 테이블 생성 (subscription_tier를 TEXT로)
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users_new (
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
                    subscription_tier TEXT DEFAULT 'basic',
                    subscription_expires DATETIME,
                    is_active BOOLEAN DEFAULT 1,
                    is_verified BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login DATETIME
                )
            """))
            
            # 기존 데이터 복사 (있다면)
            conn.execute(text("""
                INSERT OR IGNORE INTO users_new 
                SELECT id, email, hashed_password, name, birth_date, gender, 
                       height, weight, profile_image, health_score, 
                       'basic' as subscription_tier, subscription_expires, 
                       is_active, is_verified, created_at, updated_at, last_login
                FROM users
            """))
            
            # 기존 테이블 삭제하고 새 테이블로 교체
            conn.execute(text("DROP TABLE IF EXISTS users"))
            conn.execute(text("ALTER TABLE users_new RENAME TO users"))
            
            # 테스트 사용자 추가/업데이트
            conn.execute(text("""
                INSERT OR REPLACE INTO users 
                (id, email, hashed_password, name, birth_date, gender, height, weight, subscription_tier, is_active, is_verified)
                VALUES (1, 'test@example.com', '$2b$12$dummy_hash_for_testing', 'Test User', '1990-01-01', 'male', 175, 70.0, 'basic', 1, 1)
            """))
            
            conn.commit()
            
        print("✅ 구독 tier 문제가 해결되었습니다!")
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

if __name__ == "__main__":
    fix_subscription_enum()