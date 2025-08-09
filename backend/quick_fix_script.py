#!/usr/bin/env python3
"""
Healthcare AI 백엔드 빠른 수정 스크립트
"""

import os
import sys
from pathlib import Path

def main():
    print("🔧 Healthcare AI 백엔드 빠른 수정 시작...")
    
    # 1. SQLite 데이터베이스 파일 삭제 (새로 시작)
    db_file = Path("healthcare_ai.db")
    if db_file.exists():
        db_file.unlink()
        print("✅ 기존 SQLite 파일 삭제")
    
    # 2. SQLite 테이블 생성
    try:
        from sqlalchemy import create_engine, text
        from app.models.models import Base
        
        engine = create_engine("sqlite:///./healthcare_ai.db", echo=False)
        
        # 모든 테이블 생성
        Base.metadata.create_all(engine)
        print("✅ SQLite 테이블 생성 완료")
        
        # 테스트 사용자 생성
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO users (
                    id, email, hashed_password, name, birth_date, gender, 
                    height, weight, subscription_tier, is_active, is_verified, created_at
                ) VALUES (
                    1, 'test@example.com', '$2b$12$dummy_hash_for_testing', 
                    'Test User', '1990-01-01', 'male', 175, 70.0, 
                    'FREE', 1, 1, datetime('now')
                )
            """))
            conn.commit()
        print("✅ 테스트 사용자 생성 완료")
        
    except Exception as e:
        print(f"❌ 데이터베이스 설정 실패: {e}")
        return False
    
    # 3. 테스트 실행
    print("\n🧪 간단한 테스트 실행...")
    try:
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        
        # API 상태 확인
        response = client.get("/")
        if response.status_code == 200:
            print("✅ API 기본 엔드포인트 작동")
        
        # 회원가입 테스트
        test_user = {
            "email": "quicktest@example.com",
            "password": "test123",
            "name": "Quick Test",
            "birth_date": "1990-01-01",
            "gender": "male",
            "height": 175,
            "weight": 70.0
        }
        
        response = client.post("/api/v1/auth/register", json=test_user)
        if response.status_code == 200:
            print("✅ 회원가입 API 작동")
            print(f"   응답: {response.json()}")
        else:
            print(f"❌ 회원가입 실패: {response.status_code}")
            print(f"   에러: {response.text}")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False
    
    print("\n🎉 빠른 수정 완료!")
    print("이제 다음을 실행해보세요:")
    print("  python debug_test.py")
    
    return True

if __name__ == "__main__":
    # 현재 디렉토리를 Python 경로에 추가
    sys.path.insert(0, str(Path(__file__).parent))
    
    success = main()
    sys.exit(0 if success else 1)