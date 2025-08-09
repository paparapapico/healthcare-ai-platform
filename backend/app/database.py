# app/db/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 데이터베이스 URL 가져오기
DATABASE_URL = "postgresql://healthcare_user:healthcare123@localhost:5432/healthcare_db"

if not DATABASE_URL:
    # .env 파일이 없거나 DATABASE_URL이 설정되지 않은 경우 기본값 사용
    DATABASE_URL = "postgresql://healthcare_user:healthcare123@localhost:5432/healthcare_db"
    print(f"⚠️  Using default DATABASE_URL: {DATABASE_URL}")

# SQLAlchemy 엔진 생성
engine = create_engine(DATABASE_URL)

# 세션 팩토리 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 베이스 클래스 생성
Base = declarative_base()

# 데이터베이스 세션 의존성
def get_db():
    """FastAPI 의존성 주입용 데이터베이스 세션"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 데이터베이스 초기화
def init_db():
    """데이터베이스 테이블 생성"""
    try:
        # 모든 모델 임포트
        from app.models import user, workout  # 기존 모델들
        
        # 테이블 생성
        Base.metadata.create_all(bind=engine)
        print("✅ 데이터베이스 테이블 생성/업데이트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 데이터베이스 초기화 실패: {e}")
        return False

# 데이터베이스 연결 테스트
def test_connection():
    """데이터베이스 연결 테스트"""
    try:
        with engine.connect() as connection:
            result = connection.execute("SELECT 1")
            print("✅ 데이터베이스 연결 성공!")
            return True
    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        return False

if __name__ == "__main__":
    test_connection()