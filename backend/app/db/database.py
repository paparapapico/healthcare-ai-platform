from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLite 데이터베이스 URL (개발/테스트용) - PostgreSQL 대신 사용
SQLALCHEMY_DATABASE_URL = "sqlite:///./healthcare_ai.db"

# PostgreSQL을 사용하는 경우 (현재 문제가 있어서 주석 처리):
# SQLALCHEMY_DATABASE_URL = "postgresql://postgres:healthcare123@localhost:5432/healthcare_ai"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}  # SQLite용 설정
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# 의존성 주입용 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()