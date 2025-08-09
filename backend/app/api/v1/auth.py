from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import date, datetime, timedelta
from app.db.database import get_db
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
import bcrypt
import jwt


from app.db.database import get_db
from app.models.models import User
from app.schemas.auth import UserCreate, UserResponse, Token


# OAuth2 설정
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 라우터 생성 - 이 부분이 중요!
router = APIRouter()

# JWT 설정
SECRET_KEY = "your-secret-key-here"  # 실제로는 환경변수에서 가져와야 함
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def hash_password(password: str) -> str:
    """비밀번호 해시화"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """비밀번호 검증"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """JWT 토큰 생성"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user_id() -> int:
    """현재 사용자 ID 반환 (테스트용 - 실제로는 JWT에서 추출)"""
    return 1  # 테스트용 고정값

# Pydantic 모델들
class UserCreate(BaseModel):
    email: str
    password: str
    name: str
    birth_date: date  # 자동으로 문자열을 date로 변환
    gender: Optional[str] = None
    height: Optional[int] = None
    weight: Optional[float] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    birth_date: date
    gender: str
    height: Optional[int] = None
    weight: Optional[float] = None
    subscription_tier: str
    is_active: bool
    created_at: datetime  # 이 필드 추가
    updated_at: Optional[datetime] = None  # 선택적으로 추가
    
    class Config:
        from_attributes = True

@router.post("/register", response_model=UserResponse)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """회원가입"""
    try:
        # 이메일 중복 확인
        existing_user = db.query(User).filter(User.email == user.email).first()
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
        
        # 새 사용자 생성
        hashed_pw = hash_password(user.password)
        db_user = User(
            email=user.email,
            hashed_password=hashed_pw,
            name=user.name,
            birth_date=user.birth_date,
            gender=user.gender,
            height=user.height,
            weight=user.weight,
            is_active=True,
            is_verified=False
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        return UserResponse(
            id=db_user.id,
            email=db_user.email,
            name=db_user.name,
            is_active=db_user.is_active
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.post("/login", response_model=Token)
async def login_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """로그인"""
    try:
        # 사용자 찾기
        user = db.query(User).filter(User.email == form_data.username).first()
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 토큰 생성
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user.id)}, expires_delta=access_token_expires
        )
        
        # 마지막 로그인 시간 업데이트
        user.last_login = datetime.utcnow()
        db.commit()
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@router.get("/me", response_model=UserResponse)
async def get_current_user(current_user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    """현재 사용자 정보 조회"""
    user = db.query(User).filter(User.id == current_user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        is_active=user.is_active
    )