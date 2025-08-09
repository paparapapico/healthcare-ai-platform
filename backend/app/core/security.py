"""
JWT Security System
JWT 기반 인증 시스템
파일 위치: backend/app/core/security.py
"""

from datetime import datetime, timedelta
from typing import Optional, Union, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import os
from pydantic import BaseModel

from app.db.database import get_db
from app.models.models import User

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# ========================
# Token Models
# ========================

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None
    scopes: list[str] = []

class RefreshTokenRequest(BaseModel):
    refresh_token: str

# ========================
# Password Utilities
# ========================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """비밀번호 검증"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """비밀번호 해싱"""
    return pwd_context.hash(password)

# ========================
# JWT Token Management
# ========================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """액세스 토큰 생성"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict) -> str:
    """리프레시 토큰 생성"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, token_type: str = "access") -> TokenData:
    """토큰 검증"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # 토큰 타입 확인
        if payload.get("type") != token_type:
            raise credentials_exception
        
        # 만료 시간 확인
        exp = payload.get("exp")
        if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id: int = payload.get("sub")
        email: str = payload.get("email")
        scopes: list = payload.get("scopes", [])
        
        if user_id is None:
            raise credentials_exception
        
        return TokenData(user_id=user_id, email=email, scopes=scopes)
        
    except JWTError:
        raise credentials_exception

def create_tokens(user_id: int, email: str, scopes: list[str] = []) -> Token:
    """액세스 토큰과 리프레시 토큰 생성"""
    token_data = {
        "sub": str(user_id),
        "email": email,
        "scopes": scopes
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )

# ========================
# Authentication Dependencies
# ========================

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """현재 인증된 사용자 가져오기"""
    token_data = verify_token(token)
    
    user = db.query(User).filter(User.id == token_data.user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """활성 사용자만 허용"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

def require_scopes(*required_scopes: str):
    """특정 권한 요구 데코레이터"""
    async def scope_checker(
        token: str = Depends(oauth2_scheme)
    ):
        token_data = verify_token(token)
        
        for scope in required_scopes:
            if scope not in token_data.scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Not enough permissions. Required scope: {scope}"
                )
        
        return token_data
    
    return scope_checker

# ========================
# Rate Limiting
# ========================

from functools import wraps
import time
from collections import defaultdict
from typing import Dict, Tuple

class RateLimiter:
    """Rate Limiter"""
    
    def __init__(self):
        self.requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> Tuple[bool, int]:
        """요청 허용 여부 확인"""
        now = time.time()
        
        # 만료된 요청 제거
        self.requests[key] = [
            req_time for req_time in self.requests[key] 
            if req_time > now - window_seconds
        ]
        
        # 요청 수 확인
        if len(self.requests[key]) >= max_requests:
            # 다음 요청 가능 시간 계산
            oldest_request = min(self.requests[key])
            retry_after = int(oldest_request + window_seconds - now)
            return False, retry_after
        
        # 요청 기록
        self.requests[key].append(now)
        return True, 0

rate_limiter = RateLimiter()

def rate_limit(max_requests: int = 60, window_seconds: int = 60):
    """Rate limiting 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            # IP 주소를 키로 사용
            client_ip = request.client.host
            
            allowed, retry_after = rate_limiter.is_allowed(
                client_ip, 
                max_requests, 
                window_seconds
            )
            
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests",
                    headers={"Retry-After": str(retry_after)}
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator

# ========================
# API Key Authentication (for external services)
# ========================

from fastapi.security import APIKeyHeader
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)) -> bool:
    """API 키 검증"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key is missing"
        )
    
    # 실제로는 DB에서 검증
    valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")
    
    if api_key not in valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    return True