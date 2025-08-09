from pydantic import BaseModel, EmailStr, validator
from typing import Optional
from datetime import datetime, date  # 이 라인을 추가!
from enum import Enum

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str
    birth_date: date  # 자동으로 문자열을 date로 변환
    gender: Optional[str] = None
    height: Optional[int] = None
    weight: Optional[float] = None

# 현재 문제가 있는 UserResponse 모델을 이렇게 수정:

class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    birth_date: Optional[date] = None  # 또는 Optional[str]
    gender: Optional[str] = None
    subscription_tier: Optional[str] = "FREE"  # 기본값 설정
    created_at: Optional[datetime] = None
    is_active: bool = True
    
    class Config:
        from_attributes = True  # SQLAlchemy 모델과 호환

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None