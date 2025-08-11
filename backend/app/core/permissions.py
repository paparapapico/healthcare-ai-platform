# backend/app/core/permissions.py
from functools import wraps
from fastapi import HTTPException, status, Depends
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..services.payment_service import PaymentService

def require_subscription(feature: str = None):
    """구독 필요 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 의존성에서 current_user 찾기
            current_user = None
            for key, value in kwargs.items():
                if isinstance(value, User):
                    current_user = value
                    break
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # 구독 확인
            if not current_user.subscription or current_user.subscription.status != "active":
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail="Active subscription required"
                )
            
            # 특정 기능 권한 확인
            if feature and not PaymentService.check_user_permissions(current_user, feature):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Your subscription plan doesn't include {feature}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def check_usage_limit(limit_type: str, count: int = 1):
    """사용량 제한 확인 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = None
            db = None
            
            # 의존성에서 current_user와 db 찾기
            for key, value in kwargs.items():
                if isinstance(value, User):
                    current_user = value
                elif hasattr(value, 'query'):  # Session 객체
                    db = value
            
            if not current_user or not db:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Required dependencies not found"
                )
            
            # 사용량 제한 확인
            limits = PaymentService.get_usage_limits(current_user)
            
            if limit_type in limits:
                max_limit = limits[limit_type]
                if max_limit != -1:  # -1은 무제한
                    # 현재 사용량 확인 (실제 구현에서는 Redis나 DB에서 조회)
                    current_usage = 0  # TODO: 실제 사용량 조회 로직
                    
                    if current_usage + count > max_limit:
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"Usage limit exceeded for {limit_type}. Upgrade your plan for more access."
                        )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator