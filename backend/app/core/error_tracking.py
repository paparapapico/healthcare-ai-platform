"""
Error Tracking with Sentry
에러 추적 및 모니터링
파일 위치: backend/app/core/error_tracking.py
"""

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.redis import RedisIntegration
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

def initialize_sentry(dsn: Optional[str] = None, environment: str = "development"):
    """Sentry 초기화"""
    
    dsn = dsn or os.getenv("SENTRY_DSN")
    
    if not dsn:
        logger.warning("Sentry DSN not provided, error tracking disabled")
        return
    
    try:
        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            integrations=[
                FastApiIntegration(
                    transaction_style="endpoint",
                    failed_request_status_codes=[400, 401, 403, 404, 405, 500, 502, 503, 504]
                ),
                SqlalchemyIntegration(),
                RedisIntegration(),
            ],
            traces_sample_rate=0.1 if environment == "production" else 1.0,
            profiles_sample_rate=0.1 if environment == "production" else 1.0,
            attach_stacktrace=True,
            send_default_pii=False,  # PII 정보 전송 안함
            before_send=before_send_filter,
            release=os.getenv("APP_VERSION", "1.0.0"),
        )
        
        logger.info(f"Sentry initialized for {environment} environment")
        
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}")

def before_send_filter(event, hint):
    """Sentry 이벤트 필터링"""
    
    # 민감한 정보 제거
    if 'request' in event and 'data' in event['request']:
        sensitive_fields = ['password', 'token', 'secret', 'api_key', 'credit_card']
        for field in sensitive_fields:
            if field in event['request']['data']:
                event['request']['data'][field] = '[FILTERED]'
    
    # 특정 에러 무시
    if 'exc_info' in hint:
        exc_type, exc_value, tb = hint['exc_info']
        
        # 무시할 예외 목록
        ignored_exceptions = [
            'KeyboardInterrupt',
            'SystemExit',
            'GeneratorExit'
        ]
        
        if exc_type.__name__ in ignored_exceptions:
            return None
    
    # 404 에러는 warning 레벨로
    if event.get('level') == 'error':
        if 'tags' in event and event['tags'].get('status_code') == 404:
            event['level'] = 'warning'
    
    return event

def capture_exception(error: Exception, context: dict = None):
    """예외 캡처 및 전송"""
    with sentry_sdk.push_scope() as scope:
        if context:
            for key, value in context.items():
                scope.set_context(key, value)
        
        sentry_sdk.capture_exception(error)

def capture_message(message: str, level: str = "info", context: dict = None):
    """메시지 캡처 및 전송"""
    with sentry_sdk.push_scope() as scope:
        if context:
            for key, value in context.items():
                scope.set_context(key, value)
        
        sentry_sdk.capture_message(message, level=level)

def set_user_context(user_id: int, email: str = None, username: str = None):
    """사용자 컨텍스트 설정"""
    sentry_sdk.set_user({
        "id": user_id,
        "email": email,
        "username": username
    })

def add_breadcrumb(message: str, category: str = "custom", level: str = "info", data: dict = None):
    """Breadcrumb 추가"""
    sentry_sdk.add_breadcrumb(
        message=message,
        category=category,
        level=level,
        data=data or {}
    )

# Custom error classes
class HealthcareAppError(Exception):
    """기본 앱 에러"""
    pass

class ValidationError(HealthcareAppError):
    """유효성 검사 에러"""
    pass

class AuthenticationError(HealthcareAppError):
    """인증 에러"""
    pass

class AuthorizationError(HealthcareAppError):
    """권한 에러"""
    pass

class NotFoundError(HealthcareAppError):
    """리소스 없음 에러"""
    pass

class ExternalServiceError(HealthcareAppError):
    """외부 서비스 에러"""
    pass