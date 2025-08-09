"""
Redis Caching System
성능 최적화를 위한 캐싱 시스템
파일 위치: backend/app/core/cache.py
"""

import json
import pickle
import hashlib
from typing import Any, Optional, Union, Callable
from datetime import timedelta
from functools import wraps
import logging

import redis
from redis import Redis
import asyncio

logger = logging.getLogger(__name__)

# ========================
# Redis Configuration
# ========================

class RedisConfig:
    """Redis 설정"""
    HOST = "localhost"
    PORT = 6379
    DB = 0
    PASSWORD = None
    DECODE_RESPONSES = False
    MAX_CONNECTIONS = 50
    
    # TTL 설정 (초)
    DEFAULT_TTL = 3600  # 1시간
    SHORT_TTL = 300     # 5분
    MEDIUM_TTL = 1800   # 30분
    LONG_TTL = 86400    # 24시간
    
    # 키 프리픽스
    KEY_PREFIX = "healthcare:"

# ========================
# Cache Manager
# ========================

class CacheManager:
    """캐시 매니저"""
    
    def __init__(self):
        self.redis_client: Optional[Redis] = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Redis 연결 초기화"""
        try:
            self.redis_client = redis.Redis(
                host=RedisConfig.HOST,
                port=RedisConfig.PORT,
                db=RedisConfig.DB,
                password=RedisConfig.PASSWORD,
                decode_responses=RedisConfig.DECODE_RESPONSES,
                max_connections=RedisConfig.MAX_CONNECTIONS
            )
            
            # 연결 테스트
            self.redis_client.ping()
            logger.info("Redis connection established")
            
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _make_key(self, key: str) -> str:
        """캐시 키 생성"""
        return f"{RedisConfig.KEY_PREFIX}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """캐시 조회"""
        if not self.redis_client:
            return None
        
        try:
            full_key = self._make_key(key)
            data = self.redis_client.get(full_key)
            
            if data:
                # 역직렬화
                return pickle.loads(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = RedisConfig.DEFAULT_TTL) -> bool:
        """캐시 저장"""
        if not self.redis_client:
            return False
        
        try:
            full_key = self._make_key(key)
            
            # 직렬화
            serialized = pickle.dumps(value)
            
            # TTL과 함께 저장
            return self.redis_client.setex(full_key, ttl, serialized)
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """캐시 삭제"""
        if not self.redis_client:
            return False
        
        try:
            full_key = self._make_key(key)
            return bool(self.redis_client.delete(full_key))
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """패턴 매칭으로 캐시 삭제"""
        if not self.redis_client:
            return 0
        
        try:
            full_pattern = self._make_key(pattern)
            keys = self.redis_client.keys(full_pattern)
            
            if keys:
                return self.redis_client.delete(*keys)
            
            return 0
            
        except Exception as e:
            logger.error(f"Cache delete pattern error: {e}")
            return 0
    
    def exists(self, key: str) -> bool:
        """캐시 존재 확인"""
        if not self.redis_client:
            return False
        
        try:
            full_key = self._make_key(key)
            return bool(self.redis_client.exists(full_key))
            
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """TTL 설정"""
        if not self.redis_client:
            return False
        
        try:
            full_key = self._make_key(key)
            return bool(self.redis_client.expire(full_key, ttl))
            
        except Exception as e:
            logger.error(f"Cache expire error: {e}")
            return False
    
    def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """카운터 증가"""
        if not self.redis_client:
            return None
        
        try:
            full_key = self._make_key(key)
            return self.redis_client.incr(full_key, amount)
            
        except Exception as e:
            logger.error(f"Cache incr error: {e}")
            return None
    
    def get_json(self, key: str) -> Optional[dict]:
        """JSON 캐시 조회"""
        if not self.redis_client:
            return None
        
        try:
            full_key = self._make_key(key)
            data = self.redis_client.get(full_key)
            
            if data:
                return json.loads(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get_json error: {e}")
            return None
    
    def set_json(self, key: str, value: dict, ttl: int = RedisConfig.DEFAULT_TTL) -> bool:
        """JSON 캐시 저장"""
        if not self.redis_client:
            return False
        
        try:
            full_key = self._make_key(key)
            serialized = json.dumps(value)
            
            return self.redis_client.setex(full_key, ttl, serialized)
            
        except Exception as e:
            logger.error(f"Cache set_json error: {e}")
            return False

# Singleton instance
cache_manager = CacheManager()

# ========================
# Cache Decorators
# ========================

def cache_result(ttl: int = RedisConfig.DEFAULT_TTL, key_prefix: str = ""):
    """함수 결과 캐싱 데코레이터"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 캐시 키 생성
            cache_key = f"{key_prefix}:{func.__name__}"
            if args:
                # 인자를 해시화하여 키에 추가
                args_hash = hashlib.md5(str(args).encode()).hexdigest()[:8]
                cache_key += f":{args_hash}"
            if kwargs:
                kwargs_hash = hashlib.md5(str(kwargs).encode()).hexdigest()[:8]
                cache_key += f":{kwargs_hash}"
            
            # 캐시 조회
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_result
            
            # 함수 실행
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # 결과 캐싱
            cache_manager.set(cache_key, result, ttl)
            logger.debug(f"Cache set: {cache_key}")
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 캐시 키 생성
            cache_key = f"{key_prefix}:{func.__name__}"
            if args:
                args_hash = hashlib.md5(str(args).encode()).hexdigest()[:8]
                cache_key += f":{args_hash}"
            if kwargs:
                kwargs_hash = hashlib.md5(str(kwargs).encode()).hexdigest()[:8]
                cache_key += f":{kwargs_hash}"
            
            # 캐시 조회
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_result
            
            # 함수 실행
            result = func(*args, **kwargs)
            
            # 결과 캐싱
            cache_manager.set(cache_key, result, ttl)
            logger.debug(f"Cache set: {cache_key}")
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def invalidate_cache(pattern: str):
    """캐시 무효화 데코레이터"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 함수 실행
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # 캐시 무효화
            deleted = cache_manager.delete_pattern(pattern)
            logger.debug(f"Cache invalidated: {pattern} ({deleted} keys)")
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 함수 실행
            result = func(*args, **kwargs)
            
            # 캐시 무효화
            deleted = cache_manager.delete_pattern(pattern)
            logger.debug(f"Cache invalidated: {pattern} ({deleted} keys)")
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# ========================
# Cache Utilities
# ========================

class CacheKeys:
    """캐시 키 상수"""
    
    # User
    USER_PROFILE = "user:profile:{user_id}"
    USER_HEALTH_SCORE = "user:health_score:{user_id}"
    USER_WORKOUTS = "user:workouts:{user_id}"
    USER_ACHIEVEMENTS = "user:achievements:{user_id}"
    
    # Social
    LEADERBOARD_GLOBAL = "leaderboard:global:{period}"
    CHALLENGE_INFO = "challenge:{challenge_id}"
    FRIEND_LIST = "friends:{user_id}"
    
    # Health
    HEALTH_DASHBOARD = "health:dashboard:{user_id}"
    HEALTH_METRICS = "health:metrics:{user_id}:{date}"
    
    # Stats
    WORKOUT_STATS = "stats:workout:{user_id}:{period}"
    APP_STATS = "stats:app:daily"

def clear_user_cache(user_id: int):
    """사용자 관련 캐시 전체 삭제"""
    patterns = [
        f"user:*:{user_id}",
        f"health:*:{user_id}*",
        f"stats:*:{user_id}*",
        f"friends:{user_id}"
    ]
    
    total_deleted = 0
    for pattern in patterns:
        deleted = cache_manager.delete_pattern(pattern)
        total_deleted += deleted
    
    logger.info(f"Cleared {total_deleted} cache keys for user {user_id}")
    return total_deleted

# ========================
# Usage Examples
# ========================

"""
사용 예시:

1. 데코레이터로 캐싱:
@cache_result(ttl=300, key_prefix="api")
async def get_user_profile(user_id: int):
    # DB 조회 로직
    return user_data

2. 수동 캐싱:
# 저장
cache_manager.set("my_key", my_data, ttl=600)

# 조회
cached_data = cache_manager.get("my_key")

3. 캐시 무효화:
@invalidate_cache("user:*:123")
async def update_user_profile(user_id: int, data: dict):
    # 업데이트 로직
    pass
"""