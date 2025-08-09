"""
Monitoring and Metrics
Prometheus 메트릭 및 모니터링
파일 위치: backend/app/core/monitoring.py
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time
from functools import wraps
import psutil
import logging

logger = logging.getLogger(__name__)

# ========================
# Prometheus Metrics
# ========================

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# Business metrics
workouts_created_total = Counter(
    'workouts_created_total',
    'Total workouts created',
    ['exercise_type']
)

user_registrations_total = Counter(
    'user_registrations_total',
    'Total user registrations'
)

active_users_gauge = Gauge(
    'active_users',
    'Number of active users'
)

health_score_histogram = Histogram(
    'user_health_score',
    'Distribution of user health scores',
    buckets=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
)

# WebSocket metrics
websocket_connections = Gauge(
    'websocket_connections',
    'Current WebSocket connections'
)

websocket_messages_total = Counter(
    'websocket_messages_total',
    'Total WebSocket messages',
    ['direction']  # sent, received
)

# AI metrics
pose_analysis_duration = Histogram(
    'pose_analysis_duration_seconds',
    'Pose analysis processing time',
    ['exercise_type']
)

pose_analysis_errors = Counter(
    'pose_analysis_errors_total',
    'Pose analysis errors',
    ['error_type']
)

# Cache metrics
cache_hits = Counter(
    'cache_hits_total',
    'Cache hit count',
    ['cache_key_prefix']
)

cache_misses = Counter(
    'cache_misses_total',
    'Cache miss count',
    ['cache_key_prefix']
)

# Database metrics
db_query_duration = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['query_type']
)

db_connection_pool_size = Gauge(
    'db_connection_pool_size',
    'Database connection pool size'
)

# System metrics
system_cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

system_memory_usage = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage'
)

system_disk_usage = Gauge(
    'system_disk_usage_percent',
    'System disk usage percentage'
)

# ========================
# Metric Decorators
# ========================

def track_request_metrics(method: str, endpoint: str):
    """HTTP 요청 메트릭 추적 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                raise e
            finally:
                duration = time.time() - start_time
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status=status
                ).inc()
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
        
        return wrapper
    return decorator

def track_db_query(query_type: str):
    """데이터베이스 쿼리 메트릭 추적 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                db_query_duration.labels(query_type=query_type).observe(duration)
        
        return wrapper
    return decorator

def track_cache_access(key_prefix: str):
    """캐시 접근 메트릭 추적 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            if result is not None:
                cache_hits.labels(cache_key_prefix=key_prefix).inc()
            else:
                cache_misses.labels(cache_key_prefix=key_prefix).inc()
            
            return result
        
        return wrapper
    return decorator

# ========================
# Metric Collection
# ========================

class MetricsCollector:
    """메트릭 수집기"""
    
    @staticmethod
    def collect_system_metrics():
        """시스템 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            system_cpu_usage.set(cpu_percent)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            system_memory_usage.set(memory.percent)
            
            # 디스크 사용률
            disk = psutil.disk_usage('/')
            system_disk_usage.set(disk.percent)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    @staticmethod
    def update_active_users(count: int):
        """활성 사용자 수 업데이트"""
        active_users_gauge.set(count)
    
    @staticmethod
    def track_workout_created(exercise_type: str):
        """운동 생성 추적"""
        workouts_created_total.labels(exercise_type=exercise_type).inc()
    
    @staticmethod
    def track_user_registration():
        """사용자 등록 추적"""
        user_registrations_total.inc()
    
    @staticmethod
    def track_health_score(score: float):
        """건강 점수 분포 추적"""
        health_score_histogram.observe(score)
    
    @staticmethod
    def track_websocket_connection(delta: int):
        """WebSocket 연결 수 추적"""
        websocket_connections.inc(delta)
    
    @staticmethod
    def track_websocket_message(direction: str):
        """WebSocket 메시지 추적"""
        websocket_messages_total.labels(direction=direction).inc()
    
    @staticmethod
    def track_pose_analysis(exercise_type: str, duration: float, error: bool = False):
        """포즈 분석 메트릭 추적"""
        if error:
            pose_analysis_errors.labels(error_type=exercise_type).inc()
        else:
            pose_analysis_duration.labels(exercise_type=exercise_type).observe(duration)

# ========================
# Metrics Endpoint
# ========================

async def metrics_endpoint() -> Response:
    """Prometheus 메트릭 엔드포인트"""
    # 시스템 메트릭 수집
    MetricsCollector.collect_system_metrics()
    
    # Prometheus 형식으로 메트릭 생성
    metrics = generate_latest()
    
    return Response(
        content=metrics,
        media_type=CONTENT_TYPE_LATEST
    )

# ========================
# Health Check
# ========================

class HealthChecker:
    """헬스 체크"""
    
    @staticmethod
    async def check_database(db) -> bool:
        """데이터베이스 연결 체크"""
        try:
            db.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    @staticmethod
    async def check_redis(redis_client) -> bool:
        """Redis 연결 체크"""
        try:
            redis_client.ping()
            return True
        except Exception:
            return False
    
    @staticmethod
    async def get_health_status(db, redis_client) -> dict:
        """전체 헬스 상태"""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "database": await HealthChecker.check_database(db),
                "redis": await HealthChecker.check_redis(redis_client),
                "api": True
            },
            "metrics": {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        }

# ========================
# Alert Rules
# ========================

ALERT_RULES = """
# Alert Rules for Prometheus
# 파일 위치: monitoring/alerts.yml

groups:
  - name: healthcare_alerts
    interval: 30s
    rules:
      # High CPU usage
      - alert: HighCPUUsage
        expr: system_cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% (current value: {{ $value }}%)"
      
      # High memory usage
      - alert: HighMemoryUsage
        expr: system_memory_usage_percent > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 90% (current value: {{ $value }}%)"
      
      # API response time
      - alert: SlowAPIResponse
        expr: rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m]) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow API response time"
          description: "Average API response time is above 1 second"
      
      # Database connection pool
      - alert: DatabaseConnectionPoolExhausted
        expr: db_connection_pool_size > 40
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool nearly exhausted"
          description: "Database connection pool size is {{ $value }}"
      
      # Error rate
      - alert: HighErrorRate
        expr: rate(http_requests_total{status="error"}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 5%"
"""

# 파일 저장
def save_alert_rules():
    """Alert rules 파일 저장"""
    with open("monitoring/alerts.yml", "w") as f:
        f.write(ALERT_RULES)