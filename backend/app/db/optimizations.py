"""
Database Optimizations
데이터베이스 인덱싱 및 쿼리 최적화
파일 위치: backend/app/db/optimizations.py
"""

from sqlalchemy import create_engine, text, Index
from sqlalchemy.orm import Session
import logging

from app.db.database import engine
from app.models.models import (
    User, Workout, HealthMetric, Challenge, 
    ChallengeParticipant, Friendship, Achievement
)

logger = logging.getLogger(__name__)

# ========================
# Index Creation
# ========================

def create_indexes():
    """데이터베이스 인덱스 생성"""
    
    indexes = [
        # User indexes
        Index('idx_user_email', User.email),
        Index('idx_user_health_score', User.health_score),
        Index('idx_user_created_at', User.created_at),
        
        # Workout indexes
        Index('idx_workout_user_id', Workout.user_id),
        Index('idx_workout_start_time', Workout.start_time),
        Index('idx_workout_exercise_type', Workout.exercise_type),
        Index('idx_workout_user_time', Workout.user_id, Workout.start_time),
        
        # HealthMetric indexes
        Index('idx_health_user_id', HealthMetric.user_id),
        Index('idx_health_recorded_at', HealthMetric.recorded_at),
        Index('idx_health_user_date', HealthMetric.user_id, HealthMetric.recorded_at),
        
        # Challenge indexes
        Index('idx_challenge_active', Challenge.is_active),
        Index('idx_challenge_end_date', Challenge.end_date),
        
        # ChallengeParticipant indexes
        Index('idx_participant_challenge', ChallengeParticipant.challenge_id),
        Index('idx_participant_user', ChallengeParticipant.user_id),
        Index('idx_participant_challenge_user', 
              ChallengeParticipant.challenge_id, 
              ChallengeParticipant.user_id),
        
        # Friendship indexes
        Index('idx_friendship_user', Friendship.user_id),
        Index('idx_friendship_friend', Friendship.friend_id),
        Index('idx_friendship_status', Friendship.status),
    ]
    
    created_count = 0
    for index in indexes:
        try:
            index.create(engine, checkfirst=True)
            created_count += 1
            logger.info(f"Created index: {index.name}")
        except Exception as e:
            logger.error(f"Failed to create index {index.name}: {e}")
    
    logger.info(f"Created {created_count} indexes")
    return created_count

# ========================
# Query Optimization Views
# ========================

def create_materialized_views():
    """구체화된 뷰 생성 (PostgreSQL)"""
    
    views = [
        # 사용자 통계 뷰
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS user_stats AS
        SELECT 
            u.id as user_id,
            u.name,
            u.health_score,
            COUNT(DISTINCT w.id) as total_workouts,
            SUM(w.calories_burned) as total_calories,
            AVG(w.avg_form_score) as avg_form_score,
            MAX(w.start_time) as last_workout_date
        FROM users u
        LEFT JOIN workouts w ON u.id = w.user_id
        GROUP BY u.id, u.name, u.health_score
        WITH DATA;
        """,
        
        # 주간 리더보드 뷰
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS weekly_leaderboard AS
        SELECT 
            u.id as user_id,
            u.name,
            COUNT(w.id) as workout_count,
            SUM(w.calories_burned) as total_calories,
            AVG(w.avg_form_score) as avg_form_score
        FROM users u
        LEFT JOIN workouts w ON u.id = w.user_id
        WHERE w.start_time >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY u.id, u.name
        ORDER BY total_calories DESC
        WITH DATA;
        """,
        
        # 챌린지 진행 상황 뷰
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS challenge_progress AS
        SELECT 
            c.id as challenge_id,
            c.title,
            cp.user_id,
            u.name as user_name,
            cp.current_reps,
            cp.current_calories,
            RANK() OVER (PARTITION BY c.id ORDER BY cp.current_calories DESC) as rank
        FROM challenges c
        JOIN challenge_participants cp ON c.id = cp.challenge_id
        JOIN users u ON cp.user_id = u.id
        WHERE c.is_active = true
        WITH DATA;
        """
    ]
    
    created_count = 0
    with engine.connect() as conn:
        for view_sql in views:
            try:
                conn.execute(text(view_sql))
                conn.commit()
                created_count += 1
                logger.info(f"Created materialized view")
            except Exception as e:
                logger.error(f"Failed to create view: {e}")
    
    return created_count

def refresh_materialized_views():
    """구체화된 뷰 새로고침"""
    
    view_names = [
        'user_stats',
        'weekly_leaderboard',
        'challenge_progress'
    ]
    
    refreshed_count = 0
    with engine.connect() as conn:
        for view_name in view_names:
            try:
                conn.execute(text(f"REFRESH MATERIALIZED VIEW {view_name}"))
                conn.commit()
                refreshed_count += 1
                logger.info(f"Refreshed view: {view_name}")
            except Exception as e:
                logger.error(f"Failed to refresh view {view_name}: {e}")
    
    return refreshed_count

# ========================
# Query Optimization Helpers
# ========================

class QueryOptimizer:
    """쿼리 최적화 헬퍼"""
    
    @staticmethod
    def get_user_with_stats(db: Session, user_id: int):
        """사용자 정보와 통계를 한 번에 조회 (N+1 문제 해결)"""
        from sqlalchemy.orm import joinedload
        
        return db.query(User).options(
            joinedload(User.workouts),
            joinedload(User.achievements),
            joinedload(User.health_metrics)
        ).filter(User.id == user_id).first()
    
    @staticmethod
    def get_recent_workouts_batch(db: Session, user_ids: list, days: int = 7):
        """여러 사용자의 최근 운동 배치 조회"""
        from datetime import datetime, timedelta
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        return db.query(Workout).filter(
            Workout.user_id.in_(user_ids),
            Workout.start_time >= start_date
        ).all()
    
    @staticmethod
    def get_leaderboard_optimized(db: Session, limit: int = 10):
        """최적화된 리더보드 조회"""
        from sqlalchemy import func
        
        # 구체화된 뷰 사용
        result = db.execute(
            text("""
                SELECT user_id, name, workout_count, total_calories, avg_form_score
                FROM weekly_leaderboard
                LIMIT :limit
            """),
            {"limit": limit}
        )
        
        return result.fetchall()

# ========================
# Database Maintenance
# ========================

class DatabaseMaintenance:
    """데이터베이스 유지보수"""
    
    @staticmethod
    def vacuum_analyze():
        """VACUUM 및 ANALYZE 실행 (PostgreSQL)"""
        try:
            with engine.connect() as conn:
                conn.execute(text("VACUUM ANALYZE"))
                conn.commit()
            logger.info("VACUUM ANALYZE completed")
            return True
        except Exception as e:
            logger.error(f"VACUUM ANALYZE failed: {e}")
            return False
    
    @staticmethod
    def update_statistics():
        """통계 정보 업데이트"""
        tables = ['users', 'workouts', 'health_metrics', 'challenges']
        
        updated_count = 0
        with engine.connect() as conn:
            for table in tables:
                try:
                    conn.execute(text(f"ANALYZE {table}"))
                    conn.commit()
                    updated_count += 1
                    logger.info(f"Updated statistics for {table}")
                except Exception as e:
                    logger.error(f"Failed to update statistics for {table}: {e}")
        
        return updated_count
    
    @staticmethod
    def clean_old_data(days: int = 90):
        """오래된 데이터 정리"""
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        with Session(engine) as db:
            try:
                # 오래된 건강 메트릭 삭제
                deleted = db.query(HealthMetric).filter(
                    HealthMetric.recorded_at < cutoff_date
                ).delete()
                
                db.commit()
                logger.info(f"Deleted {deleted} old health metrics")
                return deleted
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to clean old data: {e}")
                return 0

# ========================
# Connection Pool Optimization
# ========================

def optimize_connection_pool():
    """연결 풀 최적화"""
    from sqlalchemy.pool import QueuePool
    
    # 연결 풀 설정
    engine.pool._recycle = 3600  # 1시간마다 연결 재활용
    engine.pool._timeout = 30    # 30초 타임아웃
    engine.pool._overflow = 10   # 최대 10개 오버플로우 연결
    
    logger.info("Connection pool optimized")

# ========================
# Initialization
# ========================

def initialize_optimizations():
    """모든 최적화 초기화"""
    logger.info("Initializing database optimizations...")
    
    # 인덱스 생성
    create_indexes()
    
    # 구체화된 뷰 생성
    # create_materialized_views()  # PostgreSQL only
    
    # 연결 풀 최적화
    optimize_connection_pool()
    
    # 통계 업데이트
    DatabaseMaintenance.update_statistics()
    
    logger.info("Database optimizations completed")

if __name__ == "__main__":
    initialize_optimizations()