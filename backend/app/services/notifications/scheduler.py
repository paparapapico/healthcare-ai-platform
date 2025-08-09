"""
Notification Scheduler
알림 예약 및 자동화 시스템
파일 위치: backend/app/services/notifications/scheduler.py
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Dict, List
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.db.database import SessionLocal
from app.services.notifications.push_service import notification_service

logger = logging.getLogger(__name__)

class NotificationScheduler:
    """알림 스케줄러"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.jobs = {}
        
    def initialize(self):
        """스케줄러 초기화 및 작업 등록"""
        try:
            # 매일 오전 8시: 운동 리마인더
            self.scheduler.add_job(
                self._send_morning_reminders,
                CronTrigger(hour=8, minute=0),
                id="morning_workout_reminder",
                name="Morning Workout Reminder",
                replace_existing=True
            )
            
            # 매일 오후 12시: 점심 후 활동 리마인더
            self.scheduler.add_job(
                self._send_lunch_activity_reminder,
                CronTrigger(hour=12, minute=30),
                id="lunch_activity_reminder",
                name="Lunch Activity Reminder",
                replace_existing=True
            )
            
            # 매일 오후 6시: 저녁 운동 리마인더
            self.scheduler.add_job(
                self._send_evening_workout_reminder,
                CronTrigger(hour=18, minute=0),
                id="evening_workout_reminder",
                name="Evening Workout Reminder",
                replace_existing=True
            )
            
            # 매일 밤 10시: 수면 리마인더
            self.scheduler.add_job(
                self._send_sleep_reminder,
                CronTrigger(hour=22, minute=0),
                id="sleep_reminder",
                name="Sleep Reminder",
                replace_existing=True
            )
            
            # 3시간마다: 물 섭취 리마인더 (9시-21시)
            self.scheduler.add_job(
                self._send_water_reminder,
                CronTrigger(hour='9,12,15,18,21', minute=0),
                id="water_reminder",
                name="Water Intake Reminder",
                replace_existing=True
            )
            
            # 매주 일요일 오후 8시: 주간 리포트
            self.scheduler.add_job(
                self._send_weekly_report,
                CronTrigger(day_of_week='sun', hour=20, minute=0),
                id="weekly_report",
                name="Weekly Health Report",
                replace_existing=True
            )
            
            # 매일 자정: 데이터 정리 및 통계 업데이트
            self.scheduler.add_job(
                self._daily_cleanup,
                CronTrigger(hour=0, minute=0),
                id="daily_cleanup",
                name="Daily Data Cleanup",
                replace_existing=True
            )
            
            # 스케줄러 시작
            self.scheduler.start()
            logger.info("Notification scheduler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {e}")
    
    async def _send_morning_reminders(self):
        """아침 운동 리마인더"""
        logger.info("Sending morning workout reminders...")
        
        db = SessionLocal()
        try:
            await notification_service.send_workout_reminder(db)
        finally:
            db.close()
    
    async def _send_lunch_activity_reminder(self):
        """점심 활동 리마인더"""
        logger.info("Sending lunch activity reminders...")
        
        # 점심 후 스트레칭이나 짧은 운동 권장
        # 구현 생략 (send_workout_reminder와 유사)
    
    async def _send_evening_workout_reminder(self):
        """저녁 운동 리마인더"""
        logger.info("Sending evening workout reminders...")
        
        db = SessionLocal()
        try:
            await notification_service.send_workout_reminder(db)
        finally:
            db.close()
    
    async def _send_sleep_reminder(self):
        """수면 리마인더"""
        logger.info("Sending sleep reminders...")
        
        db = SessionLocal()
        try:
            await notification_service.send_sleep_reminder(db)
        finally:
            db.close()
    
    async def _send_water_reminder(self):
        """물 섭취 리마인더"""
        logger.info("Sending water intake reminders...")
        
        db = SessionLocal()
        try:
            await notification_service.send_water_reminders(db)
        finally:
            db.close()
    
    async def _send_weekly_report(self):
        """주간 건강 리포트"""
        logger.info("Generating and sending weekly reports...")
        
        from app.models.models import User, Workout, HealthMetric
        
        db = SessionLocal()
        try:
            # 모든 활성 사용자 조회
            users = db.query(User).filter(User.is_active == True).all()
            
            for user in users:
                # 주간 통계 계산
                week_start = datetime.now() - timedelta(days=7)
                
                # 운동 통계
                workouts = db.query(Workout).filter(
                    Workout.user_id == user.id,
                    Workout.start_time >= week_start
                ).all()
                
                total_workouts = len(workouts)
                total_calories = sum(w.calories_burned or 0 for w in workouts)
                avg_form_score = sum(w.avg_form_score or 0 for w in workouts) / len(workouts) if workouts else 0
                
                # 건강 메트릭
                health_metrics = db.query(HealthMetric).filter(
                    HealthMetric.user_id == user.id,
                    HealthMetric.recorded_at >= week_start
                ).all()
                
                avg_sleep = sum(m.sleep_hours or 0 for m in health_metrics if m.sleep_hours) / 7
                
                # 리포트 생성 및 전송
                report = f"""
                📊 주간 건강 리포트
                
                운동: {total_workouts}회
                칼로리: {total_calories:.0f}kcal
                평균 자세: {avg_form_score:.0f}점
                평균 수면: {avg_sleep:.1f}시간
                
                다음 주도 화이팅! 💪
                """
                
                # 알림 전송 (구현 생략)
                
        finally:
            db.close()
    
    async def _daily_cleanup(self):
        """일일 데이터 정리"""
        logger.info("Running daily cleanup...")
        
        # 오래된 알림 히스토리 삭제
        # 통계 데이터 업데이트
        # 캐시 정리 등
    
    def add_custom_job(self, job_id: str, func, trigger, **kwargs):
        """커스텀 작업 추가"""
        try:
            self.scheduler.add_job(
                func,
                trigger,
                id=job_id,
                replace_existing=True,
                **kwargs
            )
            logger.info(f"Added custom job: {job_id}")
        except Exception as e:
            logger.error(f"Failed to add custom job: {e}")
    
    def remove_job(self, job_id: str):
        """작업 제거"""
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed job: {job_id}")
        except Exception as e:
            logger.error(f"Failed to remove job: {e}")
    
    def get_jobs(self) -> List[Dict]:
        """등록된 작업 목록 조회"""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time,
                "trigger": str(job.trigger)
            })
        return jobs
    
    def shutdown(self):
        """스케줄러 종료"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler shut down")

# Singleton instance
scheduler = NotificationScheduler()

# FastAPI 앱 시작 시 스케줄러 초기화
def init_scheduler():
    """스케줄러 초기화 (main.py에서 호출)"""
    scheduler.initialize()