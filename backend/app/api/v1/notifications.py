"""
Notification API Endpoints
알림 관련 API 엔드포인트
파일 위치: backend/app/api/v1/notifications.py
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from app.db.database import get_db
from app.services.notifications.push_service import (
    push_manager, 
    notification_service,
    NotificationType,
    NotificationPayload
)
from app.services.notifications.scheduler import scheduler

router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])

# ========================
# Schemas
# ========================

class DeviceTokenRegister(BaseModel):
    fcm_token: str
    device_type: str = "ios"  # ios, android
    device_model: Optional[str] = None
    app_version: Optional[str] = None

class NotificationPreferences(BaseModel):
    workout_reminder: bool = True
    friend_activity: bool = True
    challenge_updates: bool = True
    water_reminder: bool = True
    sleep_reminder: bool = True
    weekly_report: bool = True
    reminder_time: Optional[str] = "09:00"  # HH:MM format

class TestNotification(BaseModel):
    title: str
    body: str
    type: str = "test"

class NotificationHistory(BaseModel):
    id: int
    type: str
    title: str
    body: str
    sent_at: datetime
    read: bool = False

# ========================
# Device Token Management
# ========================

@router.post("/register-device")
async def register_device_token(
    token_data: DeviceTokenRegister,
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """FCM 디바이스 토큰 등록"""
    
    success = await push_manager.register_device_token(
        user_id=current_user_id,
        fcm_token=token_data.fcm_token,
        device_type=token_data.device_type
    )
    
    if success:
        # DB에도 저장 (구현 생략)
        return {"message": "디바이스 토큰이 등록되었습니다"}
    else:
        raise HTTPException(status_code=400, detail="토큰 등록 실패")

@router.delete("/unregister-device")
async def unregister_device(
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """디바이스 토큰 등록 해제"""
    
    # 토큰 삭제 로직
    if current_user_id in push_manager.user_tokens:
        del push_manager.user_tokens[current_user_id]
    
    return {"message": "디바이스 토큰이 해제되었습니다"}

# ========================
# Notification Preferences
# ========================

@router.get("/preferences", response_model=NotificationPreferences)
async def get_notification_preferences(
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """알림 설정 조회"""
    
    # 실제로는 DB에서 조회
    preferences = NotificationPreferences(
        workout_reminder=True,
        friend_activity=True,
        challenge_updates=True,
        water_reminder=True,
        sleep_reminder=True,
        weekly_report=True,
        reminder_time="09:00"
    )
    
    return preferences

@router.put("/preferences")
async def update_notification_preferences(
    preferences: NotificationPreferences,
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """알림 설정 업데이트"""
    
    # DB에 저장 (구현 생략)
    
    # 스케줄러 업데이트
    if not preferences.workout_reminder:
        scheduler.remove_job("morning_workout_reminder")
        scheduler.remove_job("evening_workout_reminder")
    
    if not preferences.water_reminder:
        scheduler.remove_job("water_reminder")
    
    if not preferences.sleep_reminder:
        scheduler.remove_job("sleep_reminder")
    
    return {"message": "알림 설정이 업데이트되었습니다"}

# ========================
# Send Notifications
# ========================

@router.post("/send-test")
async def send_test_notification(
    notification: TestNotification,
    current_user_id: int = 1,  # TODO: Get from auth
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """테스트 알림 전송"""
    
    payload = NotificationPayload(
        user_id=current_user_id,
        type=NotificationType.WORKOUT_REMINDER,  # 임시
        title=notification.title,
        body=notification.body
    )
    
    # 백그라운드에서 전송
    background_tasks.add_task(push_manager.send_notification, payload)
    
    return {"message": "테스트 알림을 전송했습니다"}

@router.post("/send-workout-reminder")
async def trigger_workout_reminder(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """운동 리마인더 수동 트리거"""
    
    background_tasks.add_task(notification_service.send_workout_reminder, db)
    
    return {"message": "운동 리마인더를 전송 중입니다"}

# ========================
# Notification History
# ========================

@router.get("/history", response_model=List[NotificationHistory])
async def get_notification_history(
    limit: int = 20,
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """알림 히스토리 조회"""
    
    # 실제로는 DB에서 조회
    history = [
        NotificationHistory(
            id=1,
            type="workout_reminder",
            title="운동할 시간이에요!",
            body="오늘의 운동을 시작해볼까요?",
            sent_at=datetime.utcnow(),
            read=False
        ),
        NotificationHistory(
            id=2,
            type="achievement",
            title="업적 달성!",
            body="첫 운동 업적을 달성했습니다",
            sent_at=datetime.utcnow(),
            read=True
        )
    ]
    
    return history

@router.put("/history/{notification_id}/read")
async def mark_notification_as_read(
    notification_id: int,
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """알림 읽음 처리"""
    
    # DB 업데이트 (구현 생략)
    
    return {"message": "알림을 읽음으로 표시했습니다"}

# ========================
# Scheduler Management
# ========================

@router.get("/scheduled-jobs")
async def get_scheduled_jobs():
    """예약된 작업 목록 조회 (관리자용)"""
    
    jobs = scheduler.get_jobs()
    return {"jobs": jobs}

@router.post("/schedule-custom")
async def schedule_custom_notification(
    title: str,
    body: str,
    scheduled_time: datetime,
    user_ids: List[int],
    background_tasks: BackgroundTasks
):
    """커스텀 알림 예약 (관리자용)"""
    
    for user_id in user_ids:
        payload = NotificationPayload(
            user_id=user_id,
            type=NotificationType.DAILY_REPORT,
            title=title,
            body=body,
            scheduled_time=scheduled_time
        )
        
        background_tasks.add_task(push_manager.schedule_notification, payload)
    
    return {
        "message": f"{len(user_ids)}명에게 알림을 예약했습니다",
        "scheduled_time": scheduled_time
    }