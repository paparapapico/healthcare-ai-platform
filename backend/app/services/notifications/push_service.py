"""
Push Notification Service
FCM을 사용한 푸시 알림 시스템
파일 위치: backend/app/services/notifications/push_service.py
"""

import json
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from enum import Enum

# Firebase Admin SDK (설치 필요: pip install firebase-admin)
import firebase_admin
from firebase_admin import credentials, messaging
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.models import User, Workout, Challenge, Friendship

logger = logging.getLogger(__name__)

# ========================
# Notification Types
# ========================

class NotificationType(Enum):
    WORKOUT_REMINDER = "workout_reminder"
    FRIEND_REQUEST = "friend_request"
    CHALLENGE_INVITE = "challenge_invite"
    ACHIEVEMENT_EARNED = "achievement_earned"
    DAILY_REPORT = "daily_report"
    STREAK_REMINDER = "streak_reminder"
    FRIEND_ACTIVITY = "friend_activity"
    WATER_REMINDER = "water_reminder"
    SLEEP_REMINDER = "sleep_reminder"

@dataclass
class NotificationPayload:
    user_id: int
    type: NotificationType
    title: str
    body: str
    data: Optional[Dict] = None
    image_url: Optional[str] = None
    action_url: Optional[str] = None
    scheduled_time: Optional[datetime] = None

# ========================
# Push Notification Manager
# ========================

class PushNotificationManager:
    def __init__(self):
        """푸시 알림 매니저 초기화"""
        # Firebase 초기화 (서비스 계정 키 필요)
        try:
            # Firebase 서비스 계정 키 파일 경로
            cred = credentials.Certificate("path/to/serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
            self.initialized = True
            logger.info("Firebase Admin SDK initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            self.initialized = False
        
        # 알림 큐
        self.notification_queue = asyncio.Queue()
        
        # 사용자별 FCM 토큰 캐시 (실제로는 Redis 사용)
        self.user_tokens = {}
        
    async def register_device_token(self, user_id: int, fcm_token: str, device_type: str = "ios"):
        """디바이스 FCM 토큰 등록"""
        try:
            # 토큰 유효성 검증
            if not fcm_token or len(fcm_token) < 10:
                raise ValueError("Invalid FCM token")
            
            # 토큰 저장 (실제로는 DB에 저장)
            self.user_tokens[user_id] = {
                "token": fcm_token,
                "device_type": device_type,
                "registered_at": datetime.utcnow()
            }
            
            logger.info(f"Registered FCM token for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register token: {e}")
            return False
    
    async def send_notification(self, payload: NotificationPayload) -> bool:
        """단일 푸시 알림 전송"""
        if not self.initialized:
            logger.warning("Firebase not initialized, skipping notification")
            return False
        
        try:
            # 사용자 FCM 토큰 조회
            user_token_data = self.user_tokens.get(payload.user_id)
            if not user_token_data:
                logger.warning(f"No FCM token found for user {payload.user_id}")
                return False
            
            fcm_token = user_token_data["token"]
            
            # FCM 메시지 구성
            notification = messaging.Notification(
                title=payload.title,
                body=payload.body,
                image=payload.image_url
            )
            
            # 데이터 페이로드
            data = {
                "type": payload.type.value,
                "timestamp": datetime.utcnow().isoformat(),
                "action_url": payload.action_url or ""
            }
            
            if payload.data:
                data.update({k: str(v) for k, v in payload.data.items()})
            
            # 플랫폼별 설정
            apns_config = messaging.APNSConfig(
                headers={'apns-priority': '10'},
                payload=messaging.APNSPayload(
                    aps=messaging.Aps(
                        alert=messaging.ApsAlert(
                            title=payload.title,
                            body=payload.body
                        ),
                        badge=1,
                        sound='default',
                        category=payload.type.value
                    )
                )
            )
            
            android_config = messaging.AndroidConfig(
                priority='high',
                notification=messaging.AndroidNotification(
                    priority='high',
                    default_sound=True,
                    default_vibrate_timings=True,
                    notification_count=1
                )
            )
            
            # 메시지 생성
            message = messaging.Message(
                notification=notification,
                data=data,
                token=fcm_token,
                apns=apns_config,
                android=android_config
            )
            
            # 메시지 전송
            response = messaging.send(message)
            logger.info(f"Successfully sent notification to user {payload.user_id}: {response}")
            
            # 알림 히스토리 저장 (DB에 기록)
            await self._save_notification_history(payload)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False
    
    async def send_batch_notifications(self, payloads: List[NotificationPayload]) -> Dict[int, bool]:
        """배치 푸시 알림 전송"""
        results = {}
        
        # 병렬 전송을 위한 태스크 생성
        tasks = []
        for payload in payloads:
            task = asyncio.create_task(self.send_notification(payload))
            tasks.append((payload.user_id, task))
        
        # 모든 태스크 완료 대기
        for user_id, task in tasks:
            try:
                result = await task
                results[user_id] = result
            except Exception as e:
                logger.error(f"Batch notification failed for user {user_id}: {e}")
                results[user_id] = False
        
        return results
    
    async def schedule_notification(self, payload: NotificationPayload):
        """예약 알림 설정"""
        if not payload.scheduled_time:
            payload.scheduled_time = datetime.utcnow()
        
        # 알림 큐에 추가
        await self.notification_queue.put(payload)
        logger.info(f"Scheduled notification for user {payload.user_id} at {payload.scheduled_time}")
    
    async def _save_notification_history(self, payload: NotificationPayload):
        """알림 히스토리 저장"""
        # 실제로는 DB에 저장
        pass

# ========================
# Notification Templates
# ========================

class NotificationTemplates:
    """알림 템플릿 관리"""
    
    @staticmethod
    def workout_reminder(user_name: str, last_workout_days: int) -> Dict:
        """운동 리마인더 템플릿"""
        if last_workout_days == 0:
            return {
                "title": "오늘도 화이팅! 💪",
                "body": f"{user_name}님, 오늘의 운동을 시작해볼까요?",
            }
        elif last_workout_days == 1:
            return {
                "title": "운동할 시간이에요! 🏃‍♂️",
                "body": f"{user_name}님, 어제의 기록을 이어가볼까요?",
            }
        else:
            return {
                "title": "오랜만이에요! 😊",
                "body": f"{user_name}님, {last_workout_days}일만의 운동! 함께 시작해요",
            }
    
    @staticmethod
    def friend_request(friend_name: str) -> Dict:
        """친구 요청 템플릿"""
        return {
            "title": "새로운 친구 요청 👥",
            "body": f"{friend_name}님이 친구 요청을 보냈습니다",
        }
    
    @staticmethod
    def challenge_invite(challenge_title: str, inviter_name: str) -> Dict:
        """챌린지 초대 템플릿"""
        return {
            "title": "챌린지 초대 🎯",
            "body": f"{inviter_name}님이 '{challenge_title}' 챌린지에 초대했습니다",
        }
    
    @staticmethod
    def achievement_earned(achievement_name: str, achievement_icon: str) -> Dict:
        """업적 달성 템플릿"""
        return {
            "title": f"업적 달성! {achievement_icon}",
            "body": f"축하합니다! '{achievement_name}' 업적을 달성했습니다",
        }
    
    @staticmethod
    def streak_reminder(streak_days: int) -> Dict:
        """연속 운동 리마인더 템플릿"""
        return {
            "title": f"연속 {streak_days}일째! 🔥",
            "body": f"대단해요! 오늘도 기록을 이어가세요",
        }
    
    @staticmethod
    def water_reminder(glasses_remaining: int) -> Dict:
        """물 섭취 리마인더 템플릿"""
        return {
            "title": "💧 수분 보충 시간",
            "body": f"오늘 {glasses_remaining}잔 더 마셔주세요",
        }
    
    @staticmethod
    def sleep_reminder() -> Dict:
        """수면 리마인더 템플릿"""
        return {
            "title": "😴 잠들 시간이에요",
            "body": "충분한 수면은 건강의 기본! 편안한 밤 되세요",
        }
    
    @staticmethod
    def daily_report(health_score: float, calories: float) -> Dict:
        """일일 리포트 템플릿"""
        return {
            "title": "📊 오늘의 건강 리포트",
            "body": f"건강점수 {health_score}점 | 소모 칼로리 {calories}kcal",
        }

# ========================
# Notification Service
# ========================

class NotificationService:
    """알림 서비스 (비즈니스 로직)"""
    
    def __init__(self):
        self.manager = PushNotificationManager()
        self.templates = NotificationTemplates()
    
    async def send_workout_reminder(self, db: Session):
        """운동 리마인더 전송"""
        # 오늘 운동하지 않은 사용자 조회
        today_start = datetime.combine(datetime.now().date(), datetime.min.time())
        
        users_without_workout = db.query(User).filter(
            ~User.workouts.any(Workout.start_time >= today_start),
            User.is_active == True
        ).all()
        
        payloads = []
        for user in users_without_workout:
            # 마지막 운동일 계산
            last_workout = db.query(Workout).filter(
                Workout.user_id == user.id
            ).order_by(Workout.start_time.desc()).first()
            
            if last_workout:
                days_since = (datetime.now() - last_workout.start_time).days
            else:
                days_since = 999
            
            # 템플릿 생성
            template = self.templates.workout_reminder(user.name, days_since)
            
            payload = NotificationPayload(
                user_id=user.id,
                type=NotificationType.WORKOUT_REMINDER,
                title=template["title"],
                body=template["body"],
                action_url="app://workout/start"
            )
            payloads.append(payload)
        
        # 배치 전송
        if payloads:
            results = await self.manager.send_batch_notifications(payloads)
            logger.info(f"Sent workout reminders to {len(results)} users")
    
    async def send_friend_activity_notification(
        self, 
        user_id: int, 
        friend_name: str, 
        activity_type: str,
        db: Session
    ):
        """친구 활동 알림"""
        template = {
            "title": "친구 활동 소식 🏃‍♂️",
            "body": f"{friend_name}님이 {activity_type}을(를) 완료했습니다!"
        }
        
        payload = NotificationPayload(
            user_id=user_id,
            type=NotificationType.FRIEND_ACTIVITY,
            title=template["title"],
            body=template["body"],
            data={"friend_name": friend_name, "activity": activity_type}
        )
        
        await self.manager.send_notification(payload)
    
    async def send_water_reminders(self, db: Session):
        """물 섭취 리마인더 (3시간마다)"""
        from app.models.models import HealthMetric
        
        # 오늘 물 섭취량이 부족한 사용자
        today_start = datetime.combine(datetime.now().date(), datetime.min.time())
        
        users = db.query(User).filter(User.is_active == True).all()
        
        payloads = []
        for user in users:
            # 오늘 물 섭취량 계산
            today_water = db.query(HealthMetric).filter(
                HealthMetric.user_id == user.id,
                HealthMetric.recorded_at >= today_start,
                HealthMetric.water_intake.isnot(None)
            ).all()
            
            total_water = sum(m.water_intake for m in today_water)
            
            if total_water < 1500:  # 1.5L 미만인 경우
                glasses_remaining = (2000 - total_water) // 250
                template = self.templates.water_reminder(glasses_remaining)
                
                payload = NotificationPayload(
                    user_id=user.id,
                    type=NotificationType.WATER_REMINDER,
                    title=template["title"],
                    body=template["body"],
                    action_url="app://health/water"
                )
                payloads.append(payload)
        
        if payloads:
            await self.manager.send_batch_notifications(payloads)
    
    async def send_sleep_reminder(self, db: Session):
        """수면 리마인더 (밤 10시)"""
        users = db.query(User).filter(User.is_active == True).all()
        
        template = self.templates.sleep_reminder()
        payloads = []
        
        for user in users:
            payload = NotificationPayload(
                user_id=user.id,
                type=NotificationType.SLEEP_REMINDER,
                title=template["title"],
                body=template["body"],
                action_url="app://health/sleep"
            )
            payloads.append(payload)
        
        if payloads:
            await self.manager.send_batch_notifications(payloads)

# Singleton instances
push_manager = PushNotificationManager()
notification_service = NotificationService()