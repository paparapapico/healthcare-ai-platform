"""
사용자 제출 운동 데이터 크라우드소싱 플랫폼
User-Submitted Exercise Data Crowdsourcing Platform
"""

import os
import json
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import cv2
import mediapipe as mp
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import asyncio
import aiofiles
from PIL import Image
import io

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 데이터베이스 설정
Base = declarative_base()
engine = create_engine('sqlite:///crowdsource_data.db')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPI 앱
app = FastAPI(title="Exercise Data Crowdsourcing Platform")

class UserSubmission(Base):
    """사용자 제출 데이터 모델"""
    __tablename__ = "submissions"
    
    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(String, unique=True, index=True)
    user_id = Column(String, index=True)
    username = Column(String)
    email = Column(String)
    
    # 운동 정보
    exercise_type = Column(String)
    skill_level = Column(String)  # beginner, intermediate, advanced, professional
    years_experience = Column(Integer)
    
    # 파일 정보
    file_path = Column(String)
    file_type = Column(String)  # video, image
    file_size = Column(Integer)
    duration = Column(Float)  # 비디오 길이 (초)
    
    # 품질 평가
    quality_score = Column(Float)
    pose_confidence = Column(Float)
    visibility_score = Column(Float)
    
    # 검증 상태
    is_verified = Column(Boolean, default=False)
    verification_notes = Column(Text)
    verified_by = Column(String)
    
    # 보상
    reward_points = Column(Integer, default=0)
    reward_tier = Column(String)  # bronze, silver, gold, platinum
    
    # 메타데이터
    device_info = Column(String)
    location = Column(String)
    submission_date = Column(DateTime, default=datetime.utcnow)
    
    # 처리 상태
    processing_status = Column(String, default='pending')  # pending, processing, completed, rejected
    pose_data_path = Column(String)
    feedback = Column(Text)

class SubmissionRequest(BaseModel):
    """제출 요청 모델"""
    user_id: str
    username: str
    email: str
    exercise_type: str
    skill_level: str = Field(default="intermediate", description="beginner, intermediate, advanced, professional")
    years_experience: int = Field(default=1, ge=0, le=50)
    device_info: Optional[str] = None
    location: Optional[str] = None

class QualityAnalyzer:
    """제출된 데이터 품질 분석기"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 운동별 품질 기준
        self.quality_criteria = {
            'squat': {
                'min_frames': 30,  # 최소 1초
                'min_visibility': 0.6,
                'required_joints': ['hip', 'knee', 'ankle'],
                'angle_ranges': {
                    'knee': (60, 180),
                    'hip': (45, 180)
                }
            },
            'push_up': {
                'min_frames': 30,
                'min_visibility': 0.6,
                'required_joints': ['shoulder', 'elbow', 'wrist'],
                'angle_ranges': {
                    'elbow': (60, 180)
                }
            },
            'deadlift': {
                'min_frames': 30,
                'min_visibility': 0.7,
                'required_joints': ['hip', 'knee', 'ankle', 'shoulder'],
                'angle_ranges': {
                    'hip': (20, 180),
                    'knee': (90, 180)
                }
            }
        }
    
    async def analyze_video_quality(self, video_path: str, exercise_type: str) -> Dict:
        """비디오 품질 분석"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        quality_metrics = {
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'valid_frames': 0,
            'avg_visibility': 0,
            'pose_confidence': 0,
            'quality_score': 0,
            'issues': [],
            'pose_data': []
        }
        
        # 최소 요구사항 체크
        criteria = self.quality_criteria.get(exercise_type, self.quality_criteria['squat'])
        
        if total_frames < criteria['min_frames']:
            quality_metrics['issues'].append(f"비디오가 너무 짧습니다 (최소 {criteria['min_frames']} 프레임 필요)")
        
        visibility_scores = []
        confidence_scores = []
        pose_frames = []
        
        frame_count = 0
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 포즈 감지
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # 랜드마크 추출
                landmarks = []
                visibility_sum = 0
                
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                    visibility_sum += landmark.visibility
                
                avg_visibility = visibility_sum / len(results.pose_landmarks.landmark)
                visibility_scores.append(avg_visibility)
                
                if avg_visibility >= criteria['min_visibility']:
                    quality_metrics['valid_frames'] += 1
                    pose_frames.append({
                        'frame_id': frame_count,
                        'timestamp': frame_count / fps,
                        'landmarks': landmarks,
                        'visibility': avg_visibility
                    })
                    
                    # 관절 각도 체크
                    angles = self.calculate_joint_angles(landmarks)
                    if self.validate_angles(angles, criteria['angle_ranges']):
                        confidence_scores.append(1.0)
                    else:
                        confidence_scores.append(0.5)
            
            frame_count += 1
        
        cap.release()
        
        # 메트릭 계산
        if visibility_scores:
            quality_metrics['avg_visibility'] = np.mean(visibility_scores)
        
        if confidence_scores:
            quality_metrics['pose_confidence'] = np.mean(confidence_scores)
        
        # 품질 점수 계산
        quality_score = 0
        
        # 프레임 비율 (30%)
        frame_ratio = quality_metrics['valid_frames'] / max(total_frames, 1)
        quality_score += frame_ratio * 30
        
        # 가시성 점수 (30%)
        quality_score += quality_metrics['avg_visibility'] * 30
        
        # 포즈 신뢰도 (30%)
        quality_score += quality_metrics['pose_confidence'] * 30
        
        # 지속 시간 보너스 (10%)
        if duration >= 5:  # 5초 이상
            quality_score += 10
        elif duration >= 3:
            quality_score += 5
        
        quality_metrics['quality_score'] = min(100, quality_score)
        quality_metrics['pose_data'] = pose_frames
        
        # 피드백 생성
        if quality_metrics['quality_score'] < 60:
            quality_metrics['issues'].append("전반적인 품질이 낮습니다. 조명과 카메라 각도를 개선해주세요.")
        
        return quality_metrics
    
    def calculate_joint_angles(self, landmarks: List[Dict]) -> Dict:
        """관절 각도 계산"""
        angles = {}
        
        # MediaPipe 랜드마크 인덱스
        POSE_LANDMARKS = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        
        # 무릎 각도
        if all(landmarks[i] for i in [POSE_LANDMARKS['left_hip'], 
                                      POSE_LANDMARKS['left_knee'], 
                                      POSE_LANDMARKS['left_ankle']]):
            angles['knee'] = self.calculate_angle(
                landmarks[POSE_LANDMARKS['left_hip']],
                landmarks[POSE_LANDMARKS['left_knee']],
                landmarks[POSE_LANDMARKS['left_ankle']]
            )
        
        # 엉덩이 각도
        if all(landmarks[i] for i in [POSE_LANDMARKS['left_shoulder'],
                                      POSE_LANDMARKS['left_hip'],
                                      POSE_LANDMARKS['left_knee']]):
            angles['hip'] = self.calculate_angle(
                landmarks[POSE_LANDMARKS['left_shoulder']],
                landmarks[POSE_LANDMARKS['left_hip']],
                landmarks[POSE_LANDMARKS['left_knee']]
            )
        
        # 팔꿈치 각도
        if all(landmarks[i] for i in [POSE_LANDMARKS['left_shoulder'],
                                      POSE_LANDMARKS['left_elbow'],
                                      POSE_LANDMARKS['left_wrist']]):
            angles['elbow'] = self.calculate_angle(
                landmarks[POSE_LANDMARKS['left_shoulder']],
                landmarks[POSE_LANDMARKS['left_elbow']],
                landmarks[POSE_LANDMARKS['left_wrist']]
            )
        
        return angles
    
    def calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """3점 사이 각도 계산"""
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def validate_angles(self, angles: Dict, angle_ranges: Dict) -> bool:
        """각도 유효성 검증"""
        for joint, (min_angle, max_angle) in angle_ranges.items():
            if joint in angles:
                if not (min_angle <= angles[joint] <= max_angle):
                    return False
        return True

class RewardSystem:
    """보상 시스템"""
    
    # 보상 포인트 체계
    REWARD_POINTS = {
        'submission': 10,  # 기본 제출
        'high_quality': 20,  # 고품질 (80점 이상)
        'professional': 50,  # 전문가 레벨
        'verified': 30,  # 검증 완료
        'rare_exercise': 25,  # 희귀 운동
        'long_duration': 15,  # 긴 영상 (30초 이상)
        'perfect_form': 40,  # 완벽한 자세
    }
    
    # 티어 시스템
    TIERS = {
        'bronze': {'min_points': 0, 'benefits': ['기본 피드백']},
        'silver': {'min_points': 100, 'benefits': ['상세 피드백', '월간 리포트']},
        'gold': {'min_points': 500, 'benefits': ['AI 코칭', '개인 분석', '프리미엄 콘텐츠']},
        'platinum': {'min_points': 1000, 'benefits': ['1:1 전문가 상담', 'VIP 기능', '수익 공유']}
    }
    
    @classmethod
    def calculate_rewards(cls, quality_metrics: Dict, exercise_type: str, skill_level: str) -> Dict:
        """보상 계산"""
        points = cls.REWARD_POINTS['submission']
        bonuses = []
        
        # 품질 보너스
        if quality_metrics['quality_score'] >= 80:
            points += cls.REWARD_POINTS['high_quality']
            bonuses.append('고품질 데이터')
        
        # 전문가 레벨 보너스
        if skill_level == 'professional':
            points += cls.REWARD_POINTS['professional']
            bonuses.append('전문가 레벨')
        
        # 긴 영상 보너스
        if quality_metrics.get('duration', 0) >= 30:
            points += cls.REWARD_POINTS['long_duration']
            bonuses.append('긴 영상')
        
        # 희귀 운동 보너스
        rare_exercises = ['snatch', 'clean_and_jerk', 'muscle_up', 'handstand_push_up']
        if exercise_type in rare_exercises:
            points += cls.REWARD_POINTS['rare_exercise']
            bonuses.append('희귀 운동')
        
        # 완벽한 자세 보너스
        if quality_metrics.get('pose_confidence', 0) >= 0.95:
            points += cls.REWARD_POINTS['perfect_form']
            bonuses.append('완벽한 자세')
        
        # 티어 결정
        tier = 'bronze'
        for tier_name, tier_data in cls.TIERS.items():
            if points >= tier_data['min_points']:
                tier = tier_name
        
        return {
            'points': points,
            'tier': tier,
            'bonuses': bonuses,
            'benefits': cls.TIERS[tier]['benefits']
        }

# API 엔드포인트
@app.post("/submit")
async def submit_exercise_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    submission_data: str = Form(...),
    db: Session = SessionLocal()
):
    """운동 데이터 제출 API"""
    try:
        # JSON 파싱
        submission_info = json.loads(submission_data)
        request = SubmissionRequest(**submission_info)
        
        # 제출 ID 생성
        submission_id = hashlib.md5(f"{request.user_id}_{datetime.now().isoformat()}".encode()).hexdigest()
        
        # 파일 저장
        os.makedirs('crowdsource_uploads', exist_ok=True)
        file_path = f"crowdsource_uploads/{submission_id}_{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # DB 레코드 생성
        submission = UserSubmission(
            submission_id=submission_id,
            user_id=request.user_id,
            username=request.username,
            email=request.email,
            exercise_type=request.exercise_type,
            skill_level=request.skill_level,
            years_experience=request.years_experience,
            file_path=file_path,
            file_type='video' if file.filename.endswith(('.mp4', '.avi', '.mov')) else 'image',
            file_size=len(content),
            device_info=request.device_info,
            location=request.location,
            processing_status='pending'
        )
        
        db.add(submission)
        db.commit()
        
        # 백그라운드 처리
        background_tasks.add_task(process_submission, submission_id)
        
        return JSONResponse({
            'status': 'success',
            'submission_id': submission_id,
            'message': '제출이 완료되었습니다. 처리 중입니다.'
        })
    
    except Exception as e:
        logger.error(f"Submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        db.close()

async def process_submission(submission_id: str):
    """제출 데이터 백그라운드 처리"""
    db = SessionLocal()
    
    try:
        # 제출 데이터 조회
        submission = db.query(UserSubmission).filter(
            UserSubmission.submission_id == submission_id
        ).first()
        
        if not submission:
            return
        
        # 상태 업데이트
        submission.processing_status = 'processing'
        db.commit()
        
        # 품질 분석
        analyzer = QualityAnalyzer()
        quality_metrics = await analyzer.analyze_video_quality(
            submission.file_path,
            submission.exercise_type
        )
        
        # 품질 메트릭 저장
        submission.quality_score = quality_metrics['quality_score']
        submission.pose_confidence = quality_metrics['pose_confidence']
        submission.visibility_score = quality_metrics['avg_visibility']
        submission.duration = quality_metrics['duration']
        
        # 포즈 데이터 저장
        pose_data_path = f"crowdsource_uploads/poses_{submission_id}.json"
        with open(pose_data_path, 'w') as f:
            json.dump(quality_metrics['pose_data'], f)
        submission.pose_data_path = pose_data_path
        
        # 보상 계산
        rewards = RewardSystem.calculate_rewards(
            quality_metrics,
            submission.exercise_type,
            submission.skill_level
        )
        
        submission.reward_points = rewards['points']
        submission.reward_tier = rewards['tier']
        
        # 피드백 생성
        feedback = generate_feedback(quality_metrics, rewards)
        submission.feedback = json.dumps(feedback)
        
        # 상태 업데이트
        if quality_metrics['quality_score'] >= 60:
            submission.processing_status = 'completed'
        else:
            submission.processing_status = 'rejected'
            submission.feedback = json.dumps({
                'message': '품질 기준을 충족하지 못했습니다.',
                'issues': quality_metrics['issues']
            })
        
        db.commit()
        
        # 이메일 알림 (선택적)
        # await send_notification_email(submission.email, submission_id, rewards)
        
    except Exception as e:
        logger.error(f"Processing error for {submission_id}: {e}")
        submission.processing_status = 'error'
        submission.feedback = str(e)
        db.commit()
    
    finally:
        db.close()

def generate_feedback(quality_metrics: Dict, rewards: Dict) -> Dict:
    """피드백 생성"""
    feedback = {
        'quality_score': quality_metrics['quality_score'],
        'rewards': rewards,
        'strengths': [],
        'improvements': [],
        'tips': []
    }
    
    # 강점
    if quality_metrics['avg_visibility'] >= 0.8:
        feedback['strengths'].append('훌륭한 카메라 각도와 가시성')
    
    if quality_metrics['pose_confidence'] >= 0.9:
        feedback['strengths'].append('매우 정확한 자세')
    
    if quality_metrics['duration'] >= 10:
        feedback['strengths'].append('충분한 운동 시간')
    
    # 개선점
    if quality_metrics['avg_visibility'] < 0.6:
        feedback['improvements'].append('카메라 각도를 조정하여 전신이 보이도록 하세요')
    
    if quality_metrics['pose_confidence'] < 0.7:
        feedback['improvements'].append('자세 정확도를 높여주세요')
    
    if quality_metrics['valid_frames'] / max(quality_metrics['total_frames'], 1) < 0.7:
        feedback['improvements'].append('더 안정적인 촬영 환경이 필요합니다')
    
    # 팁
    feedback['tips'] = [
        '밝은 조명에서 촬영하면 더 정확한 분석이 가능합니다',
        '카메라를 삼각대에 고정하면 안정적인 영상을 얻을 수 있습니다',
        '전신이 화면에 들어오도록 충분한 거리를 확보하세요'
    ]
    
    return feedback

@app.get("/submission/{submission_id}")
async def get_submission_status(submission_id: str, db: Session = SessionLocal()):
    """제출 상태 조회 API"""
    try:
        submission = db.query(UserSubmission).filter(
            UserSubmission.submission_id == submission_id
        ).first()
        
        if not submission:
            raise HTTPException(status_code=404, detail="제출을 찾을 수 없습니다")
        
        return {
            'submission_id': submission.submission_id,
            'status': submission.processing_status,
            'quality_score': submission.quality_score,
            'reward_points': submission.reward_points,
            'reward_tier': submission.reward_tier,
            'feedback': json.loads(submission.feedback) if submission.feedback else None
        }
    
    finally:
        db.close()

@app.get("/leaderboard")
async def get_leaderboard(db: Session = SessionLocal()):
    """리더보드 API"""
    try:
        # 상위 사용자 조회
        top_users = db.query(
            UserSubmission.username,
            UserSubmission.user_id,
            db.func.sum(UserSubmission.reward_points).label('total_points'),
            db.func.count(UserSubmission.id).label('total_submissions'),
            db.func.avg(UserSubmission.quality_score).label('avg_quality')
        ).group_by(
            UserSubmission.user_id,
            UserSubmission.username
        ).order_by(
            db.func.sum(UserSubmission.reward_points).desc()
        ).limit(100).all()
        
        leaderboard = []
        for rank, user in enumerate(top_users, 1):
            # 티어 결정
            tier = 'bronze'
            for tier_name, tier_data in RewardSystem.TIERS.items():
                if user.total_points >= tier_data['min_points']:
                    tier = tier_name
            
            leaderboard.append({
                'rank': rank,
                'username': user.username,
                'user_id': user.user_id,
                'total_points': user.total_points,
                'total_submissions': user.total_submissions,
                'avg_quality': round(user.avg_quality, 1) if user.avg_quality else 0,
                'tier': tier
            })
        
        return {
            'leaderboard': leaderboard,
            'updated_at': datetime.now().isoformat()
        }
    
    finally:
        db.close()

@app.get("/stats")
async def get_platform_stats(db: Session = SessionLocal()):
    """플랫폼 통계 API"""
    try:
        total_submissions = db.query(UserSubmission).count()
        total_users = db.query(UserSubmission.user_id).distinct().count()
        avg_quality = db.query(db.func.avg(UserSubmission.quality_score)).scalar()
        
        exercise_stats = db.query(
            UserSubmission.exercise_type,
            db.func.count(UserSubmission.id).label('count')
        ).group_by(UserSubmission.exercise_type).all()
        
        skill_distribution = db.query(
            UserSubmission.skill_level,
            db.func.count(UserSubmission.id).label('count')
        ).group_by(UserSubmission.skill_level).all()
        
        return {
            'total_submissions': total_submissions,
            'total_users': total_users,
            'avg_quality_score': round(avg_quality, 1) if avg_quality else 0,
            'exercise_distribution': {ex.exercise_type: ex.count for ex in exercise_stats},
            'skill_distribution': {skill.skill_level: skill.count for skill in skill_distribution},
            'updated_at': datetime.now().isoformat()
        }
    
    finally:
        db.close()

# 데이터베이스 초기화
Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    import uvicorn
    
    print("""
    🚀 크라우드소싱 플랫폼 시작!
    
    📱 API 엔드포인트:
    - POST /submit - 운동 데이터 제출
    - GET /submission/{id} - 제출 상태 조회
    - GET /leaderboard - 리더보드
    - GET /stats - 플랫폼 통계
    
    📊 보상 시스템:
    - 기본 제출: 10 포인트
    - 고품질 데이터: +20 포인트
    - 전문가 레벨: +50 포인트
    - 완벽한 자세: +40 포인트
    
    🏆 티어 시스템:
    - Bronze: 0+ 포인트
    - Silver: 100+ 포인트
    - Gold: 500+ 포인트
    - Platinum: 1000+ 포인트
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)