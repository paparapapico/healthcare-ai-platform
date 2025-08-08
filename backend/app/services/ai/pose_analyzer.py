"""
AI Pose Analysis Engine
실시간 자세 분석 및 운동 감지 시스템
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import base64
from io import BytesIO
from PIL import Image
import json
import time

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ========================
# Data Classes
# ========================

class ExerciseType(Enum):
    SQUAT = "squat"
    PUSHUP = "pushup"
    PLANK = "plank"
    LUNGE = "lunge"
    JUMPING_JACK = "jumping_jack"
    SHOULDER_PRESS = "shoulder_press"

class ExerciseStage(Enum):
    READY = "ready"
    UP = "up"
    DOWN = "down"
    HOLD = "hold"
    REST = "rest"

@dataclass
class JointPosition:
    x: float
    y: float
    z: float
    visibility: float

@dataclass
class PoseAnalysisResult:
    exercise_type: str
    form_score: float
    stage: str
    rep_count: int
    angles: Dict[str, float]
    feedback: List[str]
    corrections: List[str]
    calories_burned: float
    duration: float

# ========================
# Pose Analyzer Class
# ========================

class PoseAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2,  # 더 정확한 모델 사용
            enable_segmentation=True,
            smooth_landmarks=True
        )
        
        # 운동별 상태 추적
        self.exercise_states = {
            exercise_type: {
                "stage": ExerciseStage.READY,
                "rep_count": 0,
                "start_time": None,
                "last_rep_time": None,
                "calories": 0,
                "form_scores": []
            }
            for exercise_type in ExerciseType
        }
        
    def calculate_angle(self, a: JointPosition, b: JointPosition, c: JointPosition) -> float:
        """3개 관절 포인트로 각도 계산"""
        a_arr = np.array([a.x, a.y])
        b_arr = np.array([b.x, b.y])
        c_arr = np.array([c.x, c.y])
        
        radians = np.arctan2(c_arr[1] - b_arr[1], c_arr[0] - b_arr[0]) - \
                  np.arctan2(a_arr[1] - b_arr[1], a_arr[0] - b_arr[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def get_joint_position(self, landmarks, joint_idx: int) -> JointPosition:
        """랜드마크에서 관절 위치 추출"""
        landmark = landmarks.landmark[joint_idx]
        return JointPosition(
            x=landmark.x,
            y=landmark.y,
            z=landmark.z,
            visibility=landmark.visibility
        )
    
    # ========================
    # Exercise-Specific Analysis
    # ========================
    
    def analyze_squat(self, landmarks) -> Dict:
        """스쿼트 분석"""
        # 관절 위치 추출
        left_hip = self.get_joint_position(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        left_knee = self.get_joint_position(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
        left_ankle = self.get_joint_position(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
        left_shoulder = self.get_joint_position(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        
        right_hip = self.get_joint_position(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
        right_knee = self.get_joint_position(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
        right_ankle = self.get_joint_position(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
        
        # 각도 계산
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        # 등 각도 (자세 확인)
        back_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        
        # 발 너비 확인
        stance_width = abs(left_ankle.x - right_ankle.x)
        shoulder_width = abs(left_shoulder.x - self.get_joint_position(
            landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER).x)
        
        # 자세 평가
        feedback = []
        corrections = []
        form_score = 100.0
        
        # 무릎 각도 체크
        if knee_angle > 160:
            stage = ExerciseStage.UP
            feedback.append("시작 자세 - 준비되었습니다")
        elif knee_angle < 90:
            stage = ExerciseStage.DOWN
            feedback.append("훌륭합니다! 충분히 내려갔습니다")
        else:
            stage = ExerciseStage.HOLD
            feedback.append(f"무릎 각도: {knee_angle:.1f}°")
        
        # 자세 교정 피드백
        if knee_angle < 70:
            corrections.append("너무 깊게 내려갔습니다. 무릎 각도를 90도 정도로 유지하세요")
            form_score -= 10
            
        if back_angle < 160:
            corrections.append("상체를 더 세우세요. 등을 곧게 펴주세요")
            form_score -= 15
            
        if stance_width < shoulder_width * 0.8:
            corrections.append("발을 조금 더 벌려주세요")
            form_score -= 10
        elif stance_width > shoulder_width * 1.5:
            corrections.append("발 간격이 너무 넓습니다")
            form_score -= 10
            
        # 무릎이 발끝을 넘는지 체크
        if left_knee.x > left_ankle.x + 0.1:
            corrections.append("왼쪽 무릎이 발끝을 넘었습니다. 엉덩이를 뒤로 빼세요")
            form_score -= 10
            
        return {
            "angles": {
                "knee": knee_angle,
                "back": back_angle,
                "left_knee": left_knee_angle,
                "right_knee": right_knee_angle
            },
            "stage": stage,
            "form_score": max(0, form_score),
            "feedback": feedback,
            "corrections": corrections,
            "stance_width": stance_width
        }
    
    def analyze_pushup(self, landmarks) -> Dict:
        """푸시업 분석"""
        # 관절 위치 추출
        left_shoulder = self.get_joint_position(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        left_elbow = self.get_joint_position(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
        left_wrist = self.get_joint_position(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
        left_hip = self.get_joint_position(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        left_ankle = self.get_joint_position(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
        
        # 팔꿈치 각도
        elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # 몸통 일직선 확인 (플랭크 자세)
        body_alignment = self.calculate_angle(left_shoulder, left_hip, left_ankle)
        
        feedback = []
        corrections = []
        form_score = 100.0
        
        # 푸시업 단계 판정
        if elbow_angle > 160:
            stage = ExerciseStage.UP
            feedback.append("팔을 완전히 폈습니다")
        elif elbow_angle < 90:
            stage = ExerciseStage.DOWN
            feedback.append("충분히 내려갔습니다")
        else:
            stage = ExerciseStage.HOLD
            
        # 자세 교정
        if body_alignment < 160:
            corrections.append("엉덩이가 처졌습니다. 코어에 힘을 주세요")
            form_score -= 20
        elif body_alignment > 190:
            corrections.append("엉덩이가 너무 올라갔습니다")
            form_score -= 15
            
        if elbow_angle < 45:
            corrections.append("너무 깊게 내려갔습니다")
            form_score -= 10
            
        return {
            "angles": {
                "elbow": elbow_angle,
                "body_alignment": body_alignment
            },
            "stage": stage,
            "form_score": max(0, form_score),
            "feedback": feedback,
            "corrections": corrections
        }
    
    def analyze_plank(self, landmarks) -> Dict:
        """플랭크 분석"""
        # 관절 위치 추출
        left_shoulder = self.get_joint_position(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        left_elbow = self.get_joint_position(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
        left_hip = self.get_joint_position(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        left_ankle = self.get_joint_position(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
        
        # 몸통 일직선 확인
        body_alignment = self.calculate_angle(left_shoulder, left_hip, left_ankle)
        
        # 팔꿈치 위치 확인
        elbow_under_shoulder = abs(left_shoulder.x - left_elbow.x) < 0.1
        
        feedback = []
        corrections = []
        form_score = 100.0
        
        stage = ExerciseStage.HOLD
        feedback.append("플랭크 자세 유지 중")
        
        # 자세 교정
        if body_alignment < 160:
            corrections.append("엉덩이를 올려 일직선을 만드세요")
            form_score -= 25
        elif body_alignment > 190:
            corrections.append("엉덩이를 낮춰 일직선을 만드세요")
            form_score -= 25
            
        if not elbow_under_shoulder:
            corrections.append("팔꿈치를 어깨 바로 아래 위치시키세요")
            form_score -= 15
            
        return {
            "angles": {
                "body_alignment": body_alignment
            },
            "stage": stage,
            "form_score": max(0, form_score),
            "feedback": feedback,
            "corrections": corrections
        }
    
    # ========================
    # Main Analysis Function
    # ========================
    
    def analyze_frame(self, frame: np.ndarray, exercise_type: ExerciseType) -> PoseAnalysisResult:
        """단일 프레임 분석"""
        # BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # 포즈 감지
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return PoseAnalysisResult(
                exercise_type=exercise_type.value,
                form_score=0,
                stage="no_pose_detected",
                rep_count=0,
                angles={},
                feedback=["사람이 감지되지 않습니다. 카메라 앞에 서주세요"],
                corrections=[],
                calories_burned=0,
                duration=0
            )
        
        # 운동 종류별 분석
        if exercise_type == ExerciseType.SQUAT:
            analysis = self.analyze_squat(results.pose_landmarks)
        elif exercise_type == ExerciseType.PUSHUP:
            analysis = self.analyze_pushup(results.pose_landmarks)
        elif exercise_type == ExerciseType.PLANK:
            analysis = self.analyze_plank(results.pose_landmarks)
        else:
            analysis = self.analyze_squat(results.pose_landmarks)  # 기본값
        
        # 상태 업데이트 및 반복 횟수 계산
        state = self.exercise_states[exercise_type]
        current_time = time.time()
        
        if state["start_time"] is None:
            state["start_time"] = current_time
            
        # 반복 횟수 계산 (UP -> DOWN -> UP 사이클)
        if analysis["stage"] != state["stage"]:
            if state["stage"] == ExerciseStage.DOWN and analysis["stage"] == ExerciseStage.UP:
                state["rep_count"] += 1
                state["last_rep_time"] = current_time
                
                # 칼로리 계산 (운동 종류별 MET 값 적용)
                met_values = {
                    ExerciseType.SQUAT: 5.0,
                    ExerciseType.PUSHUP: 8.0,
                    ExerciseType.PLANK: 3.5,
                    ExerciseType.LUNGE: 4.0
                }
                met = met_values.get(exercise_type, 4.0)
                # 칼로리 = MET * 체중(kg) * 시간(hours)
                # 여기서는 한 rep당 예상 칼로리
                state["calories"] += met * 70 * (1/360)  # 70kg 기준, 10초당
                
            state["stage"] = analysis["stage"]
        
        # Form score 추적
        state["form_scores"].append(analysis["form_score"])
        avg_form_score = np.mean(state["form_scores"][-30:])  # 최근 30프레임 평균
        
        duration = current_time - state["start_time"] if state["start_time"] else 0
        
        return PoseAnalysisResult(
            exercise_type=exercise_type.value,
            form_score=avg_form_score,
            stage=analysis["stage"].value,
            rep_count=state["rep_count"],
            angles=analysis["angles"],
            feedback=analysis["feedback"],
            corrections=analysis["corrections"],
            calories_burned=round(state["calories"], 1),
            duration=round(duration, 1)
        )
    
    def analyze_base64_image(self, image_base64: str, exercise_type: str) -> Dict:
        """Base64 이미지 분석"""
        try:
            # Base64 디코딩
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            
            # PIL to numpy array
            frame = np.array(image)
            
            # BGR to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 운동 타입 변환
            exercise_enum = ExerciseType(exercise_type)
            
            # 분석 수행
            result = self.analyze_frame(frame, exercise_enum)
            
            return {
                "success": True,
                "data": {
                    "exercise_type": result.exercise_type,
                    "form_score": result.form_score,
                    "stage": result.stage,
                    "rep_count": result.rep_count,
                    "angles": result.angles,
                    "feedback": result.feedback,
                    "corrections": result.corrections,
                    "calories_burned": result.calories_burned,
                    "duration": result.duration
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def draw_pose(self, image: np.ndarray, landmarks) -> np.ndarray:
        """포즈 스켈레톤 그리기"""
        if landmarks:
            mp_drawing.draw_landmarks(
                image,
                landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
        return image
    
    def reset_exercise_state(self, exercise_type: ExerciseType):
        """운동 상태 초기화"""
        self.exercise_states[exercise_type] = {
            "stage": ExerciseStage.READY,
            "rep_count": 0,
            "start_time": None,
            "last_rep_time": None,
            "calories": 0,
            "form_scores": []
        }

# ========================
# Singleton Instance
# ========================

pose_analyzer = PoseAnalyzer()