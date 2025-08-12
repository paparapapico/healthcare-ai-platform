"""
전문 운동선수 데이터 수집 모듈
올림픽 선수, 프로 운동선수들의 동작 데이터를 수집하고 처리
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpertDataCollector:
    """전문가 수준 운동 데이터 수집기"""
    
    # MediaPipe 랜드마크 인덱스
    POSE_LANDMARKS = {
        'nose': 0,
        'left_eye': 2,
        'right_eye': 5,
        'left_ear': 7,
        'right_ear': 8,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
        'left_heel': 29,
        'right_heel': 30,
        'left_foot': 31,
        'right_foot': 32
    }
    
    def __init__(self, model_complexity: int = 2):
        """
        Args:
            model_complexity: MediaPipe 모델 복잡도 (0, 1, 2)
                            2 = 최고 정확도
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=True,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.collected_data = []
        self.frame_buffer = []
        
    def collect_from_video(self, 
                          video_path: str, 
                          athlete_info: Dict,
                          exercise_type: str) -> List[Dict]:
        """
        비디오에서 전문가 데이터 수집
        
        Args:
            video_path: 비디오 파일 경로
            athlete_info: 선수 정보 (이름, 레벨, 국가 등)
            exercise_type: 운동 종류
            
        Returns:
            수집된 데이터 리스트
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"FPS: {fps}, Exercise: {exercise_type}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # 랜드마크 추출
                landmarks = self.extract_landmarks(results.pose_landmarks)
                
                # 운동역학적 특징 계산
                biomechanics = self.calculate_biomechanics(landmarks)
                
                # 시간적 특징 (이전 프레임과 비교)
                temporal_features = self.calculate_temporal_features(
                    landmarks, 
                    self.frame_buffer[-1] if self.frame_buffer else None
                )
                
                # 데이터 저장
                frame_data = {
                    'frame_id': frame_count,
                    'timestamp': frame_count / fps,
                    'athlete': athlete_info,
                    'exercise_type': exercise_type,
                    'landmarks': landmarks,
                    'biomechanics': biomechanics,
                    'temporal': temporal_features,
                    'quality_score': self.estimate_quality_score(biomechanics, exercise_type)
                }
                
                self.collected_data.append(frame_data)
                self.frame_buffer.append(landmarks)
                
                # 버퍼 크기 제한 (메모리 관리)
                if len(self.frame_buffer) > 30:
                    self.frame_buffer.pop(0)
            
            frame_count += 1
            
            # 진행 상황 출력
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames...")
        
        cap.release()
        logger.info(f"Total frames processed: {frame_count}")
        
        return self.collected_data
    
    def extract_landmarks(self, pose_landmarks) -> Dict:
        """3D 랜드마크 추출 및 정규화"""
        landmarks = {}
        
        for name, idx in self.POSE_LANDMARKS.items():
            if idx < len(pose_landmarks.landmark):
                landmark = pose_landmarks.landmark[idx]
                landmarks[name] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
        
        # 정규화 (hip center 기준)
        if 'left_hip' in landmarks and 'right_hip' in landmarks:
            hip_center = {
                'x': (landmarks['left_hip']['x'] + landmarks['right_hip']['x']) / 2,
                'y': (landmarks['left_hip']['y'] + landmarks['right_hip']['y']) / 2,
                'z': (landmarks['left_hip']['z'] + landmarks['right_hip']['z']) / 2
            }
            
            for name in landmarks:
                landmarks[name]['x'] -= hip_center['x']
                landmarks[name]['y'] -= hip_center['y']
                landmarks[name]['z'] -= hip_center['z']
        
        return landmarks
    
    def calculate_biomechanics(self, landmarks: Dict) -> Dict:
        """운동역학적 특징 계산"""
        features = {}
        
        # 1. 관절 각도
        features['angles'] = {
            'left_elbow': self.calculate_angle(
                landmarks.get('left_shoulder'),
                landmarks.get('left_elbow'),
                landmarks.get('left_wrist')
            ),
            'right_elbow': self.calculate_angle(
                landmarks.get('right_shoulder'),
                landmarks.get('right_elbow'),
                landmarks.get('right_wrist')
            ),
            'left_knee': self.calculate_angle(
                landmarks.get('left_hip'),
                landmarks.get('left_knee'),
                landmarks.get('left_ankle')
            ),
            'right_knee': self.calculate_angle(
                landmarks.get('right_hip'),
                landmarks.get('right_knee'),
                landmarks.get('right_ankle')
            ),
            'left_hip': self.calculate_angle(
                landmarks.get('left_shoulder'),
                landmarks.get('left_hip'),
                landmarks.get('left_knee')
            ),
            'right_hip': self.calculate_angle(
                landmarks.get('right_shoulder'),
                landmarks.get('right_hip'),
                landmarks.get('right_knee')
            )
        }
        
        # 2. 신체 정렬
        features['alignment'] = {
            'spine': self.calculate_spine_angle(landmarks),
            'shoulder_tilt': self.calculate_shoulder_tilt(landmarks),
            'hip_tilt': self.calculate_hip_tilt(landmarks),
            'knee_valgus': self.calculate_knee_valgus(landmarks)
        }
        
        # 3. 무게중심 (Center of Mass)
        features['center_of_mass'] = self.calculate_center_of_mass(landmarks)
        
        # 4. 안정성 지표
        features['stability'] = {
            'base_of_support': self.calculate_base_of_support(landmarks),
            'sway_area': self.calculate_sway_area(landmarks)
        }
        
        return features
    
    def calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """3점 사이의 각도 계산 (3D)"""
        if not all([p1, p2, p3]):
            return 0.0
            
        # 벡터 계산
        v1 = np.array([
            p1['x'] - p2['x'],
            p1['y'] - p2['y'],
            p1['z'] - p2['z']
        ])
        v2 = np.array([
            p3['x'] - p2['x'],
            p3['y'] - p2['y'],
            p3['z'] - p2['z']
        ])
        
        # 각도 계산
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_spine_angle(self, landmarks: Dict) -> float:
        """척추 각도 계산"""
        if not all(k in landmarks for k in ['nose', 'left_hip', 'right_hip']):
            return 0.0
            
        # 중간 지점 계산
        hip_center = {
            'x': (landmarks['left_hip']['x'] + landmarks['right_hip']['x']) / 2,
            'y': (landmarks['left_hip']['y'] + landmarks['right_hip']['y']) / 2,
            'z': (landmarks['left_hip']['z'] + landmarks['right_hip']['z']) / 2
        }
        
        # 수직선과의 각도
        spine_vector = np.array([
            landmarks['nose']['x'] - hip_center['x'],
            landmarks['nose']['y'] - hip_center['y'],
            0  # z축 무시 (정면 각도만)
        ])
        
        vertical = np.array([0, -1, 0])
        
        cosine = np.dot(spine_vector, vertical) / (np.linalg.norm(spine_vector) * np.linalg.norm(vertical) + 1e-8)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_shoulder_tilt(self, landmarks: Dict) -> float:
        """어깨 기울기 계산"""
        if not all(k in landmarks for k in ['left_shoulder', 'right_shoulder']):
            return 0.0
            
        # 어깨 라인 벡터
        shoulder_vector = np.array([
            landmarks['right_shoulder']['x'] - landmarks['left_shoulder']['x'],
            landmarks['right_shoulder']['y'] - landmarks['left_shoulder']['y'],
            0
        ])
        
        # 수평선과의 각도
        horizontal = np.array([1, 0, 0])
        
        cosine = np.dot(shoulder_vector, horizontal) / (np.linalg.norm(shoulder_vector) + 1e-8)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_hip_tilt(self, landmarks: Dict) -> float:
        """골반 기울기 계산"""
        if not all(k in landmarks for k in ['left_hip', 'right_hip']):
            return 0.0
            
        hip_vector = np.array([
            landmarks['right_hip']['x'] - landmarks['left_hip']['x'],
            landmarks['right_hip']['y'] - landmarks['left_hip']['y'],
            0
        ])
        
        horizontal = np.array([1, 0, 0])
        
        cosine = np.dot(hip_vector, horizontal) / (np.linalg.norm(hip_vector) + 1e-8)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_knee_valgus(self, landmarks: Dict) -> Dict:
        """무릎 외반 각도 (Knee Valgus) 계산"""
        result = {}
        
        # 왼쪽 무릎
        if all(k in landmarks for k in ['left_hip', 'left_knee', 'left_ankle']):
            hip_knee = np.array([
                landmarks['left_knee']['x'] - landmarks['left_hip']['x'],
                landmarks['left_knee']['y'] - landmarks['left_hip']['y'],
                0
            ])
            knee_ankle = np.array([
                landmarks['left_ankle']['x'] - landmarks['left_knee']['x'],
                landmarks['left_ankle']['y'] - landmarks['left_knee']['y'],
                0
            ])
            
            # Q-angle 계산
            cosine = np.dot(hip_knee, knee_ankle) / (np.linalg.norm(hip_knee) * np.linalg.norm(knee_ankle) + 1e-8)
            result['left'] = 180 - np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        
        # 오른쪽 무릎
        if all(k in landmarks for k in ['right_hip', 'right_knee', 'right_ankle']):
            hip_knee = np.array([
                landmarks['right_knee']['x'] - landmarks['right_hip']['x'],
                landmarks['right_knee']['y'] - landmarks['right_hip']['y'],
                0
            ])
            knee_ankle = np.array([
                landmarks['right_ankle']['x'] - landmarks['right_knee']['x'],
                landmarks['right_ankle']['y'] - landmarks['right_knee']['y'],
                0
            ])
            
            cosine = np.dot(hip_knee, knee_ankle) / (np.linalg.norm(hip_knee) * np.linalg.norm(knee_ankle) + 1e-8)
            result['right'] = 180 - np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        
        return result
    
    def calculate_center_of_mass(self, landmarks: Dict) -> Dict:
        """무게중심 계산 (세그먼트 가중치 적용)"""
        # 신체 세그먼트별 가중치 (Dempster's data)
        segment_weights = {
            'head': 0.081,
            'trunk': 0.497,
            'upper_arm': 0.028,
            'forearm': 0.016,
            'hand': 0.006,
            'thigh': 0.100,
            'shank': 0.0465,
            'foot': 0.0145
        }
        
        com_x, com_y, com_z = 0, 0, 0
        total_weight = 0
        
        # 머리
        if 'nose' in landmarks:
            weight = segment_weights['head']
            com_x += landmarks['nose']['x'] * weight
            com_y += landmarks['nose']['y'] * weight
            com_z += landmarks['nose']['z'] * weight
            total_weight += weight
        
        # 몸통 (어깨와 엉덩이 중점)
        if all(k in landmarks for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            trunk_x = (landmarks['left_shoulder']['x'] + landmarks['right_shoulder']['x'] + 
                      landmarks['left_hip']['x'] + landmarks['right_hip']['x']) / 4
            trunk_y = (landmarks['left_shoulder']['y'] + landmarks['right_shoulder']['y'] + 
                      landmarks['left_hip']['y'] + landmarks['right_hip']['y']) / 4
            trunk_z = (landmarks['left_shoulder']['z'] + landmarks['right_shoulder']['z'] + 
                      landmarks['left_hip']['z'] + landmarks['right_hip']['z']) / 4
            
            weight = segment_weights['trunk']
            com_x += trunk_x * weight
            com_y += trunk_y * weight
            com_z += trunk_z * weight
            total_weight += weight
        
        if total_weight > 0:
            return {
                'x': com_x / total_weight,
                'y': com_y / total_weight,
                'z': com_z / total_weight
            }
        
        return {'x': 0, 'y': 0, 'z': 0}
    
    def calculate_base_of_support(self, landmarks: Dict) -> float:
        """지지 기저면 계산"""
        foot_points = []
        
        for foot in ['left_ankle', 'right_ankle', 'left_heel', 'right_heel', 
                    'left_foot', 'right_foot']:
            if foot in landmarks:
                foot_points.append([landmarks[foot]['x'], landmarks[foot]['y']])
        
        if len(foot_points) < 2:
            return 0.0
        
        # Convex hull 면적 계산
        foot_points = np.array(foot_points)
        from scipy.spatial import ConvexHull
        
        try:
            hull = ConvexHull(foot_points)
            return hull.volume  # 2D에서는 area
        except:
            return 0.0
    
    def calculate_sway_area(self, landmarks: Dict) -> float:
        """흔들림 면적 계산 (안정성 지표)"""
        if 'center_of_mass' not in landmarks:
            return 0.0
        
        # 프레임 버퍼에서 COM 추출
        if len(self.frame_buffer) < 10:
            return 0.0
        
        com_points = []
        for frame in self.frame_buffer[-10:]:
            if isinstance(frame, dict) and 'center_of_mass' in frame:
                com = frame['center_of_mass']
                com_points.append([com['x'], com['y']])
        
        if len(com_points) < 3:
            return 0.0
        
        # 95% confidence ellipse 면적
        com_points = np.array(com_points)
        covariance = np.cov(com_points.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        
        # Ellipse area = π * a * b
        area = np.pi * np.sqrt(eigenvalues[0]) * np.sqrt(eigenvalues[1])
        
        return area
    
    def calculate_temporal_features(self, 
                                   current_landmarks: Dict, 
                                   previous_landmarks: Optional[Dict]) -> Dict:
        """시간적 특징 계산 (속도, 가속도)"""
        if not previous_landmarks:
            return {
                'velocity': {},
                'acceleration': {},
                'jerk': {}
            }
        
        features = {
            'velocity': {},
            'acceleration': {},
            'angular_velocity': {}
        }
        
        # 주요 관절의 속도 계산
        for joint in ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle']:
            if joint in current_landmarks and joint in previous_landmarks:
                velocity = {
                    'x': current_landmarks[joint]['x'] - previous_landmarks[joint]['x'],
                    'y': current_landmarks[joint]['y'] - previous_landmarks[joint]['y'],
                    'z': current_landmarks[joint]['z'] - previous_landmarks[joint]['z']
                }
                
                # 속도 크기
                features['velocity'][joint] = np.linalg.norm([
                    velocity['x'], velocity['y'], velocity['z']
                ])
        
        # 관절 각속도
        current_angles = self.calculate_biomechanics(current_landmarks).get('angles', {})
        previous_angles = self.calculate_biomechanics(previous_landmarks).get('angles', {})
        
        for angle_name in current_angles:
            if angle_name in previous_angles:
                features['angular_velocity'][angle_name] = current_angles[angle_name] - previous_angles[angle_name]
        
        return features
    
    def estimate_quality_score(self, biomechanics: Dict, exercise_type: str) -> float:
        """운동 품질 점수 추정 (0-100)"""
        score = 100.0
        
        if exercise_type == 'squat':
            # 무릎 각도 체크
            knee_angles = biomechanics.get('angles', {})
            left_knee = knee_angles.get('left_knee', 90)
            right_knee = knee_angles.get('right_knee', 90)
            
            # 이상적인 스쿼트 깊이 (70-90도)
            if left_knee < 70 or left_knee > 110:
                score -= 10
            if right_knee < 70 or right_knee > 110:
                score -= 10
            
            # 좌우 대칭성
            if abs(left_knee - right_knee) > 10:
                score -= 5
            
            # 척추 정렬
            spine_angle = biomechanics.get('alignment', {}).get('spine', 0)
            if abs(spine_angle) > 20:
                score -= 15
            
            # 무릎 외반
            knee_valgus = biomechanics.get('alignment', {}).get('knee_valgus', {})
            if knee_valgus.get('left', 0) > 15 or knee_valgus.get('right', 0) > 15:
                score -= 10
                
        elif exercise_type == 'push_up':
            # 팔꿈치 각도
            elbow_angles = biomechanics.get('angles', {})
            left_elbow = elbow_angles.get('left_elbow', 90)
            right_elbow = elbow_angles.get('right_elbow', 90)
            
            # 좌우 대칭
            if abs(left_elbow - right_elbow) > 15:
                score -= 10
            
            # 몸통 정렬
            spine_angle = biomechanics.get('alignment', {}).get('spine', 0)
            if abs(spine_angle - 180) > 15:  # 플랭크 자세
                score -= 15
        
        return max(0, score)
    
    def save_dataset(self, output_path: str, format: str = 'json'):
        """수집한 데이터 저장"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.collected_data, f, indent=2)
        elif format == 'npy':
            np.save(output_path, self.collected_data)
        
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Total samples: {len(self.collected_data)}")
    
    def visualize_pose(self, frame: np.ndarray, landmarks: Dict) -> np.ndarray:
        """포즈 시각화"""
        # MediaPipe 연결선 정의
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle')
        ]
        
        h, w = frame.shape[:2]
        
        # 랜드마크 그리기
        for name, landmark in landmarks.items():
            x = int((landmark['x'] + 0.5) * w)  # 정규화 복원
            y = int((landmark['y'] + 0.5) * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # 연결선 그리기
        for connection in connections:
            if connection[0] in landmarks and connection[1] in landmarks:
                pt1 = landmarks[connection[0]]
                pt2 = landmarks[connection[1]]
                
                x1 = int((pt1['x'] + 0.5) * w)
                y1 = int((pt1['y'] + 0.5) * h)
                x2 = int((pt2['x'] + 0.5) * w)
                y2 = int((pt2['y'] + 0.5) * h)
                
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        return frame


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect expert exercise data')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--exercise', type=str, required=True, 
                       choices=['squat', 'push_up', 'deadlift', 'bench_press'],
                       help='Exercise type')
    parser.add_argument('--athlete', type=str, default='Unknown', help='Athlete name')
    parser.add_argument('--level', type=str, default='expert',
                       choices=['beginner', 'intermediate', 'expert', 'olympic'],
                       help='Athlete level')
    parser.add_argument('--output', type=str, default='dataset.json', help='Output file path')
    
    args = parser.parse_args()
    
    # 데이터 수집기 초기화
    collector = ExpertDataCollector()
    
    # 선수 정보
    athlete_info = {
        'name': args.athlete,
        'level': args.level,
        'timestamp': datetime.now().isoformat()
    }
    
    # 데이터 수집
    data = collector.collect_from_video(
        args.video,
        athlete_info,
        args.exercise
    )
    
    # 데이터 저장
    collector.save_dataset(args.output)
    
    print(f"Data collection complete! Saved {len(data)} frames to {args.output}")


if __name__ == "__main__":
    main()