"""
가상 운동 데이터 생성기
올림픽 수준의 운동 데이터를 시뮬레이션하여 생성
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
from pathlib import Path

class VirtualAthleteDataGenerator:
    def __init__(self):
        self.base_path = Path("data/virtual_athletes")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.exercises = {
            'push_up': {
                'keypoints': 17,
                'phases': ['ready', 'down', 'up'],
                'olympic_form_score': 95,
                'amateur_form_score': 65,
                'rep_duration': 2.0  # seconds
            },
            'squat': {
                'keypoints': 17,
                'phases': ['standing', 'down', 'up'],
                'olympic_form_score': 97,
                'amateur_form_score': 70,
                'rep_duration': 3.0
            },
            'deadlift': {
                'keypoints': 17,
                'phases': ['ready', 'lift', 'lock', 'lower'],
                'olympic_form_score': 96,
                'amateur_form_score': 60,
                'rep_duration': 4.0
            },
            'plank': {
                'keypoints': 17,
                'phases': ['holding'],
                'olympic_form_score': 98,
                'amateur_form_score': 75,
                'rep_duration': 60.0  # hold time
            }
        }
        
    def generate_keypoints(self, exercise_type, phase, athlete_level='olympic'):
        """운동 단계별 키포인트 생성"""
        config = self.exercises[exercise_type]
        
        # 기본 키포인트 위치
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        keypoints = []
        
        # 운동별, 단계별 키포인트 위치 계산
        if exercise_type == 'push_up':
            keypoints = self._generate_pushup_keypoints(phase, athlete_level)
        elif exercise_type == 'squat':
            keypoints = self._generate_squat_keypoints(phase, athlete_level)
        elif exercise_type == 'deadlift':
            keypoints = self._generate_deadlift_keypoints(phase, athlete_level)
        elif exercise_type == 'plank':
            keypoints = self._generate_plank_keypoints(phase, athlete_level)
            
        # 키포인트에 이름 추가
        for i, name in enumerate(keypoint_names):
            if i < len(keypoints):
                keypoints[i]['name'] = name
                
        return keypoints
    
    def _generate_pushup_keypoints(self, phase, level):
        """팔굽혀펴기 키포인트 생성"""
        noise = 0.01 if level == 'olympic' else 0.03
        
        if phase == 'ready':
            # 준비 자세
            base_positions = [
                {'x': 0.5, 'y': 0.2, 'confidence': 0.95},  # nose
                {'x': 0.48, 'y': 0.18, 'confidence': 0.93},  # left_eye
                {'x': 0.52, 'y': 0.18, 'confidence': 0.93},  # right_eye
                {'x': 0.45, 'y': 0.19, 'confidence': 0.90},  # left_ear
                {'x': 0.55, 'y': 0.19, 'confidence': 0.90},  # right_ear
                {'x': 0.35, 'y': 0.35, 'confidence': 0.95},  # left_shoulder
                {'x': 0.65, 'y': 0.35, 'confidence': 0.95},  # right_shoulder
                {'x': 0.25, 'y': 0.45, 'confidence': 0.92},  # left_elbow
                {'x': 0.75, 'y': 0.45, 'confidence': 0.92},  # right_elbow
                {'x': 0.20, 'y': 0.55, 'confidence': 0.90},  # left_wrist
                {'x': 0.80, 'y': 0.55, 'confidence': 0.90},  # right_wrist
                {'x': 0.40, 'y': 0.60, 'confidence': 0.93},  # left_hip
                {'x': 0.60, 'y': 0.60, 'confidence': 0.93},  # right_hip
                {'x': 0.38, 'y': 0.75, 'confidence': 0.91},  # left_knee
                {'x': 0.62, 'y': 0.75, 'confidence': 0.91},  # right_knee
                {'x': 0.37, 'y': 0.90, 'confidence': 0.88},  # left_ankle
                {'x': 0.63, 'y': 0.90, 'confidence': 0.88},  # right_ankle
            ]
        elif phase == 'down':
            # 내려간 자세
            base_positions = [
                {'x': 0.5, 'y': 0.45, 'confidence': 0.94},
                {'x': 0.48, 'y': 0.43, 'confidence': 0.92},
                {'x': 0.52, 'y': 0.43, 'confidence': 0.92},
                {'x': 0.45, 'y': 0.44, 'confidence': 0.89},
                {'x': 0.55, 'y': 0.44, 'confidence': 0.89},
                {'x': 0.35, 'y': 0.50, 'confidence': 0.94},
                {'x': 0.65, 'y': 0.50, 'confidence': 0.94},
                {'x': 0.20, 'y': 0.52, 'confidence': 0.91},
                {'x': 0.80, 'y': 0.52, 'confidence': 0.91},
                {'x': 0.15, 'y': 0.58, 'confidence': 0.89},
                {'x': 0.85, 'y': 0.58, 'confidence': 0.89},
                {'x': 0.40, 'y': 0.62, 'confidence': 0.92},
                {'x': 0.60, 'y': 0.62, 'confidence': 0.92},
                {'x': 0.38, 'y': 0.76, 'confidence': 0.90},
                {'x': 0.62, 'y': 0.76, 'confidence': 0.90},
                {'x': 0.37, 'y': 0.90, 'confidence': 0.87},
                {'x': 0.63, 'y': 0.90, 'confidence': 0.87},
            ]
        else:  # up
            # 올라간 자세 (ready와 유사)
            base_positions = self._generate_pushup_keypoints('ready', level)
            
        # 노이즈 추가
        for kp in base_positions:
            kp['x'] += np.random.normal(0, noise)
            kp['y'] += np.random.normal(0, noise)
            kp['confidence'] *= (1 - np.random.uniform(0, 0.05))
            
        return base_positions
    
    def _generate_squat_keypoints(self, phase, level):
        """스쿼트 키포인트 생성"""
        noise = 0.01 if level == 'olympic' else 0.03
        
        if phase == 'standing':
            # 서있는 자세
            base_positions = [
                {'x': 0.5, 'y': 0.1, 'confidence': 0.96},
                {'x': 0.48, 'y': 0.08, 'confidence': 0.94},
                {'x': 0.52, 'y': 0.08, 'confidence': 0.94},
                {'x': 0.45, 'y': 0.09, 'confidence': 0.91},
                {'x': 0.55, 'y': 0.09, 'confidence': 0.91},
                {'x': 0.40, 'y': 0.25, 'confidence': 0.95},
                {'x': 0.60, 'y': 0.25, 'confidence': 0.95},
                {'x': 0.38, 'y': 0.40, 'confidence': 0.93},
                {'x': 0.62, 'y': 0.40, 'confidence': 0.93},
                {'x': 0.37, 'y': 0.50, 'confidence': 0.91},
                {'x': 0.63, 'y': 0.50, 'confidence': 0.91},
                {'x': 0.42, 'y': 0.50, 'confidence': 0.94},
                {'x': 0.58, 'y': 0.50, 'confidence': 0.94},
                {'x': 0.41, 'y': 0.70, 'confidence': 0.92},
                {'x': 0.59, 'y': 0.70, 'confidence': 0.92},
                {'x': 0.40, 'y': 0.90, 'confidence': 0.89},
                {'x': 0.60, 'y': 0.90, 'confidence': 0.89},
            ]
        elif phase == 'down':
            # 앉은 자세
            base_positions = [
                {'x': 0.5, 'y': 0.3, 'confidence': 0.95},
                {'x': 0.48, 'y': 0.28, 'confidence': 0.93},
                {'x': 0.52, 'y': 0.28, 'confidence': 0.93},
                {'x': 0.45, 'y': 0.29, 'confidence': 0.90},
                {'x': 0.55, 'y': 0.29, 'confidence': 0.90},
                {'x': 0.40, 'y': 0.40, 'confidence': 0.94},
                {'x': 0.60, 'y': 0.40, 'confidence': 0.94},
                {'x': 0.35, 'y': 0.48, 'confidence': 0.92},
                {'x': 0.65, 'y': 0.48, 'confidence': 0.92},
                {'x': 0.33, 'y': 0.55, 'confidence': 0.90},
                {'x': 0.67, 'y': 0.55, 'confidence': 0.90},
                {'x': 0.42, 'y': 0.60, 'confidence': 0.93},
                {'x': 0.58, 'y': 0.60, 'confidence': 0.93},
                {'x': 0.38, 'y': 0.65, 'confidence': 0.91},
                {'x': 0.62, 'y': 0.65, 'confidence': 0.91},
                {'x': 0.40, 'y': 0.90, 'confidence': 0.88},
                {'x': 0.60, 'y': 0.90, 'confidence': 0.88},
            ]
        else:  # up
            base_positions = self._generate_squat_keypoints('standing', level)
            
        # 노이즈 추가
        for kp in base_positions:
            kp['x'] += np.random.normal(0, noise)
            kp['y'] += np.random.normal(0, noise)
            kp['confidence'] *= (1 - np.random.uniform(0, 0.05))
            
        return base_positions
    
    def _generate_deadlift_keypoints(self, phase, level):
        """데드리프트 키포인트 생성"""
        # 스쿼트와 유사하지만 상체 각도가 다름
        base_positions = self._generate_squat_keypoints('standing' if phase in ['ready', 'lock'] else 'down', level)
        
        # 데드리프트 특성 반영 (상체 기울기)
        if phase in ['lift', 'lower']:
            for i in range(6):  # 상체 키포인트
                base_positions[i]['y'] += 0.1
                
        return base_positions
    
    def _generate_plank_keypoints(self, phase, level):
        """플랭크 키포인트 생성"""
        noise = 0.005 if level == 'olympic' else 0.02
        
        # 플랭크 홀드 자세
        base_positions = [
            {'x': 0.5, 'y': 0.4, 'confidence': 0.96},
            {'x': 0.48, 'y': 0.38, 'confidence': 0.94},
            {'x': 0.52, 'y': 0.38, 'confidence': 0.94},
            {'x': 0.45, 'y': 0.39, 'confidence': 0.91},
            {'x': 0.55, 'y': 0.39, 'confidence': 0.91},
            {'x': 0.40, 'y': 0.45, 'confidence': 0.95},
            {'x': 0.60, 'y': 0.45, 'confidence': 0.95},
            {'x': 0.35, 'y': 0.50, 'confidence': 0.93},
            {'x': 0.65, 'y': 0.50, 'confidence': 0.93},
            {'x': 0.30, 'y': 0.55, 'confidence': 0.91},
            {'x': 0.70, 'y': 0.55, 'confidence': 0.91},
            {'x': 0.42, 'y': 0.50, 'confidence': 0.94},
            {'x': 0.58, 'y': 0.50, 'confidence': 0.94},
            {'x': 0.41, 'y': 0.65, 'confidence': 0.92},
            {'x': 0.59, 'y': 0.65, 'confidence': 0.92},
            {'x': 0.40, 'y': 0.80, 'confidence': 0.89},
            {'x': 0.60, 'y': 0.80, 'confidence': 0.89},
        ]
        
        # 노이즈 추가 (플랭크는 흔들림이 적음)
        for kp in base_positions:
            kp['x'] += np.random.normal(0, noise)
            kp['y'] += np.random.normal(0, noise)
            kp['confidence'] *= (1 - np.random.uniform(0, 0.03))
            
        return base_positions
    
    def generate_workout_session(self, exercise_type, athlete_name, athlete_level='olympic', num_reps=20):
        """완전한 운동 세션 데이터 생성"""
        config = self.exercises[exercise_type]
        session_data = {
            'athlete_name': athlete_name,
            'athlete_level': athlete_level,
            'exercise_type': exercise_type,
            'timestamp': datetime.now().isoformat(),
            'total_reps': num_reps,
            'frames': []
        }
        
        # 각 반복 동작 생성
        for rep in range(num_reps):
            # 각 phase별로 프레임 생성
            for phase in config['phases']:
                # 여러 프레임 생성 (phase당 5-10 프레임)
                num_frames = random.randint(5, 10)
                for frame_idx in range(num_frames):
                    frame_data = {
                        'rep_number': rep + 1,
                        'phase': phase,
                        'frame_number': len(session_data['frames']),
                        'timestamp': (datetime.now() + timedelta(milliseconds=len(session_data['frames']) * 100)).isoformat(),
                        'keypoints': self.generate_keypoints(exercise_type, phase, athlete_level),
                        'form_score': self._calculate_form_score(athlete_level, config)
                    }
                    session_data['frames'].append(frame_data)
                    
        return session_data
    
    def _calculate_form_score(self, level, config):
        """자세 점수 계산"""
        if level == 'olympic':
            base_score = config['olympic_form_score']
            variation = 3
        else:
            base_score = config['amateur_form_score']
            variation = 10
            
        return max(0, min(100, base_score + np.random.normal(0, variation)))
    
    def generate_dataset(self, num_athletes=10):
        """전체 데이터셋 생성"""
        print("가상 운동 데이터 생성 시작...")
        
        athletes = []
        
        # 올림픽 선수 데이터
        for i in range(num_athletes // 2):
            athlete_name = f"Olympic_Athlete_{i+1}"
            print(f"  {athlete_name} 데이터 생성 중...")
            
            for exercise_type in self.exercises.keys():
                session = self.generate_workout_session(
                    exercise_type=exercise_type,
                    athlete_name=athlete_name,
                    athlete_level='olympic',
                    num_reps=random.randint(15, 25)
                )
                
                # 파일로 저장
                filename = self.base_path / f"{athlete_name}_{exercise_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(session, f, indent=2, ensure_ascii=False)
                    
                athletes.append({
                    'name': athlete_name,
                    'level': 'olympic',
                    'exercise': exercise_type,
                    'file': str(filename)
                })
        
        # 아마추어 선수 데이터
        for i in range(num_athletes // 2):
            athlete_name = f"Amateur_Athlete_{i+1}"
            print(f"  {athlete_name} 데이터 생성 중...")
            
            for exercise_type in self.exercises.keys():
                session = self.generate_workout_session(
                    exercise_type=exercise_type,
                    athlete_name=athlete_name,
                    athlete_level='amateur',
                    num_reps=random.randint(10, 20)
                )
                
                # 파일로 저장
                filename = self.base_path / f"{athlete_name}_{exercise_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(session, f, indent=2, ensure_ascii=False)
                    
                athletes.append({
                    'name': athlete_name,
                    'level': 'amateur',
                    'exercise': exercise_type,
                    'file': str(filename)
                })
        
        # 메타데이터 저장
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'total_athletes': num_athletes,
            'total_sessions': len(athletes),
            'athletes': athletes
        }
        
        with open(self.base_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        print(f"총 {len(athletes)}개 세션 데이터 생성 완료!")
        return metadata

if __name__ == "__main__":
    generator = VirtualAthleteDataGenerator()
    metadata = generator.generate_dataset(num_athletes=20)
    print(f"데이터 생성 완료: {metadata['total_sessions']}개 세션")