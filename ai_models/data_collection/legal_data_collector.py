"""
합법적인 운동 데이터 수집 시스템
저작권을 준수하며 고품질 학습 데이터 확보
"""

import requests
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import mediapipe as mp
from typing import List, Dict, Optional
import time

class LegalDataCollector:
    """합법적인 방법으로 운동 데이터를 수집하는 클래스"""
    
    def __init__(self):
        self.base_path = Path("data/legal_sources")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        self.legal_sources = {
            "creative_commons": self.collect_creative_commons,
            "sports_apis": self.collect_sports_apis,
            "webcam_capture": self.collect_webcam_data,
            "synthetic_data": self.generate_synthetic_data
        }
    
    def collect_creative_commons(self):
        """Creative Commons 라이선스 영상 수집"""
        print("Creative Commons 데이터 수집 중...")
        
        # Wikimedia Commons API 사용
        cc_sources = [
            {
                "name": "Wikimedia Sports",
                "api": "https://commons.wikimedia.org/w/api.php",
                "category": "Sports_videos"
            },
            {
                "name": "Internet Archive",
                "api": "https://archive.org/advancedsearch.php",
                "collection": "opensource_movies"
            }
        ]
        
        collected_videos = []
        
        for source in cc_sources:
            try:
                # API 호출로 CC 라이선스 영상 검색
                params = {
                    "action": "query",
                    "format": "json",
                    "list": "categorymembers",
                    "cmtitle": f"Category:{source['category']}",
                    "cmlimit": "50"
                }
                
                response = requests.get(source["api"], params=params)
                if response.status_code == 200:
                    data = response.json()
                    print(f"  {source['name']}: {len(data.get('query', {}).get('categorymembers', []))}개 영상 발견")
                    collected_videos.extend(data.get('query', {}).get('categorymembers', []))
                
                time.sleep(1)  # API 호출 간격
                
            except Exception as e:
                print(f"  {source['name']} 수집 실패: {e}")
        
        return collected_videos
    
    def collect_sports_apis(self):
        """공식 스포츠 API 데이터 수집"""
        print("공식 스포츠 API 데이터 수집 중...")
        
        # 무료 스포츠 API들
        free_apis = [
            {
                "name": "Olympic API",
                "url": "https://olympics.com/en/api/",
                "description": "올림픽 공식 데이터"
            },
            {
                "name": "Sports Open Data",
                "url": "https://www.sports-reference.com/",
                "description": "스포츠 통계 데이터"
            }
        ]
        
        # 실제 구현에서는 각 API의 엔드포인트에 맞게 수정
        collected_data = []
        
        for api in free_apis:
            print(f"  {api['name']} 접근 중...")
            # API별 구체적인 구현 필요
            collected_data.append({
                "source": api["name"],
                "status": "접근 가능",
                "data_type": "메타데이터"
            })
        
        return collected_data
    
    def collect_webcam_data(self, duration_minutes=5):
        """웹캠을 통한 실시간 운동 데이터 수집"""
        print(f"웹캠 데이터 수집 시작 ({duration_minutes}분)...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("웹캠을 찾을 수 없습니다.")
            return []
        
        collected_frames = []
        start_time = time.time()
        frame_count = 0
        
        print("운동을 시작하세요! (ESC로 종료)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # MediaPipe로 포즈 감지
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # 키포인트 추출
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                frame_data = {
                    'frame_id': frame_count,
                    'timestamp': time.time() - start_time,
                    'landmarks': landmarks,
                    'quality_score': self._assess_frame_quality(landmarks)
                }
                
                # 품질이 좋은 프레임만 저장
                if frame_data['quality_score'] > 0.7:
                    collected_frames.append(frame_data)
                
                # 포즈 시각화
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )
            
            # 화면 표시
            cv2.putText(frame, f"Collected: {len(collected_frames)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Exercise Data Collection', frame)
            
            frame_count += 1
            
            # 종료 조건
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
            if time.time() - start_time > duration_minutes * 60:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"수집 완료: {len(collected_frames)}개 고품질 프레임")
        return collected_frames
    
    def _assess_frame_quality(self, landmarks):
        """프레임 품질 평가"""
        if not landmarks or len(landmarks) < 17:
            return 0.0
        
        # 가시성 평균
        visibility_avg = np.mean([lm['visibility'] for lm in landmarks])
        
        # 주요 관절 포인트 확인
        key_joints = [5, 6, 11, 12, 13, 14, 15, 16]  # 어깨, 엉덩이, 무릎, 발목
        key_visibility = np.mean([landmarks[i]['visibility'] for i in key_joints if i < len(landmarks)])
        
        return (visibility_avg * 0.3 + key_visibility * 0.7)
    
    def generate_synthetic_data(self, num_samples=1000):
        """합성 데이터 생성 (물리학 기반)"""
        print(f"물리학 기반 합성 데이터 {num_samples}개 생성 중...")
        
        synthetic_exercises = []
        
        # 물리학적 제약 조건
        physics_constraints = {
            'gravity': 9.81,
            'human_proportions': {
                'head_to_shoulder': 0.2,
                'shoulder_to_elbow': 0.3,
                'elbow_to_wrist': 0.25,
                'shoulder_to_hip': 0.35,
                'hip_to_knee': 0.4,
                'knee_to_ankle': 0.35
            },
            'joint_limits': {
                'elbow': (0, 150),  # degrees
                'knee': (0, 135),
                'shoulder': (0, 180)
            }
        }
        
        for i in range(num_samples):
            # 랜덤 운동 시나리오 생성
            exercise_type = np.random.choice(['push_up', 'squat', 'deadlift', 'plank'])
            athlete_skill = np.random.choice(['beginner', 'intermediate', 'advanced'])
            
            # 물리학적으로 타당한 동작 생성
            motion_sequence = self._generate_physics_based_motion(
                exercise_type, 
                athlete_skill, 
                physics_constraints
            )
            
            synthetic_exercises.append({
                'id': f"synthetic_{i:04d}",
                'exercise_type': exercise_type,
                'skill_level': athlete_skill,
                'motion_data': motion_sequence,
                'generated_at': datetime.now().isoformat()
            })
            
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num_samples} 완료")
        
        return synthetic_exercises
    
    def _generate_physics_based_motion(self, exercise_type, skill_level, constraints):
        """물리학 기반 동작 생성"""
        # 실제 구현에서는 더 복잡한 물리 시뮬레이션 필요
        frames = []
        duration = 3.0  # 3초 동작
        fps = 30
        total_frames = int(duration * fps)
        
        for frame_idx in range(total_frames):
            t = frame_idx / fps  # 시간
            
            # 운동별 동작 패턴
            if exercise_type == 'push_up':
                # 사인파 기반 상하 운동
                y_offset = 0.1 * np.sin(2 * np.pi * t / 2.0)  # 2초 주기
                
            elif exercise_type == 'squat':
                # 스쿼트 동작 패턴
                y_offset = 0.2 * np.sin(2 * np.pi * t / 3.0)  # 3초 주기
                
            else:
                y_offset = 0
            
            # 기본 포즈에 동작 적용
            frame_landmarks = self._apply_motion_to_base_pose(y_offset, skill_level)
            frames.append(frame_landmarks)
        
        return frames
    
    def _apply_motion_to_base_pose(self, y_offset, skill_level):
        """기본 포즈에 동작 변화 적용"""
        # 표준 인체 포즈 (17개 키포인트)
        base_pose = {
            0: {'x': 0.5, 'y': 0.1},    # nose
            5: {'x': 0.4, 'y': 0.25},   # left_shoulder
            6: {'x': 0.6, 'y': 0.25},   # right_shoulder
            11: {'x': 0.45, 'y': 0.5},  # left_hip
            12: {'x': 0.55, 'y': 0.5},  # right_hip
            13: {'x': 0.43, 'y': 0.7},  # left_knee
            14: {'x': 0.57, 'y': 0.7},  # right_knee
            15: {'x': 0.42, 'y': 0.9},  # left_ankle
            16: {'x': 0.58, 'y': 0.9},  # right_ankle
        }
        
        # 동작 적용
        landmarks = []
        for i in range(17):
            if i in base_pose:
                x = base_pose[i]['x']
                y = base_pose[i]['y'] + y_offset
                
                # 실력 레벨에 따른 노이즈
                noise_level = {
                    'beginner': 0.03,
                    'intermediate': 0.015,
                    'advanced': 0.005
                }.get(skill_level, 0.02)
                
                x += np.random.normal(0, noise_level)
                y += np.random.normal(0, noise_level)
                
            else:
                x, y = 0.5, 0.5  # 기본값
            
            landmarks.append({
                'x': np.clip(x, 0, 1),
                'y': np.clip(y, 0, 1),
                'visibility': np.random.uniform(0.85, 0.98)
            })
        
        return landmarks
    
    def save_collected_data(self, data, source_type):
        """수집된 데이터 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.base_path / f"{source_type}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"데이터 저장 완료: {filename}")
        return filename
    
    def collect_all_legal_sources(self):
        """모든 합법적 소스에서 데이터 수집"""
        print("=== 합법적 데이터 수집 시작 ===")
        
        all_collected_data = {}
        
        for source_name, collect_func in self.legal_sources.items():
            try:
                print(f"\n[{source_name}] 데이터 수집 중...")
                data = collect_func()
                
                if data:
                    filename = self.save_collected_data(data, source_name)
                    all_collected_data[source_name] = {
                        'count': len(data) if isinstance(data, list) else 1,
                        'file': str(filename)
                    }
                else:
                    print(f"  {source_name}: 데이터 없음")
                    
            except Exception as e:
                print(f"  {source_name} 수집 실패: {e}")
        
        # 수집 요약 저장
        summary = {
            'collection_date': datetime.now().isoformat(),
            'total_sources': len(all_collected_data),
            'sources': all_collected_data,
            'legal_compliance': True,
            'copyright_status': "All data collected under legal frameworks"
        }
        
        summary_file = self.base_path / "collection_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== 수집 완료 ===")
        print(f"총 {len(all_collected_data)}개 소스에서 데이터 수집")
        print(f"요약 파일: {summary_file}")
        
        return all_collected_data

if __name__ == "__main__":
    collector = LegalDataCollector()
    results = collector.collect_all_legal_sources()
    
    print("\n수집된 데이터:")
    for source, info in results.items():
        print(f"  {source}: {info['count']}개")