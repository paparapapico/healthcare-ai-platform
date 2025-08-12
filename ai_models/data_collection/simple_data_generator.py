"""
간단하고 빠른 데이터 생성기
즉시 실행 가능한 대량 운동 데이터 생성
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import random

class SimpleDataGenerator:
    def __init__(self):
        self.base_path = Path("data/simple_generated")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.exercises = {
            'push_up': {'phases': ['up', 'down'], 'duration': 2.0},
            'squat': {'phases': ['up', 'down'], 'duration': 3.0},
            'deadlift': {'phases': ['up', 'down'], 'duration': 4.0},
            'plank': {'phases': ['hold'], 'duration': 30.0}
        }
        
    def generate_keypoints(self, exercise, phase, skill_level):
        """17개 키포인트 생성 (COCO 포맷)"""
        
        # 기본 포즈 템플릿
        if exercise == 'push_up':
            if phase == 'up':
                base_points = [
                    [0.5, 0.2], [0.48, 0.18], [0.52, 0.18], [0.45, 0.19], [0.55, 0.19],
                    [0.35, 0.35], [0.65, 0.35], [0.25, 0.45], [0.75, 0.45], 
                    [0.20, 0.55], [0.80, 0.55], [0.40, 0.60], [0.60, 0.60],
                    [0.38, 0.75], [0.62, 0.75], [0.37, 0.90], [0.63, 0.90]
                ]
            else:  # down
                base_points = [
                    [0.5, 0.45], [0.48, 0.43], [0.52, 0.43], [0.45, 0.44], [0.55, 0.44],
                    [0.35, 0.50], [0.65, 0.50], [0.20, 0.52], [0.80, 0.52],
                    [0.15, 0.58], [0.85, 0.58], [0.40, 0.62], [0.60, 0.62],
                    [0.38, 0.76], [0.62, 0.76], [0.37, 0.90], [0.63, 0.90]
                ]
                
        elif exercise == 'squat':
            if phase == 'up':
                base_points = [
                    [0.5, 0.1], [0.48, 0.08], [0.52, 0.08], [0.45, 0.09], [0.55, 0.09],
                    [0.40, 0.25], [0.60, 0.25], [0.38, 0.40], [0.62, 0.40],
                    [0.37, 0.50], [0.63, 0.50], [0.42, 0.50], [0.58, 0.50],
                    [0.41, 0.70], [0.59, 0.70], [0.40, 0.90], [0.60, 0.90]
                ]
            else:  # down
                base_points = [
                    [0.5, 0.3], [0.48, 0.28], [0.52, 0.28], [0.45, 0.29], [0.55, 0.29],
                    [0.40, 0.40], [0.60, 0.40], [0.35, 0.48], [0.65, 0.48],
                    [0.33, 0.55], [0.67, 0.55], [0.42, 0.60], [0.58, 0.60],
                    [0.38, 0.65], [0.62, 0.65], [0.40, 0.90], [0.60, 0.90]
                ]
                
        elif exercise == 'deadlift':
            if phase == 'up':
                base_points = [
                    [0.5, 0.1], [0.48, 0.08], [0.52, 0.08], [0.45, 0.09], [0.55, 0.09],
                    [0.40, 0.25], [0.60, 0.25], [0.38, 0.35], [0.62, 0.35],
                    [0.36, 0.45], [0.64, 0.45], [0.42, 0.50], [0.58, 0.50],
                    [0.41, 0.70], [0.59, 0.70], [0.40, 0.90], [0.60, 0.90]
                ]
            else:  # down
                base_points = [
                    [0.5, 0.25], [0.48, 0.23], [0.52, 0.23], [0.45, 0.24], [0.55, 0.24],
                    [0.40, 0.35], [0.60, 0.35], [0.38, 0.45], [0.62, 0.45],
                    [0.36, 0.55], [0.64, 0.55], [0.42, 0.65], [0.58, 0.65],
                    [0.41, 0.75], [0.59, 0.75], [0.40, 0.90], [0.60, 0.90]
                ]
                
        else:  # plank
            base_points = [
                [0.5, 0.4], [0.48, 0.38], [0.52, 0.38], [0.45, 0.39], [0.55, 0.39],
                [0.40, 0.45], [0.60, 0.45], [0.35, 0.50], [0.65, 0.50],
                [0.30, 0.55], [0.70, 0.55], [0.42, 0.50], [0.58, 0.50],
                [0.41, 0.65], [0.59, 0.65], [0.40, 0.80], [0.60, 0.80]
            ]
        
        # 실력별 노이즈 추가
        noise_levels = {'beginner': 0.03, 'intermediate': 0.015, 'advanced': 0.005}
        noise = noise_levels.get(skill_level, 0.02)
        
        keypoints = []
        for i, (x, y) in enumerate(base_points):
            # 노이즈 추가
            x_noise = x + np.random.normal(0, noise)
            y_noise = y + np.random.normal(0, noise)
            
            # 범위 제한
            x_final = max(0, min(1, x_noise))
            y_final = max(0, min(1, y_noise))
            
            # 가시성 점수
            visibility = np.random.uniform(0.85, 0.98)
            
            keypoints.append({
                'x': x_final,
                'y': y_final,
                'visibility': visibility
            })
        
        return keypoints
    
    def calculate_form_score(self, exercise, skill_level):
        """폼 점수 계산"""
        base_scores = {
            'beginner': {'push_up': 65, 'squat': 70, 'deadlift': 60, 'plank': 75},
            'intermediate': {'push_up': 80, 'squat': 82, 'deadlift': 78, 'plank': 85},
            'advanced': {'push_up': 92, 'squat': 95, 'deadlift': 90, 'plank': 96}
        }
        
        base_score = base_scores[skill_level][exercise]
        variation = np.random.normal(0, 5)
        
        return max(0, min(100, base_score + variation))
    
    def generate_workout_session(self, exercise, athlete_name, skill_level, num_reps=15):
        """운동 세션 생성"""
        session = {
            'athlete_name': athlete_name,
            'skill_level': skill_level,
            'exercise_type': exercise,
            'total_reps': num_reps,
            'timestamp': datetime.now().isoformat(),
            'frames': []
        }
        
        config = self.exercises[exercise]
        
        for rep in range(num_reps):
            for phase in config['phases']:
                # 각 페이즈당 5-8 프레임
                frames_per_phase = random.randint(5, 8)
                
                for frame_idx in range(frames_per_phase):
                    keypoints = self.generate_keypoints(exercise, phase, skill_level)
                    form_score = self.calculate_form_score(exercise, skill_level)
                    
                    frame_data = {
                        'rep_number': rep + 1,
                        'phase': phase,
                        'frame_number': len(session['frames']),
                        'keypoints': keypoints,
                        'form_score': form_score,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    session['frames'].append(frame_data)
        
        return session
    
    def generate_large_dataset(self, total_samples=5000):
        """대량 데이터셋 생성"""
        print(f"대량 데이터셋 생성 시작 (목표: {total_samples}개 세션)")
        
        all_sessions = []
        
        # 운동별, 실력별 균등 분배
        exercises = list(self.exercises.keys())
        skill_levels = ['beginner', 'intermediate', 'advanced']
        
        sessions_per_combination = total_samples // (len(exercises) * len(skill_levels))
        
        session_count = 0
        
        for exercise in exercises:
            for skill in skill_levels:
                print(f"  {exercise} - {skill} 레벨 생성 중...")
                
                for i in range(sessions_per_combination):
                    athlete_name = f"{skill.capitalize()}_{exercise}_{i+1:04d}"
                    num_reps = random.randint(10, 25)
                    
                    session = self.generate_workout_session(
                        exercise, athlete_name, skill, num_reps
                    )
                    
                    all_sessions.append(session)
                    session_count += 1
                    
                    if session_count % 100 == 0:
                        progress = (session_count / total_samples) * 100
                        print(f"    진행률: {session_count}/{total_samples} ({progress:.1f}%)")
        
        print(f"총 {len(all_sessions)}개 세션 생성 완료!")
        return all_sessions
    
    def save_dataset(self, sessions, filename="large_dataset"):
        """데이터셋 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 메타데이터 생성
        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_sessions': len(sessions),
            'total_frames': sum(len(s['frames']) for s in sessions),
            'exercises': list(self.exercises.keys()),
            'skill_levels': ['beginner', 'intermediate', 'advanced']
        }
        
        # 분할 저장 (너무 크면 여러 파일로)
        max_sessions_per_file = 1000
        
        if len(sessions) > max_sessions_per_file:
            # 여러 파일로 분할
            for i in range(0, len(sessions), max_sessions_per_file):
                chunk = sessions[i:i+max_sessions_per_file]
                chunk_filename = f"{filename}_part{i//max_sessions_per_file + 1}_{timestamp}.json"
                filepath = self.base_path / chunk_filename
                
                data = {
                    'metadata': metadata,
                    'part_info': {
                        'part_number': i//max_sessions_per_file + 1,
                        'total_parts': (len(sessions) + max_sessions_per_file - 1) // max_sessions_per_file,
                        'sessions_in_part': len(chunk)
                    },
                    'sessions': chunk
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"파일 저장: {filepath}")
        else:
            # 단일 파일
            filepath = self.base_path / f"{filename}_{timestamp}.json"
            data = {
                'metadata': metadata,
                'sessions': sessions
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"파일 저장: {filepath}")
        
        # 요약 정보 저장
        summary = {
            'generation_summary': {
                'timestamp': timestamp,
                'total_sessions': len(sessions),
                'total_frames': sum(len(s['frames']) for s in sessions),
                'avg_frames_per_session': sum(len(s['frames']) for s in sessions) / len(sessions),
                'exercises_distribution': {},
                'skill_distribution': {}
            }
        }
        
        # 분포 계산
        for session in sessions:
            exercise = session['exercise_type']
            skill = session['skill_level']
            
            summary['generation_summary']['exercises_distribution'][exercise] = \
                summary['generation_summary']['exercises_distribution'].get(exercise, 0) + 1
            summary['generation_summary']['skill_distribution'][skill] = \
                summary['generation_summary']['skill_distribution'].get(skill, 0) + 1
        
        summary_file = self.base_path / f"generation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"요약 저장: {summary_file}")
        return summary
    
    def quick_generate(self, size=2000):
        """빠른 데이터 생성"""
        print("=" * 50)
        print("빠른 고품질 운동 데이터 생성기")
        print("=" * 50)
        
        # 데이터 생성
        sessions = self.generate_large_dataset(size)
        
        # 저장
        summary = self.save_dataset(sessions, "quick_generated")
        
        # 결과 출력
        print("=" * 50)
        print("생성 완료 결과:")
        print(f"  총 세션 수: {summary['generation_summary']['total_sessions']}")
        print(f"  총 프레임 수: {summary['generation_summary']['total_frames']}")
        print(f"  평균 프레임/세션: {summary['generation_summary']['avg_frames_per_session']:.1f}")
        print("  운동별 분포:")
        for exercise, count in summary['generation_summary']['exercises_distribution'].items():
            print(f"    {exercise}: {count}개")
        print("  실력별 분포:")
        for skill, count in summary['generation_summary']['skill_distribution'].items():
            print(f"    {skill}: {count}개")
        print("=" * 50)
        
        return sessions, summary

if __name__ == "__main__":
    generator = SimpleDataGenerator()
    sessions, summary = generator.quick_generate(3000)  # 3천개 세션 생성