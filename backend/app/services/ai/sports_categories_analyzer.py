# backend/app/services/ai/sports_categories_analyzer.py
"""
스포츠 종목별 전문가 수준 AI 분석 시스템
Sports Categories Expert-Level AI Analysis System
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum
import cv2

class SportCategory(Enum):
    """스포츠 카테고리 정의"""
    BODYWEIGHT = "bodyweight"  # 맨몸운동
    FITNESS = "fitness"        # 헬스
    GOLF = "golf"             # 골프
    SOCCER = "soccer"         # 축구
    BASKETBALL = "basketball"  # 농구
    BADMINTON = "badminton"   # 배드민턴
    TENNIS = "tennis"         # 테니스
    SWIMMING = "swimming"     # 수영
    BASEBALL = "baseball"     # 야구
    VOLLEYBALL = "volleyball" # 배구

class BaseExerciseAnalyzer(ABC):
    """기본 운동 분석기 추상 클래스"""
    
    def __init__(self, category: SportCategory):
        self.category = category
        self.exercise_standards = self._load_professional_standards()
        self.common_mistakes = self._load_common_mistakes()
        self.pro_techniques = self._load_pro_techniques()
    
    @abstractmethod
    def _load_professional_standards(self) -> Dict:
        """프로 선수급 기준 로드"""
        pass
    
    @abstractmethod
    def _load_common_mistakes(self) -> Dict:
        """일반적인 실수 패턴 로드"""
        pass
    
    @abstractmethod
    def _load_pro_techniques(self) -> Dict:
        """프로 선수 기법 로드"""
        pass
    
    @abstractmethod
    def analyze_movement(self, keypoints: List[Dict], video_frame: np.ndarray) -> Dict[str, Any]:
        """움직임 분석"""
        pass

class BodyweightAnalyzer(BaseExerciseAnalyzer):
    """맨몸운동 전문 분석기"""
    
    def __init__(self):
        super().__init__(SportCategory.BODYWEIGHT)
    
    def _load_professional_standards(self) -> Dict:
        return {
            'push_up': {
                'elite_form': {
                    'elbow_angle_bottom': (85, 95),
                    'body_alignment': (175, 180),
                    'descent_speed': (1.5, 2.5),  # 초
                    'hold_time_bottom': (0.5, 1.0),
                    'shoulder_stability': 95  # 백분율
                },
                'competition_standards': {
                    'minimum_depth': 90,  # 팔꿈치 각도
                    'perfect_alignment': True,
                    'controlled_movement': True
                }
            },
            'squat': {
                'elite_form': {
                    'knee_angle_bottom': (70, 90),
                    'hip_depth': 'below_knee',
                    'spine_neutrality': (85, 95),
                    'foot_position': 'shoulder_width',
                    'knee_tracking': 'over_toes'
                },
                'powerlifting_standards': {
                    'depth_requirement': 'hip_crease_below_knee',
                    'heel_contact': True,
                    'knee_lockout': True
                }
            },
            'plank': {
                'elite_form': {
                    'body_line': (178, 182),  # 거의 완전한 직선
                    'hold_duration': 300,      # 5분 (엘리트 기준)
                    'minimal_sway': 2,         # cm 이내
                    'breathing_control': True
                }
            }
        }
    
    def _load_common_mistakes(self) -> Dict:
        return {
            'push_up': [
                'sagging_hips',
                'incomplete_range',
                'elbow_flaring',
                'neck_strain',
                'hand_position'
            ],
            'squat': [
                'knee_valgus',
                'forward_lean',
                'heel_rise',
                'partial_depth',
                'knee_past_toe'
            ],
            'plank': [
                'hip_sag',
                'hip_pike',
                'shoulder_collapse',
                'head_drop',
                'breathing_hold'
            ]
        }
    
    def _load_pro_techniques(self) -> Dict:
        return {
            'push_up': {
                'military_standard': '완벽한 폼으로 연속 100회',
                'archer_pushup': '단일 팔 집중 고난이도 기법',
                'explosive_pushup': '파워와 속도 개발 기법'
            },
            'squat': {
                'olympic_squat': '올림픽 리프팅 표준 스쿼트',
                'pistol_squat': '단일 다리 스쿼트 (체조 선수급)',
                'jump_squat': '폭발력 개발 스쿼트'
            }
        }
    
    def analyze_movement(self, keypoints: List[Dict], video_frame: np.ndarray) -> Dict[str, Any]:
        """맨몸운동 움직임 분석"""
        # 기존 AdvancedExerciseAnalyzer와 연동
        from .advanced_exercise_analyzer import advanced_analyzer
        
        # 운동 종목 자동 감지
        exercise_type = self._detect_exercise_type(keypoints)
        
        # 전문가급 분석 수행
        analysis = advanced_analyzer.analyze_exercise_advanced(
            exercise_type, 
            keypoints
        )
        
        # 프로 선수급 추가 분석
        pro_analysis = self._analyze_pro_level(exercise_type, keypoints, analysis)
        
        return {
            **analysis,
            'pro_level_analysis': pro_analysis,
            'category': self.category.value,
            'professional_rating': self._calculate_pro_rating(analysis, pro_analysis),
            'improvement_path': self._generate_improvement_path(exercise_type, analysis)
        }
    
    def _detect_exercise_type(self, keypoints: List[Dict]) -> str:
        """운동 종목 자동 감지"""
        # MediaPipe 키포인트 기반 운동 분류
        kp_dict = {kp['name']: kp for kp in keypoints}
        
        # 손목과 어깨 높이 차이로 푸시업 감지
        if all(k in kp_dict for k in ['left_wrist', 'right_wrist', 'left_shoulder', 'right_shoulder']):
            avg_wrist_y = (kp_dict['left_wrist']['y'] + kp_dict['right_wrist']['y']) / 2
            avg_shoulder_y = (kp_dict['left_shoulder']['y'] + kp_dict['right_shoulder']['y']) / 2
            
            if avg_wrist_y > avg_shoulder_y + 0.1:
                return 'push_up'
        
        # 무릎 굽힘 정도로 스쿼트 감지
        if all(k in kp_dict for k in ['left_knee', 'right_knee', 'left_hip', 'right_hip']):
            avg_knee_y = (kp_dict['left_knee']['y'] + kp_dict['right_knee']['y']) / 2
            avg_hip_y = (kp_dict['left_hip']['y'] + kp_dict['right_hip']['y']) / 2
            
            if avg_knee_y > avg_hip_y + 0.05:
                return 'squat'
        
        # 수평 자세로 플랭크 감지
        if all(k in kp_dict for k in ['nose', 'left_ankle', 'right_ankle']):
            body_angle = abs(kp_dict['nose']['y'] - (kp_dict['left_ankle']['y'] + kp_dict['right_ankle']['y']) / 2)
            if body_angle < 0.1:
                return 'plank'
        
        return 'push_up'  # 기본값

class FitnessAnalyzer(BaseExerciseAnalyzer):
    """헬스 전문 분석기 (웨이트 트레이닝)"""
    
    def __init__(self):
        super().__init__(SportCategory.FITNESS)
    
    def _load_professional_standards(self) -> Dict:
        return {
            'bench_press': {
                'powerlifting_form': {
                    'arch_angle': (10, 20),
                    'bar_path': 'straight_vertical',
                    'pause_time': 1.0,  # 가슴 터치 후 정지
                    'lockout_complete': True
                },
                'bodybuilding_form': {
                    'controlled_negative': (2, 4),  # 초
                    'stretch_position': True,
                    'mind_muscle_connection': 95
                }
            },
            'squat_barbell': {
                'powerlifting': {
                    'depth': 'hip_crease_below_knee',
                    'bar_position': 'low_bar',
                    'stance_width': 'wide',
                    'knee_tracking': 'toes_out'
                },
                'olympic_lifting': {
                    'depth': 'ass_to_grass',
                    'bar_position': 'high_bar',
                    'stance_width': 'shoulder_width',
                    'upright_torso': True
                }
            },
            'deadlift': {
                'conventional': {
                    'bar_path': 'straight_up',
                    'hip_hinge': 'dominant',
                    'shoulder_position': 'over_bar',
                    'lockout': 'hip_thrust'
                },
                'sumo': {
                    'stance': 'wide',
                    'grip': 'narrow',
                    'upright_torso': True,
                    'knee_tracking': 'out'
                }
            }
        }
    
    def _load_common_mistakes(self) -> Dict:
        return {
            'bench_press': [
                'bouncing_chest',
                'uneven_press',
                'feet_movement',
                'arch_excessive',
                'bar_path_forward'
            ],
            'squat_barbell': [
                'knee_cave',
                'forward_lean',
                'heel_rise',
                'butt_wink',
                'uneven_depth'
            ],
            'deadlift': [
                'bar_drift',
                'rounded_back',
                'knee_lockout_early',
                'hip_rise_first',
                'uneven_pull'
            ]
        }
    
    def _load_pro_techniques(self) -> Dict:
        return {
            'bench_press': {
                'powerlifting': '아치, 레그드라이브, 파워 활용',
                'bodybuilding': '느린 템포, MMC 집중',
                'strength': '폭발적 컨센트릭'
            }
        }
    
    def analyze_movement(self, keypoints: List[Dict], video_frame: np.ndarray) -> Dict[str, Any]:
        """헬스 움직임 분석"""
        # 바벨/덤벨 위치 감지 (YOLO 객체 탐지 필요)
        equipment = self._detect_equipment(video_frame)
        exercise_type = self._classify_fitness_exercise(keypoints, equipment)
        
        # 전문가급 분석
        analysis = self._analyze_lifting_form(exercise_type, keypoints)
        
        return {
            'exercise_type': exercise_type,
            'equipment_detected': equipment,
            'form_analysis': analysis,
            'category': self.category.value,
            'powerlifting_score': self._calculate_powerlifting_score(analysis),
            'bodybuilding_score': self._calculate_bodybuilding_score(analysis)
        }

class GolfAnalyzer(BaseExerciseAnalyzer):
    """골프 전문 분석기"""
    
    def __init__(self):
        super().__init__(SportCategory.GOLF)
    
    def _load_professional_standards(self) -> Dict:
        return {
            'full_swing': {
                'pga_standards': {
                    'backswing_shoulder_turn': (90, 110),
                    'hip_turn_ratio': 0.5,  # 어깨 회전의 절반
                    'wrist_cock_angle': (70, 90),
                    'club_shaft_plane': True,
                    'weight_transfer': 'right_to_left',
                    'follow_through': 'full_finish'
                },
                'tour_player_metrics': {
                    'club_head_speed': (110, 130),  # mph
                    'smash_factor': (1.45, 1.50),
                    'attack_angle': (-2, 2),        # 드라이버
                    'swing_plane_consistency': 95   # 퍼센트
                }
            },
            'putting': {
                'pga_standards': {
                    'pendulum_motion': True,
                    'shoulder_dominant': True,
                    'putter_face_square': (0, 2),  # 도
                    'stroke_tempo': '3:1_ratio',
                    'follow_through_distance': 'equal_backswing'
                }
            },
            'chipping': {
                'short_game_standards': {
                    'weight_forward': 60,  # 왼발 체중 비율
                    'hands_ahead': True,
                    'minimal_wrist_action': True,
                    'controlled_distance': True
                }
            }
        }
    
    def _load_common_mistakes(self) -> Dict:
        return {
            'full_swing': [
                'over_the_top',
                'early_extension',
                'casting',
                'reverse_pivot',
                'flying_elbow',
                'chicken_wing'
            ],
            'putting': [
                'wrist_breakdown',
                'head_movement',
                'deceleration',
                'off_plane_stroke'
            ]
        }
    
    def _load_pro_techniques(self) -> Dict:
        return {
            'full_swing': {
                'tiger_woods': '파워와 정확성의 완벽한 조화',
                'rory_mcilroy': '현대적 파워 스윙',
                'jason_day': '완벽한 템포와 리듬'
            },
            'putting': {
                'jordan_spieth': '뛰어난 거리감과 라인 읽기',
                'tiger_woods': '압박감 하의 정확성'
            }
        }
    
    def analyze_movement(self, keypoints: List[Dict], video_frame: np.ndarray) -> Dict[str, Any]:
        """골프 스윙 분석"""
        # 골프 클럽 감지
        club_detected = self._detect_golf_club(video_frame)
        
        # 스윙 단계 분류
        swing_phase = self._classify_swing_phase(keypoints)
        
        # 전문가급 스윙 분석
        analysis = self._analyze_golf_swing(keypoints, swing_phase)
        
        return {
            'swing_type': self._determine_swing_type(keypoints),
            'swing_phase': swing_phase,
            'club_detected': club_detected,
            'swing_analysis': analysis,
            'category': self.category.value,
            'pga_tour_similarity': self._compare_to_tour_players(analysis),
            'handicap_estimate': self._estimate_handicap(analysis)
        }

class SportsAnalysisSystem:
    """통합 스포츠 분석 시스템"""
    
    def __init__(self):
        self.analyzers = {
            SportCategory.BODYWEIGHT: BodyweightAnalyzer(),
            SportCategory.FITNESS: FitnessAnalyzer(),
            SportCategory.GOLF: GolfAnalyzer(),
            # 추가 스포츠 분석기들을 여기에 등록
        }
        
        self.supported_exercises = {
            SportCategory.BODYWEIGHT: [
                'push_up', 'squat', 'plank', 'pull_up', 'burpee', 
                'mountain_climber', 'jumping_jack', 'lunge'
            ],
            SportCategory.FITNESS: [
                'bench_press', 'squat_barbell', 'deadlift', 'overhead_press',
                'barbell_row', 'bicep_curl', 'tricep_extension'
            ],
            SportCategory.GOLF: [
                'full_swing', 'putting', 'chipping', 'pitching', 'bunker_shot'
            ]
        }
    
    def analyze_sport_movement(self, 
                              category: SportCategory,
                              keypoints: List[Dict], 
                              video_frame: np.ndarray,
                              user_profile: Optional[Dict] = None) -> Dict[str, Any]:
        """스포츠 움직임 통합 분석"""
        
        if category not in self.analyzers:
            return {
                'error': f'{category.value} 종목은 아직 지원되지 않습니다',
                'supported_categories': list(self.analyzers.keys())
            }
        
        analyzer = self.analyzers[category]
        
        try:
            # 카테고리별 전문 분석 수행
            analysis_result = analyzer.analyze_movement(keypoints, video_frame)
            
            # 공통 메타데이터 추가
            analysis_result.update({
                'timestamp': datetime.utcnow().isoformat(),
                'user_profile': user_profile,
                'supported_exercises': self.supported_exercises[category],
                'analysis_version': '2.0.0',
                'confidence_level': self._calculate_confidence(keypoints, video_frame)
            })
            
            return analysis_result
            
        except Exception as e:
            return {
                'error': f'분석 중 오류 발생: {str(e)}',
                'category': category.value,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_available_categories(self) -> Dict[str, Any]:
        """지원 가능한 스포츠 카테고리 정보"""
        return {
            category.value: {
                'name': category.value,
                'analyzer_available': category in self.analyzers,
                'supported_exercises': self.supported_exercises.get(category, []),
                'ai_analysis_level': 'professional' if category in self.analyzers else 'coming_soon'
            }
            for category in SportCategory
        }
    
    def _calculate_confidence(self, keypoints: List[Dict], video_frame: np.ndarray) -> float:
        """분석 신뢰도 계산"""
        # 키포인트 가시성 확인
        visible_points = sum(1 for kp in keypoints if kp.get('visibility', 0) > 0.5)
        total_points = len(keypoints)
        
        keypoint_confidence = visible_points / total_points if total_points > 0 else 0
        
        # 이미지 품질 확인 (해상도, 밝기 등)
        height, width = video_frame.shape[:2]
        resolution_quality = min(1.0, (height * width) / (640 * 480))
        
        # 전체적인 신뢰도
        overall_confidence = (keypoint_confidence * 0.7 + resolution_quality * 0.3) * 100
        
        return min(100, max(0, overall_confidence))
    
    def get_exercise_recommendations(self, 
                                   category: SportCategory, 
                                   current_analysis: Dict,
                                   user_level: str = 'beginner') -> List[str]:
        """운동 추천 시스템"""
        recommendations = []
        
        if category == SportCategory.BODYWEIGHT:
            if user_level == 'beginner':
                recommendations = [
                    "벽 푸시업으로 시작하여 기본 자세를 익히세요",
                    "체어 스쿼트로 하체 근력을 기르세요",
                    "무릎 플랭크로 코어를 강화하세요"
                ]
            elif user_level == 'advanced':
                recommendations = [
                    "원암 푸시업에 도전해보세요",
                    "피스톨 스쿼트로 단일 다리 힘을 기르세요",
                    "5분 플랭크 홀드를 목표로 하세요"
                ]
        
        return recommendations

# 전역 인스턴스
sports_analysis_system = SportsAnalysisSystem()