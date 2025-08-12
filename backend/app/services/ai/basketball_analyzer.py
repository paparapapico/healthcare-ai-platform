# backend/app/services/ai/basketball_analyzer.py
"""
농구 전문가 수준 AI 분석기
Basketball Professional-Level AI Analyzer
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import cv2
import math
from .sports_categories_analyzer import BaseExerciseAnalyzer, SportCategory

class BasketballAnalyzer(BaseExerciseAnalyzer):
    """농구 전문 분석기 - NBA/KBL 프로 선수급 기준"""
    
    def __init__(self):
        super().__init__(SportCategory.BASKETBALL)
        self.shot_tracking = {
            'arc_height': [],
            'release_angle': [],
            'follow_through': [],
            'shooting_percentage': 0
        }
        
    def _load_professional_standards(self) -> Dict:
        """NBA/WNBA/KBL 프로 선수 기준"""
        return {
            'jump_shot': {
                'nba_standards': {
                    'release_height': (2.7, 3.2),      # 미터
                    'arc_angle': (45, 50),             # 최적 궤도각
                    'release_time': (0.6, 0.8),        # 캐치부터 릴리즈까지
                    'follow_through_angle': (15, 30),   # 손목 꺾임
                    'shooting_pocket_position': 'forehead_level',
                    'elbow_alignment': 'under_ball',
                    'off_hand_position': 'side_guide_only'
                },
                'elite_shooters': {  # 스테판 커리, 클레이 탐슨 기준
                    'consistency_rating': 95,  # 동일한 폼 반복율
                    'quick_release': 0.4,      # 초 (세계 최고 수준)
                    'arc_consistency': 3,       # 도 이내 편차
                    'balance_landing': True,    # 착지시 균형
                    'rhythm_timing': 'perfect'  # 리듬감
                }
            },
            
            'free_throw': {
                'nba_standards': {
                    'routine_time': (8, 12),           # 초
                    'consistent_routine': True,
                    'arc_angle': (48, 52),             # 자유투는 더 높은 각도
                    'spin_rate': (140, 180),           # RPM
                    'entry_angle': (43, 47),           # 림 진입각
                    'left_right_miss_ratio': 1.0       # 좌우 균형
                },
                'elite_free_throw_shooters': {  # 스티브 내쉬, 더크 노비츠키 기준
                    'percentage_target': 90,    # 90% 이상
                    'pressure_consistency': 95, # 압박 상황 일관성
                    'routine_identical': True   # 동일한 루틴
                }
            },
            
            'dribbling': {
                'nba_standards': {
                    'crossover_speed': (0.3, 0.5),    # 초
                    'ball_height': 'waist_level',
                    'body_control': 'low_center_gravity',
                    'eyes_up_percentage': 80,          # 드리블 중 고개를 든 시간
                    'hand_protection': True,           # 볼 보호
                    'change_of_pace': True            # 속도 변화
                },
                'elite_ball_handlers': {  # 카이리 어빙, 크리스 폴 기준
                    'ambidextrous_skill': 90,  # 양손 사용 능력
                    'tight_space_control': 95, # 좁은 공간 제어력
                    'hesitation_timing': 'perfect',
                    'between_legs_fluidity': 95
                }
            },
            
            'layup': {
                'nba_standards': {
                    'approach_angle': 45,              # 도
                    'jump_timing': 'two_step_rhythm',
                    'ball_protection': 'away_from_defender',
                    'soft_touch': True,                # 부드러운 터치
                    'bank_shot_angle': (30, 60),       # 뱅크샷 각도
                    'finish_hand': 'outside_hand'      # 바깥손 마무리
                },
                'elite_finishers': {  # 토니 파커, 존 월 기준
                    'contact_balance': 95,     # 접촉시 균형
                    'body_control': 98,        # 공중에서 몸 제어
                    'reverse_layup_skill': 90, # 리버스 레이업
                    'ambidextrous_finish': 85  # 양손 마무리
                }
            },
            
            'defense': {
                'nba_standards': {
                    'defensive_stance': {
                        'feet_width': 'shoulder_width_plus',
                        'knee_bend': (45, 60),         # 무릎 굽힘 각도
                        'hand_position': 'active_hands',
                        'body_position': 'between_man_basket',
                        'weight_distribution': 'balls_of_feet'
                    },
                    'lateral_movement': {
                        'slide_step': 'no_crossover',
                        'hip_sink': True,              # 엉덩이 낮춤
                        'head_stability': True,        # 머리 흔들림 없음
                        'reaction_time': 0.2          # 초
                    }
                },
                'elite_defenders': {  # 카와이 레너드, 드레이몬드 그린 기준
                    'anticipation_skill': 95,   # 예측 능력
                    'help_defense_timing': 98,  # 도움 수비 타이밍
                    'steal_technique': 'clean', # 깨끗한 스틸
                    'block_timing': 'vertical'  # 수직 블록
                }
            }
        }
    
    def _load_common_mistakes(self) -> Dict:
        """농구에서 흔히 하는 실수들"""
        return {
            'jump_shot': [
                'thumb_flick',           # 엄지 튕기기
                'inconsistent_release_point',
                'off_hand_interference', # 보조손 간섭
                'low_arc',              # 낮은 궤도
                'fade_away_unnecessary', # 불필요한 페이드어웨이
                'rushed_shot',          # 급한 슛
                'poor_follow_through',  # 부실한 팔로스루
                'elbow_out'            # 팔꿈치 벌어짐
            ],
            
            'free_throw': [
                'inconsistent_routine',  # 불일치한 루틴
                'thinking_too_much',    # 과도한 생각
                'muscle_tension',       # 근육 긴장
                'target_fixation',      # 목표 고착
                'breath_holding'        # 숨참기
            ],
            
            'dribbling': [
                'ball_watching',        # 공만 보기
                'high_dribble',        # 높은 드리블
                'weak_hand_avoidance', # 약한손 기피
                'predictable_moves',   # 예측 가능한 동작
                'no_change_of_pace',   # 속도 변화 없음
                'loose_handle'         # 느슨한 핸들링
            ],
            
            'layup': [
                'wrong_foot_takeoff',   # 잘못된 발 사용
                'ball_exposure',        # 공 노출
                'rushing_finish',       # 급한 마무리
                'soft_touch_missing',   # 부드러운 터치 부족
                'poor_angle'           # 잘못된 접근 각도
            ],
            
            'defense': [
                'crossing_feet',        # 발 교차
                'reaching_in',         # 파울성 손 뻗기
                'upright_stance',      # 직립 자세
                'ball_watching',       # 공만 보기
                'lazy_closeout',       # 느린 근접수비
                'poor_help_timing'     # 잘못된 도움수비 타이밍
            ]
        }
    
    def _load_pro_techniques(self) -> Dict:
        """프로 선수들의 시그니처 기술"""
        return {
            'shooting': {
                'stephen_curry': {
                    'quick_release': '0.4초 초고속 릴리즈',
                    'deep_range': '로고 슛 마스터',
                    'off_dribble': '드리블 후 슛 정확도 세계 최고'
                },
                'klay_thompson': {
                    'catch_and_shoot': '캐치앤슛 최고 전문가',
                    'rhythm_shooting': '완벽한 리듬감',
                    'corner_three': '코너 3점슛 특화'
                },
                'ray_allen': {
                    'pure_form': '교과서적 슛폼',
                    'clutch_shooting': '클러치 상황 정확도',
                    'game_preparation': '경기 준비 루틴의 달인'
                }
            },
            
            'ball_handling': {
                'kyrie_irving': {
                    'handle_artistry': '예술적 볼핸들링',
                    'isolation_moves': '1대1 특화 기술',
                    'ambidextrous': '완벽한 양손 사용'
                },
                'chris_paul': {
                    'game_control': '경기 흐름 조절',
                    'pocket_passing': '포켓 패스 마스터',
                    'court_vision': '뛰어난 코트 비전'
                }
            },
            
            'finishing': {
                'tony_parker': {
                    'teardrop': '티어드롭 슛 창시자',
                    'contact_finishing': '접촉 상황 마무리',
                    'floater_master': '플로터 전문가'
                },
                'derrick_rose': {
                    'acrobatic_finishes': '곡예같은 마무리',
                    'body_control': '공중에서의 몸 제어',
                    'explosive_first_step': '폭발적인 첫 스텝'
                }
            }
        }
    
    def analyze_movement(self, keypoints: List[Dict], video_frame: np.ndarray) -> Dict[str, Any]:
        """농구 동작 종합 분석"""
        
        # 농구 동작 분류
        action_type = self._classify_basketball_action(keypoints, video_frame)
        
        # 동작별 전문 분석
        if action_type == 'shooting':
            analysis = self._analyze_shooting_form(keypoints, video_frame)
        elif action_type == 'dribbling':
            analysis = self._analyze_dribbling(keypoints, video_frame)
        elif action_type == 'layup':
            analysis = self._analyze_layup(keypoints, video_frame)
        elif action_type == 'defense':
            analysis = self._analyze_defense_stance(keypoints, video_frame)
        else:
            analysis = self._analyze_general_basketball_movement(keypoints, video_frame)
        
        # NBA 프로급 비교 분석
        pro_comparison = self._compare_to_nba_players(action_type, analysis)
        
        # 개선 제안 생성
        improvement_suggestions = self._generate_basketball_improvements(action_type, analysis)
        
        return {
            'category': self.category.value,
            'action_type': action_type,
            'technical_analysis': analysis,
            'nba_comparison': pro_comparison,
            'improvement_suggestions': improvement_suggestions,
            'skill_level_estimate': self._estimate_skill_level(analysis),
            'training_recommendations': self._get_training_recommendations(action_type, analysis),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _classify_basketball_action(self, keypoints: List[Dict], video_frame: np.ndarray) -> str:
        """농구 동작 분류"""
        kp_dict = {kp['name']: kp for kp in keypoints}
        
        # 슛팅 동작 감지 (팔이 위로 올라가고 손목이 꺾임)
        if all(k in kp_dict for k in ['right_wrist', 'right_elbow', 'right_shoulder']):
            wrist = kp_dict['right_wrist']
            elbow = kp_dict['right_elbow']
            shoulder = kp_dict['right_shoulder']
            
            # 손목이 어깨보다 높고 팔꿈치가 굽어있으면 슛팅
            if wrist['y'] < shoulder['y'] and elbow['y'] > shoulder['y']:
                return 'shooting'
        
        # 드리블링 감지 (한 손이 아래로 향함)
        if all(k in kp_dict for k in ['right_wrist', 'left_wrist', 'right_hip', 'left_hip']):
            avg_hip_y = (kp_dict['right_hip']['y'] + kp_dict['left_hip']['y']) / 2
            
            if (kp_dict['right_wrist']['y'] > avg_hip_y or 
                kp_dict['left_wrist']['y'] > avg_hip_y):
                return 'dribbling'
        
        # 레이업 감지 (한 발이 들려있고 팔이 위로)
        if self._detect_single_foot_jump(keypoints):
            return 'layup'
        
        # 수비 자세 감지 (낮은 자세, 팔 벌림)
        if self._detect_defensive_stance(keypoints):
            return 'defense'
        
        return 'general_movement'
    
    def _analyze_shooting_form(self, keypoints: List[Dict], video_frame: np.ndarray) -> Dict[str, Any]:
        """슛팅 폼 분석"""
        kp_dict = {kp['name']: kp for kp in keypoints}
        analysis = {}
        
        # 릴리즈 포인트 분석
        if all(k in kp_dict for k in ['right_wrist', 'right_elbow', 'right_shoulder']):
            release_height = self._calculate_release_height(kp_dict)
            analysis['release_height'] = release_height
            
            # 팔꿈치 정렬 분석
            elbow_alignment = self._analyze_elbow_alignment(kp_dict)
            analysis['elbow_alignment'] = elbow_alignment
            
            # 팔로우 스루 분석
            follow_through = self._analyze_follow_through(kp_dict)
            analysis['follow_through'] = follow_through
        
        # 슛팅 밸런스 분석
        balance_analysis = self._analyze_shooting_balance(kp_dict)
        analysis['balance'] = balance_analysis
        
        # 슛팅 궤도 추정
        if 'shot_arc' in video_frame:  # 볼 트래킹이 있다면
            arc_analysis = self._analyze_shot_arc(video_frame)
            analysis['arc_analysis'] = arc_analysis
        
        # NBA 기준 점수 계산
        nba_score = self._calculate_nba_shooting_score(analysis)
        analysis['nba_similarity_score'] = nba_score
        
        return analysis
    
    def _analyze_dribbling(self, keypoints: List[Dict], video_frame: np.ndarray) -> Dict[str, Any]:
        """드리블링 기술 분석"""
        kp_dict = {kp['name']: kp for kp in keypoints}
        analysis = {}
        
        # 드리블 높이 분석
        dribble_height = self._calculate_dribble_height(kp_dict)
        analysis['dribble_height'] = dribble_height
        
        # 몸의 중심 낮춤 정도
        stance_analysis = self._analyze_dribbling_stance(kp_dict)
        analysis['stance'] = stance_analysis
        
        # 볼 보호 자세
        ball_protection = self._analyze_ball_protection(kp_dict)
        analysis['ball_protection'] = ball_protection
        
        # 헤드업 여부 (시선 분석)
        head_position = self._analyze_head_position(kp_dict)
        analysis['head_up'] = head_position
        
        return analysis
    
    def _compare_to_nba_players(self, action_type: str, analysis: Dict) -> Dict[str, Any]:
        """NBA 선수와의 비교 분석"""
        comparison = {
            'overall_similarity': 0,
            'similar_players': [],
            'skill_gaps': [],
            'strengths': []
        }
        
        if action_type == 'shooting':
            # 스테판 커리와 비교
            curry_similarity = self._compare_to_curry_shooting(analysis)
            comparison['curry_similarity'] = curry_similarity
            
            # 클레이 탐슨과 비교
            klay_similarity = self._compare_to_klay_shooting(analysis)
            comparison['klay_similarity'] = klay_similarity
            
            # 가장 유사한 선수 결정
            if curry_similarity > klay_similarity:
                comparison['most_similar'] = 'Stephen Curry'
                comparison['overall_similarity'] = curry_similarity
            else:
                comparison['most_similar'] = 'Klay Thompson'
                comparison['overall_similarity'] = klay_similarity
        
        return comparison
    
    def _generate_basketball_improvements(self, action_type: str, analysis: Dict) -> List[str]:
        """농구 실력 향상 제안"""
        improvements = []
        
        if action_type == 'shooting':
            if analysis.get('elbow_alignment', 0) < 80:
                improvements.append("팔꿈치를 볼 아래 정확히 위치시키세요 (BEEF 원리)")
            
            if analysis.get('follow_through', 0) < 75:
                improvements.append("팔로우 스루에서 손목을 더 부드럽게 꺾으세요")
            
            if analysis.get('balance', 0) < 85:
                improvements.append("슛 후 착지시 균형을 더 잘 유지하세요")
        
        elif action_type == 'dribbling':
            if analysis.get('dribble_height', 0) > 50:  # 너무 높음
                improvements.append("드리블 높이를 허리 아래로 낮추세요")
            
            if not analysis.get('head_up', False):
                improvements.append("드리블하면서 고개를 들어 코트를 보세요")
        
        # 일반적인 향상 제안
        improvements.extend([
            "매일 500개씩 자유투 연습을 하세요",
            "양손 드리블링을 균등하게 연습하세요",
            "1대1 상황 연습을 늘려보세요"
        ])
        
        return improvements[:5]  # 최대 5개 제안
    
    def _estimate_skill_level(self, analysis: Dict) -> Dict[str, Any]:
        """농구 실력 레벨 추정"""
        
        # 종합 점수 계산
        total_score = 0
        factors = []
        
        for key, value in analysis.items():
            if isinstance(value, (int, float)):
                total_score += value
                factors.append(key)
        
        avg_score = total_score / len(factors) if factors else 0
        
        # 레벨 분류
        if avg_score >= 90:
            level = "프로/대학 선수급"
            description = "매우 뛰어난 기술 수준입니다"
        elif avg_score >= 80:
            level = "고등학교 대표급"
            description = "우수한 기술 수준입니다"
        elif avg_score >= 70:
            level = "클럽팀급"
            description = "중상급 실력입니다"
        elif avg_score >= 60:
            level = "동네 농구 상급자"
            description = "기본기가 갖춰진 수준입니다"
        else:
            level = "초보자"
            description = "기본기 연습이 필요합니다"
        
        return {
            'level': level,
            'score': avg_score,
            'description': description,
            'next_goal': self._get_next_goal(avg_score)
        }
    
    def _get_training_recommendations(self, action_type: str, analysis: Dict) -> List[Dict]:
        """훈련 추천 프로그램"""
        recommendations = []
        
        if action_type == 'shooting':
            recommendations.extend([
                {
                    'exercise': '폼 슈팅',
                    'description': '골대 근처에서 올바른 폼으로 100회',
                    'duration': '15분',
                    'frequency': '매일',
                    'focus': '정확한 폼 습관화'
                },
                {
                    'exercise': '자유투 연습',
                    'description': '루틴을 정하고 500회 연속 연습',
                    'duration': '30분',
                    'frequency': '매일',
                    'focus': '정신적 집중력과 일관성'
                },
                {
                    'exercise': '스팟 슈팅',
                    'description': '5개 지점에서 각 10회씩',
                    'duration': '20분',
                    'frequency': '격일',
                    'focus': '다양한 각도 적응'
                }
            ])
        
        elif action_type == 'dribbling':
            recommendations.extend([
                {
                    'exercise': '스테이셔너리 드리블',
                    'description': '제자리에서 양손 교대로 5분씩',
                    'duration': '10분',
                    'frequency': '매일',
                    'focus': '볼 핸들링 기초'
                },
                {
                    'exercise': '콘 드리블',
                    'description': '콘 사이를 드리블하며 이동',
                    'duration': '15분',
                    'frequency': '매일',
                    'focus': '실전 상황 적응'
                }
            ])
        
        return recommendations
    
    # 보조 메서드들 (실제 구현시 더 정교한 계산 필요)
    def _calculate_release_height(self, kp_dict: Dict) -> float:
        """릴리즈 높이 계산"""
        return 2.8  # 임시값
    
    def _analyze_elbow_alignment(self, kp_dict: Dict) -> float:
        """팔꿈치 정렬 분석"""
        return 85  # 임시값
    
    def _analyze_follow_through(self, kp_dict: Dict) -> float:
        """팔로우 스루 분석"""
        return 80  # 임시값
    
    def _analyze_shooting_balance(self, kp_dict: Dict) -> float:
        """슛팅 밸런스 분석"""
        return 88  # 임시값
    
    def _calculate_dribble_height(self, kp_dict: Dict) -> float:
        """드리블 높이 계산"""
        return 45  # 임시값 (허리 아래 기준)
    
    def _analyze_dribbling_stance(self, kp_dict: Dict) -> float:
        """드리블링 자세 분석"""
        return 82  # 임시값
    
    def _analyze_ball_protection(self, kp_dict: Dict) -> float:
        """볼 보호 자세 분석"""
        return 75  # 임시값
    
    def _analyze_head_position(self, kp_dict: Dict) -> bool:
        """헤드업 여부 분석"""
        return True  # 임시값
    
    def _detect_single_foot_jump(self, keypoints: List[Dict]) -> bool:
        """한 발 점프 감지"""
        return False  # 임시값
    
    def _detect_defensive_stance(self, keypoints: List[Dict]) -> bool:
        """수비 자세 감지"""
        return False  # 임시값
    
    def _compare_to_curry_shooting(self, analysis: Dict) -> float:
        """스테판 커리 슛팅과 비교"""
        return 75  # 임시값
    
    def _compare_to_klay_shooting(self, analysis: Dict) -> float:
        """클레이 탐슨 슛팅과 비교"""
        return 70  # 임시값
    
    def _get_next_goal(self, current_score: float) -> str:
        """다음 목표 설정"""
        if current_score < 60:
            return "기본기 완성하기"
        elif current_score < 80:
            return "실전 적용 능력 기르기"
        else:
            return "프로급 일관성 달성하기"

# 전역 인스턴스
basketball_analyzer = BasketballAnalyzer()