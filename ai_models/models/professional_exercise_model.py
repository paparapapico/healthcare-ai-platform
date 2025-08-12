"""
전문 운동선수 수준 AI 모델 아키텍처
Professional athlete-level exercise AI model architecture
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

class ProfessionalExerciseModel:
    """올림픽 수준 운동 분석 AI 모델"""
    
    # 운동별 올림픽 표준 각도 범위
    OLYMPIC_STANDARDS = {
        'squat': {
            'knee_angle_bottom': (70, 90),  # 깊은 스쿼트
            'hip_angle_bottom': (60, 80),
            'spine_angle': (75, 90),  # 척추 중립
            'knee_tracking': 5,  # 무릎 외반 허용 각도
        },
        'push_up': {
            'elbow_angle_bottom': (85, 95),
            'shoulder_angle': (40, 60),
            'body_alignment': 175,  # 거의 일직선
            'descent_time': (1.5, 2.5),  # 초
        },
        'deadlift': {
            'hip_hinge_angle': (20, 30),
            'knee_angle_start': (15, 25),
            'spine_neutrality': 5,  # 허용 오차
            'bar_path_deviation': 5,  # cm
        },
        'bench_press': {
            'elbow_angle_bottom': (85, 95),
            'arch_angle': (10, 20),
            'bar_path_angle': 5,  # 수직선 대비
            'tempo': (2, 1, 2),  # 하강-정지-상승
        }
    }
    
    def __init__(self, exercise_type: str):
        """
        Args:
            exercise_type: 운동 종류
        """
        self.exercise_type = exercise_type
        self.model = None
        self.attention_weights = None
        self.standards = self.OLYMPIC_STANDARDS.get(exercise_type, {})
    
    def build_vision_transformer(self, 
                                input_shape: Tuple[int, int] = (30, 132),
                                num_heads: int = 8,
                                ff_dim: int = 512,
                                num_transformer_blocks: int = 4,
                                mlp_units: List[int] = [512, 256, 128]) -> tf.keras.Model:
        """
        Vision Transformer 기반 모델 구축
        
        Args:
            input_shape: 입력 형태 (시퀀스 길이, 특징 수)
            num_heads: 어텐션 헤드 수
            ff_dim: Feed-forward 차원
            num_transformer_blocks: Transformer 블록 수
            mlp_units: MLP 유닛 리스트
        """
        inputs = layers.Input(shape=input_shape)
        
        # Patch + Position Embedding
        # 각 프레임을 패치로 간주
        x = layers.Dense(input_shape[1])(inputs)
        
        # Learnable position embedding
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        position_embedding = layers.Embedding(
            input_dim=input_shape[0], 
            output_dim=input_shape[1]
        )(positions)
        x = x + position_embedding
        
        # Transformer 블록들
        for _ in range(num_transformer_blocks):
            x = self.transformer_block(x, num_heads, ff_dim, input_shape[1])
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # MLP Head
        for dim in mlp_units:
            x = layers.Dense(dim, activation='gelu')(x)
            x = layers.Dropout(0.2)(x)
        
        # Multi-task outputs
        outputs = self.create_multi_task_outputs(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def transformer_block(self, x, num_heads: int, ff_dim: int, embed_dim: int):
        """Transformer 블록"""
        # Multi-Head Attention
        attention_output, attention_weights = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=0.1
        )(x, x, return_attention_scores=True)
        
        # Skip connection 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        x2 = layers.Add()([attention_output, x1])
        
        # Feed Forward Network
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(ff_dim, activation='gelu')(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(embed_dim)(x3)
        x3 = layers.Dropout(0.1)(x3)
        
        # Skip connection 2
        output = layers.Add()([x3, x2])
        
        # 어텐션 가중치 저장 (분석용)
        self.attention_weights = attention_weights
        
        return output
    
    def create_multi_task_outputs(self, x):
        """다중 작업 출력 헤드"""
        outputs = {}
        
        # 1. 운동 단계 분류 (Phase Classification)
        phase_branch = layers.Dense(128, activation='relu', name='phase_branch')(x)
        phase_output = layers.Dense(5, activation='softmax', name='phase')(phase_branch)
        outputs['phase'] = phase_output
        
        # 2. 자세 품질 점수 (Form Quality Score)
        quality_branch = layers.Dense(64, activation='relu', name='quality_branch')(x)
        quality_output = layers.Dense(1, activation='sigmoid', name='quality')(quality_branch)
        outputs['quality'] = quality_output
        
        # 3. 반복 횟수 (Rep Counting)
        rep_branch = layers.Dense(64, activation='relu', name='rep_branch')(x)
        rep_output = layers.Dense(1, activation='linear', name='rep_count')(rep_branch)
        outputs['rep_count'] = rep_output
        
        # 4. 실수 감지 (Mistake Detection) - Multi-label
        mistake_branch = layers.Dense(128, activation='relu', name='mistake_branch')(x)
        mistake_output = layers.Dense(10, activation='sigmoid', name='mistakes')(mistake_branch)
        outputs['mistakes'] = mistake_output
        
        # 5. 부상 위험도 (Injury Risk Assessment)
        injury_branch = layers.Dense(64, activation='relu', name='injury_branch')(x)
        injury_output = layers.Dense(3, activation='softmax', name='injury_risk')(injury_branch)
        outputs['injury_risk'] = injury_output
        
        # 6. 피로도 추정 (Fatigue Estimation)
        fatigue_branch = layers.Dense(32, activation='relu', name='fatigue_branch')(x)
        fatigue_output = layers.Dense(1, activation='sigmoid', name='fatigue')(fatigue_branch)
        outputs['fatigue'] = fatigue_output
        
        # 7. 템포/리듬 점수 (Tempo/Rhythm Score)
        tempo_branch = layers.Dense(32, activation='relu', name='tempo_branch')(x)
        tempo_output = layers.Dense(1, activation='sigmoid', name='tempo')(tempo_branch)
        outputs['tempo'] = tempo_output
        
        return outputs
    
    def build_expert_evaluator(self) -> tf.keras.Model:
        """전문가 평가 모델 (앙상블)"""
        input_shape = (30, 132)
        
        # 다양한 아키텍처의 모델들
        models_list = []
        
        # 1. Vision Transformer
        vit_model = self.build_vision_transformer(input_shape)
        models_list.append(vit_model)
        
        # 2. Temporal Convolutional Network (TCN)
        tcn_model = self.build_tcn_model(input_shape)
        models_list.append(tcn_model)
        
        # 3. Graph Neural Network (관절 관계 모델링)
        gnn_model = self.build_gnn_model(input_shape)
        models_list.append(gnn_model)
        
        # 앙상블 결합
        ensemble_model = self.create_ensemble(models_list, input_shape)
        
        return ensemble_model
    
    def build_tcn_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Temporal Convolutional Network"""
        inputs = layers.Input(shape=input_shape)
        
        x = inputs
        num_channels = [64, 128, 256]
        
        for i, channels in enumerate(num_channels):
            # Dilated causal convolution
            dilation_rate = 2 ** i
            x = layers.Conv1D(
                filters=channels,
                kernel_size=3,
                padding='causal',
                dilation_rate=dilation_rate,
                activation='relu'
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.SpatialDropout1D(0.2)(x)
            
            # Residual connection
            if i > 0:
                residual = layers.Conv1D(channels, 1)(inputs if i == 0 else residual)
                x = layers.Add()([x, residual])
                residual = x
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        
        outputs = self.create_multi_task_outputs(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_gnn_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Graph Neural Network for skeleton modeling"""
        inputs = layers.Input(shape=input_shape)
        
        # 관절 연결 그래프 정의
        adjacency_matrix = self.create_skeleton_adjacency_matrix()
        
        x = inputs
        
        # Graph Convolution Layers
        for _ in range(3):
            x = self.graph_convolution(x, adjacency_matrix, 128)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.2)(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        
        outputs = self.create_multi_task_outputs(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def graph_convolution(self, x, adjacency_matrix, units):
        """그래프 컨볼루션 레이어"""
        # 간단한 구현 (실제로는 spektral 같은 라이브러리 사용 권장)
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        
        # Feature transformation
        x_transformed = layers.Dense(units)(x)
        
        # Graph convolution (simplified)
        # 실제 구현시 adjacency matrix를 활용한 message passing 필요
        output = x_transformed
        
        return output
    
    def create_skeleton_adjacency_matrix(self):
        """스켈레톤 인접 행렬 생성"""
        # MediaPipe 33개 랜드마크 연결 관계
        connections = [
            (11, 12),  # 어깨
            (11, 13), (13, 15),  # 왼팔
            (12, 14), (14, 16),  # 오른팔
            (11, 23), (12, 24),  # 몸통
            (23, 24),  # 엉덩이
            (23, 25), (25, 27),  # 왼다리
            (24, 26), (26, 28),  # 오른다리
        ]
        
        # 33x33 인접 행렬
        adj_matrix = np.zeros((33, 33))
        for i, j in connections:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
        
        # Self-connections
        np.fill_diagonal(adj_matrix, 1)
        
        return tf.constant(adj_matrix, dtype=tf.float32)
    
    def create_ensemble(self, models_list: List[tf.keras.Model], 
                       input_shape: Tuple[int, int]) -> tf.keras.Model:
        """앙상블 모델 생성"""
        inputs = layers.Input(shape=input_shape)
        
        # 각 모델의 예측 수집
        predictions = []
        for model in models_list:
            pred = model(inputs)
            predictions.append(pred)
        
        # 태스크별 앙상블
        ensemble_outputs = {}
        
        for task_name in predictions[0].keys():
            task_predictions = [pred[task_name] for pred in predictions]
            
            if task_name in ['phase', 'injury_risk']:
                # 분류 태스크: 평균 후 소프트맥스
                avg_pred = layers.Average()(task_predictions)
                ensemble_outputs[task_name] = avg_pred
            else:
                # 회귀/이진 분류: 가중 평균
                # 학습 가능한 가중치
                weights = [layers.Dense(1, activation='sigmoid')(inputs) 
                          for _ in task_predictions]
                weighted_preds = [w * p for w, p in zip(weights, task_predictions)]
                ensemble_outputs[task_name] = layers.Add()(weighted_preds)
        
        ensemble_model = models.Model(inputs=inputs, outputs=ensemble_outputs)
        return ensemble_model
    
    def create_explainable_model(self) -> tf.keras.Model:
        """설명 가능한 AI 모델 (Attention 시각화 포함)"""
        input_shape = (30, 132)
        inputs = layers.Input(shape=input_shape)
        
        # 특징 중요도 학습
        feature_importance = layers.Dense(
            input_shape[1], 
            activation='sigmoid',
            name='feature_importance'
        )(inputs)
        
        # 특징 가중치 적용
        weighted_inputs = layers.Multiply()([inputs, feature_importance])
        
        # Temporal Attention
        temporal_attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            name='temporal_attention'
        )(weighted_inputs, weighted_inputs)
        
        # 특징 추출
        x = layers.LSTM(128, return_sequences=True)(temporal_attention)
        x = layers.LSTM(64)(x)
        
        # 예측
        outputs = self.create_multi_task_outputs(x)
        
        # 설명 가능성을 위한 추가 출력
        outputs['feature_importance'] = feature_importance
        outputs['temporal_attention'] = temporal_attention
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def compare_with_olympic_standard(self, 
                                     pose_sequence: np.ndarray,
                                     exercise_type: str) -> Dict:
        """올림픽 선수 표준과 비교"""
        if exercise_type not in self.OLYMPIC_STANDARDS:
            return {'error': 'No Olympic standard available for this exercise'}
        
        standards = self.OLYMPIC_STANDARDS[exercise_type]
        comparison_results = {
            'overall_similarity': 0,
            'technique_scores': {},
            'deviations': [],
            'recommendations': []
        }
        
        # 각 프레임에서 주요 각도 계산
        for frame_idx, frame in enumerate(pose_sequence):
            frame_analysis = self.analyze_frame(frame, standards)
            
            for metric, value in frame_analysis.items():
                if metric not in comparison_results['technique_scores']:
                    comparison_results['technique_scores'][metric] = []
                comparison_results['technique_scores'][metric].append(value)
        
        # 전체 유사도 계산
        total_score = 0
        for metric, scores in comparison_results['technique_scores'].items():
            avg_score = np.mean(scores)
            total_score += avg_score
            
            if avg_score < 80:
                comparison_results['deviations'].append({
                    'metric': metric,
                    'score': avg_score,
                    'recommendation': self.get_improvement_recommendation(metric, avg_score)
                })
        
        comparison_results['overall_similarity'] = total_score / len(comparison_results['technique_scores'])
        
        # 종합 추천
        if comparison_results['overall_similarity'] >= 95:
            comparison_results['recommendations'].append("올림픽 선수 수준입니다! 현재 기술을 유지하세요.")
        elif comparison_results['overall_similarity'] >= 85:
            comparison_results['recommendations'].append("준프로 수준입니다. 세부 기술을 다듬으세요.")
        else:
            comparison_results['recommendations'].append("기본기를 더 연습해야 합니다.")
        
        return comparison_results
    
    def analyze_frame(self, frame: np.ndarray, standards: Dict) -> Dict:
        """단일 프레임 분석"""
        # 랜드마크에서 각도 계산 (간단한 예시)
        analysis = {}
        
        # 실제 구현시 MediaPipe 랜드마크 인덱스 사용
        # 여기서는 예시로 간단히 구현
        if 'knee_angle_bottom' in standards:
            knee_angle = self.calculate_joint_angle(frame, 'knee')
            min_angle, max_angle = standards['knee_angle_bottom']
            if min_angle <= knee_angle <= max_angle:
                analysis['knee_angle'] = 100
            else:
                deviation = min(abs(knee_angle - min_angle), abs(knee_angle - max_angle))
                analysis['knee_angle'] = max(0, 100 - deviation)
        
        return analysis
    
    def calculate_joint_angle(self, landmarks: np.ndarray, joint: str) -> float:
        """관절 각도 계산"""
        # 실제 구현 필요
        # MediaPipe 랜드마크 인덱스 사용
        return 90.0  # 임시 값
    
    def get_improvement_recommendation(self, metric: str, score: float) -> str:
        """개선 추천사항 생성"""
        recommendations = {
            'knee_angle': {
                80: "무릎 각도가 약간 벗어났습니다. 더 깊게 앉으세요.",
                60: "무릎 각도 개선이 필요합니다. 유연성 운동을 병행하세요.",
                40: "무릎 각도가 부적절합니다. 기본 자세부터 다시 연습하세요."
            },
            'spine_angle': {
                80: "척추 정렬을 조금 더 신경쓰세요.",
                60: "허리가 굽어집니다. 코어 근육을 강화하세요.",
                40: "척추 정렬이 위험합니다. 부상 방지를 위해 자세 교정이 시급합니다."
            }
        }
        
        if metric in recommendations:
            for threshold, rec in sorted(recommendations[metric].items(), reverse=True):
                if score >= threshold:
                    return rec
        
        return "기술 개선이 필요합니다."
    
    def generate_personalized_program(self, 
                                     user_analysis: Dict,
                                     goal: str = 'olympic') -> Dict:
        """개인 맞춤 훈련 프로그램 생성"""
        program = {
            'duration_weeks': 12,
            'phases': [],
            'exercises': [],
            'progression': []
        }
        
        current_level = user_analysis.get('overall_similarity', 50)
        
        if goal == 'olympic' and current_level < 95:
            # Phase 1: 기초 강화 (4주)
            program['phases'].append({
                'phase': 1,
                'name': '기초 강화',
                'weeks': 4,
                'focus': '기본 자세와 근력',
                'exercises': [
                    {'name': '벽 스쿼트', 'sets': 3, 'reps': 15, 'progression': 'hold time'},
                    {'name': '고블릿 스쿼트', 'sets': 4, 'reps': 12, 'progression': 'weight'},
                    {'name': '코어 강화', 'sets': 3, 'duration': '30-60s'},
                ]
            })
            
            # Phase 2: 기술 개선 (4주)
            program['phases'].append({
                'phase': 2,
                'name': '기술 개선',
                'weeks': 4,
                'focus': '올바른 움직임 패턴',
                'exercises': [
                    {'name': '템포 스쿼트', 'sets': 4, 'reps': 8, 'tempo': '3-1-3-1'},
                    {'name': '페이즈 스쿼트', 'sets': 3, 'reps': 10, 'pause': '2s at bottom'},
                    {'name': '싱글 레그 워크', 'sets': 3, 'reps': 12},
                ]
            })
            
            # Phase 3: 파워 개발 (4주)
            program['phases'].append({
                'phase': 3,
                'name': '파워 개발',
                'weeks': 4,
                'focus': '폭발력과 속도',
                'exercises': [
                    {'name': '점프 스쿼트', 'sets': 5, 'reps': 5, 'rest': '2-3min'},
                    {'name': '박스 점프', 'sets': 4, 'reps': 6, 'progression': 'height'},
                    {'name': '올림픽 리프트 변형', 'sets': 4, 'reps': 3},
                ]
            })
        
        # 주별 진행도
        for week in range(1, 13):
            program['progression'].append({
                'week': week,
                'expected_improvement': current_level + (95 - current_level) * (week / 12),
                'focus_points': self.get_weekly_focus(week, user_analysis)
            })
        
        return program
    
    def get_weekly_focus(self, week: int, user_analysis: Dict) -> List[str]:
        """주별 집중 포인트"""
        weak_points = user_analysis.get('deviations', [])
        
        if week <= 4:
            return ['기본 자세 확립', '근력 기초 구축', '유연성 향상']
        elif week <= 8:
            return ['움직임 패턴 개선', '약점 보완', '협응력 향상']
        else:
            return ['파워 개발', '속도 향상', '경기 특화 훈련']


class ExerciseInference:
    """실시간 추론을 위한 최적화된 클래스"""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: TFLite 모델 경로
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 추론 시간 추적
        self.inference_times = []
    
    def predict(self, pose_sequence: np.ndarray) -> Dict:
        """실시간 추론"""
        import time
        
        start_time = time.time()
        
        # 입력 설정
        input_data = pose_sequence.astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # 추론 실행
        self.interpreter.invoke()
        
        # 출력 추출
        outputs = {}
        for detail in self.output_details:
            output_data = self.interpreter.get_tensor(detail['index'])
            outputs[detail['name']] = output_data
        
        # 추론 시간 기록
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        # 결과 후처리
        result = self.postprocess_outputs(outputs)
        result['inference_time_ms'] = inference_time
        
        return result
    
    def postprocess_outputs(self, outputs: Dict) -> Dict:
        """출력 후처리"""
        result = {}
        
        # Phase 디코딩
        if 'phase' in outputs:
            phase_probs = outputs['phase'][0]
            phase_labels = ['ready', 'descent', 'bottom', 'ascent', 'top']
            result['phase'] = phase_labels[np.argmax(phase_probs)]
            result['phase_confidence'] = float(np.max(phase_probs))
        
        # Quality score
        if 'quality' in outputs:
            result['quality_score'] = float(outputs['quality'][0][0] * 100)
        
        # Mistakes 디코딩
        if 'mistakes' in outputs:
            mistake_probs = outputs['mistakes'][0]
            mistake_labels = [
                'knee_valgus', 'forward_lean', 'heel_rise', 'back_round',
                'elbow_flare', 'partial_rom', 'asymmetry', 'speed_issue',
                'breathing', 'stability'
            ]
            detected_mistakes = [
                mistake_labels[i] for i, prob in enumerate(mistake_probs) if prob > 0.5
            ]
            result['mistakes'] = detected_mistakes
        
        # Injury risk
        if 'injury_risk' in outputs:
            risk_probs = outputs['injury_risk'][0]
            risk_labels = ['low', 'medium', 'high']
            result['injury_risk'] = risk_labels[np.argmax(risk_probs)]
            result['injury_risk_score'] = float(np.max(risk_probs))
        
        return result
    
    def get_performance_stats(self) -> Dict:
        """성능 통계"""
        if not self.inference_times:
            return {'error': 'No inference performed yet'}
        
        return {
            'avg_inference_time_ms': np.mean(self.inference_times),
            'min_inference_time_ms': np.min(self.inference_times),
            'max_inference_time_ms': np.max(self.inference_times),
            'fps': 1000 / np.mean(self.inference_times),
            'total_inferences': len(self.inference_times)
        }


if __name__ == "__main__":
    # 모델 테스트
    model = ProfessionalExerciseModel('squat')
    
    # Vision Transformer 모델 생성
    vit_model = model.build_vision_transformer()
    vit_model.summary()
    
    # 전문가 평가 모델 생성
    expert_model = model.build_expert_evaluator()
    print("\n전문가 평가 모델 생성 완료")
    
    # 설명 가능한 모델 생성
    explainable_model = model.create_explainable_model()
    print("\n설명 가능한 AI 모델 생성 완료")