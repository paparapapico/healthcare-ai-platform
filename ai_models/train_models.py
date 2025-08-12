"""
AI 모델 학습 스크립트
가상 데이터를 사용하여 운동 분석 모델 학습
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ExerciseModelTrainer:
    def __init__(self):
        self.data_path = Path("data/virtual_athletes")
        self.model_path = Path("models")
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        self.exercises = ['push_up', 'squat', 'deadlift', 'plank']
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def load_data(self):
        """데이터 로드 및 전처리"""
        print("데이터 로딩 중...")
        
        all_data = []
        
        # 모든 JSON 파일 로드
        for json_file in self.data_path.glob("*.json"):
            if json_file.name == 'metadata.json':
                continue
                
            with open(json_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                
            # 각 프레임을 개별 샘플로 변환
            for frame in session_data['frames']:
                # 키포인트를 평면 배열로 변환
                keypoint_features = []
                for kp in frame['keypoints']:
                    keypoint_features.extend([kp['x'], kp['y'], kp['confidence']])
                
                sample = {
                    'exercise_type': session_data['exercise_type'],
                    'athlete_level': session_data['athlete_level'],
                    'phase': frame['phase'],
                    'form_score': frame['form_score'],
                    'keypoints': keypoint_features
                }
                all_data.append(sample)
        
        print(f"총 {len(all_data)}개 샘플 로드 완료")
        return pd.DataFrame(all_data)
    
    def prepare_features(self, df):
        """특징 추출 및 전처리"""
        print("특징 추출 중...")
        
        # 키포인트 특징을 개별 컬럼으로 확장
        keypoint_cols = []
        for i in range(17):  # 17개 키포인트
            for coord in ['x', 'y', 'conf']:
                col_name = f'kp_{i}_{coord}'
                df[col_name] = df['keypoints'].apply(lambda x: x[i*3 + ['x', 'y', 'conf'].index(coord)])
                keypoint_cols.append(col_name)
        
        # 추가 특징 계산
        # 1. 관절 각도
        df['elbow_angle_left'] = self.calculate_angle(df, 5, 7, 9)  # shoulder-elbow-wrist
        df['elbow_angle_right'] = self.calculate_angle(df, 6, 8, 10)
        df['knee_angle_left'] = self.calculate_angle(df, 11, 13, 15)  # hip-knee-ankle
        df['knee_angle_right'] = self.calculate_angle(df, 12, 14, 16)
        
        # 2. 신체 정렬
        df['spine_alignment'] = self.calculate_spine_alignment(df)
        df['shoulder_alignment'] = abs(df['kp_5_y'] - df['kp_6_y'])  # 어깨 수평
        df['hip_alignment'] = abs(df['kp_11_y'] - df['kp_12_y'])  # 엉덩이 수평
        
        # 3. 움직임 범위
        df['vertical_range'] = df.groupby(['exercise_type'])['kp_0_y'].transform(lambda x: x.max() - x.min())
        
        feature_cols = keypoint_cols + [
            'elbow_angle_left', 'elbow_angle_right',
            'knee_angle_left', 'knee_angle_right',
            'spine_alignment', 'shoulder_alignment', 'hip_alignment',
            'vertical_range'
        ]
        
        return df, feature_cols
    
    def calculate_angle(self, df, p1_idx, p2_idx, p3_idx):
        """세 점 사이의 각도 계산"""
        p1_x = df[f'kp_{p1_idx}_x']
        p1_y = df[f'kp_{p1_idx}_y']
        p2_x = df[f'kp_{p2_idx}_x']
        p2_y = df[f'kp_{p2_idx}_y']
        p3_x = df[f'kp_{p3_idx}_x']
        p3_y = df[f'kp_{p3_idx}_y']
        
        # 벡터 계산
        v1_x = p1_x - p2_x
        v1_y = p1_y - p2_y
        v2_x = p3_x - p2_x
        v2_y = p3_y - p2_y
        
        # 각도 계산 (라디안 -> 도)
        angle = np.arctan2(v2_y, v2_x) - np.arctan2(v1_y, v1_x)
        angle = np.abs(angle * 180 / np.pi)
        angle = np.where(angle > 180, 360 - angle, angle)
        
        return angle
    
    def calculate_spine_alignment(self, df):
        """척추 정렬 계산"""
        # 코, 엉덩이 중심 사이의 수직 정렬
        nose_x = df['kp_0_x']
        hip_center_x = (df['kp_11_x'] + df['kp_12_x']) / 2
        return abs(nose_x - hip_center_x)
    
    def build_model(self, input_dim, num_classes):
        """딥러닝 모델 구축"""
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            
            # 다중 출력
            layers.Dense(num_classes + 1)  # phase classes + form_score
        ])
        
        return model
    
    def train_exercise_model(self, df, exercise_type, feature_cols):
        """특정 운동 모델 학습"""
        print(f"\n{exercise_type} 모델 학습 중...")
        
        # 해당 운동 데이터만 필터링
        exercise_df = df[df['exercise_type'] == exercise_type].copy()
        
        if len(exercise_df) == 0:
            print(f"  {exercise_type} 데이터가 없습니다.")
            return None
        
        # 특징과 레이블 준비
        X = exercise_df[feature_cols].values
        
        # Phase 인코딩
        phase_encoder = LabelEncoder()
        y_phase = phase_encoder.fit_transform(exercise_df['phase'])
        y_form = exercise_df['form_score'].values / 100.0  # 0-1 정규화
        
        # 데이터 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 학습/검증 데이터 분할
        X_train, X_val, y_phase_train, y_phase_val, y_form_train, y_form_val = train_test_split(
            X_scaled, y_phase, y_form, test_size=0.2, random_state=42
        )
        
        print(f"  학습 데이터: {len(X_train)} 샘플")
        print(f"  검증 데이터: {len(X_val)} 샘플")
        
        # 모델 구축 및 컴파일
        num_phases = len(phase_encoder.classes_)
        model = self.build_model(X_train.shape[1], num_phases)
        
        # 커스텀 손실 함수 (phase 분류 + form score 회귀)
        def custom_loss(y_true, y_pred):
            # y_true: [phase_onehot..., form_score]
            # y_pred: [phase_logits..., form_score_pred]
            
            phase_loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true[:, 0], y_pred[:, :-1], from_logits=True
            )
            form_loss = tf.keras.losses.mse(y_true[:, 1], y_pred[:, -1])
            
            return phase_loss + 0.5 * form_loss
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=custom_loss,
            metrics=['accuracy']
        )
        
        # 학습 데이터 준비 (phase와 form score 결합)
        y_train = np.column_stack([y_phase_train, y_form_train])
        y_val = np.column_stack([y_phase_val, y_form_val])
        
        # 모델 학습
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        
        # 모델 평가
        val_loss = min(history.history['val_loss'])
        print(f"  검증 손실: {val_loss:.4f}")
        
        # 모델 및 전처리기 저장
        model_filename = self.model_path / f"{exercise_type}_model.h5"
        model.save(model_filename)
        
        scaler_filename = self.model_path / f"{exercise_type}_scaler.pkl"
        joblib.dump(scaler, scaler_filename)
        
        encoder_filename = self.model_path / f"{exercise_type}_encoder.pkl"
        joblib.dump(phase_encoder, encoder_filename)
        
        print(f"  모델 저장 완료: {model_filename}")
        
        return {
            'model': model,
            'scaler': scaler,
            'encoder': phase_encoder,
            'val_loss': val_loss
        }
    
    def train_all_models(self):
        """모든 운동 모델 학습"""
        print("="*50)
        print("AI 모델 학습 시작")
        print("="*50)
        
        # 데이터 로드
        df = self.load_data()
        
        # 특징 추출
        df, feature_cols = self.prepare_features(df)
        
        # 각 운동별 모델 학습
        results = {}
        for exercise in self.exercises:
            result = self.train_exercise_model(df, exercise, feature_cols)
            if result:
                results[exercise] = result
                self.models[exercise] = result['model']
                self.scalers[exercise] = result['scaler']
                self.encoders[exercise] = result['encoder']
        
        # 학습 결과 요약
        print("\n" + "="*50)
        print("학습 완료 요약")
        print("="*50)
        
        for exercise, result in results.items():
            print(f"{exercise}: 검증 손실 = {result['val_loss']:.4f}")
        
        # 메타데이터 저장
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'exercises': list(results.keys()),
            'performance': {ex: float(res['val_loss']) for ex, res in results.items()}
        }
        
        with open(self.model_path / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n모든 모델이 {self.model_path}에 저장되었습니다.")
        return results
    
    def test_inference(self, exercise_type='push_up'):
        """모델 추론 테스트"""
        if exercise_type not in self.models:
            print(f"{exercise_type} 모델이 없습니다.")
            return
        
        print(f"\n{exercise_type} 모델 추론 테스트...")
        
        # 테스트용 더미 데이터 생성
        dummy_keypoints = np.random.rand(17 * 3)
        dummy_angles = np.random.rand(8)
        dummy_features = np.concatenate([dummy_keypoints, dummy_angles])
        
        # 스케일링
        scaled_features = self.scalers[exercise_type].transform([dummy_features])
        
        # 예측
        prediction = self.models[exercise_type].predict(scaled_features, verbose=0)
        
        # Phase 예측
        phase_logits = prediction[0][:-1]
        phase_pred = np.argmax(phase_logits)
        phase_name = self.encoders[exercise_type].inverse_transform([phase_pred])[0]
        
        # Form score 예측
        form_score = prediction[0][-1] * 100
        
        print(f"  예측 단계: {phase_name}")
        print(f"  자세 점수: {form_score:.1f}%")

if __name__ == "__main__":
    trainer = ExerciseModelTrainer()
    results = trainer.train_all_models()
    
    # 추론 테스트
    for exercise in trainer.exercises:
        if exercise in trainer.models:
            trainer.test_inference(exercise)