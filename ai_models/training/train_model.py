"""
전문 운동선수 수준 AI 모델 학습 스크립트
Olympic-level exercise model training
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple, Optional
import wandb  # Weights & Biases for experiment tracking

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OlympicExerciseTrainer:
    """올림픽 수준 운동 AI 모델 트레이너"""
    
    def __init__(self, exercise_type: str, config_path: Optional[str] = None):
        """
        Args:
            exercise_type: 운동 종류 (squat, push_up, deadlift 등)
            config_path: 학습 설정 파일 경로
        """
        self.exercise_type = exercise_type
        self.config = self.load_config(config_path)
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
        # WandB 초기화 (실험 추적)
        if self.config.get('use_wandb', False):
            wandb.init(
                project="olympic-exercise-ai",
                name=f"{exercise_type}_training",
                config=self.config
            )
    
    def load_config(self, config_path: Optional[str]) -> Dict:
        """학습 설정 로드"""
        default_config = {
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'early_stopping_patience': 15,
            'reduce_lr_patience': 10,
            'sequence_length': 30,  # 30 frames = 1 second at 30 FPS
            'landmark_features': 132,  # 33 landmarks x 4 features
            'use_wandb': False,
            'data_augmentation': True,
            'model_architecture': 'transformer',  # 'lstm', 'cnn', 'transformer'
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                default_config.update(custom_config)
        
        return default_config
    
    def build_transformer_model(self) -> tf.keras.Model:
        """Transformer 기반 모델 구축 (최신 아키텍처)"""
        inputs = layers.Input(shape=(self.config['sequence_length'], self.config['landmark_features']))
        
        # Positional Encoding
        positions = tf.range(start=0, limit=self.config['sequence_length'], delta=1)
        position_embeddings = layers.Embedding(
            input_dim=self.config['sequence_length'],
            output_dim=self.config['landmark_features']
        )(positions)
        
        x = inputs + position_embeddings
        
        # Multi-Head Self-Attention Blocks
        for _ in range(3):  # 3개의 Transformer 블록
            # Multi-Head Attention
            attention_output = layers.MultiHeadAttention(
                num_heads=8,
                key_dim=64,
                dropout=0.1
            )(x, x)
            attention_output = layers.Dropout(0.1)(attention_output)
            x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # Feed Forward Network
            ffn_output = layers.Dense(512, activation='relu')(x)
            ffn_output = layers.Dense(self.config['landmark_features'])(ffn_output)
            ffn_output = layers.Dropout(0.1)(ffn_output)
            x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Task-specific heads
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Multi-task outputs
        outputs = {}
        
        # 1. Exercise Phase Classification (5 phases)
        phase_output = layers.Dense(64, activation='relu')(x)
        phase_output = layers.Dense(5, activation='softmax', name='phase')(phase_output)
        outputs['phase'] = phase_output
        
        # 2. Form Quality Score (0-100)
        quality_output = layers.Dense(32, activation='relu')(x)
        quality_output = layers.Dense(1, activation='sigmoid', name='quality')(quality_output)
        outputs['quality'] = quality_output
        
        # 3. Rep Counting
        rep_output = layers.Dense(32, activation='relu')(x)
        rep_output = layers.Dense(1, activation='linear', name='rep_count')(rep_output)
        outputs['rep_count'] = rep_output
        
        # 4. Mistake Detection (multi-label)
        mistake_output = layers.Dense(64, activation='relu')(x)
        mistake_output = layers.Dense(10, activation='sigmoid', name='mistakes')(mistake_output)
        outputs['mistakes'] = mistake_output
        
        # 5. Injury Risk Assessment
        injury_output = layers.Dense(32, activation='relu')(x)
        injury_output = layers.Dense(3, activation='softmax', name='injury_risk')(injury_output)
        outputs['injury_risk'] = injury_output
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def build_lstm_model(self) -> tf.keras.Model:
        """LSTM 기반 시계열 모델"""
        inputs = layers.Input(shape=(self.config['sequence_length'], self.config['landmark_features']))
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(64))(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Multi-task outputs (같은 구조)
        outputs = self._create_output_heads(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_cnn_lstm_model(self) -> tf.keras.Model:
        """CNN + LSTM 하이브리드 모델"""
        inputs = layers.Input(shape=(self.config['sequence_length'], self.config['landmark_features']))
        
        # 1D CNN for feature extraction
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        # LSTM for temporal modeling
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.LSTM(64)(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Multi-task outputs
        outputs = self._create_output_heads(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def _create_output_heads(self, x):
        """공통 출력 헤드 생성"""
        outputs = {}
        
        # Phase classification
        phase_output = layers.Dense(64, activation='relu')(x)
        phase_output = layers.Dense(5, activation='softmax', name='phase')(phase_output)
        outputs['phase'] = phase_output
        
        # Quality score
        quality_output = layers.Dense(32, activation='relu')(x)
        quality_output = layers.Dense(1, activation='sigmoid', name='quality')(quality_output)
        outputs['quality'] = quality_output
        
        # Rep counting
        rep_output = layers.Dense(32, activation='relu')(x)
        rep_output = layers.Dense(1, activation='linear', name='rep_count')(rep_output)
        outputs['rep_count'] = rep_output
        
        # Mistake detection
        mistake_output = layers.Dense(64, activation='relu')(x)
        mistake_output = layers.Dense(10, activation='sigmoid', name='mistakes')(mistake_output)
        outputs['mistakes'] = mistake_output
        
        # Injury risk
        injury_output = layers.Dense(32, activation='relu')(x)
        injury_output = layers.Dense(3, activation='softmax', name='injury_risk')(injury_output)
        outputs['injury_risk'] = injury_output
        
        return outputs
    
    def prepare_data(self, data_path: str) -> Tuple:
        """데이터 준비 및 전처리"""
        logger.info(f"Loading data from {data_path}")
        
        # 데이터 로드
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # 시퀀스 데이터 생성
        X, y = self.create_sequences(raw_data)
        
        # 데이터 정규화
        X_shape = X.shape
        X_flat = X.reshape(-1, X_shape[-1])
        X_flat = self.scaler.fit_transform(X_flat)
        X = X_flat.reshape(X_shape)
        
        # 학습/검증/테스트 분할
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y['phase']
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp['phase']
        )
        
        logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # 데이터 증강 (선택적)
        if self.config.get('data_augmentation', False):
            X_train, y_train = self.augment_data(X_train, y_train)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_sequences(self, raw_data: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """원시 데이터에서 시퀀스 생성"""
        sequences = []
        labels = {
            'phase': [],
            'quality': [],
            'rep_count': [],
            'mistakes': [],
            'injury_risk': []
        }
        
        sequence_length = self.config['sequence_length']
        
        for i in range(len(raw_data) - sequence_length):
            # 시퀀스 추출
            sequence = []
            for j in range(i, i + sequence_length):
                frame_data = raw_data[j]
                # 랜드마크 데이터를 플랫하게 변환
                landmarks = frame_data['landmarks']
                flat_landmarks = []
                for landmark in landmarks:
                    flat_landmarks.extend([
                        landmark['x'],
                        landmark['y'],
                        landmark['z'],
                        landmark['visibility']
                    ])
                sequence.append(flat_landmarks)
            
            sequences.append(sequence)
            
            # 라벨 추출 (마지막 프레임 기준)
            last_frame = raw_data[i + sequence_length - 1]
            labels['phase'].append(self.encode_phase(last_frame.get('phase', 'ready')))
            labels['quality'].append(last_frame.get('quality_score', 80) / 100.0)
            labels['rep_count'].append(last_frame.get('rep_count', 0))
            labels['mistakes'].append(self.encode_mistakes(last_frame.get('mistakes', [])))
            labels['injury_risk'].append(self.encode_injury_risk(last_frame.get('injury_risk', 'low')))
        
        X = np.array(sequences)
        y = {
            'phase': np.array(labels['phase']),
            'quality': np.array(labels['quality']).reshape(-1, 1),
            'rep_count': np.array(labels['rep_count']).reshape(-1, 1),
            'mistakes': np.array(labels['mistakes']),
            'injury_risk': np.array(labels['injury_risk'])
        }
        
        return X, y
    
    def encode_phase(self, phase: str) -> np.ndarray:
        """운동 단계 원-핫 인코딩"""
        phases = ['ready', 'descent', 'bottom', 'ascent', 'top']
        encoded = np.zeros(5)
        if phase in phases:
            encoded[phases.index(phase)] = 1
        return encoded
    
    def encode_mistakes(self, mistakes: List[str]) -> np.ndarray:
        """실수 멀티-라벨 인코딩"""
        mistake_types = [
            'knee_valgus', 'forward_lean', 'heel_rise', 'back_round',
            'elbow_flare', 'partial_rom', 'asymmetry', 'speed_issue',
            'breathing', 'stability'
        ]
        encoded = np.zeros(10)
        for mistake in mistakes:
            if mistake in mistake_types:
                encoded[mistake_types.index(mistake)] = 1
        return encoded
    
    def encode_injury_risk(self, risk: str) -> np.ndarray:
        """부상 위험도 원-핫 인코딩"""
        risk_levels = ['low', 'medium', 'high']
        encoded = np.zeros(3)
        if risk in risk_levels:
            encoded[risk_levels.index(risk)] = 1
        return encoded
    
    def augment_data(self, X: np.ndarray, y: Dict) -> Tuple[np.ndarray, Dict]:
        """데이터 증강"""
        augmented_X = []
        augmented_y = {key: [] for key in y.keys()}
        
        for i in range(len(X)):
            # 원본 데이터
            augmented_X.append(X[i])
            for key in y.keys():
                augmented_y[key].append(y[key][i])
            
            # 노이즈 추가
            noisy = X[i] + np.random.normal(0, 0.01, X[i].shape)
            augmented_X.append(noisy)
            for key in y.keys():
                augmented_y[key].append(y[key][i])
            
            # 시간축 스케일링 (속도 변화)
            if np.random.random() > 0.5:
                scaled = self.time_warp(X[i], factor=np.random.uniform(0.9, 1.1))
                augmented_X.append(scaled)
                for key in y.keys():
                    augmented_y[key].append(y[key][i])
        
        return np.array(augmented_X), {k: np.array(v) for k, v in augmented_y.items()}
    
    def time_warp(self, sequence: np.ndarray, factor: float) -> np.ndarray:
        """시간축 워핑 (속도 변화 시뮬레이션)"""
        from scipy.interpolate import interp1d
        
        old_length = len(sequence)
        old_indices = np.arange(old_length)
        new_length = int(old_length * factor)
        new_indices = np.linspace(0, old_length - 1, new_length)
        
        # 각 특징에 대해 보간
        warped = []
        for feature_idx in range(sequence.shape[1]):
            f = interp1d(old_indices, sequence[:, feature_idx], kind='linear')
            warped_feature = f(new_indices)
            warped.append(warped_feature)
        
        warped = np.array(warped).T
        
        # 원래 길이로 패딩 또는 자르기
        if new_length < old_length:
            # 패딩
            padding = old_length - new_length
            warped = np.pad(warped, ((0, padding), (0, 0)), mode='edge')
        else:
            # 자르기
            warped = warped[:old_length]
        
        return warped
    
    def train(self, train_data: Tuple, val_data: Tuple, test_data: Tuple):
        """모델 학습"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # 모델 구축
        if self.config['model_architecture'] == 'transformer':
            self.model = self.build_transformer_model()
        elif self.config['model_architecture'] == 'lstm':
            self.model = self.build_lstm_model()
        elif self.config['model_architecture'] == 'cnn_lstm':
            self.model = self.build_cnn_lstm_model()
        else:
            raise ValueError(f"Unknown architecture: {self.config['model_architecture']}")
        
        # 컴파일
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss={
                'phase': 'categorical_crossentropy',
                'quality': 'mse',
                'rep_count': 'mse',
                'mistakes': 'binary_crossentropy',
                'injury_risk': 'categorical_crossentropy'
            },
            loss_weights={
                'phase': 1.0,
                'quality': 0.8,
                'rep_count': 0.5,
                'mistakes': 0.7,
                'injury_risk': 0.6
            },
            metrics={
                'phase': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2)],
                'quality': ['mae'],
                'rep_count': ['mae'],
                'mistakes': ['binary_accuracy', tf.keras.metrics.AUC()],
                'injury_risk': ['accuracy']
            }
        )
        
        # 콜백 설정
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                f'models/best_{self.exercise_type}_model.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            callbacks.TensorBoard(
                log_dir=f'logs/{self.exercise_type}',
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        
        # WandB 콜백 추가
        if self.config.get('use_wandb', False):
            callbacks_list.append(wandb.keras.WandbCallback())
        
        # 학습
        logger.info("Starting training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        # 테스트 평가
        logger.info("Evaluating on test set...")
        test_results = self.model.evaluate(X_test, y_test, verbose=1)
        
        # 결과 저장
        self.save_results(test_results)
        
        return self.history
    
    def save_results(self, test_results: List):
        """학습 결과 저장"""
        results = {
            'exercise_type': self.exercise_type,
            'config': self.config,
            'test_metrics': {},
            'training_history': {
                'loss': self.history.history['loss'],
                'val_loss': self.history.history['val_loss']
            }
        }
        
        # 테스트 메트릭 저장
        metric_names = self.model.metrics_names
        for i, metric_name in enumerate(metric_names):
            results['test_metrics'][metric_name] = float(test_results[i])
        
        # JSON 파일로 저장
        os.makedirs('results', exist_ok=True)
        with open(f'results/{self.exercise_type}_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to results/{self.exercise_type}_training_results.json")
        
        # 모델 요약 저장
        with open(f'results/{self.exercise_type}_model_summary.txt', 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    def export_model(self, format: str = 'tflite'):
        """모델 내보내기"""
        if not self.model:
            raise ValueError("No trained model to export")
        
        os.makedirs('exported_models', exist_ok=True)
        
        if format == 'tflite':
            # TensorFlow Lite 변환
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            # 저장
            with open(f'exported_models/{self.exercise_type}_model.tflite', 'wb') as f:
                f.write(tflite_model)
            
            # 모델 크기 출력
            size_mb = len(tflite_model) / 1024 / 1024
            logger.info(f"TFLite model exported: {size_mb:.2f} MB")
            
        elif format == 'onnx':
            # ONNX 변환
            import tf2onnx
            
            spec = (tf.TensorSpec((None, self.config['sequence_length'], 
                                 self.config['landmark_features']), tf.float32),)
            
            model_proto, _ = tf2onnx.convert.from_keras(
                self.model,
                input_signature=spec,
                output_path=f'exported_models/{self.exercise_type}_model.onnx'
            )
            
            logger.info("ONNX model exported")
            
        elif format == 'saved_model':
            # TensorFlow SavedModel 형식
            self.model.save(f'exported_models/{self.exercise_type}_saved_model')
            logger.info("SavedModel exported")
        
        else:
            raise ValueError(f"Unknown export format: {format}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='Train Olympic-level exercise AI model')
    parser.add_argument('--exercise', type=str, required=True,
                       choices=['squat', 'push_up', 'deadlift', 'bench_press', 'plank'],
                       help='Exercise type to train')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data JSON file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config JSON file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--architecture', type=str, default='transformer',
                       choices=['transformer', 'lstm', 'cnn_lstm'],
                       help='Model architecture to use')
    parser.add_argument('--export', type=str, default='tflite',
                       choices=['tflite', 'onnx', 'saved_model'],
                       help='Export format for the trained model')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases for experiment tracking')
    
    args = parser.parse_args()
    
    # 설정 업데이트
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # 커맨드라인 인자로 설정 오버라이드
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'model_architecture': args.architecture,
        'use_wandb': args.wandb
    })
    
    # 트레이너 초기화
    trainer = OlympicExerciseTrainer(args.exercise, config)
    
    # 데이터 준비
    train_data, val_data, test_data = trainer.prepare_data(args.data)
    
    # 모델 학습
    history = trainer.train(train_data, val_data, test_data)
    
    # 모델 내보내기
    trainer.export_model(args.export)
    
    print(f"\n✅ Training complete for {args.exercise}!")
    print(f"📊 Model exported as {args.export} format")
    print(f"📁 Check 'results/' and 'exported_models/' directories for outputs")


if __name__ == "__main__":
    main()