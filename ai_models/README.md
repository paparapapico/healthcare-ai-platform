# 🏆 전문 운동선수 수준 AI 모델 개발 가이드

## 📋 목차
1. [개요](#개요)
2. [필요한 기술 스택](#필요한-기술-스택)
3. [개발 단계](#개발-단계)
4. [구현 방법](#구현-방법)
5. [앱 통합 방법](#앱-통합-방법)

---

## 개요

전문 운동선수 수준의 AI를 만들기 위해서는 다음이 필요합니다:
- **대량의 전문가 데이터**: 올림픽 선수, 프로 운동선수들의 동작 데이터
- **고급 머신러닝 모델**: Pose Estimation + Action Recognition
- **실시간 처리 능력**: 모바일에서 30FPS 이상 처리

## 필요한 기술 스택

### 1. AI/ML 프레임워크
```bash
# 설치 명령어
pip install tensorflow==2.13.0
pip install torch torchvision
pip install mediapipe
pip install opencv-python
pip install numpy pandas scikit-learn
```

### 2. 데이터 수집 도구
- **Kinect SDK**: 3D 골격 데이터 수집
- **OpenPose**: 2D 포즈 추정
- **MediaPipe**: Google의 실시간 포즈 감지

### 3. 모델 최적화
- **TensorFlow Lite**: 모바일 배포용
- **ONNX Runtime**: 크로스 플랫폼 추론
- **Core ML**: iOS 최적화

## 개발 단계

### 🔸 Phase 1: 데이터 수집 (2-3개월)

#### 1.1 전문가 데이터셋 구축
```python
# data_collection/collect_expert_data.py

import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime

class ExpertDataCollector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 최고 정확도
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        self.collected_data = []
    
    def collect_from_video(self, video_path, athlete_info):
        """전문 선수 비디오에서 데이터 추출"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 포즈 감지
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                # 33개 랜드마크 추출
                landmarks = self.extract_landmarks(results.pose_landmarks)
                
                # 운동학적 특징 계산
                biomechanics = self.calculate_biomechanics(landmarks)
                
                self.collected_data.append({
                    'frame': frame_count,
                    'timestamp': frame_count / 30.0,  # 30 FPS 가정
                    'athlete': athlete_info,
                    'landmarks': landmarks,
                    'biomechanics': biomechanics
                })
            
            frame_count += 1
        
        cap.release()
        return self.collected_data
    
    def extract_landmarks(self, pose_landmarks):
        """3D 랜드마크 추출"""
        landmarks = []
        for landmark in pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        return landmarks
    
    def calculate_biomechanics(self, landmarks):
        """운동역학적 특징 계산"""
        features = {}
        
        # 관절 각도 계산
        features['elbow_angle_left'] = self.calculate_angle(
            landmarks[11], landmarks[13], landmarks[15]  # 어깨-팔꿈치-손목
        )
        features['elbow_angle_right'] = self.calculate_angle(
            landmarks[12], landmarks[14], landmarks[16]
        )
        features['knee_angle_left'] = self.calculate_angle(
            landmarks[23], landmarks[25], landmarks[27]  # 엉덩이-무릎-발목
        )
        features['knee_angle_right'] = self.calculate_angle(
            landmarks[24], landmarks[26], landmarks[28]
        )
        
        # 몸통 기울기
        features['torso_angle'] = self.calculate_torso_angle(landmarks)
        
        # 무게중심
        features['center_of_mass'] = self.calculate_com(landmarks)
        
        # 속도와 가속도 (프레임 간 차이)
        features['velocity'] = self.calculate_velocity(landmarks)
        
        return features
    
    def calculate_angle(self, p1, p2, p3):
        """3점 사이의 각도 계산"""
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])
        
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return np.degrees(angle)
    
    def save_dataset(self, filename):
        """수집한 데이터 저장"""
        with open(filename, 'w') as f:
            json.dump(self.collected_data, f, indent=2)
```

#### 1.2 데이터 라벨링
```python
# data_collection/label_data.py

class ExerciseLabeler:
    def __init__(self):
        self.exercise_phases = {
            'push_up': ['ready', 'descent', 'bottom', 'ascent', 'top'],
            'squat': ['standing', 'descent', 'bottom', 'ascent', 'standing'],
            'deadlift': ['setup', 'pull', 'lockout', 'descent'],
        }
        
        self.technique_scores = {
            'perfect': 100,
            'excellent': 90,
            'good': 80,
            'fair': 70,
            'poor': 60
        }
    
    def label_exercise_phase(self, landmarks, exercise_type):
        """운동 단계 자동 라벨링"""
        if exercise_type == 'push_up':
            elbow_angle = self.get_elbow_angle(landmarks)
            if elbow_angle > 160:
                return 'top'
            elif elbow_angle < 90:
                return 'bottom'
            # ... 추가 로직
    
    def label_technique_quality(self, biomechanics, exercise_type):
        """기술 품질 평가"""
        score = 100
        
        if exercise_type == 'squat':
            # 무릎이 발끝을 넘는지 확인
            if biomechanics['knee_over_toe']:
                score -= 10
            
            # 척추 정렬 확인
            if abs(biomechanics['spine_angle'] - 90) > 15:
                score -= 15
            
            # 무릎 각도 확인
            if biomechanics['knee_angle'] < 70:
                score -= 5  # 너무 깊음
            elif biomechanics['knee_angle'] > 100:
                score -= 10  # 충분히 내려가지 않음
        
        return score
```

### 🔸 Phase 2: AI 모델 개발 (2-3개월)

#### 2.1 딥러닝 모델 아키텍처
```python
# models/professional_exercise_model.py

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class ProfessionalExerciseAI:
    def __init__(self, exercise_type):
        self.exercise_type = exercise_type
        self.model = self.build_model()
        self.lstm_model = self.build_lstm_model()
        self.attention_model = self.build_attention_model()
    
    def build_model(self):
        """기본 CNN 모델 (공간적 특징 추출)"""
        model = models.Sequential([
            layers.Input(shape=(33, 4)),  # 33 landmarks x 4 features (x,y,z,visibility)
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu')
        ])
        return model
    
    def build_lstm_model(self):
        """시간적 패턴 학습을 위한 LSTM"""
        model = models.Sequential([
            layers.Input(shape=(30, 132)),  # 30 frames x 132 features
            layers.LSTM(256, return_sequences=True),
            layers.LSTM(128, return_sequences=True),
            layers.LSTM(64),
            layers.Dense(32, activation='relu'),
            layers.Dense(5, activation='softmax')  # 5 exercise phases
        ])
        return model
    
    def build_attention_model(self):
        """중요한 관절에 집중하는 Attention 메커니즘"""
        inputs = layers.Input(shape=(33, 4))
        
        # Multi-Head Attention
        attention = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=64
        )(inputs, inputs)
        
        # Feed Forward
        x = layers.LayerNormalization()(attention + inputs)
        ff = layers.Dense(256, activation='relu')(x)
        ff = layers.Dense(132)(ff)
        
        # Output layers
        x = layers.LayerNormalization()(ff + x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Multi-task outputs
        phase_output = layers.Dense(5, activation='softmax', name='phase')(x)
        quality_output = layers.Dense(1, activation='sigmoid', name='quality')(x)
        rep_count_output = layers.Dense(1, activation='linear', name='rep_count')(x)
        
        model = models.Model(
            inputs=inputs,
            outputs=[phase_output, quality_output, rep_count_output]
        )
        
        return model
    
    def train_model(self, train_data, val_data, epochs=100):
        """모델 학습"""
        self.attention_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'phase': 'categorical_crossentropy',
                'quality': 'mse',
                'rep_count': 'mse'
            },
            loss_weights={
                'phase': 1.0,
                'quality': 0.5,
                'rep_count': 0.3
            },
            metrics={
                'phase': 'accuracy',
                'quality': 'mae',
                'rep_count': 'mae'
            }
        )
        
        # 콜백 설정
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                f'best_model_{self.exercise_type}.h5',
                save_best_only=True
            )
        ]
        
        history = self.attention_model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks
        )
        
        return history
```

#### 2.2 전문가 수준 평가 시스템
```python
# models/expert_evaluator.py

class ExpertEvaluator:
    def __init__(self):
        self.olympic_standards = self.load_olympic_standards()
        self.technique_database = self.load_technique_database()
    
    def evaluate_like_coach(self, pose_sequence, exercise_type):
        """올림픽 코치처럼 평가"""
        evaluation = {
            'technical_score': 0,
            'power_score': 0,
            'rhythm_score': 0,
            'stability_score': 0,
            'detailed_feedback': [],
            'improvement_plan': []
        }
        
        # 1. 기술적 정확도 평가
        evaluation['technical_score'] = self.evaluate_technique(
            pose_sequence, 
            self.olympic_standards[exercise_type]
        )
        
        # 2. 파워와 폭발력 평가
        evaluation['power_score'] = self.evaluate_power(pose_sequence)
        
        # 3. 리듬과 템포 평가
        evaluation['rhythm_score'] = self.evaluate_rhythm(pose_sequence)
        
        # 4. 안정성과 균형 평가
        evaluation['stability_score'] = self.evaluate_stability(pose_sequence)
        
        # 5. 상세 피드백 생성
        evaluation['detailed_feedback'] = self.generate_expert_feedback(
            evaluation, exercise_type
        )
        
        # 6. 개선 계획 수립
        evaluation['improvement_plan'] = self.create_training_plan(
            evaluation, exercise_type
        )
        
        return evaluation
    
    def evaluate_technique(self, pose_sequence, olympic_standard):
        """올림픽 표준과 비교"""
        score = 100
        deductions = []
        
        for i, pose in enumerate(pose_sequence):
            # 각 프레임별 표준과의 차이 계산
            deviation = self.calculate_deviation(pose, olympic_standard[i])
            
            if deviation > 0.1:  # 10% 이상 차이
                score -= min(10, deviation * 50)
                deductions.append(f"Frame {i}: {deviation:.2%} deviation")
        
        return max(0, score)
    
    def evaluate_power(self, pose_sequence):
        """파워 메트릭 계산"""
        velocities = []
        accelerations = []
        
        for i in range(1, len(pose_sequence)):
            # 속도 계산
            velocity = self.calculate_velocity(
                pose_sequence[i-1], 
                pose_sequence[i]
            )
            velocities.append(velocity)
            
            # 가속도 계산
            if i > 1:
                acceleration = velocities[-1] - velocities[-2]
                accelerations.append(acceleration)
        
        # 폭발력 점수 (피크 가속도 기반)
        peak_acceleration = max(accelerations) if accelerations else 0
        power_score = min(100, peak_acceleration * 10)
        
        return power_score
    
    def generate_expert_feedback(self, evaluation, exercise_type):
        """전문가 수준 피드백 생성"""
        feedback = []
        
        total_score = np.mean([
            evaluation['technical_score'],
            evaluation['power_score'],
            evaluation['rhythm_score'],
            evaluation['stability_score']
        ])
        
        if total_score >= 90:
            feedback.append("올림픽 선수 수준의 완벽한 자세입니다!")
        elif total_score >= 80:
            feedback.append("프로 운동선수 수준입니다. 작은 디테일만 개선하면 완벽합니다.")
        elif total_score >= 70:
            feedback.append("상급자 수준입니다. 전문가의 지도를 받으면 큰 발전이 있을 것입니다.")
        else:
            feedback.append("기본기를 더 다져야 합니다.")
        
        # 구체적 피드백
        if evaluation['technical_score'] < 80:
            feedback.append("🎯 기술: 관절 각도와 신체 정렬을 개선하세요.")
        
        if evaluation['power_score'] < 80:
            feedback.append("💪 파워: 폭발적인 움직임을 더 연습하세요.")
        
        if evaluation['rhythm_score'] < 80:
            feedback.append("🎵 리듬: 일정한 템포를 유지하는 연습이 필요합니다.")
        
        if evaluation['stability_score'] < 80:
            feedback.append("⚖️ 안정성: 코어 근육을 강화하여 균형을 개선하세요.")
        
        return feedback
```

### 🔸 Phase 3: 모델 최적화 (1개월)

#### 3.1 모바일 최적화
```python
# optimization/mobile_optimizer.py

import tensorflow as tf
import coremltools as ct
import onnx

class MobileOptimizer:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
    
    def optimize_for_mobile(self):
        """모바일 배포를 위한 최적화"""
        
        # 1. 양자화 (Quantization)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        tflite_model = converter.convert()
        
        # 2. 모델 크기 확인
        model_size = len(tflite_model) / 1024 / 1024  # MB
        print(f"Optimized model size: {model_size:.2f} MB")
        
        # 3. iOS용 Core ML 변환
        mlmodel = ct.convert(
            self.model,
            inputs=[ct.ImageType(shape=(1, 224, 224, 3))],
            minimum_deployment_target=ct.target.iOS14
        )
        
        # 4. ONNX 변환 (크로스 플랫폼)
        onnx_model = tf2onnx.convert.from_keras(self.model)
        
        return {
            'tflite': tflite_model,
            'coreml': mlmodel,
            'onnx': onnx_model
        }
    
    def benchmark_performance(self, optimized_model):
        """성능 벤치마크"""
        import time
        
        # TFLite 인터프리터 설정
        interpreter = tf.lite.Interpreter(model_content=optimized_model)
        interpreter.allocate_tensors()
        
        # 추론 시간 측정
        times = []
        for _ in range(100):
            start = time.time()
            
            # 더미 입력
            input_data = np.random.random((1, 33, 4)).astype(np.float32)
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # ms
        fps = 1000 / avg_time
        
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"FPS: {fps:.1f}")
        
        return fps > 30  # 30 FPS 이상이면 실시간 처리 가능
```

### 🔸 Phase 4: 앱 통합 (1개월)

#### 4.1 React Native 통합
```typescript
// mobile/src/ai/ProfessionalAI.ts

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';

export class ProfessionalExerciseAI {
  private model: tf.LayersModel | null = null;
  private isReady: boolean = false;
  
  async initialize() {
    // TensorFlow.js 초기화
    await tf.ready();
    
    // 모델 로드
    this.model = await tf.loadLayersModel('assets://models/professional_ai.json');
    
    // 웜업 실행
    const dummy = tf.zeros([1, 33, 4]);
    await this.model.predict(dummy).data();
    dummy.dispose();
    
    this.isReady = true;
  }
  
  async analyzePose(landmarks: any[]): Promise<ExerciseAnalysis> {
    if (!this.isReady || !this.model) {
      throw new Error('Model not ready');
    }
    
    // 입력 텐서 생성
    const input = tf.tensor([landmarks]);
    
    // 추론 실행
    const predictions = await this.model.predict(input) as tf.Tensor[];
    
    // 결과 파싱
    const [phase, quality, repCount] = await Promise.all([
      predictions[0].data(),
      predictions[1].data(),
      predictions[2].data()
    ]);
    
    // 메모리 정리
    input.dispose();
    predictions.forEach(p => p.dispose());
    
    return {
      phase: this.getPhaseLabel(phase),
      quality: quality[0] * 100,
      repCount: Math.round(repCount[0]),
      feedback: this.generateFeedback(quality[0])
    };
  }
  
  private generateFeedback(quality: number): string {
    if (quality > 0.9) {
      return "완벽합니다! 올림픽 선수 같은 자세예요! 🏆";
    } else if (quality > 0.8) {
      return "훌륭해요! 프로 수준입니다! 💪";
    } else if (quality > 0.7) {
      return "좋아요! 조금만 더 개선하면 완벽해집니다!";
    } else {
      return "자세를 교정해주세요. 천천히 정확하게!";
    }
  }
}
```

#### 4.2 백엔드 API 통합
```python
# backend/app/api/v1/professional_ai.py

from fastapi import APIRouter, UploadFile, File
import numpy as np
from app.ai_models.professional_model import ProfessionalAI

router = APIRouter()
ai_model = ProfessionalAI()

@router.post("/analyze-professional")
async def analyze_professional(
    exercise_type: str,
    video: UploadFile = File(...)
):
    """전문가 수준 분석"""
    
    # 비디오에서 포즈 추출
    poses = await extract_poses_from_video(video)
    
    # AI 분석
    analysis = ai_model.analyze_sequence(poses, exercise_type)
    
    # 올림픽 선수와 비교
    comparison = ai_model.compare_with_olympic_standard(
        poses, 
        exercise_type
    )
    
    return {
        "overall_score": analysis['total_score'],
        "technique_score": analysis['technique_score'],
        "power_score": analysis['power_score'],
        "olympic_similarity": comparison['similarity_percentage'],
        "detailed_feedback": analysis['feedback'],
        "improvement_areas": analysis['improvements'],
        "training_recommendations": generate_training_plan(analysis)
    }
```

## 📦 폴더 구조

```
HealthcareAI/
├── ai_models/
│   ├── data_collection/
│   │   ├── collect_expert_data.py
│   │   ├── label_data.py
│   │   └── augment_data.py
│   ├── models/
│   │   ├── professional_exercise_model.py
│   │   ├── expert_evaluator.py
│   │   └── olympic_standards.json
│   ├── optimization/
│   │   ├── mobile_optimizer.py
│   │   └── performance_test.py
│   ├── training/
│   │   ├── train_model.py
│   │   ├── validate_model.py
│   │   └── hyperparameter_tuning.py
│   └── deployment/
│       ├── export_tflite.py
│       ├── export_coreml.py
│       └── export_onnx.py
```

## 🚀 시작하기

1. **환경 설정**
```bash
cd ai_models
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **데이터 수집**
```bash
python data_collection/collect_expert_data.py --video olympic_weightlifting.mp4
```

3. **모델 학습**
```bash
python training/train_model.py --exercise squat --epochs 100
```

4. **모바일 최적화**
```bash
python optimization/mobile_optimizer.py --model best_model_squat.h5
```

5. **배포**
```bash
python deployment/export_tflite.py --model optimized_model.h5
```

## 📊 예상 성능

- **정확도**: 95%+ (올림픽 선수 데이터 기준)
- **처리 속도**: 30+ FPS (모바일)
- **모델 크기**: < 10MB (양자화 후)
- **배터리 소모**: 최소화 (Edge AI)

## 🎯 핵심 차별점

1. **올림픽 수준 데이터**: 실제 올림픽 선수들의 동작 데이터 학습
2. **3D 분석**: 2D 영상에서 3D 동작 복원
3. **실시간 피드백**: 30FPS 이상 실시간 분석
4. **개인 맞춤**: 사용자별 진도에 따른 맞춤 피드백
5. **부상 예방**: 잘못된 자세 즉시 감지 및 경고

이 AI는 단순한 각도 측정을 넘어서 실제 올림픽 코치처럼 동작의 미묘한 차이까지 감지하고 피드백을 제공합니다!