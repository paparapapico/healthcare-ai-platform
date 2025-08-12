"""
모바일용 모델 최적화 스크립트
TensorFlow Lite로 변환하여 모바일 디바이스에서 효율적으로 실행
"""

import tensorflow as tf
from pathlib import Path
import json
import numpy as np

class MobileModelOptimizer:
    def __init__(self):
        self.model_path = Path("models")
        self.mobile_path = Path("models/mobile")
        self.mobile_path.mkdir(parents=True, exist_ok=True)
        
        self.exercises = ['push_up', 'squat', 'deadlift', 'plank']
        
    def convert_to_tflite(self, model_path, output_path, quantize=True):
        """TensorFlow Lite로 모델 변환"""
        print(f"변환 중: {model_path} -> {output_path}")
        
        # 모델 로드
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'custom_loss': lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred))}
        )
        
        # TFLite 변환기 생성
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantize:
            # 양자화 설정 (모델 크기 감소)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        # 변환
        tflite_model = converter.convert()
        
        # 저장
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # 모델 크기 비교
        original_size = model_path.stat().st_size / (1024 * 1024)  # MB
        optimized_size = output_path.stat().st_size / (1024 * 1024)  # MB
        compression_ratio = (1 - optimized_size / original_size) * 100
        
        print(f"  원본 크기: {original_size:.2f} MB")
        print(f"  최적화 크기: {optimized_size:.2f} MB")
        print(f"  압축률: {compression_ratio:.1f}%")
        
        return {
            'original_size': original_size,
            'optimized_size': optimized_size,
            'compression_ratio': compression_ratio
        }
    
    def test_tflite_inference(self, tflite_path, input_shape):
        """TFLite 모델 추론 테스트"""
        # 인터프리터 생성
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        # 입출력 텐서 정보
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # 테스트 입력 생성
        test_input = np.random.randn(1, input_shape).astype(np.float32)
        
        # 추론 실행
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        # 결과 가져오기
        output = interpreter.get_tensor(output_details[0]['index'])
        
        return output
    
    def optimize_all_models(self):
        """모든 모델 최적화"""
        print("="*50)
        print("모바일 최적화 시작")
        print("="*50)
        
        results = {}
        
        for exercise in self.exercises:
            h5_path = self.model_path / f"{exercise}_model.h5"
            
            if not h5_path.exists():
                print(f"\n{exercise} 모델이 없습니다. 건너뜁니다.")
                continue
            
            print(f"\n{exercise} 모델 최적화 중...")
            
            # TFLite 변환 (양자화 포함)
            tflite_path = self.mobile_path / f"{exercise}_model.tflite"
            result = self.convert_to_tflite(h5_path, tflite_path, quantize=True)
            
            # 추론 테스트
            print("  추론 테스트 중...")
            try:
                # 입력 크기 추정 (키포인트 51개 + 추가 특징 8개)
                input_dim = 51 + 8
                output = self.test_tflite_inference(tflite_path, input_dim)
                print(f"  추론 성공! 출력 shape: {output.shape}")
                result['inference_success'] = True
            except Exception as e:
                print(f"  추론 실패: {e}")
                result['inference_success'] = False
            
            results[exercise] = result
            
            # Float32 버전도 생성 (호환성을 위해)
            tflite_fp32_path = self.mobile_path / f"{exercise}_model_fp32.tflite"
            print(f"\n  Float32 버전 생성 중...")
            fp32_result = self.convert_to_tflite(h5_path, tflite_fp32_path, quantize=False)
            results[f"{exercise}_fp32"] = fp32_result
        
        # 최적화 결과 저장
        print("\n" + "="*50)
        print("최적화 결과")
        print("="*50)
        
        total_original = 0
        total_optimized = 0
        
        for name, result in results.items():
            if 'fp32' not in name:
                print(f"\n{name}:")
                print(f"  원본: {result['original_size']:.2f} MB")
                print(f"  최적화: {result['optimized_size']:.2f} MB")
                print(f"  압축률: {result['compression_ratio']:.1f}%")
                
                total_original += result['original_size']
                total_optimized += result['optimized_size']
        
        print(f"\n전체:")
        print(f"  총 원본 크기: {total_original:.2f} MB")
        print(f"  총 최적화 크기: {total_optimized:.2f} MB")
        print(f"  평균 압축률: {(1 - total_optimized/total_original)*100:.1f}%")
        
        # 메타데이터 저장
        metadata = {
            'optimized_at': str(Path.cwd()),
            'models': list(results.keys()),
            'total_original_size_mb': total_original,
            'total_optimized_size_mb': total_optimized,
            'average_compression_ratio': (1 - total_optimized/total_original)*100
        }
        
        with open(self.mobile_path / 'optimization_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n최적화된 모델이 {self.mobile_path}에 저장되었습니다.")
        return results

def create_model_config():
    """모바일 앱용 모델 설정 파일 생성"""
    config = {
        'models': {
            'push_up': {
                'tflite_path': 'models/mobile/push_up_model.tflite',
                'input_shape': [1, 59],  # 51 keypoints + 8 features
                'phases': ['ready', 'down', 'up'],
                'min_confidence': 0.7
            },
            'squat': {
                'tflite_path': 'models/mobile/squat_model.tflite',
                'input_shape': [1, 59],
                'phases': ['standing', 'down', 'up'],
                'min_confidence': 0.7
            },
            'deadlift': {
                'tflite_path': 'models/mobile/deadlift_model.tflite',
                'input_shape': [1, 59],
                'phases': ['ready', 'lift', 'lock', 'lower'],
                'min_confidence': 0.7
            },
            'plank': {
                'tflite_path': 'models/mobile/plank_model.tflite',
                'input_shape': [1, 59],
                'phases': ['holding'],
                'min_confidence': 0.7
            }
        },
        'preprocessing': {
            'normalize_keypoints': True,
            'calculate_angles': True,
            'smooth_predictions': True,
            'smoothing_window': 3
        },
        'feedback': {
            'update_interval_ms': 500,
            'rep_count_threshold': 0.8,
            'form_warning_threshold': 60
        }
    }
    
    config_path = Path('models/mobile/model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"모델 설정 파일 생성: {config_path}")
    return config

if __name__ == "__main__":
    # 모델 최적화
    optimizer = MobileModelOptimizer()
    results = optimizer.optimize_all_models()
    
    # 설정 파일 생성
    config = create_model_config()
    
    print("\n모바일 최적화 완료!")