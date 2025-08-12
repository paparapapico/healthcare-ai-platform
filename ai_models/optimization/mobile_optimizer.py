"""
모바일 배포를 위한 AI 모델 최적화
Mobile deployment optimization for AI models
"""

import os
import numpy as np
import tensorflow as tf
import coremltools as ct
from typing import Dict, List, Tuple, Optional
import json
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MobileModelOptimizer:
    """모바일 최적화 전문 클래스"""
    
    def __init__(self, model_path: str, target_platform: str = 'all'):
        """
        Args:
            model_path: Keras 모델 경로
            target_platform: 타겟 플랫폼 ('ios', 'android', 'all')
        """
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)
        self.target_platform = target_platform
        self.optimization_results = {}
    
    def optimize_for_mobile(self) -> Dict:
        """전체 모바일 최적화 파이프라인"""
        logger.info("Starting mobile optimization pipeline...")
        
        results = {}
        
        # 1. 모델 프루닝 (가지치기)
        pruned_model = self.prune_model()
        results['pruning'] = self.evaluate_model_size(pruned_model)
        
        # 2. 양자화 (Quantization)
        if self.target_platform in ['android', 'all']:
            tflite_models = self.quantize_for_tflite(pruned_model)
            results['tflite'] = tflite_models
        
        # 3. iOS Core ML 변환
        if self.target_platform in ['ios', 'all']:
            coreml_model = self.convert_to_coreml(pruned_model)
            results['coreml'] = coreml_model
        
        # 4. ONNX 변환 (크로스 플랫폼)
        onnx_model = self.convert_to_onnx(pruned_model)
        results['onnx'] = onnx_model
        
        # 5. 엣지 TPU 최적화 (Google Coral)
        edge_tpu_model = self.optimize_for_edge_tpu(pruned_model)
        results['edge_tpu'] = edge_tpu_model
        
        # 6. 성능 벤치마크
        results['benchmarks'] = self.benchmark_all_models(results)
        
        self.optimization_results = results
        return results
    
    def prune_model(self, sparsity: float = 0.5) -> tf.keras.Model:
        """모델 프루닝 (불필요한 가중치 제거)"""
        import tensorflow_model_optimization as tfmot
        
        logger.info(f"Pruning model with {sparsity*100}% sparsity...")
        
        # 프루닝 설정
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=sparsity,
                begin_step=0,
                end_step=1000
            )
        }
        
        # 각 레이어에 프루닝 적용
        def apply_pruning_to_layer(layer):
            if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv1D, tf.keras.layers.Conv2D)):
                return prune_low_magnitude(layer, **pruning_params)
            return layer
        
        # 모델 복제 및 프루닝 적용
        pruned_model = tf.keras.models.clone_model(
            self.model,
            clone_function=apply_pruning_to_layer
        )
        
        # 가중치 복사
        pruned_model.set_weights(self.model.get_weights())
        
        # 컴파일
        pruned_model.compile(
            optimizer='adam',
            loss=self.model.loss,
            metrics=self.model.metrics
        )
        
        return pruned_model
    
    def quantize_for_tflite(self, model: tf.keras.Model) -> Dict:
        """TensorFlow Lite 양자화 (다양한 방법)"""
        logger.info("Quantizing model for TensorFlow Lite...")
        
        results = {}
        
        # 1. Dynamic Range Quantization (가장 간단)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        dynamic_quant_model = converter.convert()
        results['dynamic'] = {
            'model': dynamic_quant_model,
            'size_mb': len(dynamic_quant_model) / 1024 / 1024
        }
        
        # 2. Integer Quantization (INT8)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.representative_dataset_gen
        
        # INT8 only
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        int8_model = converter.convert()
        results['int8'] = {
            'model': int8_model,
            'size_mb': len(int8_model) / 1024 / 1024
        }
        
        # 3. Float16 Quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        float16_model = converter.convert()
        results['float16'] = {
            'model': float16_model,
            'size_mb': len(float16_model) / 1024 / 1024
        }
        
        # 4. Hybrid Quantization (weights only)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        
        hybrid_model = converter.convert()
        results['hybrid'] = {
            'model': hybrid_model,
            'size_mb': len(hybrid_model) / 1024 / 1024
        }
        
        # 모델 저장
        os.makedirs('optimized_models/tflite', exist_ok=True)
        for name, data in results.items():
            with open(f'optimized_models/tflite/model_{name}.tflite', 'wb') as f:
                f.write(data['model'])
            logger.info(f"{name.upper()} model size: {data['size_mb']:.2f} MB")
        
        return results
    
    def representative_dataset_gen(self):
        """대표 데이터셋 생성 (양자화용)"""
        # 실제 데이터 분포를 대표하는 샘플 생성
        for _ in range(100):
            # 30 frames x 132 features (33 landmarks x 4)
            data = np.random.randn(1, 30, 132).astype(np.float32)
            yield [data]
    
    def convert_to_coreml(self, model: tf.keras.Model) -> Dict:
        """iOS Core ML 변환"""
        logger.info("Converting to Core ML...")
        
        try:
            # Core ML 변환
            mlmodel = ct.convert(
                model,
                convert_to="mlprogram",  # 최신 Core ML 형식
                inputs=[ct.TensorType(shape=(1, 30, 132))],
                minimum_deployment_target=ct.target.iOS15,
                compute_units=ct.ComputeUnit.ALL  # CPU, GPU, Neural Engine 모두 사용
            )
            
            # 메타데이터 추가
            mlmodel.author = "Olympic AI Training System"
            mlmodel.short_description = "Professional exercise analysis model"
            mlmodel.version = "1.0.0"
            
            # 모델 압축
            mlmodel_compressed = self.compress_coreml_model(mlmodel)
            
            # 저장
            os.makedirs('optimized_models/coreml', exist_ok=True)
            mlmodel.save('optimized_models/coreml/exercise_model.mlpackage')
            
            # 모델 크기 계산
            import shutil
            size_mb = shutil.disk_usage('optimized_models/coreml/exercise_model.mlpackage').used / 1024 / 1024
            
            return {
                'model': mlmodel,
                'compressed': mlmodel_compressed,
                'size_mb': size_mb,
                'compute_units': 'ALL',
                'ios_version': 'iOS 15+'
            }
            
        except Exception as e:
            logger.error(f"Core ML conversion failed: {e}")
            return {'error': str(e)}
    
    def compress_coreml_model(self, mlmodel):
        """Core ML 모델 압축"""
        # 가중치 압축 (16-bit)
        spec = mlmodel.get_spec()
        
        # Linear quantization
        from coremltools.models.neural_network.quantization_utils import quantize_weights
        compressed_model = quantize_weights(mlmodel, nbits=16)
        
        return compressed_model
    
    def convert_to_onnx(self, model: tf.keras.Model) -> Dict:
        """ONNX 변환 (크로스 플랫폼)"""
        logger.info("Converting to ONNX...")
        
        try:
            import tf2onnx
            
            # ONNX 변환
            spec = (tf.TensorSpec((None, 30, 132), tf.float32, name="input"),)
            
            model_proto, _ = tf2onnx.convert.from_keras(
                model,
                input_signature=spec,
                opset=13,  # ONNX opset version
                output_path='optimized_models/onnx/exercise_model.onnx'
            )
            
            # ONNX Runtime 최적화
            import onnxruntime as ort
            
            # 최적화 세션 옵션
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.optimized_model_filepath = 'optimized_models/onnx/exercise_model_opt.onnx'
            
            # 모델 크기
            size_mb = os.path.getsize('optimized_models/onnx/exercise_model.onnx') / 1024 / 1024
            
            return {
                'model_path': 'optimized_models/onnx/exercise_model.onnx',
                'optimized_path': 'optimized_models/onnx/exercise_model_opt.onnx',
                'size_mb': size_mb,
                'opset_version': 13
            }
            
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            return {'error': str(e)}
    
    def optimize_for_edge_tpu(self, model: tf.keras.Model) -> Dict:
        """Google Coral Edge TPU 최적화"""
        logger.info("Optimizing for Edge TPU...")
        
        # Edge TPU는 INT8 양자화가 필요
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        # Edge TPU 컴파일러용 모델 저장
        edge_tpu_path = 'optimized_models/edge_tpu/model_edgetpu.tflite'
        os.makedirs(os.path.dirname(edge_tpu_path), exist_ok=True)
        
        with open(edge_tpu_path, 'wb') as f:
            f.write(tflite_model)
        
        # Edge TPU 컴파일 명령 (실제 컴파일은 Edge TPU Compiler 필요)
        compile_command = f"edgetpu_compiler -s {edge_tpu_path}"
        
        return {
            'model_path': edge_tpu_path,
            'size_mb': len(tflite_model) / 1024 / 1024,
            'compile_command': compile_command,
            'note': 'Run Edge TPU Compiler to generate final model'
        }
    
    def benchmark_all_models(self, models: Dict) -> Dict:
        """모든 최적화 모델 벤치마크"""
        logger.info("Benchmarking optimized models...")
        
        benchmarks = {}
        
        # TFLite 벤치마크
        if 'tflite' in models:
            for variant, data in models['tflite'].items():
                bench_result = self.benchmark_tflite_model(data['model'])
                benchmarks[f'tflite_{variant}'] = bench_result
        
        # ONNX 벤치마크
        if 'onnx' in models and 'model_path' in models['onnx']:
            benchmarks['onnx'] = self.benchmark_onnx_model(models['onnx']['model_path'])
        
        # 원본 모델 벤치마크 (비교용)
        benchmarks['original'] = self.benchmark_keras_model(self.model)
        
        # 요약 통계
        benchmarks['summary'] = self.generate_benchmark_summary(benchmarks)
        
        return benchmarks
    
    def benchmark_tflite_model(self, model_content: bytes, num_runs: int = 100) -> Dict:
        """TFLite 모델 벤치마크"""
        import time
        
        # 인터프리터 생성
        interpreter = tf.lite.Interpreter(model_content=model_content)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # 더미 입력
        input_shape = input_details[0]['shape']
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # 웜업
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
        
        # 실제 벤치마크
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            times.append((time.perf_counter() - start) * 1000)  # ms
        
        return {
            'mean_latency_ms': np.mean(times),
            'std_latency_ms': np.std(times),
            'min_latency_ms': np.min(times),
            'max_latency_ms': np.max(times),
            'p50_latency_ms': np.percentile(times, 50),
            'p95_latency_ms': np.percentile(times, 95),
            'p99_latency_ms': np.percentile(times, 99),
            'fps': 1000 / np.mean(times)
        }
    
    def benchmark_onnx_model(self, model_path: str, num_runs: int = 100) -> Dict:
        """ONNX 모델 벤치마크"""
        import time
        import onnxruntime as ort
        
        # 세션 생성
        sess = ort.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name
        
        # 더미 입력
        input_data = np.random.randn(1, 30, 132).astype(np.float32)
        
        # 웜업
        for _ in range(10):
            sess.run(None, {input_name: input_data})
        
        # 실제 벤치마크
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            output = sess.run(None, {input_name: input_data})
            times.append((time.perf_counter() - start) * 1000)  # ms
        
        return {
            'mean_latency_ms': np.mean(times),
            'std_latency_ms': np.std(times),
            'min_latency_ms': np.min(times),
            'max_latency_ms': np.max(times),
            'p50_latency_ms': np.percentile(times, 50),
            'p95_latency_ms': np.percentile(times, 95),
            'p99_latency_ms': np.percentile(times, 99),
            'fps': 1000 / np.mean(times)
        }
    
    def benchmark_keras_model(self, model: tf.keras.Model, num_runs: int = 100) -> Dict:
        """Keras 모델 벤치마크"""
        import time
        
        # 더미 입력
        input_data = np.random.randn(1, 30, 132).astype(np.float32)
        
        # 웜업
        for _ in range(10):
            _ = model.predict(input_data, verbose=0)
        
        # 실제 벤치마크
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            output = model.predict(input_data, verbose=0)
            times.append((time.perf_counter() - start) * 1000)  # ms
        
        return {
            'mean_latency_ms': np.mean(times),
            'std_latency_ms': np.std(times),
            'min_latency_ms': np.min(times),
            'max_latency_ms': np.max(times),
            'p50_latency_ms': np.percentile(times, 50),
            'p95_latency_ms': np.percentile(times, 95),
            'p99_latency_ms': np.percentile(times, 99),
            'fps': 1000 / np.mean(times)
        }
    
    def generate_benchmark_summary(self, benchmarks: Dict) -> Dict:
        """벤치마크 요약 생성"""
        summary = {
            'best_latency': {},
            'best_fps': {},
            'size_comparison': {},
            'recommendation': ''
        }
        
        # 최고 성능 모델 찾기
        latencies = {k: v.get('mean_latency_ms', float('inf')) 
                    for k, v in benchmarks.items() if isinstance(v, dict) and 'mean_latency_ms' in v}
        
        if latencies:
            best_model = min(latencies, key=latencies.get)
            summary['best_latency'] = {
                'model': best_model,
                'latency_ms': latencies[best_model]
            }
        
        # FPS 비교
        fps_values = {k: v.get('fps', 0) 
                     for k, v in benchmarks.items() if isinstance(v, dict) and 'fps' in v}
        
        if fps_values:
            best_fps_model = max(fps_values, key=fps_values.get)
            summary['best_fps'] = {
                'model': best_fps_model,
                'fps': fps_values[best_fps_model]
            }
        
        # 추천
        if summary['best_fps'].get('fps', 0) >= 30:
            summary['recommendation'] = f"✅ {summary['best_fps']['model']} 모델이 실시간 처리에 적합합니다 (30+ FPS)"
        else:
            summary['recommendation'] = "⚠️ 추가 최적화가 필요합니다 (30 FPS 미달)"
        
        return summary
    
    def save_optimization_report(self, output_path: str = 'optimization_report.json'):
        """최적화 보고서 저장"""
        report = {
            'model_path': self.model_path,
            'target_platform': self.target_platform,
            'optimization_results': self.optimization_results,
            'timestamp': str(np.datetime64('now'))
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Optimization report saved to {output_path}")
        
        # Markdown 보고서도 생성
        self.generate_markdown_report(report, output_path.replace('.json', '.md'))
    
    def generate_markdown_report(self, report: Dict, output_path: str):
        """Markdown 형식 보고서 생성"""
        md_content = f"""# 모델 최적화 보고서

## 📊 개요
- **원본 모델**: {report['model_path']}
- **타겟 플랫폼**: {report['target_platform']}
- **생성 시간**: {report['timestamp']}

## 🎯 최적화 결과

### TensorFlow Lite
"""
        
        if 'tflite' in report['optimization_results']:
            for variant, data in report['optimization_results']['tflite'].items():
                md_content += f"- **{variant.upper()}**: {data['size_mb']:.2f} MB\n"
        
        md_content += "\n### 성능 벤치마크\n"
        
        if 'benchmarks' in report['optimization_results']:
            benchmarks = report['optimization_results']['benchmarks']
            if 'summary' in benchmarks:
                summary = benchmarks['summary']
                if 'best_fps' in summary:
                    md_content += f"- **최고 FPS**: {summary['best_fps']['model']} - {summary['best_fps']['fps']:.1f} FPS\n"
                if 'best_latency' in summary:
                    md_content += f"- **최저 지연시간**: {summary['best_latency']['model']} - {summary['best_latency']['latency_ms']:.2f} ms\n"
                if 'recommendation' in summary:
                    md_content += f"\n### 💡 추천사항\n{summary['recommendation']}\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Markdown report saved to {output_path}")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize model for mobile deployment')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to Keras model (.h5 or SavedModel)')
    parser.add_argument('--platform', type=str, default='all',
                       choices=['ios', 'android', 'all'],
                       help='Target platform')
    parser.add_argument('--output', type=str, default='optimization_report.json',
                       help='Output report path')
    
    args = parser.parse_args()
    
    # 최적화 실행
    optimizer = MobileModelOptimizer(args.model, args.platform)
    results = optimizer.optimize_for_mobile()
    
    # 보고서 저장
    optimizer.save_optimization_report(args.output)
    
    print("\n✅ Mobile optimization complete!")
    print(f"📊 Report saved to {args.output}")
    print("📁 Optimized models saved in 'optimized_models/' directory")
    
    # 요약 출력
    if 'benchmarks' in results and 'summary' in results['benchmarks']:
        summary = results['benchmarks']['summary']
        if 'recommendation' in summary:
            print(f"\n{summary['recommendation']}")


if __name__ == "__main__":
    main()