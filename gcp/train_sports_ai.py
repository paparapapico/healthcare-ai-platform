"""
GCP에서 실행할 Sports AI 학습 메인 스크립트
Main Training Script for Sports AI on GCP
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import wandb
from google.cloud import storage
import time
from datetime import datetime
import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GCPTrainingManager:
    """GCP 학습 관리자"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler()  # Mixed Precision
        
        # GCS 설정
        self.bucket_name = 'sports-ai-data-bucket'
        self.checkpoint_bucket = 'sports-ai-data-bucket-checkpoints'
        self.model_bucket = 'sports-ai-data-bucket-models'
        
        # WandB 초기화
        if not args.no_wandb:
            wandb.init(
                project="sports-ai-training",
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args)
            )
        
        logger.info(f"🚀 학습 시작 - Device: {self.device}")
        self._log_system_info()
    
    def _log_system_info(self):
        """시스템 정보 로깅"""
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            logger.info(f"CUDA: {torch.version.cuda}")
            logger.info(f"cuDNN: {torch.backends.cudnn.version()}")
    
    def train_phase_1_experiments(self):
        """Phase 1: T4로 실험 및 테스트 ($50 예산)"""
        logger.info("="*50)
        logger.info("📊 Phase 1: 실험 단계 시작 (Tesla T4)")
        logger.info("="*50)
        
        # 작은 데이터셋으로 시작
        experiments = [
            {
                'name': 'lightweight_model',
                'batch_size': 16,
                'learning_rate': 1e-4,
                'epochs': 10,
                'data_fraction': 0.01  # 1% 데이터
            },
            {
                'name': 'medium_model',
                'batch_size': 8,
                'learning_rate': 5e-5,
                'epochs': 10,
                'data_fraction': 0.05  # 5% 데이터
            },
            {
                'name': 'transfer_learning',
                'batch_size': 12,
                'learning_rate': 2e-5,
                'epochs': 5,
                'data_fraction': 0.1  # 10% 데이터
            }
        ]
        
        best_config = None
        best_accuracy = 0
        
        for exp in experiments:
            logger.info(f"\n🔬 실험: {exp['name']}")
            
            # 모델 생성
            model = self._create_model(exp['name'])
            
            # 데이터 로더
            train_loader, val_loader = self._prepare_data(
                batch_size=exp['batch_size'],
                data_fraction=exp['data_fraction']
            )
            
            # 학습
            accuracy = self._train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=exp['epochs'],
                learning_rate=exp['learning_rate'],
                experiment_name=exp['name']
            )
            
            # 최고 성능 기록
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = exp
                self._save_checkpoint(model, f"best_phase1_{exp['name']}.pth")
            
            # 메모리 정리
            del model
            torch.cuda.empty_cache()
            
            # 비용 추적
            self._log_cost_estimate('t4', hours=2)
        
        logger.info(f"\n✅ Phase 1 완료!")
        logger.info(f"최고 설정: {best_config['name']}")
        logger.info(f"최고 정확도: {best_accuracy:.2%}")
        
        return best_config
    
    def train_phase_2_full_training(self, best_config):
        """Phase 2: V100으로 본격 학습 ($200 예산)"""
        logger.info("="*50)
        logger.info("🚀 Phase 2: 본격 학습 (Tesla V100)")
        logger.info("="*50)
        
        # 스포츠별 학습
        sports = ['basketball', 'soccer', 'bodyweight', 'golf']
        trained_models = {}
        
        for sport in sports:
            logger.info(f"\n🏃 {sport.upper()} 모델 학습 시작")
            
            # 전체 데이터 사용
            model = self._create_model('full_model')
            
            # 스포츠별 데이터 로드
            train_loader, val_loader = self._prepare_sport_data(
                sport=sport,
                batch_size=32,  # V100은 더 큰 배치 가능
                data_fraction=1.0  # 100% 데이터
            )
            
            # 학습
            accuracy = self._train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=50,
                learning_rate=1e-4,
                experiment_name=f"{sport}_full"
            )
            
            # 모델 저장
            model_path = f"models/{sport}_model_v100.pth"
            self._save_to_gcs(model, model_path)
            trained_models[sport] = accuracy
            
            logger.info(f"✅ {sport} 완료: {accuracy:.2%}")
            
            # 메모리 정리
            del model
            torch.cuda.empty_cache()
            
            # 비용 추적
            self._log_cost_estimate('v100', hours=18)
        
        logger.info(f"\n✅ Phase 2 완료!")
        for sport, acc in trained_models.items():
            logger.info(f"{sport}: {acc:.2%}")
        
        return trained_models
    
    def train_phase_3_optimization(self, trained_models):
        """Phase 3: 최종 최적화 및 배포 준비 ($50 예산)"""
        logger.info("="*50)
        logger.info("⚡ Phase 3: 최적화 단계 (Tesla T4)")
        logger.info("="*50)
        
        optimization_tasks = [
            'model_ensemble',
            'quantization',
            'tensorrt_optimization',
            'mobile_conversion'
        ]
        
        for task in optimization_tasks:
            logger.info(f"\n🔧 {task} 진행 중...")
            
            if task == 'model_ensemble':
                ensemble_model = self._create_ensemble(trained_models)
                self._save_to_gcs(ensemble_model, "models/ensemble_model.pth")
                
            elif task == 'quantization':
                for sport in trained_models.keys():
                    quantized = self._quantize_model(f"models/{sport}_model_v100.pth")
                    self._save_to_gcs(quantized, f"models/{sport}_quantized.pth")
            
            elif task == 'tensorrt_optimization':
                self._optimize_tensorrt(trained_models)
            
            elif task == 'mobile_conversion':
                self._convert_to_mobile(trained_models)
            
            # 비용 추적
            self._log_cost_estimate('t4', hours=3)
        
        logger.info("\n✅ Phase 3 완료!")
        logger.info("모든 모델이 배포 준비 완료되었습니다!")
    
    def _create_model(self, model_type):
        """모델 생성"""
        if model_type == 'lightweight_model':
            from ai_models.models.unified_sports_ai_model import UnifiedSportsAIModel
            model = UnifiedSportsAIModel(
                sports_types=['basketball', 'soccer'],
                architecture='lightweight'
            ).build_unified_model()
        
        elif model_type == 'full_model':
            from ai_models.models.unified_sports_ai_model import UnifiedSportsAIModel
            model = UnifiedSportsAIModel(
                sports_types=['basketball', 'soccer', 'bodyweight', 'golf'],
                architecture='hybrid_ensemble'
            ).build_unified_model()
        
        else:  # transfer_learning
            # 사전학습 모델 로드
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
            # 마지막 레이어 수정
            model.fc = nn.Linear(2048, 100)  # 100개 클래스
        
        return model.to(self.device)
    
    def _train_model(self, model, train_loader, val_loader, epochs, learning_rate, experiment_name):
        """모델 학습"""
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        best_accuracy = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Mixed Precision Training
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Backward
                self.scaler.scale(loss).backward()
                
                # Gradient Accumulation
                if (batch_idx + 1) % self.args.accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'loss': train_loss/(batch_idx+1),
                    'acc': 100.*train_correct/train_total
                })
            
            # Validation
            accuracy = self._validate(model, val_loader)
            
            # Logging
            if not self.args.no_wandb:
                wandb.log({
                    f'{experiment_name}/train_loss': train_loss/len(train_loader),
                    f'{experiment_name}/train_acc': 100.*train_correct/train_total,
                    f'{experiment_name}/val_acc': accuracy,
                    'epoch': epoch
                })
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self._save_checkpoint(model, f"{experiment_name}_best.pth")
            
            # Preemptible 대비 주기적 저장
            if epoch % 5 == 0:
                self._save_checkpoint(model, f"{experiment_name}_epoch_{epoch}.pth")
        
        return best_accuracy
    
    def _validate(self, model, val_loader):
        """검증"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        logger.info(f"Validation Accuracy: {accuracy:.2f}%")
        return accuracy
    
    def _save_checkpoint(self, model, filename):
        """체크포인트 저장"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat(),
            'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
        }
        
        # 로컬 저장
        local_path = Path('checkpoints') / filename
        local_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, local_path)
        
        # GCS 업로드
        self._upload_to_gcs(local_path, f"checkpoints/{filename}")
        logger.info(f"✅ 체크포인트 저장: {filename}")
    
    def _save_to_gcs(self, model, path):
        """GCS에 모델 저장"""
        local_path = Path(path)
        local_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), local_path)
        self._upload_to_gcs(local_path, path)
    
    def _upload_to_gcs(self, local_path, gcs_path):
        """GCS 업로드"""
        client = storage.Client()
        bucket = client.bucket(self.checkpoint_bucket)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))
        logger.info(f"☁️ GCS 업로드: gs://{self.checkpoint_bucket}/{gcs_path}")
    
    def _log_cost_estimate(self, gpu_type, hours):
        """비용 추정 로깅"""
        costs = {
            't4': 0.10,  # Preemptible T4
            'v100': 2.48  # Regular V100
        }
        cost = costs.get(gpu_type, 0) * hours
        total_cost = getattr(self, 'total_cost', 0) + cost
        self.total_cost = total_cost
        
        logger.info(f"💰 비용: ${cost:.2f} (누적: ${total_cost:.2f}/$300)")
        
        if not self.args.no_wandb:
            wandb.log({
                'cost/session': cost,
                'cost/total': total_cost,
                'cost/remaining': 300 - total_cost
            })
    
    def _prepare_data(self, batch_size, data_fraction):
        """데이터 준비 (간단한 예시)"""
        # 실제로는 GCS에서 데이터 로드
        from torch.utils.data import TensorDataset
        
        # 더미 데이터 (실제로는 collected_sports_data 사용)
        n_samples = int(10000 * data_fraction)
        X = torch.randn(n_samples, 3, 224, 224)
        y = torch.randint(0, 10, (n_samples,))
        
        dataset = TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _prepare_sport_data(self, sport, batch_size, data_fraction):
        """스포츠별 데이터 준비"""
        # 실제 구현에서는 수집된 데이터 사용
        return self._prepare_data(batch_size, data_fraction)
    
    def run_complete_training(self):
        """전체 학습 파이프라인 실행"""
        try:
            start_time = time.time()
            
            # Phase 1: 실험
            best_config = self.train_phase_1_experiments()
            
            # Phase 2: 본격 학습
            trained_models = self.train_phase_2_full_training(best_config)
            
            # Phase 3: 최적화
            self.train_phase_3_optimization(trained_models)
            
            # 완료
            total_time = (time.time() - start_time) / 3600
            logger.info("="*50)
            logger.info("🎉 전체 학습 완료!")
            logger.info(f"⏱️ 총 시간: {total_time:.1f}시간")
            logger.info(f"💰 총 비용: ${self.total_cost:.2f}")
            logger.info(f"💾 모델 저장 위치: gs://{self.model_bucket}/")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"❌ 학습 중 오류 발생: {e}")
            self._save_checkpoint(None, "error_checkpoint.pth")
            raise


def main():
    parser = argparse.ArgumentParser(description='Sports AI Training on GCP')
    parser.add_argument('--phase', type=str, default='all', choices=['1', '2', '3', 'all'])
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
    
    args = parser.parse_args()
    
    # 학습 시작
    trainer = GCPTrainingManager(args)
    
    if args.phase == 'all':
        trainer.run_complete_training()
    elif args.phase == '1':
        trainer.train_phase_1_experiments()
    elif args.phase == '2':
        best_config = {'name': 'transfer_learning'}  # 기본값
        trainer.train_phase_2_full_training(best_config)
    elif args.phase == '3':
        trained_models = {}  # 기본값
        trainer.train_phase_3_optimization(trained_models)


if __name__ == "__main__":
    main()