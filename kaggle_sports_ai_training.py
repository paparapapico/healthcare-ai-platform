"""
Kaggle Sports AI Training - 완전 무료 연속 학습
Professional Sports Analysis AI Model Training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import os
import json
from datetime import datetime
import pickle

print("=" * 60)
print("🏆 Sports AI Professional Training - Kaggle Edition")
print("=" * 60)

# GPU 확인
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("❌ GPU를 사용할 수 없습니다!")

# ============================================================================
# 1. 자동 재시작 학습 시스템
# ============================================================================

class AutoRestartTrainer:
    """11시간 50분마다 자동 체크포인트 저장 및 재시작"""
    
    def __init__(self, checkpoint_dir='/kaggle/working'):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = f'{checkpoint_dir}/sports_ai_checkpoint.pth'
        self.start_time = time.time()
        self.session_duration = 11.8 * 3600  # 11시간 48분 (안전 마진)
        
        # 체크포인트 로드
        self.load_checkpoint()
        
    def load_checkpoint(self):
        """이전 세션 체크포인트 로드"""
        if os.path.exists(self.checkpoint_path):
            print("📂 이전 체크포인트 발견! 로딩 중...")
            checkpoint = torch.load(self.checkpoint_path)
            self.epoch = checkpoint['epoch']
            self.best_accuracy = checkpoint['best_accuracy']
            self.training_history = checkpoint['training_history']
            print(f"✅ Epoch {self.epoch}부터 재개, 최고 정확도: {self.best_accuracy:.2%}")
            return checkpoint
        else:
            print("🆕 새로운 학습 시작!")
            self.epoch = 0
            self.best_accuracy = 0
            self.training_history = []
            return None
    
    def save_checkpoint(self, model, optimizer, metrics):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, self.checkpoint_path)
        print(f"💾 체크포인트 저장 완료! (Epoch {self.epoch})")
        
        # 백업 저장
        backup_path = f'{self.checkpoint_dir}/backup_epoch_{self.epoch}.pth'
        torch.save(checkpoint, backup_path)
    
    def should_stop(self):
        """세션 종료 시간 체크"""
        elapsed = time.time() - self.start_time
        remaining = (self.session_duration - elapsed) / 60  # 분 단위
        
        if remaining < 10:  # 10분 남았을 때
            print(f"⏰ 세션 종료까지 {remaining:.1f}분 남음!")
            return True
        
        if self.epoch % 10 == 0:  # 10 에폭마다 시간 체크
            hours_elapsed = elapsed / 3600
            print(f"⏱️ 경과 시간: {hours_elapsed:.1f}시간")
        
        return False

# ============================================================================
# 2. Sports AI 모델 정의
# ============================================================================

class SportsAIModel(nn.Module):
    """프로 스포츠 분석 AI 모델"""
    
    def __init__(self, num_sports=4, num_keypoints=17):
        super().__init__()
        
        # Pose Encoder (MediaPipe 17 keypoints)
        self.pose_encoder = nn.Sequential(
            nn.Linear(num_keypoints * 3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Temporal Encoder (시간 정보)
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True, bidirectional=True)
        
        # Sport Classifier
        self.sport_classifier = nn.Linear(512, num_sports)
        
        # Movement Quality Scorer
        self.quality_scorer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100)  # 0-100 점수
        )
        
        # Professional Comparison
        self.pro_comparator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # Top 10 프로 선수와 유사도
        )
    
    def forward(self, x):
        # x shape: (batch, sequence_length, keypoints*3)
        batch_size, seq_len, _ = x.shape
        
        # Pose encoding
        x = x.reshape(-1, x.shape[-1])
        pose_features = self.pose_encoder(x)
        pose_features = pose_features.reshape(batch_size, seq_len, -1)
        
        # Temporal encoding
        lstm_out, _ = self.lstm(pose_features)
        
        # Global features
        global_features = torch.mean(lstm_out, dim=1)
        
        # Multi-task outputs
        sport = self.sport_classifier(global_features)
        quality = self.quality_scorer(global_features)
        pro_match = self.pro_comparator(global_features)
        
        return {
            'sport': sport,
            'quality': quality,
            'pro_match': pro_match
        }

# ============================================================================
# 3. 데이터셋 클래스
# ============================================================================

class SportsDataset(Dataset):
    """스포츠 동작 데이터셋"""
    
    def __init__(self, num_samples=5000, seq_length=30):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_keypoints = 17
        
        # 더미 데이터 생성 (실제로는 수집된 데이터 사용)
        self.data = torch.randn(num_samples, seq_length, self.num_keypoints * 3)
        self.sport_labels = torch.randint(0, 4, (num_samples,))  # 4개 스포츠
        self.quality_scores = torch.randint(0, 100, (num_samples,)).float()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'pose': self.data[idx],
            'sport': self.sport_labels[idx],
            'quality': self.quality_scores[idx]
        }

# ============================================================================
# 4. 학습 함수
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 데이터 준비
        poses = batch['pose'].to(device)
        sport_labels = batch['sport'].to(device)
        quality_scores = batch['quality'].to(device)
        
        # Forward pass
        outputs = model(poses)
        
        # Loss 계산
        sport_loss = criterion['sport'](outputs['sport'], sport_labels)
        quality_loss = criterion['quality'](outputs['quality'].squeeze(), quality_scores)
        total_batch_loss = sport_loss + 0.5 * quality_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()
        
        # 통계
        total_loss += total_batch_loss.item()
        _, predicted = outputs['sport'].max(1)
        total += sport_labels.size(0)
        correct += predicted.eq(sport_labels).sum().item()
        
        # 진행상황 출력
        if batch_idx % 10 == 0:
            print(f'  Batch [{batch_idx}/{len(dataloader)}] '
                  f'Loss: {total_batch_loss.item():.4f} '
                  f'Acc: {100.*correct/total:.2f}%')
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

# ============================================================================
# 5. 메인 학습 루프
# ============================================================================

def main():
    """메인 학습 함수"""
    
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")
    
    # 자동 재시작 매니저
    trainer = AutoRestartTrainer()
    
    # 모델 생성
    model = SportsAIModel().to(device)
    
    # 데이터 로더
    dataset = SportsDataset(num_samples=5000)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 옵티마이저 & 손실 함수
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = {
        'sport': nn.CrossEntropyLoss(),
        'quality': nn.MSELoss()
    }
    
    # 체크포인트 로드
    checkpoint = trainer.load_checkpoint()
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 학습 시작
    print("\n🚀 학습 시작!")
    print("=" * 60)
    
    for epoch in range(trainer.epoch, 1000):
        trainer.epoch = epoch
        
        print(f"\n📍 Epoch {epoch+1}/1000")
        print("-" * 40)
        
        # 학습
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 검증
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                poses = batch['pose'].to(device)
                sport_labels = batch['sport'].to(device)
                outputs = model(poses)
                _, predicted = outputs['sport'].max(1)
                val_total += sport_labels.size(0)
                val_correct += predicted.eq(sport_labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # 결과 출력
        print(f"📊 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"📊 Val Acc: {val_acc:.2f}%")
        
        # 최고 모델 저장
        if val_acc > trainer.best_accuracy:
            trainer.best_accuracy = val_acc
            print(f"🏆 새로운 최고 정확도! {val_acc:.2f}%")
        
        # 히스토리 저장
        trainer.training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        })
        
        # 체크포인트 저장
        if epoch % 5 == 0 or trainer.should_stop():
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }
            trainer.save_checkpoint(model, optimizer, metrics)
        
        # 세션 종료 체크
        if trainer.should_stop():
            print("\n⏰ 세션 종료 시간입니다!")
            print("📝 다음 스텝:")
            print("1. Output 탭에서 checkpoint 파일 다운로드")
            print("2. 새 노트북 생성")
            print("3. checkpoint 업로드 후 이 코드 다시 실행")
            print("=" * 60)
            break
    
    print("\n✅ 학습 완료!")
    return model

# ============================================================================
# 6. 실행
# ============================================================================

if __name__ == "__main__":
    # 패키지 설치
    os.system('pip install -q torch torchvision')
    
    # 학습 실행
    trained_model = main()
    
    # 최종 모델 저장
    torch.save(trained_model.state_dict(), '/kaggle/working/final_sports_ai_model.pth')
    print("\n🎉 모든 작업 완료!")