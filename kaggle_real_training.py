"""
Kaggle Sports AI - 실제 학습 버전
11시간 연속 학습 with 체크포인트
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import time
import os
import gc
from datetime import datetime

print("=" * 60)
print("🏆 Sports AI Professional Training - Real Version")
print("=" * 60)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.set_per_process_memory_fraction(0.9)
else:
    print("⚠️ CPU mode")

# ============================================================================
# 1. 실제 스포츠 AI 모델
# ============================================================================

class SportsAIModel(nn.Module):
    """4가지 스포츠 분석 모델"""
    
    def __init__(self):
        super().__init__()
        
        # Pose Encoder (17 keypoints * 3 = 51 features)
        self.pose_encoder = nn.Sequential(
            nn.Linear(51, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Sports Classifier (농구, 축구, 골프, 맨몸운동)
        self.sports_head = nn.Linear(128, 4)
        
        # Quality Score (0-100점)
        self.quality_head = nn.Linear(128, 1)
        
    def forward(self, x):
        features = self.pose_encoder(x)
        sport = self.sports_head(features)
        quality = torch.sigmoid(self.quality_head(features)) * 100
        return sport, quality

# ============================================================================
# 2. 실제 데이터셋 (스포츠별)
# ============================================================================

def create_sports_dataset(num_samples=10000):
    """실제 스포츠 데이터 시뮬레이션"""
    
    print("📊 스포츠 데이터 생성 중...")
    
    data = []
    labels = []
    qualities = []
    
    sports_patterns = {
        0: "농구 - 점프샷, 드리블",
        1: "축구 - 킥, 드리블", 
        2: "골프 - 스윙",
        3: "맨몸운동 - 스쿼트, 푸시업"
    }
    
    for sport_id in range(4):
        sport_samples = num_samples // 4
        
        # 각 스포츠별 특징 패턴 생성
        if sport_id == 0:  # 농구
            # 팔 위치가 높음 (슈팅)
            sport_data = torch.randn(sport_samples, 51)
            sport_data[:, 15:21] += 1.0  # 팔 키포인트 강조
        elif sport_id == 1:  # 축구
            # 다리 움직임 많음
            sport_data = torch.randn(sport_samples, 51)
            sport_data[:, 33:45] += 1.0  # 다리 키포인트 강조
        elif sport_id == 2:  # 골프
            # 회전 동작
            sport_data = torch.randn(sport_samples, 51)
            sport_data[:, 0:6] += 0.5  # 몸통 회전
        else:  # 맨몸운동
            # 전신 균형
            sport_data = torch.randn(sport_samples, 51)
        
        data.append(sport_data)
        labels.extend([sport_id] * sport_samples)
        
        # 품질 점수 (프로: 80-100, 아마추어: 40-79, 초보: 0-39)
        qualities.extend(torch.randint(40, 95, (sport_samples,)).float())
    
    X = torch.cat(data, dim=0)
    y = torch.tensor(labels)
    q = torch.tensor(qualities)
    
    print(f"✅ 데이터 준비 완료: {len(X)} 샘플")
    for sport_id, name in sports_patterns.items():
        count = (y == sport_id).sum().item()
        print(f"  - {name}: {count} 샘플")
    
    return TensorDataset(X, y, q)

# ============================================================================
# 3. 체크포인트 시스템
# ============================================================================

class CheckpointManager:
    """11시간마다 자동 저장"""
    
    def __init__(self):
        self.start_time = time.time()
        self.checkpoint_dir = '/kaggle/working'
        self.checkpoint_path = f'{self.checkpoint_dir}/sports_ai_checkpoint.pth'
        self.best_path = f'{self.checkpoint_dir}/best_model.pth'
        self.session_limit = 11.5 * 3600  # 11시간 30분
        
        self.epoch = 0
        self.best_accuracy = 0
        self.history = []
        
        # 이전 체크포인트 로드
        if os.path.exists(self.checkpoint_path):
            self.load()
    
    def load(self):
        """체크포인트 로드"""
        print("📂 이전 체크포인트 로드 중...")
        checkpoint = torch.load(self.checkpoint_path)
        self.epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        self.history = checkpoint.get('history', [])
        print(f"✅ Epoch {self.epoch}부터 재개")
        return checkpoint
    
    def save(self, model, optimizer, accuracy):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'current_accuracy': accuracy,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, self.checkpoint_path)
        print(f"💾 체크포인트 저장 (Epoch {self.epoch})")
        
        # 최고 모델 저장
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(model.state_dict(), self.best_path)
            print(f"🏆 최고 모델 갱신! 정확도: {accuracy:.2f}%")
    
    def should_stop(self):
        """시간 체크"""
        elapsed = time.time() - self.start_time
        remaining = (self.session_limit - elapsed) / 60
        
        if remaining < 20:  # 20분 남음
            print(f"⏰ 세션 종료 임박! {remaining:.0f}분 남음")
            return True
        
        if self.epoch % 50 == 0:
            hours = elapsed / 3600
            print(f"⏱️ 경과: {hours:.1f}시간")
        
        return False

# ============================================================================
# 4. 메인 학습 루프
# ============================================================================

def train():
    """실제 11시간 학습"""
    
    # 체크포인트 매니저
    ckpt_manager = CheckpointManager()
    
    # 모델 생성
    model = SportsAIModel().to(device)
    print(f"📦 모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # 데이터 준비
    dataset = create_sports_dataset(10000)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 옵티마이저
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    sport_criterion = nn.CrossEntropyLoss()
    quality_criterion = nn.MSELoss()
    
    # 체크포인트 로드
    if os.path.exists(ckpt_manager.checkpoint_path):
        checkpoint = ckpt_manager.load()
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    print("\n" + "=" * 60)
    print("🚀 학습 시작! (최대 11시간)")
    print("=" * 60)
    
    # 학습 시작
    for epoch in range(ckpt_manager.epoch, 1000):
        ckpt_manager.epoch = epoch
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        print(f"\n📍 Epoch {epoch+1}/1000")
        
        for batch_idx, (data, sport_labels, quality_labels) in enumerate(train_loader):
            data = data.to(device)
            sport_labels = sport_labels.to(device)
            quality_labels = quality_labels.to(device)
            
            # Forward
            sport_pred, quality_pred = model(data)
            
            # Loss
            sport_loss = sport_criterion(sport_pred, sport_labels)
            quality_loss = quality_criterion(quality_pred.squeeze(), quality_labels)
            total_loss = sport_loss + 0.1 * quality_loss
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 통계
            train_loss += total_loss.item()
            _, predicted = sport_pred.max(1)
            train_total += sport_labels.size(0)
            train_correct += predicted.eq(sport_labels).sum().item()
            
            # 진행 표시
            if batch_idx % 50 == 0:
                acc = 100. * train_correct / train_total
                print(f"  Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {total_loss.item():.4f} | "
                      f"Acc: {acc:.2f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_quality_error = 0
        
        with torch.no_grad():
            for data, sport_labels, quality_labels in val_loader:
                data = data.to(device)
                sport_labels = sport_labels.to(device)
                quality_labels = quality_labels.to(device)
                
                sport_pred, quality_pred = model(data)
                
                _, predicted = sport_pred.max(1)
                val_total += sport_labels.size(0)
                val_correct += predicted.eq(sport_labels).sum().item()
                
                val_quality_error += torch.abs(quality_pred.squeeze() - quality_labels).mean().item()
        
        val_accuracy = 100. * val_correct / val_total
        avg_quality_error = val_quality_error / len(val_loader)
        
        print(f"📊 Validation: Acc {val_accuracy:.2f}% | Quality Error: {avg_quality_error:.1f}점")
        
        # 히스토리 저장
        ckpt_manager.history.append({
            'epoch': epoch,
            'train_acc': 100. * train_correct / train_total,
            'val_acc': val_accuracy,
            'quality_error': avg_quality_error
        })
        
        # 체크포인트 저장 (10 에폭마다)
        if epoch % 10 == 0:
            ckpt_manager.save(model, optimizer, val_accuracy)
            
            # 메모리 정리
            gc.collect()
            torch.cuda.empty_cache()
        
        # 세션 종료 체크
        if ckpt_manager.should_stop():
            print("\n" + "=" * 60)
            print("⏰ 세션 종료 시간!")
            print("📝 다음 단계:")
            print("1. Output에서 checkpoint 다운로드")
            print("2. 새 노트북에서 이 코드 다시 실행")
            print("3. 자동으로 이어서 학습됩니다!")
            print("=" * 60)
            
            ckpt_manager.save(model, optimizer, val_accuracy)
            break
    
    return model

# ============================================================================
# 5. 실행
# ============================================================================

if __name__ == "__main__":
    try:
        # 학습 실행
        model = train()
        
        # 최종 테스트
        print("\n🧪 최종 테스트...")
        model.eval()
        
        sports = ['🏀 농구', '⚽ 축구', '⛳ 골프', '🏃 맨몸운동']
        
        with torch.no_grad():
            # 각 스포츠별 테스트
            for i in range(4):
                test_input = torch.randn(1, 51).to(device)
                sport_pred, quality_pred = model(test_input)
                
                pred_sport = sport_pred.argmax().item()
                pred_quality = quality_pred.item()
                
                print(f"테스트 {i+1}: {sports[pred_sport]} (품질: {pred_quality:.1f}점)")
        
        print("\n✨ 학습 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()