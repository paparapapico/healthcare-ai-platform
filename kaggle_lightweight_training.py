"""
Kaggle Lightweight Sports AI Training - 메모리 최적화 버전
Optimized for Kaggle P100 GPU (16GB VRAM)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import gc
import os
from datetime import datetime

print("=" * 60)
print("🏆 Sports AI Training - Lightweight Version")
print("=" * 60)

# GPU 설정 및 메모리 최적화
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 메모리 제한 설정
    torch.cuda.set_per_process_memory_fraction(0.8)
else:
    device = torch.device('cpu')
    print("⚠️ CPU mode")

# ============================================================================
# 1. 초경량 모델 (메모리 사용 최소화)
# ============================================================================

class LightweightSportsModel(nn.Module):
    """메모리 효율적인 경량 모델"""
    
    def __init__(self):
        super().__init__()
        # 작은 네트워크
        self.encoder = nn.Sequential(
            nn.Linear(51, 64),  # 17 keypoints * 3 = 51
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(32, 4)  # 4 sports
        
    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)

# ============================================================================
# 2. 작은 데이터셋
# ============================================================================

class SmallDataset(Dataset):
    """메모리 효율적인 작은 데이터셋"""
    
    def __init__(self, size=1000):
        # 작은 크기로 시작
        self.data = torch.randn(size, 51)
        self.labels = torch.randint(0, 4, (size,))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ============================================================================
# 3. 메인 학습 (메모리 최적화)
# ============================================================================

def train_lightweight():
    """경량 학습 함수"""
    
    try:
        # 1. 모델 생성
        print("\n📦 모델 생성 중...")
        model = LightweightSportsModel().to(device)
        print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
        
        # 2. 데이터 준비 (작은 배치)
        print("\n📊 데이터 준비 중...")
        dataset = SmallDataset(size=500)  # 작은 데이터
        dataloader = DataLoader(
            dataset, 
            batch_size=8,  # 작은 배치
            shuffle=True,
            num_workers=0  # 메모리 절약
        )
        
        # 3. 옵티마이저
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 4. 학습 시작
        print("\n🚀 학습 시작!")
        print("-" * 40)
        
        for epoch in range(10):  # 10 에폭만
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                # GPU로 이동
                data = data.to(device)
                target = target.to(device)
                
                # Forward
                output = model(data)
                loss = criterion(output, target)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 통계
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # 메모리 정리
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # 에폭 결과
            accuracy = 100. * correct / total
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/10 | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")
            
            # 메모리 상태
            if device.type == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"  └─ VRAM: {allocated:.2f}/{reserved:.2f} GB")
        
        # 5. 모델 저장
        print("\n💾 모델 저장 중...")
        save_path = '/kaggle/working/lightweight_model.pth'
        torch.save({
            'model': model.state_dict(),
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat()
        }, save_path)
        print(f"✅ 저장 완료: {save_path}")
        
        # 6. 테스트
        print("\n🧪 간단한 테스트...")
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 51).to(device)
            test_output = model(test_input)
            predicted_sport = test_output.argmax().item()
            sports = ['농구', '축구', '골프', '맨몸운동']
            print(f"예측 결과: {sports[predicted_sport]}")
        
        print("\n" + "=" * 60)
        print("🎉 학습 완료!")
        print("=" * 60)
        
        return model
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("메모리 정리 중...")
        torch.cuda.empty_cache()
        gc.collect()
        
        # 더 작은 모델로 재시도
        print("\n🔄 더 작은 모델로 재시도...")
        return train_minimal()

def train_minimal():
    """최소 모델 (오류 시 폴백)"""
    
    print("\n🐤 최소 모델 실행...")
    
    # 아주 작은 모델
    model = nn.Sequential(
        nn.Linear(51, 16),
        nn.ReLU(),
        nn.Linear(16, 4)
    ).to(device)
    
    # 아주 작은 데이터
    X = torch.randn(100, 51).to(device)
    y = torch.randint(0, 4, (100,)).to(device)
    
    # 간단한 학습
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        output = model(X)
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Mini Epoch {epoch+1}: Loss {loss.item():.4f}")
    
    print("✅ 최소 학습 완료!")
    return model

# ============================================================================
# 4. 실행
# ============================================================================

if __name__ == "__main__":
    # 패키지 확인
    print("\n📦 패키지 버전:")
    print(f"PyTorch: {torch.__version__}")
    
    # 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 학습 실행
    trained_model = train_lightweight()
    
    print("\n📊 최종 메모리 상태:")
    if torch.cuda.is_available():
        print(f"할당: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"예약: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    
    print("\n✨ 모든 작업 완료!")