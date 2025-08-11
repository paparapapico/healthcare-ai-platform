# 구독 플랜 초기 생성 스크립트
# backend/scripts/init_subscription_plans.py
import asyncio
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.services.payment_service import PaymentService

async def init_plans():
    """구독 플랜 초기화"""
    db = SessionLocal()
    try:
        print("🔄 구독 플랜 생성 중...")
        await PaymentService.create_subscription_plans(db)
        print("✅ 구독 플랜 생성 완료!")
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(init_plans())