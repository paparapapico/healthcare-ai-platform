# backend/app/api/content.py (새 파일)
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..core.auth import get_current_user
from ..core.permissions import require_subscription
from ..models.user import User

router = APIRouter(prefix="/content", tags=["content"])

@router.get("/premium")
@require_subscription("premium_content")
async def get_premium_content(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """프리미엄 콘텐츠 조회"""
    premium_content = [
        {
            "id": "1",
            "title": "전문가 운동 가이드",
            "description": "개인 트레이너가 직접 설계한 맞춤형 운동 프로그램",
            "type": "video",
            "duration": "45분",
            "level": "고급"
        },
        {
            "id": "2", 
            "title": "영양 관리 전문 컨설팅",
            "description": "영양사와의 1:1 상담 및 맞춤형 식단 계획",
            "type": "consultation",
            "duration": "60분",
            "level": "전문"
        },
        {
            "id": "3",
            "title": "AI 기반 부상 예방 분석",
            "description": "운동 패턴 분석을 통한 부상 위험도 예측 및 예방법 제공",
            "type": "analysis",
            "duration": "실시간",
            "level": "고급"
        }
    ]
    
    return {
        "success": True,
        "data": premium_content
    }