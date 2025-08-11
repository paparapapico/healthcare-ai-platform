# 기존 API에 구독 제한 적용
# backend/app/api/exercises.py 수정
from ..core.permissions import require_subscription, check_usage_limit

@router.post("/")
@check_usage_limit("max_exercises_per_day", 1)
async def create_exercise(
    exercise_data: ExerciseCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """운동 기록 생성 (구독 제한 적용)"""
    # 기존 로직...

@router.post("/ai-analysis") 
@require_subscription("advanced_ai")
@check_usage_limit("max_ai_analysis", 1)
async def analyze_exercise_with_ai(
    exercise_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """AI 운동 분석 (프리미엄 기능)"""
    # 기존 로직...