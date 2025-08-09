"""
Health Data API
건강 메트릭, 수면, 영양 추적
"""

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta, date
from pydantic import BaseModel
import base64

from app.db.database import get_db
from app.models.models import User, HealthMetric
from app.services.ai.nutrition_analyzer import (
    nutrition_analyzer, water_tracker, sleep_analyzer
)

router = APIRouter(prefix="/api/v1/health", tags=["health"])

# ========================
# Schemas
# ========================

class HealthMetricCreate(BaseModel):
    weight: Optional[float] = None
    body_fat_percentage: Optional[float] = None
    muscle_mass: Optional[float] = None
    heart_rate_resting: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    steps: Optional[int] = None
    water_intake: Optional[float] = None

class SleepDataCreate(BaseModel):
    sleep_hours: float
    deep_sleep_hours: Optional[float] = None
    rem_sleep_hours: Optional[float] = None
    wake_ups: Optional[int] = 0
    sleep_quality_score: Optional[float] = None

class NutritionLogCreate(BaseModel):
    meal_type: str  # breakfast, lunch, dinner, snack
    foods: List[str]
    calories: Optional[float] = None
    protein: Optional[float] = None
    carbs: Optional[float] = None
    fat: Optional[float] = None

class HealthDashboardResponse(BaseModel):
    health_score: float
    weight_trend: str  # up, down, stable
    sleep_quality: float
    nutrition_score: float
    activity_level: float
    hydration_status: dict
    recommendations: List[str]

class HealthInsightResponse(BaseModel):
    category: str
    title: str
    description: str
    action_required: bool
    priority: str  # high, medium, low

# ========================
# Health Metrics
# ========================

@router.post("/metrics")
async def log_health_metrics(
    metrics: HealthMetricCreate,
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """건강 지표 기록"""
    
    # Create new health metric entry
    health_metric = HealthMetric(
        user_id=current_user_id,
        weight=metrics.weight,
        body_fat_percentage=metrics.body_fat_percentage,
        muscle_mass=metrics.muscle_mass,
        heart_rate_resting=metrics.heart_rate_resting,
        blood_pressure_systolic=metrics.blood_pressure_systolic,
        blood_pressure_diastolic=metrics.blood_pressure_diastolic,
        steps=metrics.steps,
        water_intake=metrics.water_intake,
        recorded_at=datetime.utcnow()
    )
    
    # Calculate BMI if weight is provided
    user = db.query(User).filter(User.id == current_user_id).first()
    if metrics.weight and user.height:
        health_metric.bmi = metrics.weight / ((user.height / 100) ** 2)
    
    db.add(health_metric)
    
    # Update user's weight if provided
    if metrics.weight:
        user.weight = metrics.weight
    
    # Recalculate health score
    user.health_score = calculate_health_score(current_user_id, db)
    
    db.commit()
    
    return {
        "message": "건강 지표가 기록되었습니다",
        "health_score": user.health_score
    }

@router.get("/metrics/history")
async def get_health_metrics_history(
    days: int = 30,
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """건강 지표 히스토리 조회"""
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    metrics = db.query(HealthMetric).filter(
        HealthMetric.user_id == current_user_id,
        HealthMetric.recorded_at >= start_date
    ).order_by(HealthMetric.recorded_at.desc()).all()
    
    # Format data for charts
    history = {
        "weight": [],
        "body_fat": [],
        "muscle_mass": [],
        "heart_rate": [],
        "blood_pressure": [],
        "steps": [],
        "water_intake": []
    }
    
    for metric in metrics:
        date_str = metric.recorded_at.strftime("%Y-%m-%d")
        
        if metric.weight:
            history["weight"].append({"date": date_str, "value": metric.weight})
        if metric.body_fat_percentage:
            history["body_fat"].append({"date": date_str, "value": metric.body_fat_percentage})
        if metric.muscle_mass:
            history["muscle_mass"].append({"date": date_str, "value": metric.muscle_mass})
        if metric.heart_rate_resting:
            history["heart_rate"].append({"date": date_str, "value": metric.heart_rate_resting})
        if metric.blood_pressure_systolic:
            history["blood_pressure"].append({
                "date": date_str,
                "systolic": metric.blood_pressure_systolic,
                "diastolic": metric.blood_pressure_diastolic
            })
        if metric.steps:
            history["steps"].append({"date": date_str, "value": metric.steps})
        if metric.water_intake:
            history["water_intake"].append({"date": date_str, "value": metric.water_intake})
    
    return history

# ========================
# Sleep Tracking
# ========================

@router.post("/sleep")
async def log_sleep_data(
    sleep_data: SleepDataCreate,
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """수면 데이터 기록"""
    
    # Analyze sleep quality
    analysis = sleep_analyzer.analyze_sleep_quality(
        sleep_hours=sleep_data.sleep_hours,
        deep_sleep_hours=sleep_data.deep_sleep_hours,
        rem_sleep_hours=sleep_data.rem_sleep_hours,
        wake_ups=sleep_data.wake_ups
    )
    
    # Save to database
    health_metric = HealthMetric(
        user_id=current_user_id,
        sleep_hours=sleep_data.sleep_hours,
        sleep_quality_score=analysis["quality_score"],
        deep_sleep_hours=sleep_data.deep_sleep_hours,
        rem_sleep_hours=sleep_data.rem_sleep_hours,
        recorded_at=datetime.utcnow()
    )
    
    db.add(health_metric)
    db.commit()
    
    return {
        "message": "수면 데이터가 기록되었습니다",
        "analysis": analysis
    }

@router.get("/sleep/analysis")
async def get_sleep_analysis(
    days: int = 7,
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """수면 분석 조회"""
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    sleep_records = db.query(HealthMetric).filter(
        HealthMetric.user_id == current_user_id,
        HealthMetric.recorded_at >= start_date,
        HealthMetric.sleep_hours.isnot(None)
    ).order_by(HealthMetric.recorded_at.desc()).all()
    
    if not sleep_records:
        return {
            "average_sleep_hours": 0,
            "average_quality_score": 0,
            "trend": "no_data",
            "recommendations": ["수면 데이터를 기록해주세요"]
        }
    
    # Calculate averages
    total_sleep = sum(r.sleep_hours for r in sleep_records)
    total_quality = sum(r.sleep_quality_score or 0 for r in sleep_records)
    
    avg_sleep = total_sleep / len(sleep_records)
    avg_quality = total_quality / len(sleep_records)
    
    # Determine trend
    if len(sleep_records) >= 3:
        recent = sleep_records[:3]
        older = sleep_records[3:6] if len(sleep_records) >= 6 else sleep_records[3:]
        
        recent_avg = sum(r.sleep_hours for r in recent) / len(recent)
        older_avg = sum(r.sleep_hours for r in older) / len(older) if older else recent_avg
        
        if recent_avg > older_avg + 0.5:
            trend = "improving"
        elif recent_avg < older_avg - 0.5:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"
    
    # Generate recommendations
    recommendations = []
    if avg_sleep < 7:
        recommendations.append("수면 시간을 7-9시간으로 늘려보세요")
    if avg_quality < 70:
        recommendations.append("수면의 질을 개선하기 위해 취침 전 스마트폰 사용을 자제하세요")
    
    return {
        "average_sleep_hours": round(avg_sleep, 1),
        "average_quality_score": round(avg_quality, 1),
        "trend": trend,
        "sleep_history": [
            {
                "date": r.recorded_at.strftime("%Y-%m-%d"),
                "hours": r.sleep_hours,
                "quality": r.sleep_quality_score
            }
            for r in sleep_records
        ],
        "recommendations": recommendations
    }

# ========================
# Nutrition Tracking
# ========================

@router.post("/nutrition/analyze-image")
async def analyze_food_image(
    file: UploadFile = File(...),
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """음식 사진 분석"""
    
    # Read image file
    contents = await file.read()
    image_base64 = base64.b64encode(contents).decode('utf-8')
    
    # Analyze image
    analysis = nutrition_analyzer.analyze_food_image(image_base64)
    
    # Save to database
    health_metric = HealthMetric(
        user_id=current_user_id,
        calories_burned=analysis.total_calories,  # Actually calories consumed
        recorded_at=datetime.utcnow()
    )
    
    db.add(health_metric)
    db.commit()
    
    return {
        "foods": [
            {
                "name": food.food_name,
                "calories": food.calories,
                "protein": food.protein,
                "carbs": food.carbs,
                "fat": food.fat,
                "serving_size": food.serving_size,
                "confidence": round(food.confidence * 100, 1)
            }
            for food in analysis.foods
        ],
        "totals": {
            "calories": analysis.total_calories,
            "protein": analysis.total_protein,
            "carbs": analysis.total_carbs,
            "fat": analysis.total_fat
        },
        "meal_type": analysis.meal_type,
        "health_score": analysis.health_score,
        "recommendations": analysis.recommendations
    }

@router.post("/nutrition/log")
async def log_nutrition(
    nutrition: NutritionLogCreate,
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """영양 섭취 기록"""
    
    # Save nutrition data
    health_metric = HealthMetric(
        user_id=current_user_id,
        calories_burned=nutrition.calories,  # Actually calories consumed
        recorded_at=datetime.utcnow()
    )
    
    db.add(health_metric)
    db.commit()
    
    return {"message": "영양 데이터가 기록되었습니다"}

# ========================
# Water Intake
# ========================

@router.post("/water/log")
async def log_water_intake(
    amount_ml: int,
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """물 섭취량 기록"""
    
    # Get today's water intake
    today_start = datetime.combine(date.today(), datetime.min.time())
    today_metric = db.query(HealthMetric).filter(
        HealthMetric.user_id == current_user_id,
        HealthMetric.recorded_at >= today_start
    ).order_by(HealthMetric.recorded_at.desc()).first()
    
    if today_metric and today_metric.water_intake:
        # Update existing record
        today_metric.water_intake += amount_ml
        total_intake = today_metric.water_intake
    else:
        # Create new record
        health_metric = HealthMetric(
            user_id=current_user_id,
            water_intake=amount_ml,
            recorded_at=datetime.utcnow()
        )
        db.add(health_metric)
        total_intake = amount_ml
    
    db.commit()
    
    # Calculate hydration status
    hydration = water_tracker.calculate_hydration_score(total_intake)
    
    return {
        "message": f"{amount_ml}ml 기록 완료",
        "today_total": total_intake,
        "hydration_status": hydration
    }

@router.get("/water/status")
async def get_hydration_status(
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """수분 섭취 현황"""
    
    # Get today's water intake
    today_start = datetime.combine(date.today(), datetime.min.time())
    
    today_metrics = db.query(HealthMetric).filter(
        HealthMetric.user_id == current_user_id,
        HealthMetric.recorded_at >= today_start,
        HealthMetric.water_intake.isnot(None)
    ).all()
    
    total_intake = sum(m.water_intake for m in today_metrics)
    
    # Calculate hydration status
    hydration = water_tracker.calculate_hydration_score(total_intake)
    
    return hydration

# ========================
# Dashboard & Insights
# ========================

@router.get("/dashboard", response_model=HealthDashboardResponse)
async def get_health_dashboard(
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """건강 대시보드 데이터"""
    
    user = db.query(User).filter(User.id == current_user_id).first()
    
    # Get recent metrics
    recent_metrics = db.query(HealthMetric).filter(
        HealthMetric.user_id == current_user_id
    ).order_by(HealthMetric.recorded_at.desc()).limit(7).all()
    
    # Calculate scores
    health_score = user.health_score
    
    # Weight trend
    weight_trend = "stable"
    if len(recent_metrics) >= 2:
        if recent_metrics[0].weight and recent_metrics[1].weight:
            diff = recent_metrics[0].weight - recent_metrics[1].weight
            if diff > 0.5:
                weight_trend = "up"
            elif diff < -0.5:
                weight_trend = "down"
    
    # Sleep quality
    sleep_metrics = [m for m in recent_metrics if m.sleep_quality_score]
    sleep_quality = sum(m.sleep_quality_score for m in sleep_metrics) / len(sleep_metrics) if sleep_metrics else 70
    
    # Nutrition score
    nutrition_score = 75  # Placeholder
    
    # Activity level
    steps_metrics = [m for m in recent_metrics if m.steps]
    avg_steps = sum(m.steps for m in steps_metrics) / len(steps_metrics) if steps_metrics else 5000
    activity_level = min(100, (avg_steps / 10000) * 100)
    
    # Hydration
    today_start = datetime.combine(date.today(), datetime.min.time())
    today_water = db.query(HealthMetric).filter(
        HealthMetric.user_id == current_user_id,
        HealthMetric.recorded_at >= today_start,
        HealthMetric.water_intake.isnot(None)
    ).all()
    
    total_water = sum(m.water_intake for m in today_water)
    hydration = water_tracker.calculate_hydration_score(total_water)
    
    # Generate recommendations
    recommendations = []
    if activity_level < 50:
        recommendations.append("운동량을 늘려보세요. 하루 30분 이상 운동을 권장합니다")
    if sleep_quality < 70:
        recommendations.append("수면의 질을 개선하세요. 규칙적인 수면 패턴을 유지하세요")
    if hydration["percentage"] < 75:
        recommendations.append(f"물을 {hydration['glasses_remaining']}잔 더 마셔주세요")
    
    return HealthDashboardResponse(
        health_score=health_score,
        weight_trend=weight_trend,
        sleep_quality=sleep_quality,
        nutrition_score=nutrition_score,
        activity_level=activity_level,
        hydration_status=hydration,
        recommendations=recommendations
    )

@router.get("/insights", response_model=List[HealthInsightResponse])
async def get_health_insights(
    current_user_id: int = 1,  # TODO: Get from auth
    db: Session = Depends(get_db)
):
    """개인화된 건강 인사이트"""
    
    insights = []
    
    # Get recent data
    recent_metrics = db.query(HealthMetric).filter(
        HealthMetric.user_id == current_user_id
    ).order_by(HealthMetric.recorded_at.desc()).limit(30).all()
    
    # Sleep insight
    sleep_metrics = [m for m in recent_metrics if m.sleep_hours]
    if sleep_metrics:
        avg_sleep = sum(m.sleep_hours for m in sleep_metrics) / len(sleep_metrics)
        if avg_sleep < 6:
            insights.append(HealthInsightResponse(
                category="sleep",
                title="수면 부족 경고",
                description=f"최근 평균 수면 시간이 {avg_sleep:.1f}시간입니다. 건강을 위해 7-9시간 수면을 권장합니다.",
                action_required=True,
                priority="high"
            ))
    
    # Activity insight
    steps_metrics = [m for m in recent_metrics if m.steps]
    if steps_metrics:
        avg_steps = sum(m.steps for m in steps_metrics) / len(steps_metrics)
        if avg_steps < 5000:
            insights.append(HealthInsightResponse(
                category="activity",
                title="활동량 부족",
                description=f"일일 평균 {int(avg_steps)}걸음입니다. WHO 권장 기준인 10,000걸음을 목표로 해보세요.",
                action_required=True,
                priority="medium"
            ))
    
    # Weight insight
    weight_metrics = [m for m in recent_metrics if m.weight]
    if len(weight_metrics) >= 2:
        weight_change = weight_metrics[0].weight - weight_metrics[-1].weight
        if abs(weight_change) > 2:
            insights.append(HealthInsightResponse(
                category="weight",
                title="체중 변화 감지",
                description=f"최근 {abs(weight_change):.1f}kg {'증가' if weight_change > 0 else '감소'}했습니다.",
                action_required=False,
                priority="low"
            ))
    
    return insights

# ========================
# Helper Functions
# ========================

def calculate_health_score(user_id: int, db: Session) -> float:
    """종합 건강 점수 계산"""
    
    score = 70.0  # Base score
    
    # Get recent metrics
    recent_metrics = db.query(HealthMetric).filter(
        HealthMetric.user_id == user_id
    ).order_by(HealthMetric.recorded_at.desc()).limit(7).all()
    
    if not recent_metrics:
        return score
    
    # Factor in various metrics
    for metric in recent_metrics:
        # Sleep
        if metric.sleep_hours:
            if 7 <= metric.sleep_hours <= 9:
                score += 2
            elif metric.sleep_hours < 6:
                score -= 2
        
        # Activity
        if metric.steps:
            if metric.steps >= 10000:
                score += 2
            elif metric.steps >= 7000:
                score += 1
            elif metric.steps < 3000:
                score -= 2
        
        # Heart rate
        if metric.heart_rate_resting:
            if 60 <= metric.heart_rate_resting <= 100:
                score += 1
        
        # Blood pressure
        if metric.blood_pressure_systolic and metric.blood_pressure_diastolic:
            if (90 <= metric.blood_pressure_systolic <= 120 and 
                60 <= metric.blood_pressure_diastolic <= 80):
                score += 1
            elif metric.blood_pressure_systolic > 140:
                score -= 3
    
    # Normalize score
    return max(0, min(100, score))