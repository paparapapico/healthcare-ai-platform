"""
Nutrition Analysis AI
음식 사진 분석 및 칼로리 계산
"""

import base64
import io
import json
import requests
from PIL import Image
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

# 임시로 로컬 DB 사용 (실제로는 API 연동)
FOOD_DATABASE = {
    "김치찌개": {"calories": 150, "protein": 12, "carbs": 10, "fat": 8, "serving_size": "1인분"},
    "비빔밥": {"calories": 550, "protein": 21, "carbs": 85, "fat": 15, "serving_size": "1그릇"},
    "삼겹살": {"calories": 330, "protein": 20, "carbs": 0, "fat": 28, "serving_size": "100g"},
    "김밥": {"calories": 350, "protein": 9, "carbs": 55, "fat": 10, "serving_size": "1줄"},
    "떡볶이": {"calories": 300, "protein": 7, "carbs": 60, "fat": 5, "serving_size": "1인분"},
    "치킨": {"calories": 250, "protein": 25, "carbs": 15, "fat": 12, "serving_size": "100g"},
    "라면": {"calories": 500, "protein": 10, "carbs": 65, "fat": 20, "serving_size": "1개"},
    "샐러드": {"calories": 150, "protein": 5, "carbs": 20, "fat": 8, "serving_size": "1그릇"},
    "밥": {"calories": 300, "protein": 6, "carbs": 65, "fat": 1, "serving_size": "1공기"},
    "사과": {"calories": 95, "protein": 0.5, "carbs": 25, "fat": 0.3, "serving_size": "1개"},
    "바나나": {"calories": 105, "protein": 1.3, "carbs": 27, "fat": 0.4, "serving_size": "1개"},
    "계란": {"calories": 70, "protein": 6, "carbs": 1, "fat": 5, "serving_size": "1개"},
    "우유": {"calories": 150, "protein": 8, "carbs": 12, "fat": 8, "serving_size": "200ml"},
    "커피": {"calories": 5, "protein": 0.3, "carbs": 0, "fat": 0, "serving_size": "1잔"},
}

logger = logging.getLogger(__name__)

@dataclass
class NutritionInfo:
    food_name: str
    calories: float
    protein: float
    carbs: float
    fat: float
    serving_size: str
    confidence: float

@dataclass
class MealAnalysisResult:
    foods: List[NutritionInfo]
    total_calories: float
    total_protein: float
    total_carbs: float
    total_fat: float
    meal_type: str  # breakfast, lunch, dinner, snack
    health_score: float
    recommendations: List[str]

class NutritionAnalyzer:
    def __init__(self):
        """영양 분석기 초기화"""
        self.food_detection_confidence_threshold = 0.7
        
        # 실제로는 다음과 같은 AI 모델 사용:
        # - Google Vision API
        # - Clarifai Food Model
        # - Custom trained YOLO/EfficientNet
        
    def analyze_food_image(self, image_base64: str) -> MealAnalysisResult:
        """음식 이미지 분석"""
        try:
            # Base64 디코딩
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # 음식 감지 (실제로는 AI 모델 사용)
            detected_foods = self._detect_foods(image)
            
            # 영양 정보 계산
            nutrition_infos = []
            total_calories = 0
            total_protein = 0
            total_carbs = 0
            total_fat = 0
            
            for food_name, confidence in detected_foods:
                if food_name in FOOD_DATABASE:
                    food_data = FOOD_DATABASE[food_name]
                    nutrition_info = NutritionInfo(
                        food_name=food_name,
                        calories=food_data["calories"],
                        protein=food_data["protein"],
                        carbs=food_data["carbs"],
                        fat=food_data["fat"],
                        serving_size=food_data["serving_size"],
                        confidence=confidence
                    )
                    nutrition_infos.append(nutrition_info)
                    
                    total_calories += food_data["calories"]
                    total_protein += food_data["protein"]
                    total_carbs += food_data["carbs"]
                    total_fat += food_data["fat"]
            
            # 식사 유형 판단
            meal_type = self._determine_meal_type(total_calories)
            
            # 건강 점수 계산
            health_score = self._calculate_health_score(
                total_calories, total_protein, total_carbs, total_fat
            )
            
            # 추천사항 생성
            recommendations = self._generate_recommendations(
                total_calories, total_protein, total_carbs, total_fat, meal_type
            )
            
            return MealAnalysisResult(
                foods=nutrition_infos,
                total_calories=total_calories,
                total_protein=total_protein,
                total_carbs=total_carbs,
                total_fat=total_fat,
                meal_type=meal_type,
                health_score=health_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Food analysis error: {e}")
            raise
    
    def _detect_foods(self, image: Image.Image) -> List[tuple]:
        """음식 감지 (시뮬레이션)"""
        # 실제로는 AI 모델 사용
        # 여기서는 랜덤하게 음식 감지 시뮬레이션
        
        import random
        
        # 시뮬레이션: 1-3개의 음식 랜덤 감지
        num_foods = random.randint(1, 3)
        food_list = list(FOOD_DATABASE.keys())
        
        detected = []
        for _ in range(num_foods):
            food = random.choice(food_list)
            confidence = random.uniform(0.7, 0.99)
            detected.append((food, confidence))
        
        return detected
    
    def _determine_meal_type(self, calories: float) -> str:
        """칼로리 기반 식사 유형 판단"""
        import datetime
        
        current_hour = datetime.datetime.now().hour
        
        if 5 <= current_hour < 10:
            return "breakfast"
        elif 11 <= current_hour < 14:
            return "lunch"
        elif 17 <= current_hour < 21:
            return "dinner"
        else:
            return "snack" if calories < 300 else "dinner"
    
    def _calculate_health_score(
        self, calories: float, protein: float, carbs: float, fat: float
    ) -> float:
        """건강 점수 계산"""
        score = 70.0  # 기본 점수
        
        # 칼로리 체크
        if 400 <= calories <= 700:
            score += 10
        elif calories > 1000:
            score -= 20
        elif calories < 200:
            score -= 10
        
        # 영양 균형 체크
        total_macros = protein + carbs + fat
        if total_macros > 0:
            protein_ratio = protein / total_macros
            carbs_ratio = carbs / total_macros
            fat_ratio = fat / total_macros
            
            # 이상적인 비율: 단백질 30%, 탄수화물 40%, 지방 30%
            if 0.25 <= protein_ratio <= 0.35:
                score += 10
            if 0.35 <= carbs_ratio <= 0.45:
                score += 5
            if 0.25 <= fat_ratio <= 0.35:
                score += 5
        
        return max(0, min(100, score))
    
    def _generate_recommendations(
        self, calories: float, protein: float, carbs: float, fat: float, meal_type: str
    ) -> List[str]:
        """영양 추천사항 생성"""
        recommendations = []
        
        # 칼로리 관련
        if calories > 800:
            recommendations.append("칼로리가 높습니다. 다음 식사는 가볍게 드세요")
        elif calories < 300 and meal_type != "snack":
            recommendations.append("칼로리가 부족합니다. 영양소를 더 섭취하세요")
        
        # 단백질 관련
        if protein < 15:
            recommendations.append("단백질이 부족합니다. 닭가슴살, 계란, 두부 등을 추가하세요")
        elif protein > 40:
            recommendations.append("단백질이 충분합니다")
        
        # 탄수화물 관련
        if carbs > 80:
            recommendations.append("탄수화물이 많습니다. 다음 식사에서는 줄여보세요")
        
        # 지방 관련
        if fat > 30:
            recommendations.append("지방이 많습니다. 포화지방 섭취를 주의하세요")
        
        # 야채 추천
        if not any("샐러드" in food for food in FOOD_DATABASE.keys()):
            recommendations.append("채소를 더 섭취하시면 좋겠습니다")
        
        # 수분 섭취
        recommendations.append("식사 후 충분한 물을 마시는 것을 잊지 마세요")
        
        return recommendations[:3]  # 최대 3개 추천

class WaterIntakeTracker:
    """물 섭취량 추적기"""
    
    def __init__(self):
        self.daily_goal = 2000  # ml
        self.glass_size = 250  # ml
    
    def calculate_hydration_score(self, water_intake_ml: int) -> Dict:
        """수분 섭취 점수 계산"""
        percentage = (water_intake_ml / self.daily_goal) * 100
        
        if percentage >= 100:
            score = 100
            status = "완벽해요! 💧"
            color = "#4CAF50"
        elif percentage >= 75:
            score = 85
            status = "좋아요!"
            color = "#8BC34A"
        elif percentage >= 50:
            score = 70
            status = "조금 더 마셔주세요"
            color = "#FFC107"
        else:
            score = 50
            status = "물을 더 마셔야 해요"
            color = "#FF9800"
        
        glasses_drunk = water_intake_ml // self.glass_size
        glasses_remaining = max(0, (self.daily_goal - water_intake_ml) // self.glass_size)
        
        return {
            "score": score,
            "percentage": min(100, percentage),
            "status": status,
            "color": color,
            "glasses_drunk": glasses_drunk,
            "glasses_remaining": glasses_remaining,
            "ml_drunk": water_intake_ml,
            "ml_remaining": max(0, self.daily_goal - water_intake_ml)
        }

class SleepAnalyzer:
    """수면 분석기"""
    
    def analyze_sleep_quality(
        self, 
        sleep_hours: float,
        deep_sleep_hours: float = None,
        rem_sleep_hours: float = None,
        wake_ups: int = 0
    ) -> Dict:
        """수면 품질 분석"""
        
        quality_score = 70  # 기본 점수
        recommendations = []
        
        # 수면 시간 평가
        if 7 <= sleep_hours <= 9:
            quality_score += 20
            sleep_duration_status = "적정"
        elif 6 <= sleep_hours < 7:
            quality_score += 10
            sleep_duration_status = "부족"
            recommendations.append("30분-1시간 더 자는 것을 목표로 하세요")
        elif sleep_hours < 6:
            quality_score -= 10
            sleep_duration_status = "매우 부족"
            recommendations.append("수면 시간이 부족합니다. 건강에 악영향을 줄 수 있어요")
        else:
            quality_score += 5
            sleep_duration_status = "과다"
            recommendations.append("너무 많이 자는 것도 좋지 않아요")
        
        # 깊은 수면 평가
        if deep_sleep_hours:
            deep_sleep_ratio = deep_sleep_hours / sleep_hours
            if 0.15 <= deep_sleep_ratio <= 0.25:
                quality_score += 10
            else:
                recommendations.append("깊은 수면 시간을 늘리기 위해 운동을 해보세요")
        
        # REM 수면 평가
        if rem_sleep_hours:
            rem_ratio = rem_sleep_hours / sleep_hours
            if 0.20 <= rem_ratio <= 0.25:
                quality_score += 10
            else:
                recommendations.append("REM 수면 개선을 위해 규칙적인 수면 패턴을 유지하세요")
        
        # 수면 중 깨어남 평가
        if wake_ups <= 1:
            quality_score += 10
        elif wake_ups >= 4:
            quality_score -= 10
            recommendations.append("수면 중 자주 깨어납니다. 수면 환경을 개선해보세요")
        
        # 수면 단계 계산
        light_sleep_hours = sleep_hours
        if deep_sleep_hours:
            light_sleep_hours -= deep_sleep_hours
        if rem_sleep_hours:
            light_sleep_hours -= rem_sleep_hours
        
        sleep_stages = {
            "deep": deep_sleep_hours or sleep_hours * 0.2,
            "rem": rem_sleep_hours or sleep_hours * 0.22,
            "light": light_sleep_hours
        }
        
        # 최종 점수 조정
        quality_score = max(0, min(100, quality_score))
        
        return {
            "quality_score": quality_score,
            "sleep_duration": sleep_hours,
            "sleep_duration_status": sleep_duration_status,
            "sleep_stages": sleep_stages,
            "wake_ups": wake_ups,
            "recommendations": recommendations[:3],
            "emoji": self._get_sleep_emoji(quality_score)
        }
    
    def _get_sleep_emoji(self, score: float) -> str:
        """점수에 따른 이모지 반환"""
        if score >= 90:
            return "😴💯"
        elif score >= 75:
            return "😊"
        elif score >= 60:
            return "😐"
        else:
            return "😫"

# Singleton instances
nutrition_analyzer = NutritionAnalyzer()
water_tracker = WaterIntakeTracker()
sleep_analyzer = SleepAnalyzer()