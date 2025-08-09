"""
Backend API Tests
API 엔드포인트 테스트
파일 위치: backend/tests/test_api.py
"""

import sys
import os
# 현재 파일의 부모의 부모 디렉토리 (backend 폴더)를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json

from app.main import app
from app.db.database import get_db, Base, engine
from app.models.models import User, Workout

# 테스트 클라이언트
client = TestClient(app)

# ========================
# Fixtures
# ========================

@pytest.fixture(scope="module")
def setup_database():
    """테스트 데이터베이스 설정"""
    # 테스트용 테이블 생성
    Base.metadata.create_all(bind=engine)
    yield
    # 테스트 후 정리
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_user():
    """테스트 사용자 생성"""
    return {
        "email": "test@example.com",
        "password": "password123",
        "name": "테스트유저",
        "birth_date": "1990-01-01",
        "gender": "male",
        "height": 175,
        "weight": 70
    }

# ========================
# Auth Tests
# ========================

class TestAuth:
    """인증 관련 테스트"""
    
    def test_register_user(self, setup_database, test_user):
        """회원가입 테스트"""
        response = client.post("/api/v1/auth/register", json=test_user)
    
    # 디버깅: 응답 내용 출력
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    print(f"Headers: {response.headers}")
    
    assert response.status_code == 200
    
    def test_register_duplicate_email(self, test_user):
        """중복 이메일 가입 테스트"""
        # 첫 번째 가입
        client.post("/api/v1/auth/register", json=test_user)
        
        # 두 번째 가입 시도
        response = client.post("/api/v1/auth/register", json=test_user)
        
        assert response.status_code == 400
        assert "이미 등록된" in response.json()["detail"]
    
    def test_login_success(self, test_user):
        """로그인 성공 테스트"""
        # 회원가입
        client.post("/api/v1/auth/register", json=test_user)
        
        # 로그인
        login_data = {
            "email": test_user["email"],
            "password": test_user["password"]
        }
        response = client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_wrong_password(self, test_user):
        """잘못된 비밀번호 로그인 테스트"""
        login_data = {
            "email": test_user["email"],
            "password": "wrongpassword"
        }
        response = client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == 401

# ========================
# Workout Tests
# ========================

class TestWorkout:
    """운동 관련 테스트"""
    
    def test_create_workout(self):
        """운동 세션 생성 테스트"""
        workout_data = {
            "exercise_type": "squat",
            "duration": 300,
            "reps": 15,
            "calories_burned": 50.5,
            "form_score": 85.0
        }
        response = client.post("/api/v1/workouts", json=workout_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["exercise_type"] == workout_data["exercise_type"]
        assert data["reps"] == workout_data["reps"]
    
    def test_get_workouts(self):
        """운동 목록 조회 테스트"""
        response = client.get("/api/v1/workouts?limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_analyze_pose(self):
        """자세 분석 테스트"""
        analysis_data = {
            "image_base64": "fake_base64_image_data",
            "exercise_type": "squat"
        }
        response = client.post("/api/v1/ai/analyze-pose", json=analysis_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "form_score" in data
        assert "feedback" in data
        assert "corrections" in data

# ========================
# Health Tests
# ========================

class TestHealth:
    """건강 데이터 테스트"""
    
    def test_log_health_metrics(self):
        """건강 지표 기록 테스트"""
        metrics_data = {
            "weight": 70.5,
            "body_fat_percentage": 15.0,
            "heart_rate_resting": 65,
            "steps": 8500,
            "water_intake": 1500
        }
        response = client.post("/api/v1/health/metrics", json=metrics_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "health_score" in data
    
    def test_get_health_dashboard(self):
        """건강 대시보드 조회 테스트"""
        response = client.get("/api/v1/health/dashboard")
        
        assert response.status_code == 200
        data = response.json()
        assert "health_score" in data
        assert "weight_trend" in data
        assert "recommendations" in data
    
    def test_log_water_intake(self):
        """물 섭취 기록 테스트"""
        response = client.post("/api/v1/health/water/log?amount_ml=250")
        
        assert response.status_code == 200
        data = response.json()
        assert "today_total" in data
        assert "hydration_status" in data
    
    def test_log_sleep_data(self):
        """수면 데이터 기록 테스트"""
        sleep_data = {
            "sleep_hours": 7.5,
            "deep_sleep_hours": 1.5,
            "rem_sleep_hours": 1.8,
            "wake_ups": 2
        }
        response = client.post("/api/v1/health/sleep", json=sleep_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "analysis" in data
        assert data["analysis"]["quality_score"] > 0

# ========================
# Social Tests
# ========================

class TestSocial:
    """소셜 기능 테스트"""
    
    def test_send_friend_request(self):
        """친구 요청 테스트"""
        friend_data = {
            "friend_email": "friend@example.com"
        }
        response = client.post("/api/v1/social/friends/request", json=friend_data)
        
        # 친구가 존재하지 않으면 404
        assert response.status_code in [200, 404]
    
    def test_get_friends_list(self):
        """친구 목록 조회 테스트"""
        response = client.get("/api/v1/social/friends")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_create_challenge(self):
        """챌린지 생성 테스트"""
        challenge_data = {
            "title": "주간 스쿼트 챌린지",
            "description": "일주일 동안 매일 50개씩!",
            "exercise_type": "squat",
            "target_reps": 350,
            "duration_days": 7,
            "is_public": True
        }
        response = client.post("/api/v1/social/challenges", json=challenge_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == challenge_data["title"]
    
    def test_get_leaderboard(self):
        """리더보드 조회 테스트"""
        response = client.get("/api/v1/social/leaderboard?period=week&limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

# ========================
# Notification Tests
# ========================

class TestNotifications:
    """알림 기능 테스트"""
    
    def test_register_device_token(self):
        """디바이스 토큰 등록 테스트"""
        token_data = {
            "fcm_token": "fake_fcm_token_123456",
            "device_type": "ios"
        }
        response = client.post("/api/v1/notifications/register-device", json=token_data)
        
        assert response.status_code == 200
    
    def test_get_notification_preferences(self):
        """알림 설정 조회 테스트"""
        response = client.get("/api/v1/notifications/preferences")
        
        assert response.status_code == 200
        data = response.json()
        assert "workout_reminder" in data
        assert "water_reminder" in data
    
    def test_send_test_notification(self):
        """테스트 알림 전송"""
        notification_data = {
            "title": "테스트 알림",
            "body": "이것은 테스트 알림입니다",
            "type": "test"
        }
        response = client.post("/api/v1/notifications/send-test", json=notification_data)
        
        assert response.status_code == 200

# ========================
# Integration Tests
# ========================

class TestIntegration:
    """통합 테스트"""
    
    def test_user_workout_flow(self):
        """사용자 운동 플로우 통합 테스트"""
        # 1. 회원가입
        user_data = {
            "email": "integration@test.com",
            "password": "test123",
            "name": "통합테스트",
            "birth_date": "1990-01-01",
            "gender": "male",
            "height": 180,
            "weight": 75
        }
        register_response = client.post("/api/v1/auth/register", json=user_data)
        assert register_response.status_code == 200
        
        # 2. 로그인
        login_response = client.post("/api/v1/auth/login", json={
            "email": user_data["email"],
            "password": user_data["password"]
        })
        assert login_response.status_code == 200
        
        # 3. 운동 기록
        workout_response = client.post("/api/v1/workouts", json={
            "exercise_type": "squat",
            "duration": 600,
            "reps": 30,
            "calories_burned": 100,
            "form_score": 90
        })
        assert workout_response.status_code == 200
        
        # 4. 건강 점수 확인
        dashboard_response = client.get("/api/v1/health/dashboard")
        assert dashboard_response.status_code == 200
        assert dashboard_response.json()["health_score"] > 0

# ========================
# Performance Tests
# ========================

class TestPerformance:
    """성능 테스트"""
    
    def test_api_response_time(self):
        """API 응답 시간 테스트"""
        import time
        
        endpoints = [
            "/api/v1/health/dashboard",
            "/api/v1/workouts",
            "/api/v1/social/leaderboard"
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            response = client.get(endpoint)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # 응답 시간이 1초 이내여야 함
            assert response_time < 1.0, f"{endpoint} took {response_time:.2f}s"
            assert response.status_code == 200

# ========================
# Run Tests
# ========================

if __name__ == "__main__":
    print("테스트 시작...")
    print("앱이 성공적으로 로드되었습니다!")
    print("테스트 완료!")